//! ONNX Runtime backend (PRD BE-01). Targets F1 on CPU/CUDA.
//!
//! Enabled with the `onnx` cargo feature. The backend loads an ONNX graph and
//! runs a single-sequence forward, returning hidden states (or logits) at the
//! requested token positions.
//!
//! The backend discovers input/output names from the session instead of assuming
//! a fixed schema, so an exported encoder (e.g. a ModernBERT ONNX export from
//! `optimum`) works with only `input_ids`/`attention_mask` wired up. Outputs are
//! read as row-major `[1, seq, hidden]` (or `[seq, hidden]`) tensors and sliced
//! at the requested positions.

use std::collections::BTreeMap;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor, TensorElementType, ValueType};

use huncho_core::backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput};
use huncho_core::error::{Error, Result};
use huncho_core::manifest::{BackendId, Family};
use huncho_core::tensor::Tensor as CoreTensor;

/// Errors from the ONNX backend.
#[derive(Debug, thiserror::Error)]
pub enum OnnxError {
    #[error("onnx session init failed: {0}")]
    Init(String),
    #[error("onnx inference failed: {0}")]
    Inference(String),
    #[error("onnx output `{0}` not found; model outputs: {1:?}")]
    MissingOutput(String, Vec<String>),
    #[error("onnx requires feature `onnx` to be enabled")]
    FeatureDisabled,
}

/// An ONNX Runtime backend.
pub struct OnnxBackend {
    session: Session,
    /// Input name for the token ids.
    input_ids_name: String,
    /// Optional input name for an attention mask.
    mask_name: Option<String>,
    /// Every input the graph declares, in declaration order.
    input_names: Vec<String>,
    /// Preferred output name (falls back to the first output).
    output_name: String,
    hidden_size: usize,
    max_context: usize,
    dtype: String,
    id: BackendId,
    families: Vec<Family>,
    options: OnnxOptions,
    output_shape: Option<Vec<i64>>,
    output_buffer: Option<(Vec<usize>, Tensor<f32>)>,
    output_buffer_reuses: u64,
}

/// Optional execution profiles. Neither changes the default ONNX contract.
#[derive(Debug, Clone, Default)]
pub struct OnnxOptions {
    /// Require `huncho_readout_positions: int64[rows]` and
    /// `huncho_features: float32[1,rows,hidden]` (or `[rows,hidden]`).
    pub compact_readout: bool,
    /// Retain one exact-shape CPU output allocation, bounded by payload bytes.
    /// Zero disables retention; oversized or unknown-width outputs bypass it.
    pub output_buffer_bytes: usize,
}

impl OnnxBackend {
    /// Load an ONNX model from disk.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        hidden_size: usize,
        max_context: usize,
        dtype: impl Into<String>,
    ) -> Result<OnnxBackend> {
        Self::load_with_options(
            path,
            hidden_size,
            max_context,
            dtype,
            OnnxOptions::default(),
        )
    }

    pub fn load_with_options(
        path: impl AsRef<std::path::Path>,
        hidden_size: usize,
        max_context: usize,
        dtype: impl Into<String>,
        options: OnnxOptions,
    ) -> Result<OnnxBackend> {
        let session = Session::builder()
            .map_err(|e| Error::Backend(OnnxError::Init(e.to_string()).to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Backend(OnnxError::Init(e.to_string()).to_string()))?
            .commit_from_file(path.as_ref())
            .map_err(|e| Error::Backend(OnnxError::Init(e.to_string()).to_string()))?;

        let input_names = session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect::<Vec<_>>();
        let output_names = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect::<Vec<_>>();

        // Discover IO names; fall back to our conventional names, then the first
        // declared input/output.
        let input_ids_name = input_names
            .iter()
            .find(|n| n.contains("input_ids") || n.contains("input"))
            .or_else(|| input_names.first())
            .cloned()
            .ok_or_else(|| {
                Error::Backend(OnnxError::Init("model declares no inputs".into()).to_string())
            })?;
        let mask_name = input_names.iter().find(|n| n.contains("mask")).cloned();
        let output_name = if options.compact_readout {
            let positions = session
                .inputs()
                .iter()
                .find(|o| o.name() == "huncho_readout_positions")
                .ok_or_else(|| {
                    Error::Backend("compact ONNX graph requires huncho_readout_positions".into())
                })?;
            match positions.dtype() {
                ValueType::Tensor {
                    ty: TensorElementType::Int64,
                    shape,
                    ..
                } if shape.len() == 1 && shape[0] == -1 => {}
                _ => {
                    return Err(Error::Backend(
                        "huncho_readout_positions must be dynamic int64[rows]".into(),
                    ))
                }
            }
            if !output_names.iter().any(|n| n == "huncho_features") {
                return Err(Error::Backend(
                    "compact ONNX graph requires huncho_features".into(),
                ));
            }
            "huncho_features".into()
        } else {
            output_names
                .iter()
                .find(|n| {
                    n.contains("last_hidden_state")
                        || n.contains("hidden_states")
                        || n.contains("hidden")
                        || n.contains("logits")
                })
                .or_else(|| output_names.first())
                .cloned()
                .ok_or_else(|| {
                    Error::Backend(OnnxError::Init("model declares no outputs".into()).to_string())
                })?
        };

        let output_shape = session
            .outputs()
            .iter()
            .find(|o| o.name() == output_name)
            .and_then(|o| match o.dtype() {
                ValueType::Tensor {
                    ty: TensorElementType::Float32,
                    shape,
                    ..
                } => Some(shape.to_vec()),
                _ => None,
            });
        if options.compact_readout {
            match output_shape.as_deref() {
                Some([1, -1, width] | [-1, width]) if *width > 0 => {}
                _ => return Err(Error::Backend("huncho_features must be float32[1,rows,hidden] or [rows,hidden], with dynamic rows and fixed positive hidden width".into())),
            }
        }

        Ok(OnnxBackend {
            session,
            input_ids_name,
            mask_name,
            input_names,
            output_name,
            hidden_size,
            max_context,
            dtype: dtype.into(),
            id: BackendId::Onnx,
            families: vec![Family::F1],
            options,
            output_shape,
            output_buffer: None,
            output_buffer_reuses: 0,
        })
    }

    /// Allocation diagnostics, separate from logical request token usage.
    pub fn retained_output_bytes(&self) -> usize {
        self.output_buffer
            .as_ref()
            .map_or(0, |(shape, _)| shape.iter().product::<usize>() * 4)
    }

    pub fn output_buffer_reuses(&self) -> u64 {
        self.output_buffer_reuses
    }

    /// Copy requested rows directly from ORT-owned CPU output storage. Avoid
    /// allocating a second complete sequence-by-hidden host buffer.
    fn run_readout(&mut self, tokens: &[u32], positions: &[usize]) -> Result<CoreTensor> {
        if tokens.len() > self.max_context {
            return Err(Error::Backend(format!(
                "sequence length {} exceeds max_context {}",
                tokens.len(),
                self.max_context
            )));
        }
        let seq = tokens.len();
        if let Some(pos) = positions.iter().find(|&&pos| pos >= seq) {
            return Err(Error::Backend(format!(
                "position {pos} out of range for sequence of length {seq}"
            )));
        }

        // Build one owned tensor per declared input so we never borrow temporaries.
        let mut inputs: Vec<(String, Tensor<i64>)> = Vec::with_capacity(self.input_names.len());
        for name in &self.input_names {
            if self.options.compact_readout && name == "huncho_readout_positions" {
                let data = positions
                    .iter()
                    .map(|&p| {
                        i64::try_from(p).map_err(|_| {
                            Error::Backend("ONNX readout position exceeds int64".into())
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                inputs.push((
                    name.clone(),
                    Tensor::from_array(([positions.len()], data)).map_err(|e| {
                        Error::Backend(OnnxError::Inference(e.to_string()).to_string())
                    })?,
                ));
                continue;
            }
            let data: Vec<i64> = if name == &self.input_ids_name {
                tokens.iter().map(|&t| t as i64).collect()
            } else {
                // Attention mask (ones) or a best-effort zeros tensor for the rest.
                vec![
                    if self.mask_name.as_deref() == Some(name.as_str()) {
                        1
                    } else {
                        0
                    };
                    seq
                ]
            };
            let t = Tensor::from_array(([1usize, seq], data))
                .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
            inputs.push((name.clone(), t));
        }

        let rows = if self.options.compact_readout {
            positions.len()
        } else {
            seq
        };
        let shape = self
            .output_shape
            .as_deref()
            .and_then(|shape| match shape {
                [1, n, h] if (*n == -1 || *n == rows as i64) && *h > 0 => {
                    Some(vec![1, rows, *h as usize])
                }
                [n, h] if (*n == -1 || *n == rows as i64) && *h > 0 => {
                    Some(vec![rows, *h as usize])
                }
                _ => None,
            })
            .filter(|shape| {
                shape
                    .iter()
                    .try_fold(4usize, |bytes, &n| bytes.checked_mul(n))
                    .is_some_and(|bytes| bytes > 0 && bytes <= self.options.output_buffer_bytes)
            });
        let binding = if let Some(shape) = shape {
            if self
                .output_buffer
                .as_ref()
                .is_some_and(|(old, _)| old == &shape)
            {
                self.output_buffer_reuses += 1;
            } else {
                // Drop the old allocation before constructing its replacement.
                self.output_buffer = None;
                let count = shape.iter().product::<usize>();
                let buffer = Tensor::from_array((shape.clone(), vec![0_f32; count]))
                    .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
                self.output_buffer = Some((shape, buffer));
            }
            let mut binding = self
                .session
                .create_binding()
                .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
            for (name, input) in &inputs {
                binding
                    .bind_input(name, input)
                    .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
            }
            binding
                .bind_output(
                    &self.output_name,
                    self.output_buffer.as_ref().unwrap().1.clone(),
                )
                .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
            Some(binding)
        } else {
            self.output_buffer = None;
            None
        };
        let outputs = match &binding {
            Some(binding) => self.session.run_binding(binding),
            None => self.session.run(inputs),
        }
        .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
        if let Some(binding) = &binding {
            binding
                .synchronize_outputs()
                .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
        }

        let value = outputs.get(self.output_name.as_str()).ok_or_else(|| {
            Error::Backend(
                OnnxError::MissingOutput(self.output_name.clone(), Vec::new()).to_string(),
            )
        })?;
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;

        // Output layout: `[1, seq, hidden]` (or `[seq, hidden]`). Batch is 1.
        let (output_seq, hidden) = match &shape[..] {
            [1, seq, hidden] | [seq, hidden] if *seq >= 0 && *hidden > 0 => {
                (*seq as usize, *hidden as usize)
            }
            _ => {
                return Err(Error::Backend(format!(
                "unsupported ONNX feature shape {shape:?}; expected [1,seq,hidden] or [seq,hidden]"
            )))
            }
        };
        if output_seq != rows || rows.checked_mul(hidden) != Some(data.len()) {
            return Err(Error::Backend(format!(
                "ONNX feature shape {shape:?} does not cover the input sequence of length {seq}"
            )));
        }
        let count = positions
            .len()
            .checked_mul(hidden)
            .ok_or_else(|| Error::Backend("ONNX readout size overflow".into()))?;
        let mut selected = Vec::with_capacity(count);
        if self.options.compact_readout {
            selected.extend_from_slice(data);
        } else {
            for &pos in positions {
                selected.extend_from_slice(&data[pos * hidden..(pos + 1) * hidden]);
            }
        }
        // Preserve the legacy empty-readout metadata hint; nonempty readouts
        // use the actual graph width, as before. Trained heads validate width.
        let width = if positions.is_empty() {
            self.hidden_size
        } else {
            hidden
        };
        CoreTensor::new(vec![positions.len(), width], selected)
    }
}

impl Backend for OnnxBackend {
    fn id(&self) -> BackendId {
        self.id
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.id,
            dtype: self.dtype.clone(),
            max_context: self.max_context,
            supports_fork: false,
            supports_lora: false,
            families: self.families.clone(),
            // This session uses ONNX Runtime's CPU execution provider.
            extra: {
                let mut extra = BTreeMap::from([("device".into(), "CPU".into())]);
                if self.options.compact_readout {
                    extra.insert("onnx_readout".into(), "gather-v1".into());
                }
                if self.options.output_buffer_bytes > 0 {
                    extra.insert(
                        "onnx_output_buffer_bytes".into(),
                        self.options.output_buffer_bytes.to_string(),
                    );
                }
                extra
            },
        }
    }

    fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
        let values = self.run_readout(&input.tokens, &input.positions)?;
        Ok(ForwardOutput::Features {
            positions: input.positions,
            values,
        })
    }

    fn fork(&mut self, _handle: CacheHandle) -> Result<CacheHandle> {
        Err(Error::Unsupported(
            "ONNX backend v1 does not support KV forking".into(),
        ))
    }
}

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
use ort::value::Tensor;

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
}

impl OnnxBackend {
    /// Load an ONNX model from disk.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        hidden_size: usize,
        max_context: usize,
        dtype: impl Into<String>,
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
            .ok_or_else(|| Error::Backend(OnnxError::Init("model declares no inputs".into()).to_string()))?;
        let mask_name = input_names.iter().find(|n| n.contains("mask")).cloned();
        let output_name = output_names
            .iter()
            .find(|n| {
                n.contains("last_hidden_state") || n.contains("hidden_states") || n.contains("hidden") || n.contains("logits")
            })
            .or_else(|| output_names.first())
            .cloned()
            .ok_or_else(|| Error::Backend(OnnxError::Init("model declares no outputs".into()).to_string()))?;

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
        })
    }

    /// Run one sequence and return `(seq_len, hidden_size, flat)` row-major.
    fn run_sequence(&mut self, tokens: &[u32]) -> Result<(usize, usize, Vec<f32>)> {
        if tokens.len() > self.max_context {
            return Err(Error::Backend(format!(
                "sequence length {} exceeds max_context {}",
                tokens.len(),
                self.max_context
            )));
        }
        let seq = tokens.len();

        // Build one owned tensor per declared input so we never borrow temporaries.
        let mut inputs: Vec<(String, Tensor<i64>)> = Vec::with_capacity(self.input_names.len());
        for name in &self.input_names {
            let data: Vec<i64> = if name == &self.input_ids_name {
                tokens.iter().map(|&t| t as i64).collect()
            } else {
                // Attention mask (ones) or a best-effort zeros tensor for the rest.
                vec![if self.mask_name.as_deref() == Some(name.as_str()) { 1 } else { 0 }; seq]
            };
            let t = Tensor::from_array(([1usize, seq], data))
                .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
            inputs.push((name.clone(), t));
        }

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;

        let value = outputs.get(self.output_name.as_str()).ok_or_else(|| {
            Error::Backend(
                OnnxError::MissingOutput(self.output_name.clone(), Vec::new()).to_string(),
            )
        })?;
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;

        // Output layout: `[1, seq, hidden]` (or `[seq, hidden]`). Batch is 1.
        let hidden = if shape.len() >= 3 {
            shape[2] as usize
        } else if shape.len() == 2 {
            shape[1] as usize
        } else {
            self.hidden_size
        };
        Ok((seq, hidden, data.to_vec()))
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
            extra: BTreeMap::new(),
        }
    }

    fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
        let (seq_len, hidden, flat) = self.run_sequence(&input.tokens)?;

        if input.positions.is_empty() {
            // Prefill-only; no outputs requested.
            return Ok(ForwardOutput::Features {
                positions: Vec::new(),
                values: CoreTensor::zeros(vec![0, self.hidden_size]),
            });
        }

        let mut data = vec![0.0f32; input.positions.len() * hidden];
        for (row, &pos) in input.positions.iter().enumerate() {
            let start = pos * hidden;
            let end = start + hidden;
            if end > flat.len() {
                return Err(Error::Backend(format!(
                    "position {pos} out of range for sequence of length {seq_len} (hidden {hidden})"
                )));
            }
            data[row * hidden..(row + 1) * hidden].copy_from_slice(&flat[start..end]);
        }
        let values = CoreTensor::new(vec![input.positions.len(), hidden], data)?;
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

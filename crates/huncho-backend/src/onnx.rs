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
use std::sync::Mutex;

use ort::session::{builder::GraphOptimizationLevel, IoBinding, Session};
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
    #[cfg(feature = "onnx-shared")]
    source: SessionSource,
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
    // IoBinding owns the one original output value. Tensor::clone() in the
    // pinned ORT crate makes a deep copy and must never bind a cached clone.
    output_buffer: Option<(Vec<usize>, Mutex<IoBinding>)>,
    output_buffer_reuses: u64,
    #[cfg(test)]
    expected_bound_output_address: Option<usize>,
    #[cfg(test)]
    last_native_output_address: Option<usize>,
}

/// A forward may return early after native execution/validation errors. Clear
/// bound inputs on every path so the persistent output binding never retains
/// a preceding request's token buffers or extends the output-only byte budget.
struct OutputBindingGuard<'a>(&'a mut IoBinding);
impl std::ops::Deref for OutputBindingGuard<'_> {
    type Target = IoBinding;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}
impl Drop for OutputBindingGuard<'_> {
    fn drop(&mut self) {
        self.0.clear_inputs();
    }
}

/// Optional execution profiles. Neither changes the default ONNX contract.
#[derive(Debug, Clone, Default)]
pub struct OnnxOptions {
    /// CPU FP32 integrated F1 scalar head: tokens[1,S], positions[N],
    /// qtype[1], optional attention_mask[1,S], and raw scores[N,1].
    /// Separate from generic feature, compact-gather and native-batch graphs.
    pub integrated_head: bool,
    /// Require `huncho_readout_positions: int64[rows]` and
    /// `huncho_features: float32[1,rows,hidden]` (or `[rows,hidden]`).
    pub compact_readout: bool,
    /// Retain one exact-shape CPU output allocation, bounded by payload bytes.
    /// Zero disables retention; oversized or unknown-width outputs bypass it.
    pub output_buffer_bytes: usize,
    pub execution_provider: OnnxExecutionProvider,
    /// Intra-op threads; zero preserves ORT's automatic choice (maximum 256).
    pub intra_threads: usize,
    /// Opt into native tensor batches on a validated dynamic feature graph.
    /// CPU graphs with an explicit attention_mask also support right-padding.
    /// Compact/integrated graphs use different contracts and cannot enable this.
    pub native_batch: bool,
    /// Snapshot a supported flat CPU graph and share initializer/prepack storage
    /// across independently owned sessions. Requires `onnx-shared`; default off.
    pub shared_initializers: bool,
}

#[derive(Clone)]
enum SessionSource {
    File(std::path::PathBuf),
    #[cfg(feature = "onnx-shared")]
    Shared(std::sync::Arc<crate::onnx_shared::SharedSource>),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OnnxExecutionProvider {
    #[default]
    Cpu,
    /// Strict CUDA placement: no CPU fallback, TF32 disabled.
    Cuda { device: i32 },
}

impl OnnxExecutionProvider {
    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim().to_ascii_lowercase();
        if value == "cpu" {
            return Ok(Self::Cpu);
        }
        if let Some(device) = value.strip_prefix("cuda:") {
            if let Ok(device) = device.parse::<i32>() {
                if device >= 0 {
                    return Ok(Self::Cuda { device });
                }
            }
        }
        Err(Error::Request(
            "ONNX provider must be cpu or cuda:NONNEGATIVE_DEVICE_ID".into(),
        ))
    }
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
        let dtype = dtype.into();
        if options.integrated_head
            && (dtype != "fp32"
                || options.execution_provider != OnnxExecutionProvider::Cpu
                || options.compact_readout
                || options.native_batch)
        {
            return Err(Error::Unsupported(
                "integrated ONNX F1 heads require CPU fp32 and cannot combine with feature-gather/native-batch contracts".into(),
            ));
        }
        if options.native_batch && options.compact_readout {
            return Err(Error::Unsupported(
                "ONNX native batches and compact graph readouts are separate contracts".into(),
            ));
        }
        if options.intra_threads > 256 {
            return Err(Error::Request(
                "ONNX intra-op threads must be from 0 to 256".into(),
            ));
        }
        match options.execution_provider {
            OnnxExecutionProvider::Cuda { device } if device < 0 => {
                return Err(Error::Request(
                    "ONNX CUDA device must be nonnegative".into(),
                ));
            }
            #[cfg(not(feature = "onnx-cuda"))]
            OnnxExecutionProvider::Cuda { .. } => {
                return Err(Error::Unsupported(
                    "ONNX CUDA requires the separate onnx-cuda feature".into(),
                ));
            }
            _ => {}
        }
        if options.shared_initializers && options.execution_provider != OnnxExecutionProvider::Cpu {
            return Err(Error::Unsupported(
                "shared ONNX initializers support CPU only".into(),
            ));
        }
        let source = if options.shared_initializers {
            #[cfg(feature = "onnx-shared")]
            {
                SessionSource::Shared(std::sync::Arc::new(crate::onnx_shared::SharedSource::load(
                    path.as_ref(),
                )?))
            }
            #[cfg(not(feature = "onnx-shared"))]
            {
                return Err(Error::Unsupported(
                    "shared ONNX initializers require onnx-shared".into(),
                ));
            }
        } else {
            SessionSource::File(path.as_ref().to_path_buf())
        };
        Self::load_source(source, hidden_size, max_context, dtype, options)
    }

    fn load_source(
        source: SessionSource,
        hidden_size: usize,
        max_context: usize,
        dtype: String,
        options: OnnxOptions,
    ) -> Result<Self> {
        let mut builder = Session::builder()
            .map_err(|e| Error::Backend(OnnxError::Init(e.to_string()).to_string()))?
            .with_no_environment_execution_providers()
            .map_err(|e| Error::Backend(OnnxError::Init(e.to_string()).to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Backend(OnnxError::Init(e.to_string()).to_string()))?;
        if options.intra_threads > 0 {
            builder = builder
                .with_independent_thread_pool()
                .map_err(|e| Error::Backend(OnnxError::Init(e.to_string()).to_string()))?
                .with_intra_threads(options.intra_threads)
                .map_err(|e| Error::Backend(OnnxError::Init(e.to_string()).to_string()))?;
        }
        #[cfg(feature = "onnx-cuda")]
        if let OnnxExecutionProvider::Cuda { device } = options.execution_provider {
            builder = builder
                .with_execution_providers([ort::ep::CUDA::default()
                    .with_device_id(device)
                    .with_tf32(false)
                    .build()
                    .error_on_failure()])
                .map_err(|e| Error::Backend(OnnxError::Init(e.to_string()).to_string()))?
                .with_disable_cpu_fallback()
                .map_err(|e| Error::Backend(OnnxError::Init(e.to_string()).to_string()))?;
        }
        let session = match &source {
            SessionSource::File(path) => builder.commit_from_file(path),
            #[cfg(feature = "onnx-shared")]
            SessionSource::Shared(shared) => {
                // Serialize session creation/prepacking only. Inference has no
                // shared session lock; every context owns its session/buffers.
                let _guard = shared
                    .creation
                    .lock()
                    .map_err(|_| Error::Backend("ONNX session creation lock poisoned".into()))?;
                for (name, value) in &shared.initializers {
                    if shared.external_names.contains(name) {
                        builder = builder
                            .with_external_initializer(name, value.clone())
                            .map_err(|e| {
                                Error::Backend(OnnxError::Init(e.to_string()).to_string())
                            })?;
                    }
                    builder = builder
                        .with_initializer(name, value.clone())
                        .map_err(|e| Error::Backend(OnnxError::Init(e.to_string()).to_string()))?;
                }
                builder = builder
                    .with_prepacked_weights(&shared.prepacked)
                    .map_err(|e| Error::Backend(OnnxError::Init(e.to_string()).to_string()))?;
                builder.commit_from_memory(&shared.model)
            }
        }
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
        let input_ids_name = if options.integrated_head {
            "tokens".into()
        } else {
            input_names
                .iter()
                .find(|n| n.contains("input_ids") || n.contains("input"))
                .or_else(|| input_names.first())
                .cloned()
                .ok_or_else(|| {
                    Error::Backend(OnnxError::Init("model declares no inputs".into()).to_string())
                })?
        };
        let mask_name = input_names.iter().find(|n| n.contains("mask")).cloned();
        let output_name = if options.integrated_head {
            let required = ["tokens", "positions", "qtype"];
            if required
                .iter()
                .any(|name| !input_names.iter().any(|input| input == name))
                || input_names.len()
                    != 3 + usize::from(input_names.iter().any(|name| name == "attention_mask"))
                || output_names != ["scores"]
            {
                return Err(Error::Unsupported("integrated ONNX head requires tokens, positions, qtype, optional attention_mask and only scores".into()));
            }
            for input in session.inputs() {
                let expected: &[i64] = match input.name() {
                    "tokens" | "attention_mask" => &[1, -1],
                    "positions" => &[-1],
                    "qtype" => &[1],
                    _ => {
                        return Err(Error::Unsupported(
                            "unknown integrated ONNX head input".into(),
                        ))
                    }
                };
                if !matches!(input.dtype(), ValueType::Tensor { ty: TensorElementType::Int64, shape, .. } if &shape[..] == expected)
                {
                    return Err(Error::Unsupported(format!(
                        "integrated ONNX input {} must be int64{expected:?}",
                        input.name()
                    )));
                }
            }
            "scores".into()
        } else if options.compact_readout {
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
        } else if options.native_batch {
            if input_ids_name != "input_ids"
                || !output_names.iter().any(|n| n == "last_hidden_state")
            {
                return Err(Error::Unsupported(
                    "ONNX native batching requires input_ids and last_hidden_state".into(),
                ));
            }
            "last_hidden_state".into()
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
        if options.integrated_head && output_shape.as_deref() != Some(&[-1, 1]) {
            return Err(Error::Unsupported(
                "integrated ONNX head scores must be dynamic float32[markers,1]".into(),
            ));
        }
        if options.compact_readout {
            match output_shape.as_deref() {
                Some([1, -1, width] | [-1, width]) if *width > 0 => {}
                _ => return Err(Error::Backend("huncho_features must be float32[1,rows,hidden] or [rows,hidden], with dynamic rows and fixed positive hidden width".into())),
            }
        }
        if options.native_batch {
            for input in session.inputs() {
                let supported = matches!(
                    input.name(),
                    "input_ids" | "attention_mask" | "token_type_ids" | "position_ids"
                );
                if !supported
                    || !matches!(input.dtype(), ValueType::Tensor {
                    ty: TensorElementType::Int64, shape, ..
                } if shape[..] == [-1, -1])
                {
                    return Err(Error::Unsupported(format!("ONNX batching requires known int64[batch,seq] inputs with dynamic batch/seq; unsupported input {}", input.name())));
                }
            }
            if !matches!(output_shape.as_deref(), Some([-1, -1, width]) if *width > 0) {
                return Err(Error::Unsupported("ONNX batching requires float32[batch,seq,hidden] with dynamic batch/seq and fixed positive hidden width".into()));
            }
        }

        Ok(OnnxBackend {
            session,
            #[cfg(feature = "onnx-shared")]
            source,
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
            #[cfg(test)]
            expected_bound_output_address: None,
            #[cfg(test)]
            last_native_output_address: None,
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
    fn run_readouts(
        &mut self,
        requests: &[(&[u32], &[usize])],
        qtype: Option<u32>,
        padded: bool,
    ) -> Result<Vec<CoreTensor>> {
        let batch = requests.len();
        if batch == 0 || batch > 64 || (batch > 1 && !self.options.native_batch) {
            return Err(Error::Unsupported(
                "ONNX readouts require 1..64 rows and a native-batch profile for multiple rows"
                    .into(),
            ));
        }
        if padded && !self.supports_padded_batch() {
            return Err(Error::Unsupported(
                "padded ONNX batches require a native CPU graph with attention_mask".into(),
            ));
        }
        let seq = if padded {
            requests
                .iter()
                .map(|(tokens, _)| tokens.len())
                .max()
                .unwrap()
        } else {
            requests[0].0.len()
        };
        if self.options.integrated_head && (seq == 0 || qtype.is_none_or(|q| q > 2)) {
            return Err(Error::Backend(
                "integrated ONNX head needs nonempty tokens and qtype 0..2".into(),
            ));
        }
        if seq > self.max_context {
            return Err(Error::Backend(format!(
                "sequence length {} exceeds max_context {}",
                seq, self.max_context
            )));
        }
        for &(tokens, positions) in requests {
            if padded && tokens.is_empty() {
                return Err(Error::Backend("padded ONNX rows must be nonempty".into()));
            }
            if !padded && tokens.len() != seq {
                return Err(Error::Unsupported(
                    "ONNX native batches require equal sequence lengths without padding".into(),
                ));
            }
            if let Some(pos) = positions.iter().find(|&&pos| pos >= tokens.len()) {
                return Err(Error::Backend(format!(
                    "position {pos} out of range for sequence of length {}",
                    tokens.len()
                )));
            }
        }
        let positions = requests[0].1;
        if self.options.integrated_head && positions.is_empty() {
            return Ok(vec![CoreTensor::zeros(vec![0, 1])]);
        }
        let input_count = batch
            .checked_mul(seq)
            .ok_or_else(|| Error::Backend("ONNX batch input size overflow".into()))?;

        // Build one owned tensor per declared input so we never borrow temporaries.
        let mut inputs: Vec<(String, Tensor<i64>)> = Vec::with_capacity(self.input_names.len());
        for name in &self.input_names {
            if self.options.integrated_head {
                let (shape, data): (Vec<usize>, Vec<i64>) = match name.as_str() {
                    "tokens" => (
                        vec![1, seq],
                        requests[0]
                            .0
                            .iter()
                            .map(|&token| i64::from(token))
                            .collect(),
                    ),
                    "attention_mask" => (vec![1, seq], vec![1; seq]),
                    "positions" => (
                        vec![positions.len()],
                        positions
                            .iter()
                            .map(|&position| {
                                i64::try_from(position).map_err(|_| {
                                    Error::Backend("integrated ONNX marker exceeds int64".into())
                                })
                            })
                            .collect::<Result<_>>()?,
                    ),
                    "qtype" => (vec![1], vec![i64::from(qtype.unwrap())]),
                    _ => unreachable!("validated integrated graph inputs"),
                };
                let tensor = Tensor::from_array((shape, data))
                    .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
                inputs.push((name.clone(), tensor));
                continue;
            }
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
                requests
                    .iter()
                    .flat_map(|(tokens, _)| {
                        tokens
                            .iter()
                            .map(|&t| i64::from(t))
                            .chain(std::iter::repeat(0))
                            .take(seq)
                    })
                    .collect()
            } else if padded && name == "attention_mask" {
                requests
                    .iter()
                    .flat_map(|(tokens, _)| (0..seq).map(move |p| i64::from(p < tokens.len())))
                    .collect()
            } else if self.options.native_batch && name == "position_ids" {
                (0..batch)
                    .flat_map(|_| (0..seq).map(|p| p as i64))
                    .collect()
            } else {
                // Attention mask (ones) or a best-effort zeros tensor for the rest.
                vec![
                    if self.mask_name.as_deref() == Some(name.as_str()) {
                        1
                    } else {
                        0
                    };
                    input_count
                ]
            };
            let t = Tensor::from_array(([batch, seq], data))
                .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
            inputs.push((name.clone(), t));
        }

        let rows = if self.options.compact_readout || self.options.integrated_head {
            positions.len()
        } else {
            seq
        };
        let shape = self
            .output_shape
            .as_deref()
            .and_then(|shape| match shape {
                [b, n, h]
                    if (*b == -1 || *b == batch as i64)
                        && (*n == -1 || *n == rows as i64)
                        && *h > 0 =>
                {
                    Some(vec![batch, rows, *h as usize])
                }
                [n, h] if batch == 1 && (*n == -1 || *n == rows as i64) && *h > 0 => {
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
                #[cfg(test)]
                {
                    self.expected_bound_output_address = Some(buffer.data_ptr() as usize);
                }
                let mut binding = self
                    .session
                    .create_binding()
                    .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
                binding
                    .bind_output(&self.output_name, buffer)
                    .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
                self.output_buffer = Some((shape, Mutex::new(binding)));
            }
            let binding = self
                .output_buffer
                .as_mut()
                .unwrap()
                .1
                .get_mut()
                .map_err(|_| Error::Backend("ONNX output binding lock poisoned".into()))?;
            let guard = OutputBindingGuard(binding);
            for (name, input) in &inputs {
                guard
                    .0
                    .bind_input(name, input)
                    .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
            }
            Some(guard)
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
        #[cfg(test)]
        {
            self.last_native_output_address = Some(data.as_ptr() as usize);
        }

        let (output_seq, hidden) = match &shape[..] {
            [b, seq, hidden] if *b == batch as i64 && *seq >= 0 && *hidden > 0 => {
                (*seq as usize, *hidden as usize)
            }
            [seq, hidden] if batch == 1 && *seq >= 0 && *hidden > 0 => {
                (*seq as usize, *hidden as usize)
            }
            _ => {
                return Err(Error::Backend(format!(
                    "unsupported ONNX feature shape {shape:?}; expected [{batch},rows,hidden]"
                )))
            }
        };
        if output_seq != rows
            || rows.checked_mul(hidden).and_then(|n| n.checked_mul(batch)) != Some(data.len())
        {
            return Err(Error::Backend(format!(
                "ONNX feature shape {shape:?} does not cover the input sequence of length {seq}"
            )));
        }
        if self.options.integrated_head
            && (hidden != 1 || data.iter().any(|value| !value.is_finite()))
        {
            return Err(Error::Backend(
                "integrated ONNX head returned nonfinite or nonscalar marker scores".into(),
            ));
        }
        requests
            .iter()
            .enumerate()
            .map(|(index, (_, positions))| {
                let count = positions
                    .len()
                    .checked_mul(hidden)
                    .ok_or_else(|| Error::Backend("ONNX readout size overflow".into()))?;
                let mut selected = Vec::with_capacity(count);
                if self.options.compact_readout || self.options.integrated_head {
                    selected.extend_from_slice(data);
                } else {
                    for &pos in *positions {
                        let offset = (index * seq + pos) * hidden;
                        selected.extend_from_slice(&data[offset..offset + hidden]);
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
            })
            .collect()
    }
}

#[cfg(test)]
mod output_binding_tests {
    use super::*;

    #[test]
    fn actual_native_output_uses_the_retained_allocation_and_preserves_owned_answers() {
        let fixture = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tiny_encoder.onnx"
        );
        let mut backend = OnnxBackend::load_with_options(
            fixture,
            8,
            512,
            "fp32",
            OnnxOptions {
                output_buffer_bytes: 1024,
                ..Default::default()
            },
        )
        .unwrap();
        let first = backend
            .forward(ForwardInput::new(vec![3, 10, 5], vec![2, 0, 2]))
            .unwrap();
        let frozen: Vec<_> = first.values().data().iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            backend.last_native_output_address, backend.expected_bound_output_address,
            "the bound output must use the original preallocated storage"
        );
        let address = backend.last_native_output_address.unwrap();
        for tokens in [vec![5, 3, 10], vec![10, 5, 3], vec![3, 10, 5]] {
            backend
                .forward(ForwardInput::new(tokens, vec![0, 2, 1]))
                .unwrap();
            assert_eq!(backend.last_native_output_address, Some(address));
            assert_eq!(
                backend.last_native_output_address,
                backend.expected_bound_output_address
            );
            assert_eq!(
                first
                    .values()
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                frozen
            );
        }
        assert_eq!(backend.output_buffer_reuses(), 3);
        assert_eq!(backend.retained_output_bytes(), 96);
    }

    #[test]
    fn retained_binding_clears_request_inputs_after_success_and_output_validation_failure() {
        for (graph, integrated_head, hidden_size) in [
            ("tiny_encoder.onnx", false, 8),
            ("integrated_f1/nonfinite.onnx", true, 4),
        ] {
            let root = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"));
            let mut backend = OnnxBackend::load_with_options(
                root.join(graph),
                hidden_size,
                512,
                "fp32",
                OnnxOptions {
                    integrated_head,
                    output_buffer_bytes: 1024,
                    intra_threads: 1,
                    ..Default::default()
                },
            )
            .unwrap();
            let input = if integrated_head {
                let reference: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(root.join("integrated_f1/reference.json")).unwrap(),
                )
                .unwrap();
                let row = &reference["readouts"][0][0];
                ForwardInput::new(
                    serde_json::from_value(row["tokens"].clone()).unwrap(),
                    serde_json::from_value(row["positions"].clone()).unwrap(),
                )
                .with_qtype(row["qtype"].as_u64().unwrap() as u32)
            } else {
                ForwardInput::new(vec![3, 10, 5], vec![2, 0])
            };
            let result = backend.forward(input.clone());
            assert_eq!(result.is_err(), integrated_head);
            if integrated_head {
                assert!(result.err().unwrap().to_string().contains("nonfinite"));
            }
            assert!(backend.retained_output_bytes() > 0);
            // The output allocation remains bound, but the native runtime must
            // refuse a run without new inputs instead of retaining token data.
            let binding = backend.output_buffer.as_mut().unwrap().1.get_mut().unwrap();
            assert!(backend.session.run_binding(binding).is_err());
            let retry = backend.forward(input);
            assert_eq!(retry.is_err(), integrated_head);
        }
    }
}

impl Backend for OnnxBackend {
    fn replica(&self) -> Result<Box<dyn Backend>> {
        #[cfg(feature = "onnx-shared")]
        if matches!(&self.source, SessionSource::Shared(_)) {
            return Ok(Box::new(Self::load_source(
                self.source.clone(),
                self.hidden_size,
                self.max_context,
                self.dtype.clone(),
                self.options.clone(),
            )?));
        }
        Err(Error::Unsupported(
            "ONNX CPU replicas require shared-initializer loading (onnx-shared)".into(),
        ))
    }

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
            extra: {
                let device = match self.options.execution_provider {
                    OnnxExecutionProvider::Cpu => "CPU".into(),
                    OnnxExecutionProvider::Cuda { device } => format!("GPU (CUDA device {device})"),
                };
                let mut extra = BTreeMap::from([
                    ("device".into(), device),
                    ("native_execution".into(), "onnxruntime-v1".into()),
                ]);
                #[cfg(feature = "onnx-shared")]
                if let SessionSource::Shared(shared) = &self.source {
                    extra.insert(
                        "onnx_initializer_residency".into(),
                        "immutable-cpu-v1".into(),
                    );
                    extra.insert(
                        "onnx_shared_initializer_bytes".into(),
                        shared.bytes.to_string(),
                    );
                    extra.insert("onnx_model_snapshot_sha256".into(), shared.sha256.clone());
                }
                if matches!(
                    self.options.execution_provider,
                    OnnxExecutionProvider::Cuda { .. }
                ) {
                    extra.insert(
                        "onnx_execution_provider".into(),
                        "cuda-strict-tf32-off-v1".into(),
                    );
                }
                if self.options.intra_threads > 0 {
                    extra.insert(
                        "onnx_intra_threads".into(),
                        self.options.intra_threads.to_string(),
                    );
                }
                if self.options.native_batch {
                    extra.insert("onnx_native_batch".into(), "equal-length-v1".into());
                }
                if self.supports_padded_batch() {
                    extra.insert(
                        "padded_batch_execution".into(),
                        "onnx-cpu-right-mask-v1".into(),
                    );
                }
                if self.options.compact_readout {
                    extra.insert("onnx_readout".into(), "gather-v1".into());
                }
                if self.options.integrated_head {
                    extra.insert(
                        "onnx_integrated_head".into(),
                        "graph-integrated-f1-v1".into(),
                    );
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
        if self.options.integrated_head && (input.retain_cache || input.logit_codes.is_some()) {
            return Err(Error::Unsupported(
                "integrated ONNX F1 heads do not support cache retention or vocabulary codes"
                    .into(),
            ));
        }
        if input.fork_from.is_some() {
            return Err(Error::Unsupported(
                "ONNX does not support cached forwards".into(),
            ));
        }
        let values = self
            .run_readouts(
                &[(&input.tokens, &input.positions)],
                Some(input.qtype),
                false,
            )?
            .remove(0);
        if self.options.integrated_head {
            return Ok(ForwardOutput::Logits {
                positions: input.positions,
                values,
            });
        }
        Ok(ForwardOutput::Features {
            positions: input.positions,
            values,
        })
    }

    fn supports_batch(&self) -> bool {
        self.options.native_batch
    }

    fn forward_batch(&mut self, inputs: Vec<ForwardInput>) -> Result<Vec<ForwardOutput>> {
        if !self.options.native_batch
            || inputs.iter().any(|input| {
                input.fork_from.is_some() || input.retain_cache || input.logit_codes.is_some()
            })
        {
            return Err(Error::Unsupported(
                "ONNX native batching needs an enabled dynamic graph and uncached inputs".into(),
            ));
        }
        let requests = inputs
            .iter()
            .map(|input| (input.tokens.as_slice(), input.positions.as_slice()))
            .collect::<Vec<_>>();
        let values = self.run_readouts(&requests, None, false)?;
        Ok(inputs
            .into_iter()
            .zip(values)
            .map(|(input, values)| ForwardOutput::Features {
                positions: input.positions,
                values,
            })
            .collect())
    }

    fn supports_padded_batch(&self) -> bool {
        self.options.native_batch
            && self.options.execution_provider == OnnxExecutionProvider::Cpu
            && self.mask_name.as_deref() == Some("attention_mask")
    }

    fn forward_padded_batch(&mut self, inputs: Vec<ForwardInput>) -> Result<Vec<ForwardOutput>> {
        if !self.supports_padded_batch()
            || inputs.iter().any(|input| {
                input.fork_from.is_some() || input.retain_cache || input.logit_codes.is_some()
            })
        {
            return Err(Error::Unsupported(
                "padded ONNX batches require independent CPU feature rows with attention_mask and no cache retention or vocabulary codes".into(),
            ));
        }
        let requests = inputs
            .iter()
            .map(|input| (input.tokens.as_slice(), input.positions.as_slice()))
            .collect::<Vec<_>>();
        let values = self.run_readouts(&requests, None, true)?;
        Ok(inputs
            .into_iter()
            .zip(values)
            .map(|(input, values)| ForwardOutput::Features {
                positions: input.positions,
                values,
            })
            .collect())
    }

    fn fork(&mut self, _handle: CacheHandle) -> Result<CacheHandle> {
        Err(Error::Unsupported(
            "ONNX backend v1 does not support KV forking".into(),
        ))
    }
}

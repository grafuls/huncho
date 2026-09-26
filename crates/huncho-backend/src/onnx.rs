//! ONNX Runtime backend (PRD BE-01). Targets F1 on CPU/CUDA.
//!
//! Enabled with the `onnx` cargo feature. The backend loads an ONNX graph and
//! runs a single-sequence forward, returning hidden states (or logits) at the
//! requested token positions.

use std::collections::BTreeMap;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

use huncho_core::backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput};
use huncho_core::error::{Error, Result};
use huncho_core::manifest::{BackendId, Family};
use huncho_core::tensor::Tensor;

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
    output_names: Vec<String>,
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
        let names = session
            .inputs()
            .iter()
            .map(|i| i.name.clone())
            .collect::<Vec<_>>();
        // Heuristic: the last input is usually `input_ids`.
        let output_names = vec!["last_hidden_state".to_string()];
        Ok(OnnxBackend {
            session,
            output_names,
            hidden_size,
            max_context,
            dtype: dtype.into(),
            id: BackendId::Onnx,
            families: vec![Family::F1],
        })
    }

    fn run_sequence(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.len() > self.max_context {
            return Err(Error::Backend(format!(
                "sequence length {} exceeds max_context {}",
                tokens.len(),
                self.max_context
            )));
        }
        let shape = [1usize, tokens.len()];
        let input = TensorRef::from_array_view(&shape, &tokens)
            .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
        let outputs = self
            .session
            .run(ort::inputs!["input_ids" => input])
            .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
        // Return the first f32 output tensor's data.
        let (_, value) = outputs.iter().next().ok_or_else(|| {
            Error::Backend(OnnxError::MissingOutput("any".into(), self.output_names.clone()).to_string())
        })?;
        let arr = value
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Backend(OnnxError::Inference(e.to_string()).to_string()))?;
        Ok(arr.view().iter().copied().collect())
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
        if !input.positions.is_empty() {
            // Read hidden states at the requested positions.
            let flat = self.run_sequence(&input.tokens)?;
            let seq_len = input.tokens.len();
            let n = input.positions.len();
            let mut data = vec![0.0f32; n * self.hidden_size];
            for (row, &pos) in input.positions.iter().enumerate() {
                let start = pos * self.hidden_size;
                let end = start + self.hidden_size;
                if end <= flat.len() {
                    data[row * self.hidden_size..(row + 1) * self.hidden_size]
                        .copy_from_slice(&flat[start..end]);
                }
            }
            let values = Tensor::new(vec![n, self.hidden_size], data)?;
            Ok(ForwardOutput::Features {
                positions: input.positions,
                values,
            })
        } else {
            // Prefill-only; no outputs requested.
            let _ = self.run_sequence(&input.tokens)?;
            Ok(ForwardOutput::Features {
                positions: Vec::new(),
                values: Tensor::zeros(vec![0, self.hidden_size]),
            })
        }
    }

    fn fork(&mut self, _handle: CacheHandle) -> Result<CacheHandle> {
        Err(Error::Unsupported(
            "ONNX backend v1 does not support KV forking".into(),
        ))
    }
}

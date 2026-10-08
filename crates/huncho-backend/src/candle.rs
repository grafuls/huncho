//! Candle backend — the primary real-model path for F1 (PRD BE-02).
//!
//! Loads a Hugging Face `safetensors` ModernBERT (the family behind
//! `convaiinnovations/laya`) directly with [`candle_core`], so an F1 model can
//! be served without an ONNX export or a Python/`optimum` toolchain. The
//! backend runs the encoder and returns hidden states (`ForwardOutput::Features`)
//! at the requested token positions, which the F1 head scores.
//!
//! Enabled with the `candle` cargo feature.
//!
//! Checkpoint compatibility:
//! * Weights may use either an `encoder.` prefix (as `convaiinnovations/laya`
//!   does) or a `model.` prefix (standard HF ModernBERT); `encoder.` is remapped
//!   to `model.` at load time so [`candle_transformers::models::modernbert`]
//!   finds them. Non-encoder tensors (e.g. `act_head.*`, `temperature`) are
//!   ignored.
//! * Weights may be `f16`/`bf16`; they are converted to `f32` for CPU inference.
//! * `config.json` may use the flat `global_rope_theta`/`local_rope_theta`
//!   fields or the newer transformers-5.0 `rope_parameters` object, which is
//!   normalized before parsing.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert::{Config, ModernBert};

use huncho_core::backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput};
use huncho_core::error::{Error, Result};
use huncho_core::manifest::{BackendId, Family};
use huncho_core::tensor::Tensor as CoreTensor;

/// Errors from the candle backend.
#[derive(Debug, thiserror::Error)]
pub enum CandleError {
    #[error("failed to read config `{0}`: {1}")]
    Config(String, String),
    #[error("failed to load `{0}`: {1}")]
    Load(String, String),
    #[error("candle inference failed: {0}")]
    Inference(String),
    #[error("candle requires feature `candle` to be enabled")]
    FeatureDisabled,
}

/// A candle-backed ModernBERT encoder, optionally with a Laya decision head.
pub struct CandleBackend {
    model: ModernBert,
    head: Option<LayaHead>,
    hidden_size: usize,
    vocab_size: usize,
    max_context: usize,
    dtype: String,
    id: BackendId,
    families: Vec<Family>,
    device: Device,
}

impl CandleBackend {
    /// Load a ModernBERT encoder from a `config.json` and a `model.safetensors`.
    pub fn load(
        config_path: impl AsRef<Path>,
        weights_path: impl AsRef<Path>,
        max_context: usize,
        dtype: impl Into<String>,
    ) -> Result<CandleBackend> {
        Self::load_on_device(config_path, weights_path, max_context, dtype, Device::Cpu)
    }

    /// Explicit device path; FP32 remains the actual computation/storage dtype.
    /// CPU staging converts BF16 checkpoints before transfer on older GPUs.
    pub fn load_on_device(
        config_path: impl AsRef<Path>,
        weights_path: impl AsRef<Path>,
        max_context: usize,
        dtype: impl Into<String>,
        device: Device,
    ) -> Result<CandleBackend> {
        let dtype = dtype.into();
        if dtype != "fp32" {
            return Err(Error::Unsupported(format!(
                "ModernBERT currently executes in fp32; cannot label it `{dtype}` for calibration"
            )));
        }
        let config = parse_config(config_path.as_ref())?;
        let hidden_size = config.hidden_size;

        let tensors = load_encoder_tensors(weights_path.as_ref(), &device).map_err(|e| {
            Error::Backend(
                CandleError::Load(weights_path.as_ref().display().to_string(), e.to_string())
                    .to_string(),
            )
        })?;

        let head = LayaHead::from_tensors(&tensors, hidden_size)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = ModernBert::load(vb, &config).map_err(|e| {
            Error::Backend(CandleError::Load("weights".into(), e.to_string()).to_string())
        })?;

        Ok(CandleBackend {
            model,
            head,
            hidden_size,
            vocab_size: config.vocab_size,
            max_context,
            dtype,
            id: BackendId::Candle,
            families: vec![Family::F1],
            device,
        })
    }

    /// The hidden size reported by the loaded model.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

/// Read `config.json` and normalize it into a [`modernbert::Config`].
///
/// Handles both the flat rope-theta layout and the transformers-5.0
/// `rope_parameters` object, and tolerates a null `pad_token_id`.
fn parse_config(path: &Path) -> Result<Config> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        Error::Backend(CandleError::Config(path.display().to_string(), e.to_string()).to_string())
    })?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        Error::Backend(CandleError::Config(path.display().to_string(), e.to_string()).to_string())
    })?;
    config_from_value(&v, path)
}

fn config_from_value(v: &serde_json::Value, path: &Path) -> Result<Config> {
    fn num(v: &serde_json::Value, key: &str, default: f64) -> f64 {
        v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
    }
    fn uint(v: &serde_json::Value, key: &str, default: usize) -> usize {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(default)
    }

    let default_theta = 10_000.0;
    let global_rope_theta = v
        .get("global_rope_theta")
        .and_then(|x| x.as_f64())
        .or_else(|| {
            v.pointer("/rope_parameters/full_attention/rope_theta")
                .and_then(|x| x.as_f64())
        })
        .unwrap_or(default_theta);
    let local_rope_theta = v
        .get("local_rope_theta")
        .and_then(|x| x.as_f64())
        .or_else(|| {
            v.pointer("/rope_parameters/sliding_attention/rope_theta")
                .and_then(|x| x.as_f64())
        })
        .unwrap_or(default_theta);

    let pad_token_id = v.get("pad_token_id").and_then(|x| x.as_u64()).unwrap_or(0) as u32;

    let normalized = serde_json::json!({
        "vocab_size": uint(v, "vocab_size", 0),
        "hidden_size": uint(v, "hidden_size", 0),
        "num_hidden_layers": uint(v, "num_hidden_layers", 12),
        "num_attention_heads": uint(v, "num_attention_heads", 12),
        "intermediate_size": uint(v, "intermediate_size", 3072),
        "max_position_embeddings": uint(v, "max_position_embeddings", 8192),
        "layer_norm_eps": num(v, "layer_norm_eps", 1e-5),
        "pad_token_id": pad_token_id,
        "global_attn_every_n_layers": uint(v, "global_attn_every_n_layers", 3),
        "global_rope_theta": global_rope_theta,
        "local_attention": uint(v, "local_attention", 128),
        "local_rope_theta": local_rope_theta,
    });

    serde_json::from_value(normalized).map_err(|e| {
        Error::Backend(CandleError::Config(path.display().to_string(), e.to_string()).to_string())
    })
}

/// Map a safetensors key to the name candle's ModernBERT expects.
///
/// * `encoder.*` (as `convaiinnovations/laya` uses) is remapped to `model.*`.
/// * `model.*` (standard HF ModernBERT) is kept as-is.
/// * Laya decision-head keys (`head.*`, `type_emb.*`, `scorer.*`) are kept
///   as-is so the backend can run the full typed option-marker head.
/// * Anything else (e.g. `act_head.*`, `temperature`) is dropped — `temperature`
///   is applied by huncho-core calibration, and `act_head` carries no useful
///   signal.
fn remap_key(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("encoder.") {
        Some(format!("model.{rest}"))
    } else if name.starts_with("model.")
        || name.starts_with("head.")
        || name.starts_with("type_emb.")
        || name.starts_with("scorer.")
    {
        Some(name.to_string())
    } else {
        None
    }
}

/// Load the model tensors from a safetensors file, remapping the `encoder.`
/// prefix to `model.` and retaining the Laya decision-head tensors. Tensors are
/// converted to `f32` on CPU before transfer to the execution device.
fn load_encoder_tensors(path: &Path, device: &Device) -> candle::Result<HashMap<String, Tensor>> {
    let raw = candle::safetensors::load(path, &Device::Cpu)?;
    let mut out = HashMap::new();
    for (name, t) in raw {
        let Some(mapped) = remap_key(&name) else {
            continue;
        };
        let t = t.to_dtype(DType::F32)?.to_device(device)?;
        out.insert(mapped, t);
    }
    Ok(out)
}

impl Backend for CandleBackend {
    fn id(&self) -> BackendId {
        self.id
    }

    fn capabilities(&self) -> Capabilities {
        let mut extra = BTreeMap::from([("device".into(), crate::device_label(&self.device))]);
        if self.device.is_cuda() {
            extra.insert("device_path".into(), "modernbert-cuda".into());
        }
        Capabilities {
            id: self.id,
            dtype: self.dtype.clone(),
            max_context: self.max_context,
            supports_fork: false,
            supports_lora: false,
            families: self.families.clone(),
            extra,
        }
    }

    fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
        self.validate_input(&input)?;
        if input.positions.is_empty() {
            return Ok(ForwardOutput::Features {
                positions: Vec::new(),
                values: CoreTensor::zeros(vec![0, self.hidden_size]),
            });
        }
        Ok(self.forward_batch(vec![input])?.remove(0))
    }

    fn supports_batch(&self) -> bool {
        true
    }

    fn forward_batch(&mut self, inputs: Vec<ForwardInput>) -> Result<Vec<ForwardOutput>> {
        if inputs.is_empty() || inputs.len() > 64 {
            return Err(Error::Backend(
                "Candle batch must contain 1..=64 independent rows".into(),
            ));
        }
        let seq = inputs[0].tokens.len();
        if seq == 0 || inputs.iter().any(|input| input.tokens.len() != seq) {
            return Err(Error::Backend(
                "Candle batches require equal nonempty sequence lengths".into(),
            ));
        }
        for input in &inputs {
            self.validate_input(input)?;
        }
        let tokens: Vec<u32> = inputs
            .iter()
            .flat_map(|input| input.tokens.iter().copied())
            .collect();
        let ids = Tensor::from_vec(tokens, (inputs.len(), seq), &self.device)
            .map_err(|e| Error::Backend(CandleError::Inference(e.to_string()).to_string()))?;
        let mask = Tensor::ones((inputs.len(), seq), DType::U32, &self.device)
            .map_err(|e| Error::Backend(CandleError::Inference(e.to_string()).to_string()))?;
        let hidden = self
            .model
            .forward(&ids, &mask)
            .map_err(|e| Error::Backend(CandleError::Inference(e.to_string()).to_string()))?;
        inputs
            .into_iter()
            .enumerate()
            .map(|(row, input)| {
                let hidden = hidden
                    .narrow(0, row, 1)
                    .map_err(|e| Error::Backend(e.to_string()))?;
                self.readout(&hidden, input)
            })
            .collect()
    }

    fn fork(&mut self, _handle: CacheHandle) -> Result<CacheHandle> {
        Err(Error::Unsupported(
            "candle ModernBERT does not support KV forking".into(),
        ))
    }
}

impl CandleBackend {
    fn validate_input(&self, input: &ForwardInput) -> Result<()> {
        if input.fork_from.is_some() || input.retain_cache {
            return Err(Error::Unsupported(
                "ModernBERT does not support causal prefix reuse".into(),
            ));
        }
        if input.positions.iter().any(|&p| p >= input.tokens.len())
            || input
                .tokens
                .iter()
                .any(|&token| token as usize >= self.vocab_size)
            || (self.head.is_some() && input.qtype > 2)
        {
            return Err(Error::Backend(
                "invalid ModernBERT token/readout/question type".into(),
            ));
        }
        if input.tokens.len() > self.max_context {
            return Err(Error::Backend(format!(
                "sequence length {} exceeds candle max_context {}",
                input.tokens.len(),
                self.max_context
            )));
        }
        Ok(())
    }

    fn readout(&self, hidden_tensor: &Tensor, input: ForwardInput) -> Result<ForwardOutput> {
        let hidden = self.hidden_size;

        if input.positions.is_empty() {
            return Ok(ForwardOutput::Features {
                positions: Vec::new(),
                values: CoreTensor::zeros(vec![0, hidden]),
            });
        }

        // Laya decision-head path: run the typed option-marker head over the full
        // sequence and read per-option logits at the mask positions.
        if let Some(head) = &self.head {
            let logits = head
                .forward(hidden_tensor, input.qtype, &input.positions)
                .map_err(|e| Error::Backend(CandleError::Inference(e.to_string()).to_string()))?;
            let values = core_from_tensor(&logits)?;
            return Ok(ForwardOutput::Logits {
                positions: input.positions,
                values,
            });
        }

        let positions: Vec<u32> = input.positions.iter().map(|&p| p as u32).collect();
        // candle `index_select` requires a 1-D index tensor.
        let pos_tensor = Tensor::new(positions.as_slice(), &self.device)
            .map_err(|e| Error::Backend(CandleError::Inference(e.to_string()).to_string()))?;
        let selected = hidden_tensor
            .index_select(&pos_tensor, 1)
            .map_err(|e| Error::Backend(CandleError::Inference(e.to_string()).to_string()))?
            .squeeze(0)
            .map_err(|e| Error::Backend(CandleError::Inference(e.to_string()).to_string()))?;
        let values = core_from_tensor(&selected)?;
        Ok(ForwardOutput::Features {
            positions: input.positions,
            values,
        })
    }
}

// ---------------------------------------------------------------------------
// Laya decision head
// ---------------------------------------------------------------------------

/// One `nn.TransformerEncoderLayer` (norm_first, batch_first) of the Laya head.
/// The FFN uses ReLU (the `nn.TransformerEncoderLayer` default), not GELU.
struct LayaLayer {
    nhead: usize,
    head_dim: usize,
    qkv_w: Tensor,
    qkv_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    norm1_w: Tensor,
    norm1_b: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
    ff1_w: Tensor,
    ff1_b: Tensor,
    ff2_w: Tensor,
    ff2_b: Tensor,
}

impl LayaLayer {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let d_model = self.qkv_w.dims()[0] / 3;
        let batch = x.dims()[0];
        let seq = x.dims()[1];
        let eps = 1e-5f32;

        // Self-attention with pre-norm (norm_first).
        let xn = candle_nn::ops::layer_norm(x, &self.norm1_w, &self.norm1_b, eps)?;
        let qkv = xn
            .broadcast_matmul(&self.qkv_w.t()?)?
            .broadcast_add(&self.qkv_b)?;
        let q = qkv.narrow(2, 0, d_model)?;
        let k = qkv.narrow(2, d_model, d_model)?;
        let v = qkv.narrow(2, 2 * d_model, d_model)?;

        let q = q
            .reshape((batch, seq, self.nhead, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq, self.nhead, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq, self.nhead, self.head_dim))?
            .transpose(1, 2)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = attn.affine(scale, 0.0)?;
        let attn = candle_nn::ops::softmax(&attn, 3)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((batch, seq, d_model))?;
        let out = out
            .broadcast_matmul(&self.out_w.t()?)?
            .broadcast_add(&self.out_b)?;
        let x = x.broadcast_add(&out)?;

        // Position-wise FFN with ReLU.
        let xn2 = candle_nn::ops::layer_norm(&x, &self.norm2_w, &self.norm2_b, eps)?;
        let ff = xn2
            .broadcast_matmul(&self.ff1_w.t()?)?
            .broadcast_add(&self.ff1_b)?
            .relu()?;
        let ff = ff
            .broadcast_matmul(&self.ff2_w.t()?)?
            .broadcast_add(&self.ff2_b)?;
        let x = x.broadcast_add(&ff)?;
        Ok(x)
    }
}

/// The Laya typed decision head: `type_emb`, two transformer layers, then a
/// LayerNorm → Linear → GELU → Linear(a,1) scorer over the gathered `[MASK]`
/// marker features.
struct LayaHead {
    type_emb: Tensor,
    layers: Vec<LayaLayer>,
    scorer_ln_w: Tensor,
    scorer_ln_b: Tensor,
    scorer_lin1_w: Tensor,
    scorer_lin1_b: Tensor,
    scorer_lin2_w: Tensor,
    scorer_lin2_b: Tensor,
}

impl LayaHead {
    /// Build the head from the checkpoint tensors, or `None` when the checkpoint
    /// has no Laya decision head (a bare ModernBERT encoder).
    fn from_tensors(tensors: &HashMap<String, Tensor>, d: usize) -> Result<Option<LayaHead>> {
        if !tensors.contains_key("type_emb.weight") {
            return Ok(None);
        }
        let nhead = (d / 64).max(1);
        let head_dim = d / nhead;

        let mut layers = Vec::new();
        let mut i = 0usize;
        loop {
            let p = format!("head.layers.{i}.");
            let qkv_key = format!("{p}self_attn.in_proj_weight");
            let Some(qkv_w) = tensors.get(&qkv_key) else {
                break;
            };
            let qkv_b = tensors
                .get(&format!("{p}self_attn.in_proj_bias"))
                .cloned()
                .ok_or_else(|| missing(&qkv_key))?;
            layers.push(LayaLayer {
                nhead,
                head_dim,
                qkv_w: qkv_w.clone(),
                qkv_b,
                out_w: get(tensors, &format!("{p}self_attn.out_proj.weight"))?,
                out_b: get(tensors, &format!("{p}self_attn.out_proj.bias"))?,
                norm1_w: get(tensors, &format!("{p}norm1.weight"))?,
                norm1_b: get(tensors, &format!("{p}norm1.bias"))?,
                norm2_w: get(tensors, &format!("{p}norm2.weight"))?,
                norm2_b: get(tensors, &format!("{p}norm2.bias"))?,
                ff1_w: get(tensors, &format!("{p}linear1.weight"))?,
                ff1_b: get(tensors, &format!("{p}linear1.bias"))?,
                ff2_w: get(tensors, &format!("{p}linear2.weight"))?,
                ff2_b: get(tensors, &format!("{p}linear2.bias"))?,
            });
            i += 1;
        }
        if layers.is_empty() {
            return Err(Error::Backend(
                "checkpoint has a Laya head (`type_emb.weight`) but no head layers".into(),
            ));
        }

        Ok(Some(LayaHead {
            type_emb: tensors["type_emb.weight"].clone(),
            layers,
            scorer_ln_w: get(tensors, "scorer.0.weight")?,
            scorer_ln_b: get(tensors, "scorer.0.bias")?,
            scorer_lin1_w: get(tensors, "scorer.1.weight")?,
            scorer_lin1_b: get(tensors, "scorer.1.bias")?,
            scorer_lin2_w: get(tensors, "scorer.3.weight")?,
            scorer_lin2_b: get(tensors, "scorer.3.bias")?,
        }))
    }

    fn forward(&self, h: &Tensor, qtype: u32, positions: &[usize]) -> candle::Result<Tensor> {
        let qrow = self.type_emb.narrow(0, qtype as usize, 1)?.unsqueeze(0)?; // [1, 1, d]
        let mut z = h.broadcast_add(&qrow)?;
        for layer in &self.layers {
            z = layer.forward(&z)?;
        }

        let pos: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let pos_t = Tensor::new(pos.as_slice(), h.device())?;
        let m = z.index_select(&pos_t, 1)?.squeeze(0)?; // [n, d]

        let y = candle_nn::ops::layer_norm(&m, &self.scorer_ln_w, &self.scorer_ln_b, 1e-5f32)?;
        let y = y
            .matmul(&self.scorer_lin1_w.t()?)?
            .broadcast_add(&self.scorer_lin1_b)?;
        let y = y.gelu_erf()?;
        let y = y
            .matmul(&self.scorer_lin2_w.t()?)?
            .broadcast_add(&self.scorer_lin2_b)?;
        Ok(y)
    }
}

fn missing(key: &str) -> Error {
    Error::Backend(format!("Laya head tensors missing `{key}`"))
}

fn get(tensors: &HashMap<String, Tensor>, key: &str) -> Result<Tensor> {
    tensors.get(key).cloned().ok_or_else(|| missing(key))
}

/// Convert a 2-D candle tensor into a core tensor (used for the per-option logits).
fn core_from_tensor(t: &Tensor) -> Result<CoreTensor> {
    let dims = t.dims();
    let data = t
        .flatten_all()
        .and_then(|flat| flat.to_vec1::<f32>())
        .map_err(|e| Error::Backend(CandleError::Inference(e.to_string()).to_string()))?;
    CoreTensor::new(dims.to_vec(), data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn extraction_matches_manual_slice() {
        // Verify the backend's position extraction equals a manual slice of the
        // full model output (i.e. index_select + squeeze is correct).
        let device = Device::Cpu;
        let config = parse_config(Path::new("tests/fixtures/tiny_modernbert/config.json")).unwrap();
        let tensors = load_encoder_tensors(
            Path::new("tests/fixtures/tiny_modernbert/model.safetensors"),
            &device,
        )
        .unwrap();
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = ModernBert::load(vb, &config).unwrap();

        let tokens = vec![1u32, 2, 3, 4, 5];
        let positions = vec![1usize, 3];

        // Backend path.
        let mut backend = CandleBackend::load(
            "tests/fixtures/tiny_modernbert/config.json",
            "tests/fixtures/tiny_modernbert/model.safetensors",
            16,
            "fp32",
        )
        .unwrap();
        let out = backend
            .forward(huncho_core::backend::ForwardInput::new(
                tokens.clone(),
                positions.clone(),
            ))
            .unwrap();

        // Manual path: full [1, seq, hidden] then slice rows at positions.
        let ids = Tensor::new(tokens.as_slice(), &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let mask = Tensor::ones(tokens.len(), DType::U32, &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let full = model
            .forward(&ids, &mask)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let v3 = full.to_vec3::<f32>().unwrap();
        let hidden = v3[0][0].len();
        let mut expected = vec![0.0f32; positions.len() * hidden];
        for (row, &pos) in positions.iter().enumerate() {
            expected[row * hidden..(row + 1) * hidden].copy_from_slice(&v3[0][pos]);
        }

        assert_eq!(out.values().data(), &expected[..]);
    }

    #[test]
    fn remap_encoder_to_model() {
        assert_eq!(
            remap_key("encoder.layers.0.attn.Wqkv.weight").as_deref(),
            Some("model.layers.0.attn.Wqkv.weight")
        );
        assert_eq!(
            remap_key("encoder.embeddings.tok_embeddings.weight").as_deref(),
            Some("model.embeddings.tok_embeddings.weight")
        );
    }

    #[test]
    fn remap_keeps_model_prefix() {
        assert_eq!(
            remap_key("model.layers.1.mlp.Wo.weight").as_deref(),
            Some("model.layers.1.mlp.Wo.weight")
        );
    }

    #[test]
    fn remap_drops_non_encoder() {
        assert_eq!(remap_key("temperature"), None);
        assert_eq!(remap_key("act_head.0.weight"), None);
        assert_eq!(remap_key("score"), None);
    }

    #[test]
    fn config_parses_rope_parameters_object() {
        // transformers-5.0 style (as `convaiinnovations/laya` uses).
        let v = serde_json::json!({
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "intermediate_size": 2624,
            "max_position_embeddings": 8192,
            "layer_norm_eps": 1e-5,
            "pad_token_id": 50283,
            "global_attn_every_n_layers": 3,
            "local_attention": 128,
            "vocab_size": 50368,
            "rope_parameters": {
                "full_attention": { "rope_theta": 160000.0, "rope_type": "default" },
                "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
            }
        });
        let cfg = config_from_value(&v, Path::new("config.json")).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.global_rope_theta, 160000.0);
        assert_eq!(cfg.local_rope_theta, 10000.0);
        assert_eq!(cfg.local_attention, 128);
        assert_eq!(cfg.pad_token_id, 50283);
    }

    #[test]
    fn config_parses_flat_rope_theta() {
        let v = serde_json::json!({
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "max_position_embeddings": 16,
            "layer_norm_eps": 1e-5,
            "pad_token_id": null,
            "global_attn_every_n_layers": 1,
            "local_attention": 8,
            "vocab_size": 20,
            "global_rope_theta": 50000.0,
            "local_rope_theta": 5000.0
        });
        let cfg = config_from_value(&v, Path::new("config.json")).unwrap();
        assert_eq!(cfg.global_rope_theta, 50000.0);
        assert_eq!(cfg.local_rope_theta, 5000.0);
        // null pad_token_id tolerates -> 0.
        assert_eq!(cfg.pad_token_id, 0);
    }
}

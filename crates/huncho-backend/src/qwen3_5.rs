//! Qwen3.5 + LoRA Candle backend for Nimble (F3) and Kev (F2).
//!
//! Bespoke-Nimble packages are PEFT LoRA adapters over a **Qwen3.5-style
//! hybrid** backbone: a stack of `linear_attention` (Gated DeltaNet) layers
//! interleaved with sparse `full_attention` layers. `candle-transformers`
//! 0.11.0 has no Qwen3.5/Qwen3-Next model, so this module ports the reference
//! `modeling_qwen3_5.py` math to raw candle:
//!
//! * zero-centered RMSNorm (`x / rsqrt(var + eps) * (1 + weight)`) and the
//!   gated variant (`rms_norm(x) * silu(gate))`)
//! * partial rotary (text-only mrope collapses to standard partial rotary)
//! * full attention with `attn_output_gate`
//! * Gated DeltaNet (causal depthwise conv + per-token gated delta rule)
//! * SwiGLU MLP
//! * standard PEFT LoRA merge (`W' = W + alpha/r * (lora_B @ lora_A)`)
//!
//! The backend returns `ForwardOutput::Logits` at the requested positions so
//! `head::candidate_logits` can read the one-token candidate codes. Kev uses its
//! trained pointer projections and returns one raw score per option instead.
//!
//! Reference: `transformers` `modeling_qwen3_5.py` (`Qwen3_5ForConditionalGeneration`).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::kev::PointerHead;
use candle::{D, Device, DType, Result, Tensor};
use candle_nn::{embedding, linear_b, Activation, Embedding, Linear, Module, VarBuilder};

use huncho_core::backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput};
use huncho_core::error::{Error, Result as CoreResult};
use huncho_core::manifest::{BackendId, Family};
use huncho_core::tensor::Tensor as CoreTensor;

/// The block type of a decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Linear,
    Full,
}

/// The Qwen3.5 text architecture.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<LayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub attention_bias: bool,
}

impl Config {
    /// Parse a Qwen3.5 config JSON, accepting both the conditional-generation
    /// shape (`text_config` nested) and a bare text config.
    pub fn from_value(v: &serde_json::Value) -> Result<Config> {
        let text = v.get("text_config").unwrap_or(v);
        fn usize_of(v: &serde_json::Value, key: &str, default: usize) -> usize {
            v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(default)
        }
        fn f64_of(v: &serde_json::Value, key: &str, default: f64) -> f64 {
            v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
        }

        let hidden_size = usize_of(text, "hidden_size", 4096);
        let num_hidden_layers = usize_of(text, "num_hidden_layers", 32);
        let num_attention_heads = usize_of(text, "num_attention_heads", 16);
        let num_key_value_heads = usize_of(text, "num_key_value_heads", 4);
        let head_dim = usize_of(text, "head_dim", hidden_size / num_attention_heads);
        let intermediate_size = usize_of(text, "intermediate_size", 12288);
        let vocab_size = usize_of(text, "vocab_size", 248320);
        let rms_norm_eps = f64_of(text, "rms_norm_eps", 1e-6) as f32;
        let max_position_embeddings = usize_of(text, "max_position_embeddings", 262144);

        // Layer types: prefer the explicit array, else derive from the
        // `full_attention_interval` (full attention at `interval-1`, `2*interval-1`, ...).
        let layer_types = match text.get("layer_types").and_then(|x| x.as_array()) {
            Some(arr) => arr
                .iter()
                .map(|s| {
                    let s = s.as_str().unwrap_or("");
                    if s.contains("full_attention") {
                        Ok(LayerType::Full)
                    } else if s.contains("linear_attention") {
                        Ok(LayerType::Linear)
                    } else {
                        candle::bail!("unknown layer type `{s}`")
                    }
                })
                .collect::<Result<Vec<_>>>()?,
            None => {
                let interval = usize_of(text, "full_attention_interval", 4).max(1);
                (0..num_hidden_layers)
                    .map(|i| {
                        if (i + 1) % interval == 0 {
                            LayerType::Full
                        } else {
                            LayerType::Linear
                        }
                    })
                    .collect()
            }
        };

        let rope_theta = text
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .and_then(|x| x.as_f64())
            .or_else(|| text.get("rope_theta").and_then(|x| x.as_f64()))
            .unwrap_or(10_000.0);
        let partial_rotary_factor = text
            .get("rope_parameters")
            .and_then(|r| r.get("partial_rotary_factor"))
            .and_then(|x| x.as_f64())
            .or_else(|| text.get("partial_rotary_factor").and_then(|x| x.as_f64()))
            .unwrap_or(1.0);

        Ok(Config {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rms_norm_eps,
            max_position_embeddings,
            layer_types,
            linear_conv_kernel_dim: usize_of(text, "linear_conv_kernel_dim", 4),
            linear_key_head_dim: usize_of(text, "linear_key_head_dim", 128),
            linear_num_key_heads: usize_of(text, "linear_num_key_heads", 16),
            linear_num_value_heads: usize_of(text, "linear_num_value_heads", 32),
            linear_value_head_dim: usize_of(text, "linear_value_head_dim", 128),
            rope_theta,
            partial_rotary_factor,
            attention_bias: text
                .get("attention_bias")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        })
    }

    /// The rotary dimension (partial) used by the full-attention layers.
    pub fn rotary_dim(&self) -> usize {
        let d = (self.head_dim as f64 * self.partial_rotary_factor).round() as usize;
        d.max(2) & !1
    }
}

// ---------------------------------------------------------------------------
// Norms
// ---------------------------------------------------------------------------

/// `rms_norm(x) * (1 + weight)` — zero-centered Qwen3.5 RMSNorm.
fn rms_norm_zero(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x_f = x.to_dtype(DType::F32)?;
    let var = x_f.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (var + eps as f64)?.sqrt()?;
    let norm = x_f.broadcast_div(&denom)?;
    let eff = weight.to_dtype(DType::F32)?.affine(1.0, 1.0)?; // 1 + weight
    norm.broadcast_mul(&eff)?.to_dtype(x.dtype())
}

/// `rms_norm(x) * silu(gate)` — Qwen3.5 gated RMSNorm.
fn rms_norm_gated(x: &Tensor, gate: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x_f = x.to_dtype(DType::F32)?;
    let var = x_f.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (var + eps as f64)?.sqrt()?;
    let norm = x_f.broadcast_div(&denom)?;
    let norm = norm.broadcast_mul(&weight.to_dtype(DType::F32)?)?;
    let g = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
    norm.broadcast_mul(&g)?.to_dtype(x.dtype())
}

/// FLA-style L2 norm along the last dim for the delta rule (fp32).
fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f = x.to_dtype(DType::F32)?;
    let inv = (x_f.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    x_f.broadcast_mul(&inv)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (x.to_dtype(DType::F32)?.exp()? + 1.0)?.log()
}

// ---------------------------------------------------------------------------
// Rotary embeddings (partial, text-only mrope collapse)
// ---------------------------------------------------------------------------

struct RotaryEmbedding {
    inv_freq: Tensor,
    dtype: DType,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, device: &Device, dtype: DType) -> Result<Self> {
        let rotary_dim = cfg.rotary_dim();
        let half = rotary_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| (cfg.rope_theta as f32).powf(-(2.0 * i as f32) / rotary_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv, (1, half), device)?;
        Ok(Self { inv_freq, dtype })
    }

    /// `cos`/`sin` for a sequence length at text positions: `[seq, rotary_dim]`.
    fn cos_sin(&self, seq: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let t = Tensor::arange(0u32, seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((seq, 1))?;
        let freqs = t.matmul(&self.inv_freq)?; // [seq, rotary_dim/2]
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?; // [seq, rotary_dim]
        let cos = emb.cos()?.to_dtype(self.dtype)?;
        let sin = emb.sin()?.to_dtype(self.dtype)?;
        Ok((cos, sin))
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dims().last().copied().unwrap_or(0);
    let half = d / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    Tensor::cat(&[x2.neg()?, x1], 3)
}

/// Apply partial rotary to the first `rotary_dim` dims of `q`/`k`.
fn apply_partial_rotary(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotary_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let head_dim = q.dims().last().copied().unwrap_or(0);
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, rotary_dim]
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    let q_rot = q.narrow(3, 0, rotary_dim)?;
    let q_pass = q.narrow(3, rotary_dim, head_dim - rotary_dim)?;
    let q_embed =
        q_rot.broadcast_mul(&cos)?.broadcast_add(&rotate_half(&q_rot)?.broadcast_mul(&sin)?)?;
    let q_out = Tensor::cat(&[&q_embed, &q_pass], 3)?;

    let k_rot = k.narrow(3, 0, rotary_dim)?;
    let k_pass = k.narrow(3, rotary_dim, head_dim - rotary_dim)?;
    let k_embed =
        k_rot.broadcast_mul(&cos)?.broadcast_add(&rotate_half(&k_rot)?.broadcast_mul(&sin)?)?;
    let k_out = Tensor::cat(&[&k_embed, &k_pass], 3)?;
    Ok((q_out, k_out))
}

fn repeat_interleave_head(t: &Tensor, n: usize, dim: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(t.clone());
    }
    let shape = t.dims().to_vec();
    let mut expanded_shape = shape.clone();
    expanded_shape.insert(dim + 1, n);
    let expanded = t.unsqueeze(dim + 1)?.expand(expanded_shape)?.contiguous()?;
    let mut out_shape = shape.clone();
    out_shape[dim] = shape[dim] * n;
    expanded.reshape(out_shape)
}

// ---------------------------------------------------------------------------
// Full attention
// ---------------------------------------------------------------------------

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let hidden = cfg.hidden_size;
        let q_proj = linear_b(hidden, num_heads * head_dim * 2, cfg.attention_bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(hidden, num_kv_heads * head_dim, cfg.attention_bias, vb.pp("k_proj"))?;
        let v_proj = linear_b(hidden, num_kv_heads * head_dim, cfg.attention_bias, vb.pp("v_proj"))?;
        let o_proj = linear_b(num_heads * head_dim, hidden, cfg.attention_bias, vb.pp("o_proj"))?;
        let q_norm = vb.get(head_dim, "q_norm.weight")?;
        let k_norm = vb.get(head_dim, "k_norm.weight")?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim: cfg.rotary_dim(),
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let b = x.dims()[0];
        let seq = x.dims()[1];
        let q_gate = self.q_proj.forward(x)?; // [B, seq, num_heads*head_dim*2]
        let q_gate = q_gate.reshape((b, seq, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(3, 0, self.head_dim)?;
        let gate = q_gate.narrow(3, self.head_dim, self.head_dim)?; // [B, seq, num_heads, head_dim]

        let q = q.transpose(1, 2)?; // [B, num_heads, seq, head_dim]
        let q = rms_norm_zero(&q, &self.q_norm, self.eps)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = rms_norm_zero(&k, &self.k_norm, self.eps)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = apply_partial_rotary(&q, &k, cos, sin, self.rotary_dim)?;

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 { repeat_interleave_head(&k, n_rep, 1)? } else { k };
        let v = if n_rep > 1 { repeat_interleave_head(&v, n_rep, 1)? } else { v };

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?; // [B, num_heads, seq, seq]
        let scores = scores.broadcast_add(&mask.to_dtype(scores.dtype())?)?;
        let probs = candle_nn::ops::softmax(&scores, 3)?;
        let attn = probs.matmul(&v)?; // [B, num_heads, seq, head_dim]
        let attn = attn.transpose(1, 2)?; // [B, seq, num_heads, head_dim]
        // `attn_output_gate`: the gate is the same shape as each head (the
        // q_proj output is split in two: query + gate), so multiply elementwise.
        // Compute the gate in fp32 for stability, then cast back to the
        // activation dtype so the broadcast_mul matches `attn`.
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?.to_dtype(attn.dtype())?;
        let attn = attn.broadcast_mul(&gate)?.reshape((b, seq, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn)
    }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (linear attention)
// ---------------------------------------------------------------------------

struct LinearAttn {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    out_proj: Linear,
    conv1d_w: Tensor, // [conv_dim, 1, K]
    norm_w: Tensor,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
    base_g: Tensor, // -exp(A_log), [num_v_heads]
    dt_bias: Tensor,
    dtype: DType,
    eps: f32,
}

impl LinearAttn {
    fn new(cfg: &Config, vb: VarBuilder, _device: &Device, dtype: DType) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let num_v_heads = cfg.linear_num_value_heads;
        let num_k_heads = cfg.linear_num_key_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel = cfg.linear_conv_kernel_dim;

        let in_proj_qkv = linear_b(hidden, conv_dim, cfg.attention_bias, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_b(hidden, value_dim, cfg.attention_bias, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_b(hidden, num_v_heads, cfg.attention_bias, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_b(hidden, num_v_heads, cfg.attention_bias, vb.pp("in_proj_a"))?;
        let out_proj = linear_b(value_dim, hidden, cfg.attention_bias, vb.pp("out_proj"))?;
        let conv1d_w = vb.get((conv_dim, 1, conv_kernel), "conv1d.weight")?.to_dtype(DType::F32)?;
        let norm_w = vb.get(head_v_dim, "norm.weight")?;
        let a_log = vb.get(num_v_heads, "A_log")?.to_dtype(DType::F32)?;
        let dt_bias = vb.get(num_v_heads, "dt_bias")?.to_dtype(DType::F32)?;
        let base_g = a_log.exp()?.neg()?; // -exp(A_log)

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            out_proj,
            conv1d_w,
            norm_w,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel,
            base_g,
            dt_bias,
            dtype,
            eps: cfg.rms_norm_eps,
        })
    }

    /// Causal depthwise conv1d over `[B, C, T]` (same length out):
    /// `out[c, l] = sum_k w[c, k] * x[c, l - (K-1) + k]`.
    fn causal_conv(&self, x: &Tensor) -> Result<Tensor> {
        let (b, conv_dim, seq) = x.dims3()?;
        let k = self.conv_kernel;
        let mut out = Tensor::zeros((b, conv_dim, seq), DType::F32, x.device())?;
        for kk in 0..k {
            let shift = (k - 1) - kk;
            if shift >= seq {
                continue;
            }
            let w = self.conv1d_w.narrow(2, kk, 1)?.squeeze(1)?.unsqueeze(0)?; // [1, conv_dim, 1]
            let contrib = w.broadcast_mul(&x.to_dtype(DType::F32)?)?; // [B, conv_dim, seq]
            if shift == 0 {
                out = out.broadcast_add(&contrib)?;
            } else {
                let pad = Tensor::zeros((b, conv_dim, shift), DType::F32, x.device())?;
                let shifted = Tensor::cat(&[&pad, &contrib.narrow(2, 0, seq - shift)?], 2)?;
                out = out.broadcast_add(&shifted)?;
            }
        }
        let out = candle_nn::ops::silu(&out)?;
        out.to_dtype(x.dtype())
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.dims()[0];
        let seq = x.dims()[1];

        let mixed = self.in_proj_qkv.forward(x)?.transpose(1, 2)?; // [B, conv_dim, seq]
        let mixed = self.causal_conv(&mixed)?; // [B, conv_dim, seq]
        let mixed = mixed.transpose(1, 2)?; // [B, seq, conv_dim]

        let q = mixed.narrow(2, 0, self.key_dim)?.reshape((b, seq, self.num_k_heads, self.head_k_dim))?;
        let k = mixed.narrow(2, self.key_dim, self.key_dim)?.reshape((b, seq, self.num_k_heads, self.head_k_dim))?;
        let v = mixed.narrow(2, 2 * self.key_dim, self.value_dim)?.reshape((b, seq, self.num_v_heads, self.head_v_dim))?;

        let z = self.in_proj_z.forward(x)?.reshape((b, seq, self.num_v_heads, self.head_v_dim))?;
        let beta = candle_nn::ops::sigmoid(&self.in_proj_b.forward(x)?)?; // [B, seq, num_v_heads]
        let a = self.in_proj_a.forward(x)?.to_dtype(DType::F32)?; // [B, seq, num_v_heads]
        let g = self
            .base_g
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_mul(&softplus(&a.broadcast_add(&self.dt_bias)?)?)?; // [B, seq, num_v_heads]

        let n_rep = if self.num_v_heads.is_multiple_of(self.num_k_heads) {
            self.num_v_heads / self.num_k_heads
        } else {
            1
        };
        let q = if n_rep > 1 { repeat_interleave_head(&q, n_rep, 2)? } else { q };
        let k = if n_rep > 1 { repeat_interleave_head(&k, n_rep, 2)? } else { k };

        let q = q.transpose(1, 2)?; // [B, num_v_heads, seq, head_k]
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?; // [B, num_v_heads, seq, head_v]
        let beta = beta.transpose(1, 2)?; // [B, num_v_heads, seq]
        let g = g.transpose(1, 2)?; // [B, num_v_heads, seq]

        let scale = 1.0 / (self.head_k_dim as f64).sqrt();
        let q = l2norm(&q, 1e-6)?.affine(scale, 0.0)?;
        let k = l2norm(&k, 1e-6)?;

        // The recurrence accumulates in fp32; cast the result back to the model
        // dtype so the following gated norm and out projection stay consistent.
        let out = recurrent_gated_delta(&q, &k, &v, &g, &beta)?; // [B, num_v_heads, seq, head_v]
        let out = out.to_dtype(self.dtype)?.transpose(1, 2)?; // [B, seq, num_v_heads, head_v]
        // Apply the per-head gated RMSNorm over the last (head_v) dim, then
        // flatten the value heads for the output projection.
        let out = rms_norm_gated(&out, &z, &self.norm_w, self.eps)?; // [B, seq, num_v_heads, head_v]
        let out = out.reshape((b, seq, self.value_dim))?;
        self.out_proj.forward(&out)
    }
}

/// Per-token gated delta rule (matches `torch_recurrent_gated_delta_rule`).
fn recurrent_gated_delta(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let (b, n_v, seq, head_k) = query.dims4()?;
    let head_v = value.dims().last().copied().unwrap_or(0);
    let mut state = Tensor::zeros((b, n_v, head_k, head_v), DType::F32, query.device())?;
    let mut outs = Vec::with_capacity(seq);
    for i in 0..seq {
        let q_i = query.narrow(2, i, 1)?.squeeze(2)?; // [B, n_v, head_k]
        let k_i = key.narrow(2, i, 1)?.squeeze(2)?; // [B, n_v, head_k]
        // `value`/`beta` arrive in the model dtype (e.g. fp16) while `state` is
        // fp32; promote them so every op in the recurrent loop is fp32.
        let v_i = value.narrow(2, i, 1)?.squeeze(2)?.to_dtype(DType::F32)?; // [B, n_v, head_v]
        let g_i = g.narrow(2, i, 1)?.squeeze(2)?.to_dtype(DType::F32)?; // [B, n_v]
        let b_i = beta.narrow(2, i, 1)?.squeeze(2)?.to_dtype(DType::F32)?; // [B, n_v]

        let decay = g_i.exp()?; // [B, n_v]
        state = state.broadcast_mul(&decay.unsqueeze(2)?.unsqueeze(2)?)?;

        let kv_mem = state.broadcast_mul(&k_i.unsqueeze(3)?)?.sum(2)?; // [B, n_v, head_v]
        let delta = v_i.broadcast_sub(&kv_mem)?.broadcast_mul(&b_i.unsqueeze(2)?)?; // [B, n_v, head_v]
        state = state.broadcast_add(&k_i.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?)?;

        let out_i = state.broadcast_mul(&q_i.unsqueeze(3)?)?.sum(2)?; // [B, n_v, head_v]
        outs.push(out_i);
    }
    Tensor::stack(&outs, 2)
}

// ---------------------------------------------------------------------------
// MLP + decoder layer
// ---------------------------------------------------------------------------

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let gate_proj = linear_b(hidden, intermediate, cfg.attention_bias, vb.pp("gate_proj"))?;
        let up_proj = linear_b(hidden, intermediate, cfg.attention_bias, vb.pp("up_proj"))?;
        let down_proj = linear_b(intermediate, hidden, cfg.attention_bias, vb.pp("down_proj"))?;
        Ok(Self { gate_proj, up_proj, down_proj, act: Activation::Silu })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&self.act)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.broadcast_mul(&up)?)
    }
}

struct DecoderLayer {
    input_layernorm: Tensor,
    linear_attn: Option<LinearAttn>,
    self_attn: Option<Attention>,
    post_attention_layernorm: Tensor,
    mlp: Mlp,
    eps: f32,
}

impl DecoderLayer {
    fn new(cfg: &Config, layer_type: LayerType, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let input_layernorm = vb.get(cfg.hidden_size, "input_layernorm.weight")?;
        let post_attention_layernorm = vb.get(cfg.hidden_size, "post_attention_layernorm.weight")?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let (linear_attn, self_attn) = match layer_type {
            LayerType::Linear => (Some(LinearAttn::new(cfg, vb.pp("linear_attn"), device, dtype)?), None),
            LayerType::Full => (None, Some(Attention::new(cfg, vb.pp("self_attn"))?)),
        };
        Ok(Self {
            input_layernorm,
            linear_attn,
            self_attn,
            post_attention_layernorm,
            mlp,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let h = rms_norm_zero(x, &self.input_layernorm, self.eps)?;
        let h = if let Some(attn) = &self.linear_attn {
            attn.forward(&h)?
        } else if let Some(attn) = &self.self_attn {
            attn.forward(&h, cos, sin, mask)?
        } else {
            candle::bail!("decoder layer has neither linear nor full attention")
        };
        let h = h.broadcast_add(&residual)?;
        let normalized = rms_norm_zero(&h, &self.post_attention_layernorm, self.eps)?;
        let x = self.mlp.forward(&normalized)?.broadcast_add(&h)?;
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: Tensor,
    rotary: RotaryEmbedding,
    eps: f32,
    device: Device,
}

impl Model {
    /// Build the text model from a `VarBuilder` rooted at the full weight
    /// prefix (`model.language_model.<module>` and `lm_head`).
    pub fn new(cfg: &Config, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let embed_tokens =
            embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.language_model.embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let lt = cfg.layer_types.get(i).copied().unwrap_or(LayerType::Linear);
            let layer = DecoderLayer::new(cfg, lt, vb.pp(format!("model.language_model.layers.{i}")), device, dtype)?;
            layers.push(layer);
        }
        let norm = vb.get(cfg.hidden_size, "model.language_model.norm.weight")?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary: RotaryEmbedding::new(cfg, device, dtype)?,
            eps: cfg.rms_norm_eps,
            device: device.clone(),
        })
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let seq = ids.dims().last().copied().unwrap_or(0);
        let (cos, sin) = self.rotary.cos_sin(seq, &self.device)?;
        let mask = causal_mask(seq)?.to_device(&self.device)?;
        let mut hidden = self.embed_tokens.forward(ids)?; // [B, seq, hidden]
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
        }
        rms_norm_zero(&hidden, &self.norm, self.eps)
    }
}

/// A strictly-upper-triangular additive mask `[seq, seq]` (`-inf` for `i < j`).
fn causal_mask(seq: usize) -> Result<Tensor> {
    let mut data = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if i < j {
                data[i * seq + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (seq, seq), &Device::Cpu)
}

// ---------------------------------------------------------------------------
// LoRA merge
// ---------------------------------------------------------------------------

fn merge_lora_into_map(
    map: &mut HashMap<String, Tensor>,
    lora: &HashMap<String, Tensor>,
    lora_scale: f32,
    dtype: DType,
) -> Result<()> {
    // Collect unique base targets by stripping the adapter prefix/suffix.
    let mut targets: Vec<String> = Vec::new();
    for key in lora.keys() {
        if let Some(rest) = key.strip_prefix("base_model.model.") {
            if let Some(t) = rest.strip_suffix(".lora_A.weight") {
                targets.push(t.to_string());
            }
        }
    }
    targets.sort();
    targets.dedup();

    for target in &targets {
        let a_key = format!("base_model.model.{target}.lora_A.weight");
        let b_key = format!("base_model.model.{target}.lora_B.weight");
        let a = &lora[&a_key];
        let b = lora
            .get(&b_key)
            .ok_or_else(|| candle::Error::Msg(format!("missing LoRA weight `{b_key}`")))?;
        let base_key = canonical_weight_name(&format!("{target}.weight"))
            .ok_or_else(|| candle::Error::Msg(format!("unsupported LoRA target `{target}`")))?;
        let base = map.get_mut(&base_key).ok_or_else(|| {
            candle::Error::Msg(format!("missing base weight `{base_key}` required for LoRA merge"))
        })?;
        // Compute the LoRA delta in fp32 (candle's CPU matmul does not support
        // bf16) and only cast the merged result back to the target dtype.
        let delta = b.to_dtype(DType::F32)?.matmul(&a.to_dtype(DType::F32)?)?; // [out, in]
        let delta = delta.affine(lora_scale as f64, 0.0)?;
        let updated = base.to_dtype(DType::F32)?.broadcast_add(&delta)?.to_dtype(dtype)?;
        *base = updated;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// Errors from the Qwen3.5 candle backend.
#[derive(Debug, thiserror::Error)]
pub enum QwenError {
    #[error("failed to read config `{0}`: {1}")]
    Config(String, String),
    #[error("failed to load tensor `{0}`: {1}")]
    Load(String, String),
    #[error("candle inference failed: {0}")]
    Inference(String),
}

pub struct Qwen3_5Backend {
    model: Model,
    head: Readout,
    vocab_size: usize,
    max_context: usize,
    dtype: String,
    device: Device,
}

enum Readout {
    LanguageModel(Linear),
    Pointer(PointerHead),
}

impl Qwen3_5Backend {
    /// Load a Qwen3.5 + optional LoRA model from a base-weights directory plus
    /// an optional adapter directory.
    ///
    /// * `base_dir` must contain `config.json` and the base `*.safetensors`
    ///   shards (the large `Qwen/Qwen3.5-*` weights).
    /// * `adapter_dir` (optional) must contain `adapter_model.safetensors` and
    ///   `adapter_config.json` (for the LoRA rank/alpha).
    pub fn load(
        base_dir: &Path,
        adapter_dir: Option<&Path>,
        max_context: usize,
        dtype: impl Into<String>,
    ) -> CoreResult<Qwen3_5Backend> {
        Self::load_with_head(base_dir, adapter_dir, None, max_context, dtype.into())
    }

    /// Load Kev's backbone and trained pointer head without materializing an
    /// unused vocabulary projection. Each forward is one isolated question row.
    pub fn load_kev(
        base_dir: &Path,
        adapter_dir: &Path,
        head_path: &Path,
        max_context: usize,
        dtype: impl Into<String>,
    ) -> CoreResult<Self> {
        let dtype = dtype.into();
        if !matches!(dtype.as_str(), "fp32" | "fp16" | "f16") {
            return Err(Error::Unsupported(format!(
                "Kev's Candle CPU backend supports fp32 or fp16, not `{dtype}`"
            )));
        }
        Self::load_with_head(
            base_dir,
            Some(adapter_dir),
            Some(head_path),
            max_context,
            dtype,
        )
    }

    fn load_with_head(
        base_dir: &Path,
        adapter_dir: Option<&Path>,
        pointer_path: Option<&Path>,
        max_context: usize,
        dtype: String,
    ) -> CoreResult<Self> {
        let dtype_str = dtype;
        let dtype = match dtype_str.as_str() {
            "fp16" | "f16" => DType::F16,
            "bf16" | "bfloat16" => DType::BF16,
            "fp32" => DType::F32,
            other => {
                return Err(Error::Unsupported(format!(
                    "unsupported Qwen3.5 dtype `{other}`"
                )))
            }
        };
        let device = Device::Cpu;

        let config_path = base_dir.join("config.json");
        let config = parse_config(&config_path)?;

        let mut tensors = load_base_tensors(base_dir, &device, dtype)?;
        if tensors.is_empty() {
            return Err(Error::Backend(format!(
                "no base-weight `*.safetensors` found in `{}`; the Qwen3.5 backend needs the \
                 full base weights (fetch the `{}` repo) in this directory",
                base_dir.display(),
                base_repo_hint(&config)
            )));
        }

        if let Some(adapter_dir) = adapter_dir {
            let adapter_path = adapter_dir.join("adapter_model.safetensors");
            let lora = candle::safetensors::load(&adapter_path, &device).map_err(|e| {
                Error::Backend(QwenError::Load(adapter_path.display().to_string(), e.to_string()).to_string())
            })?;
            let (r, alpha) = read_lora_hyperparams(&adapter_dir.join("adapter_config.json"))?;
            let scale = if r > 0 { alpha / r as f32 } else { 1.0 };
            merge_lora_into_map(&mut tensors, &lora, scale, dtype).map_err(|e| {
                Error::Backend(QwenError::Load("adapter merge".into(), e.to_string()).to_string())
            })?;
        }

        // Pull the LM head out of the shared map before constructing the
        // backbone so `tensors` can be moved (not cloned) into the model
        // `VarBuilder`. The backbone does not reference `lm_head.*`, so this
        // avoids holding a second full copy of every weight in memory during
        // construction (relevant for 9B+ backbones).
        let head = match pointer_path {
            Some(path) => {
                tensors.remove("lm_head.weight");
                Readout::Pointer(PointerHead::load(path, config.hidden_size)?)
            }
            None => {
                let lm_head_w = tensors.remove("lm_head.weight").ok_or_else(|| {
                    Error::Backend("base weights are missing `lm_head.weight`".into())
                })?;
                Readout::LanguageModel(Linear::new(lm_head_w, tensors.remove("lm_head.bias")))
            }
        };
        let vocab_size = if pointer_path.is_some() {
            1
        } else {
            config.vocab_size
        };

        let vb = VarBuilder::from_tensors(tensors, dtype, &device);
        let model = Model::new(&config, vb, &device, dtype)
            .map_err(|e| Error::Backend(QwenError::Load("model".into(), e.to_string()).to_string()))?;

        Ok(Qwen3_5Backend {
            model,
            head,
            vocab_size,
            max_context,
            dtype: dtype_str,
            device,
        })
    }
}

/// A short human hint for the base vocabulary/repo of the loaded config.
fn base_repo_hint(_cfg: &Config) -> &'static str {
    "Qwen/Qwen3.5-*"
}

fn parse_config(path: &Path) -> CoreResult<Config> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Backend(QwenError::Config(path.display().to_string(), e.to_string()).to_string()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Backend(QwenError::Config(path.display().to_string(), e.to_string()).to_string()))?;
    Config::from_value(&v)
        .map_err(|e| Error::Backend(QwenError::Config(path.display().to_string(), e.to_string()).to_string()))
}

/// Read `lora_alpha`/`r` from an `adapter_config.json`.
fn read_lora_hyperparams(path: &Path) -> CoreResult<(usize, f32)> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Backend(QwenError::Config(path.display().to_string(), e.to_string()).to_string()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Backend(QwenError::Config(path.display().to_string(), e.to_string()).to_string()))?;
    let r = v.get("r").and_then(|x| x.as_u64()).unwrap_or(16) as usize;
    let alpha = v.get("lora_alpha").and_then(|x| x.as_f64()).unwrap_or(32.0) as f32;
    Ok((r, alpha))
}

/// Load all `*.safetensors` files in a directory into a single tensor map,
/// converting to the requested dtype.
pub(crate) fn load_base_tensors(dir: &Path, device: &Device, dtype: DType) -> CoreResult<HashMap<String, Tensor>> {
    let mut map = HashMap::new();
    let entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| Error::Backend(QwenError::Load(dir.display().to_string(), e.to_string()).to_string()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .filter(|p| {
            // The adapter sits next to the base weights in the package dir; do
            // not treat `adapter_model.safetensors` as a base shard.
            p.file_name().map(|n| n != "adapter_model.safetensors" && n != "joint_head.safetensors").unwrap_or(true)
        })
        .collect();
    for p in entries {
        // Model files must remain unchanged while loading. Map the shard so
        // excluded vision/MTP weights are never materialized on the device.
        let raw = unsafe { candle::safetensors::MmapedSafetensors::new(&p) }
            .map_err(|e| Error::Backend(QwenError::Load(p.display().to_string(), e.to_string()).to_string()))?;
        for (name, _) in raw.tensors() {
            // The Qwen3.5 base is a multimodal conditional-generation model; the
            // F3 adapter and this text backend only need the text backbone and
            // the LM head. Drop the vision tower and MTP tensors so we do not
            // hold the whole ~19 GB checkpoint in memory.
            let Some(k) = canonical_weight_name(&name) else {
                continue;
            };
            let v = raw.load(&name, device).and_then(|v| v.to_dtype(dtype)).map_err(|e| {
                Error::Backend(QwenError::Load(k.clone(), e.to_string()).to_string())
            })?;
            map.insert(k, v);
        }
    }
    Ok(map)
}

/// HF conditional-generation, text-only and PEFT feature-extraction saves
/// use different prefixes for the same text backbone.
fn canonical_weight_name(name: &str) -> Option<String> {
    if name.starts_with("model.language_model.") || name.starts_with("lm_head.") {
        return Some(name.to_string());
    }
    let text = name
        .strip_prefix("language_model.")
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);
    if ["layers.", "embed_tokens.", "norm."]
        .iter()
        .any(|p| text.starts_with(p))
    {
        Some(format!("model.language_model.{text}"))
    } else {
        None
    }
}

impl Backend for Qwen3_5Backend {
    fn id(&self) -> BackendId {
        BackendId::Candle
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: BackendId::Candle,
            dtype: self.dtype.clone(),
            max_context: self.max_context,
            supports_fork: false,
            supports_lora: true,
            families: vec![match self.head {
                Readout::Pointer(_) => Family::F2,
                _ => Family::F3,
            }],
            extra: BTreeMap::new(),
        }
    }

    fn forward(&mut self, input: ForwardInput) -> CoreResult<ForwardOutput> {
        if input
            .positions
            .iter()
            .any(|&position| position >= input.tokens.len())
        {
            return Err(Error::Backend(
                "Qwen3.5 readout position is outside the token sequence".into(),
            ));
        }
        if input.tokens.len() > self.max_context {
            return Err(Error::Backend(format!(
                "sequence length {} exceeds candle max_context {}",
                input.tokens.len(),
                self.max_context
            )));
        }
        if input.positions.is_empty() {
            return Ok(ForwardOutput::Logits {
                positions: Vec::new(),
                values: CoreTensor::zeros(vec![0, self.vocab_size]),
            });
        }
        let candle = |e: candle::Error| Error::Backend(QwenError::Inference(e.to_string()).to_string());
        let ids = Tensor::new(input.tokens.as_slice(), &self.device)
            .map_err(&candle)?
            .unsqueeze(0)
            .map_err(&candle)?;
        let hidden = self
            .model
            .forward(&ids)
            .map_err(&candle)?;

        let pos: Vec<u32> = input.positions.iter().map(|&p| p as u32).collect();
        let pos_t = Tensor::new(pos.as_slice(), &self.device).map_err(&candle)?;
        // Keep the hidden states at the model dtype so the `lm_head` (also at
        // the model dtype) matmul is dtype-consistent; only the logits are
        // upcast to fp32 for the contract.
        let selected = hidden
            .index_select(&pos_t, 1)
            .map_err(&candle)?
            .squeeze(0)
            .map_err(&candle)?; // [n_positions, hidden]
        let logits = match &self.head {
            Readout::LanguageModel(head) => head.forward(&selected).map_err(&candle)?,
            Readout::Pointer(head) => {
                let decide = hidden
                    .narrow(1, input.tokens.len() - 1, 1)
                    .and_then(|t| t.squeeze(0))
                    .map_err(&candle)?;
                head.forward(&decide, &selected).map_err(&candle)?
            }
        };
        let values = core_from_tensor(&logits.to_dtype(DType::F32).map_err(&candle)?)?;
        Ok(ForwardOutput::Logits {
            positions: input.positions,
            values,
        })
    }

    fn fork(&mut self, _handle: CacheHandle) -> CoreResult<CacheHandle> {
        Err(Error::Unsupported(
            "qwen3.5 candle backend v1 does not support KV forking".into(),
        ))
    }
}

/// Convert a 2-D candle tensor into a core tensor.
fn core_from_tensor(t: &Tensor) -> CoreResult<CoreTensor> {
    let dims = t.dims();
    let rows = t
        .to_vec2::<f32>()
        .map_err(|e| Error::Backend(QwenError::Inference(e.to_string()).to_string()))?;
    let mut data = Vec::with_capacity(rows.len().saturating_mul(dims.last().copied().unwrap_or(0)));
    for row in rows {
        data.extend(row);
    }
    CoreTensor::new(dims.to_vec(), data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::shape::Shape;

    fn tiny_cfg() -> Config {
        let v = serde_json::json!({
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 32,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 32,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 4,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        });
        Config::from_value(&v).unwrap()
    }

    fn rand(shape: impl Into<Shape>, device: &Device) -> Tensor {
        Tensor::randn(0.0f32, 0.05f32, shape, device).unwrap()
    }

    fn tiny_weights(cfg: &Config, device: &Device) -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        let h = cfg.hidden_size;
        m.insert("model.language_model.embed_tokens.weight".into(), rand((cfg.vocab_size, h), device));
        m.insert("lm_head.weight".into(), rand((cfg.vocab_size, h), device));
        m.insert("model.language_model.norm.weight".into(), rand(h, device));

        let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
        let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let kv_heads = cfg.num_key_value_heads;
        let inter = cfg.intermediate_size;
        let n_v = cfg.linear_num_value_heads;

        for i in 0..2 {
            let p = format!("model.language_model.layers.{i}");
            m.insert(format!("{p}.input_layernorm.weight"), rand(h, device));
            m.insert(format!("{p}.post_attention_layernorm.weight"), rand(h, device));
            m.insert(format!("{p}.mlp.gate_proj.weight"), rand((inter, h), device));
            m.insert(format!("{p}.mlp.up_proj.weight"), rand((inter, h), device));
            m.insert(format!("{p}.mlp.down_proj.weight"), rand((h, inter), device));
            if i == 0 {
                m.insert(format!("{p}.linear_attn.in_proj_qkv.weight"), rand((conv_dim, h), device));
                m.insert(format!("{p}.linear_attn.in_proj_z.weight"), rand((value_dim, h), device));
                m.insert(format!("{p}.linear_attn.in_proj_b.weight"), rand((n_v, h), device));
                m.insert(format!("{p}.linear_attn.in_proj_a.weight"), rand((n_v, h), device));
                m.insert(format!("{p}.linear_attn.out_proj.weight"), rand((h, value_dim), device));
                m.insert(format!("{p}.linear_attn.conv1d.weight"), rand((conv_dim, 1, cfg.linear_conv_kernel_dim), device));
                m.insert(format!("{p}.linear_attn.norm.weight"), rand(cfg.linear_value_head_dim, device));
                m.insert(format!("{p}.linear_attn.A_log"), rand(n_v, device));
                m.insert(format!("{p}.linear_attn.dt_bias"), rand(n_v, device));
            } else {
                m.insert(format!("{p}.self_attn.q_proj.weight"), rand((2 * heads * head_dim, h), device));
                m.insert(format!("{p}.self_attn.k_proj.weight"), rand((kv_heads * head_dim, h), device));
                m.insert(format!("{p}.self_attn.v_proj.weight"), rand((kv_heads * head_dim, h), device));
                m.insert(format!("{p}.self_attn.o_proj.weight"), rand((h, heads * head_dim), device));
                m.insert(format!("{p}.self_attn.q_norm.weight"), rand(head_dim, device));
                m.insert(format!("{p}.self_attn.k_norm.weight"), rand(head_dim, device));
            }
        }
        m
    }

    fn build_model(cfg: &Config, device: &Device) -> Model {
        let weights = tiny_weights(cfg, device);
        let vb = VarBuilder::from_tensors(weights, DType::F32, device);
        Model::new(cfg, vb, device, DType::F32).unwrap()
    }

    #[test]
    fn forward_produces_finite_hidden() {
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let model = build_model(&cfg, &device);
        let ids = Tensor::new(&[1u32, 2, 3, 4][..], &device).unwrap().unsqueeze(0).unwrap();
        let hidden = model.forward(&ids).unwrap();
        assert_eq!(hidden.dims(), &[1, 4, 16]);
        let v3 = hidden.to_vec3::<f32>().unwrap();
        assert!(v3.iter().flatten().flatten().all(|x| x.is_finite()));
    }

    #[test]
    fn forward_is_causal() {
        // Changing later tokens must not alter position-0 hidden of either the
        // DeltaNet (layer 0) or the masked full-attention (layer 1) path.
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let model = build_model(&cfg, &device);
        let ids_a = Tensor::new(&[5u32, 6, 7][..], &device).unwrap().unsqueeze(0).unwrap();
        let ids_b = Tensor::new(&[5u32, 10, 11][..], &device).unwrap().unsqueeze(0).unwrap();
        let ha = model.forward(&ids_a).unwrap().index_select(&Tensor::new(&[0u32], &device).unwrap(), 1).unwrap().to_vec3::<f32>().unwrap();
        let hb = model.forward(&ids_b).unwrap().index_select(&Tensor::new(&[0u32], &device).unwrap(), 1).unwrap().to_vec3::<f32>().unwrap();
        for (x, y) in ha[0][0].iter().zip(hb[0][0].iter()) {
            assert!((x - y).abs() < 1e-4, "position-0 hidden changed: {x} vs {y}");
        }
    }

    #[test]
    fn backend_returns_logits_shape() {
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let weights = tiny_weights(&cfg, &device);
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &device);
        let model = Model::new(&cfg, vb, &device, DType::F32).unwrap();
        let lm_head = linear_b(cfg.hidden_size, cfg.vocab_size, false, VarBuilder::from_tensors(weights, DType::F32, &device).pp("lm_head")).unwrap();

        let mut backend = Qwen3_5Backend {
            model,
            head: Readout::LanguageModel(lm_head),
            vocab_size: cfg.vocab_size,
            max_context: 32,
            dtype: "fp32".into(),
            device,
        };
        let out = backend.forward(ForwardInput::new(vec![1, 2, 3, 4], vec![3])).unwrap();
        assert_eq!(out.values().shape(), &[1, 32]);
        assert!(out.values().data().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn lo_merge_changes_weights() {
        let device = Device::Cpu;
        let mut map = HashMap::new();
        map.insert("model.language_model.layers.0.mlp.gate_proj.weight".into(), Tensor::zeros((8, 4), DType::F32, &device).unwrap());
        let mut lora = HashMap::new();
        lora.insert(
            "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_A.weight".into(),
            Tensor::full(1.0f32, (2, 4), &device).unwrap(),
        );
        lora.insert(
            "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_B.weight".into(),
            Tensor::full(1.0f32, (8, 2), &device).unwrap(),
        );
        // B@A = ones(8,4) (each entry = sum of 2 products = 2); scaled by
        // alpha/r = 2 => each entry becomes 4.
        merge_lora_into_map(&mut map, &lora, 2.0, DType::F32).unwrap();
        let merged = map["model.language_model.layers.0.mlp.gate_proj.weight"].to_vec2::<f32>().unwrap();
        assert!(merged.iter().flatten().all(|x| (*x - 4.0).abs() < 1e-4), "expected merged=4, got {merged:?}");
    }
}

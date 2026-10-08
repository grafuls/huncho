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

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::kev::PointerHead;
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, linear_b, Activation, Embedding, Linear, Module, VarBuilder};

use huncho_core::backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput};
use huncho_core::error::{Error, Result as CoreResult};
use huncho_core::manifest::{BackendId, Family};
use huncho_core::tensor::Tensor as CoreTensor;

mod attention;
mod kv_pages;
#[cfg(feature = "quantization")]
#[path = "qwen_quantized.rs"]
pub mod quantized;
mod runtime_lora;
use kv_pages::PagedKv;

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
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(default)
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

/// Zero-centered norm weights are immutable; prepare `1 + weight` once at
/// load time using exactly the former forward dtype and arithmetic.
fn effective_norm_weight(weight: Tensor) -> Result<Tensor> {
    weight.to_dtype(DType::F32)?.affine(1.0, 1.0)
}

fn rms_norm_effective(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x_f = x.to_dtype(DType::F32)?;
    let var = x_f.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (var + eps as f64)?.sqrt()?;
    let norm = x_f.broadcast_div(&denom)?;
    norm.broadcast_mul(weight)?.to_dtype(x.dtype())
}

/// `rms_norm(x) * silu(gate)` — Qwen3.5 gated RMSNorm.
fn rms_norm_gated(x: &Tensor, gate: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x_f = x.to_dtype(DType::F32)?;
    let var = x_f.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (var + eps as f64)?.sqrt()?;
    let norm = x_f.broadcast_div(&denom)?;
    let norm = norm.broadcast_mul(weight)?;
    let g = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
    norm.broadcast_mul(&g)?.to_dtype(x.dtype())
}

/// FLA-style L2 norm along the last dim for the delta rule (fp32).
fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f = x.to_dtype(DType::F32)?;
    let inv = (x_f.sqr()?.sum_keepdim(D::Minus1)? + eps)?
        .sqrt()?
        .recip()?;
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
    fn cos_sin(&self, seq: usize, offset: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let t = Tensor::arange(offset as u32, (offset + seq) as u32, device)?
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
    let q_embed = q_rot
        .broadcast_mul(&cos)?
        .broadcast_add(&rotate_half(&q_rot)?.broadcast_mul(&sin)?)?;
    let q_out = Tensor::cat(&[&q_embed, &q_pass], 3)?;

    let k_rot = k.narrow(3, 0, rotary_dim)?;
    let k_pass = k.narrow(3, rotary_dim, head_dim - rotary_dim)?;
    let k_embed = k_rot
        .broadcast_mul(&cos)?
        .broadcast_add(&rotate_half(&k_rot)?.broadcast_mul(&sin)?)?;
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

/// Optional fixed-row projection calls make kernel selection independent of
/// request/prefix length. The native path is retained when `chunk_rows == 0`.
/// Zero padding is local to a linear projection and discarded immediately;
/// it never enters attention, recurrence, positions or logical token usage.
struct BackboneLinear {
    linear: Projection,
    chunk_rows: usize,
    #[cfg(feature = "cpu-blas")]
    blas: Option<Arc<crate::cpu_blas::Runtime>>,
}

enum Projection {
    Dense(Linear),
    RuntimeLora {
        base: Linear,
        adapter: runtime_lora::Adapter,
    },
    #[cfg(feature = "quantization")]
    Packed {
        matmul: candle::quantized::QMatMul,
        bias: Option<Tensor>,
        input_width: usize,
        packed_width: usize,
    },
}

impl Module for Projection {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(input),
            Self::RuntimeLora { base, adapter } => {
                base.forward(input)?.broadcast_add(&adapter.forward(input)?)
            }
            #[cfg(feature = "quantization")]
            Self::Packed {
                matmul,
                bias,
                input_width,
                packed_width,
            } => {
                let (batch, sequence, width) = input.dims3()?;
                if width != *input_width || input.dtype() != DType::F32 || !input.device().is_cpu()
                {
                    candle::bail!(
                        "packed Qwen projection requires CPU FP32 inputs with width {input_width}"
                    )
                }
                let input = if input_width == packed_width {
                    input.contiguous()?
                } else {
                    Tensor::cat(
                        &[
                            input,
                            &Tensor::zeros(
                                (batch, sequence, packed_width - input_width),
                                DType::F32,
                                &Device::Cpu,
                            )?,
                        ],
                        2,
                    )?
                };
                let output = matmul.forward(&input)?;
                match bias {
                    Some(bias) => output.broadcast_add(bias),
                    None => Ok(output),
                }
            }
        }
    }
}

/// Construction source keeps packed weights packed from file to projection.
/// Norms, embeddings, convolution and biases still use the ordinary builder.
enum ProjectionSource {
    Dense,
    RuntimeLora(HashMap<String, runtime_lora::Adapter>),
    #[cfg(feature = "quantization")]
    Packed(quantized::PackedWeights),
}

impl ProjectionSource {
    fn linear(
        &mut self,
        input: usize,
        output: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<BackboneLinear> {
        match self {
            Self::Dense => Ok(linear_b(input, output, bias, vb)?.into()),
            Self::RuntimeLora(adapters) => {
                let name = format!("{}.weight", vb.prefix());
                let base = linear_b(input, output, bias, vb)?;
                let mut linear: BackboneLinear = base.into();
                if let Some(adapter) = adapters.remove(&name) {
                    let Projection::Dense(base) = linear.linear else {
                        unreachable!()
                    };
                    linear.linear = Projection::RuntimeLora { base, adapter };
                }
                Ok(linear)
            }
            #[cfg(feature = "quantization")]
            Self::Packed(weights) => {
                let name = format!("{}.weight", vb.prefix());
                let (weight, original_width) = weights.remove(&name).ok_or_else(|| {
                    candle::Error::Msg(format!("missing packed projection {name}"))
                })?;
                let (rows, packed_width) = weight.shape().dims2()?;
                if rows != output || original_width != input {
                    candle::bail!("packed projection schema mismatch: {name}")
                }
                let bias = if bias {
                    Some(vb.get(output, "bias")?)
                } else {
                    None
                };
                Ok(BackboneLinear {
                    linear: Projection::Packed {
                        matmul: candle::quantized::QMatMul::QTensor(weight),
                        bias,
                        input_width: input,
                        packed_width,
                    },
                    chunk_rows: 0,
                    #[cfg(feature = "cpu-blas")]
                    blas: None,
                })
            }
        }
    }

    fn finish(&self) -> Result<()> {
        if let Self::RuntimeLora(adapters) = self {
            if !adapters.is_empty() {
                candle::bail!("unused runtime LoRA projections in model schema")
            }
        }
        #[cfg(feature = "quantization")]
        if let Self::Packed(weights) = self {
            if !weights.is_empty() {
                candle::bail!("unused packed projections in model schema")
            }
        }
        Ok(())
    }
}

impl From<Linear> for BackboneLinear {
    fn from(linear: Linear) -> Self {
        Self {
            linear: Projection::Dense(linear),
            chunk_rows: 0,
            #[cfg(feature = "cpu-blas")]
            blas: None,
        }
    }
}

impl Module for BackboneLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.chunk_rows == 0 {
            return self.project(input);
        }
        let (batch, sequence, width) = input.dims3()?;
        let rows = batch * sequence;
        let input = input.reshape((rows, width))?.contiguous()?;
        let mut outputs = Vec::with_capacity(rows.div_ceil(self.chunk_rows));
        for offset in (0..rows).step_by(self.chunk_rows) {
            let count = self.chunk_rows.min(rows - offset);
            let chunk = input.narrow(0, offset, count)?.contiguous()?;
            let chunk = if count == self.chunk_rows {
                chunk
            } else {
                Tensor::cat(
                    &[
                        chunk,
                        Tensor::zeros(
                            (self.chunk_rows - count, width),
                            input.dtype(),
                            input.device(),
                        )?,
                    ],
                    0,
                )?
            };
            // Keep the original rank-three/batch-one Linear call convention.
            let output = self.project(&chunk.unsqueeze(0)?)?.squeeze(0)?;
            outputs.push(output.narrow(0, 0, count)?);
        }
        let output = Tensor::cat(&outputs, 0)?;
        let output_width = output.dim(1)?;
        output.reshape((batch, sequence, output_width))
    }
}

impl BackboneLinear {
    fn project(&self, input: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cpu-blas")]
        if let (Some(runtime), Projection::Dense(linear)) = (&self.blas, &self.linear) {
            return runtime.forward(linear, input);
        }
        #[cfg(feature = "cpu-blas")]
        if let (Some(runtime), Projection::RuntimeLora { base, adapter }) =
            (&self.blas, &self.linear)
        {
            return runtime
                .forward(base, input)?
                .broadcast_add(&adapter.forward(input)?);
        }
        self.linear.forward(input)
    }
}

struct Attention {
    q_proj: BackboneLinear,
    k_proj: BackboneLinear,
    v_proj: BackboneLinear,
    o_proj: BackboneLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
    fp32_compute: bool,
    query_rows: usize,
    grouped_gqa: bool,
    direct_paged_attention: bool,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder, projections: &mut ProjectionSource) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let hidden = cfg.hidden_size;
        let q_proj = projections.linear(
            hidden,
            num_heads * head_dim * 2,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = projections.linear(
            hidden,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = projections.linear(
            hidden,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = projections.linear(
            num_heads * head_dim,
            hidden,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;
        let q_norm = effective_norm_weight(vb.get(head_dim, "q_norm.weight")?)?;
        let k_norm = effective_norm_weight(vb.get(head_dim, "k_norm.weight")?)?;
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
            fp32_compute: false,
            query_rows: 0,
            grouped_gqa: false,
            direct_paged_attention: false,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&mut LayerCache>,
    ) -> Result<Tensor> {
        let b = x.dims()[0];
        let seq = x.dims()[1];
        let q_gate = self.q_proj.forward(x)?; // [B, seq, num_heads*head_dim*2]
        let q_gate = q_gate.reshape((b, seq, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(3, 0, self.head_dim)?;
        let gate = q_gate.narrow(3, self.head_dim, self.head_dim)?; // [B, seq, num_heads, head_dim]

        let q = q.transpose(1, 2)?; // [B, num_heads, seq, head_dim]
        let q = rms_norm_effective(&q, &self.q_norm, self.eps)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = rms_norm_effective(&k, &self.k_norm, self.eps)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = apply_partial_rotary(&q, &k, cos, sin, self.rotary_dim)?;
        let retained_pages = cache
            .as_ref()
            .and_then(|c| c.pages.as_ref())
            .map(|pages| pages.append(&k, &v))
            .transpose()?;
        let activation_dtype = q.dtype();
        let (attn, retained) = if self.direct_paged_attention && retained_pages.is_some() {
            // Persistent single-row pages take the direct path. Independent
            // forwards and private multi-row branch workspaces retain the
            // original flat attention path and are recorded as such.
            (
                retained_pages
                    .as_ref()
                    .unwrap()
                    .attention(&q, self.query_rows)?,
                None,
            )
        } else {
            let (k, v) = if let Some(pages) = cache.as_ref().and_then(|c| c.pages.as_ref()) {
                pages.materialize_with(&k, &v)?
            } else {
                match cache
                    .as_ref()
                    .and_then(|cache| cache.key.as_ref().zip(cache.value.as_ref()))
                {
                    Some((past_key, past_value)) => (
                        Tensor::cat(&[past_key, &k], 2)?,
                        Tensor::cat(&[past_value, &v], 2)?,
                    ),
                    None => (k, v),
                }
            };
            // Retain unexpanded GQA tensors. A fork shares immutable prefix storage;
            // appending a branch produces new tensors and cannot mutate its parent.
            let retained = cache
                .as_ref()
                .filter(|_| retained_pages.is_none())
                .map(|_| (k.clone(), v.clone()));

            let n_rep = self.num_heads / self.num_kv_heads;
            let k = if n_rep > 1 && !self.grouped_gqa {
                repeat_interleave_head(&k, n_rep, 1)?
            } else {
                k
            };
            let v = if n_rep > 1 && !self.grouped_gqa {
                repeat_interleave_head(&v, n_rep, 1)?
            } else {
                v
            };

            let (q, k, v) = if self.fp32_compute {
                (
                    q.to_dtype(DType::F32)?,
                    k.to_dtype(DType::F32)?,
                    v.to_dtype(DType::F32)?,
                )
            } else {
                (q, k, v)
            };
            let attn = if self.grouped_gqa {
                attention::grouped_queries(&q, &k, &v, self.query_rows, mask)?
            } else if self.query_rows > 0 {
                attention::query_blocks(&q, &k, &v, self.query_rows)?
            } else {
                let mask =
                    mask.ok_or_else(|| candle::Error::Msg("missing full causal mask".into()))?;
                let scale = 1.0 / (self.head_dim as f64).sqrt();
                let scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
                let scores = scores.broadcast_add(&mask.to_dtype(scores.dtype())?)?;
                let probs = candle_nn::ops::softmax(&scores, 3)?;
                probs.matmul(&v)?
            }
            .to_dtype(activation_dtype)?;
            (attn, retained)
        };
        let attn = attn.transpose(1, 2)?; // [B, seq, num_heads, head_dim]
                                          // `attn_output_gate`: the gate is the same shape as each head (the
                                          // q_proj output is split in two: query + gate), so multiply elementwise.
                                          // Compute the gate in fp32 for stability, then cast back to the
                                          // activation dtype so the broadcast_mul matches `attn`.
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?.to_dtype(attn.dtype())?;
        let attn = attn
            .broadcast_mul(&gate)?
            .reshape((b, seq, self.num_heads * self.head_dim))?;
        let output = self.o_proj.forward(&attn)?;
        if let Some(cache) = cache {
            if let Some(pages) = retained_pages {
                cache.pages = Some(pages);
            } else if let Some((key, value)) = retained {
                cache.key = Some(key);
                cache.value = Some(value);
            }
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (linear attention)
// ---------------------------------------------------------------------------

struct LinearAttn {
    in_proj_qkv: BackboneLinear,
    in_proj_z: BackboneLinear,
    in_proj_b: BackboneLinear,
    in_proj_a: BackboneLinear,
    out_proj: BackboneLinear,
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
    cpu_delta_rule: bool,
    cpu_causal_conv: bool,
}

impl LinearAttn {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        _device: &Device,
        dtype: DType,
        projections: &mut ProjectionSource,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let num_v_heads = cfg.linear_num_value_heads;
        let num_k_heads = cfg.linear_num_key_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel = cfg.linear_conv_kernel_dim;

        let in_proj_qkv =
            projections.linear(hidden, conv_dim, cfg.attention_bias, vb.pp("in_proj_qkv"))?;
        let in_proj_z =
            projections.linear(hidden, value_dim, cfg.attention_bias, vb.pp("in_proj_z"))?;
        let in_proj_b =
            projections.linear(hidden, num_v_heads, cfg.attention_bias, vb.pp("in_proj_b"))?;
        let in_proj_a =
            projections.linear(hidden, num_v_heads, cfg.attention_bias, vb.pp("in_proj_a"))?;
        let out_proj =
            projections.linear(value_dim, hidden, cfg.attention_bias, vb.pp("out_proj"))?;
        let conv1d_w = vb
            .get((conv_dim, 1, conv_kernel), "conv1d.weight")?
            .to_dtype(DType::F32)?;
        let norm_w = vb.get(head_v_dim, "norm.weight")?.to_dtype(DType::F32)?;
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
            cpu_delta_rule: false,
            cpu_causal_conv: false,
        })
    }

    /// Causal depthwise conv1d over `[B, C, T]` (same length out):
    /// `out[c, l] = sum_k w[c, k] * x[c, l - (K-1) + k]`.
    fn causal_conv(&self, x: &Tensor) -> Result<Tensor> {
        if self.cpu_causal_conv {
            let out = crate::conv_cpu::causal(x, &self.conv1d_w)?;
            // Preserve the original FP32 SiLU before casting to activations.
            return candle_nn::ops::silu(&out)?.to_dtype(x.dtype());
        }
        let (b, conv_dim, seq) = x.dims3()?;
        let k = self.conv_kernel;
        let x_f = x.to_dtype(DType::F32)?;
        let mut out = Tensor::zeros((b, conv_dim, seq), DType::F32, x.device())?;
        for kk in 0..k {
            let shift = (k - 1) - kk;
            if shift >= seq {
                continue;
            }
            let w = self.conv1d_w.narrow(2, kk, 1)?.squeeze(1)?.unsqueeze(0)?; // [1, conv_dim, 1]
            let contrib = w.broadcast_mul(&x_f)?; // [B, conv_dim, seq]
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

    fn forward(&self, x: &Tensor, cache: Option<&mut LayerCache>) -> Result<Tensor> {
        let b = x.dims()[0];
        let seq = x.dims()[1];

        let mixed = self.in_proj_qkv.forward(x)?.transpose(1, 2)?; // [B, conv_dim, seq]
        let history = cache.as_ref().and_then(|cache| cache.convolution.as_ref());
        let history_len = history.map_or(0, |history| history.dims()[2]);
        let mixed = match history {
            Some(history) => Tensor::cat(&[history, &mixed], 2)?,
            None => mixed,
        };
        let keep = self.conv_kernel.saturating_sub(1).min(mixed.dims()[2]);
        let convolution = if cache.is_some() && keep > 0 {
            // Copy the small tail so it does not retain the entire prefix's
            // projection storage through a tensor view.
            Some(
                mixed
                    .narrow(2, mixed.dims()[2] - keep, keep)?
                    .force_contiguous()?
                    .detach(),
            )
        } else {
            None
        };
        let mixed = self.causal_conv(&mixed)?.narrow(2, history_len, seq)?; // [B, conv_dim, seq]
        let mixed = mixed.transpose(1, 2)?; // [B, seq, conv_dim]

        let q = mixed.narrow(2, 0, self.key_dim)?.reshape((
            b,
            seq,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let k = mixed.narrow(2, self.key_dim, self.key_dim)?.reshape((
            b,
            seq,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let v = mixed
            .narrow(2, 2 * self.key_dim, self.value_dim)?
            .reshape((b, seq, self.num_v_heads, self.head_v_dim))?;

        let z = self
            .in_proj_z
            .forward(x)?
            .reshape((b, seq, self.num_v_heads, self.head_v_dim))?;
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
        let q = if n_rep > 1 {
            repeat_interleave_head(&q, n_rep, 2)?
        } else {
            q
        };
        let k = if n_rep > 1 {
            repeat_interleave_head(&k, n_rep, 2)?
        } else {
            k
        };

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
        let initial_state = cache.as_ref().and_then(|cache| cache.recurrent.as_ref());
        let (out, recurrent) = if self.cpu_delta_rule {
            crate::delta_cpu::recurrent(&q, &k, &v, &g, &beta, initial_state)?
        } else {
            recurrent_gated_delta(&q, &k, &v, &g, &beta, initial_state)?
        };
        let out = out.to_dtype(self.dtype)?.transpose(1, 2)?; // [B, seq, num_v_heads, head_v]
                                                              // Apply the per-head gated RMSNorm over the last (head_v) dim, then
                                                              // flatten the value heads for the output projection.
        let out = rms_norm_gated(&out, &z, &self.norm_w, self.eps)?; // [B, seq, num_v_heads, head_v]
        let out = out.reshape((b, seq, self.value_dim))?;
        let output = self.out_proj.forward(&out)?;
        if let Some(cache) = cache {
            cache.recurrent = Some(recurrent);
            cache.convolution = convolution;
        }
        Ok(output)
    }
}

/// Per-token gated delta rule (matches `torch_recurrent_gated_delta_rule`).
pub(crate) fn recurrent_gated_delta(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (b, n_v, seq, head_k) = query.dims4()?;
    let head_v = value.dims().last().copied().unwrap_or(0);
    let mut state = match initial_state {
        Some(state) => state.clone(),
        None => Tensor::zeros((b, n_v, head_k, head_v), DType::F32, query.device())?,
    };
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
        let delta = v_i
            .broadcast_sub(&kv_mem)?
            .broadcast_mul(&b_i.unsqueeze(2)?)?; // [B, n_v, head_v]
        state = state.broadcast_add(&k_i.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?)?;

        let out_i = state.broadcast_mul(&q_i.unsqueeze(3)?)?.sum(2)?; // [B, n_v, head_v]
        outs.push(out_i);
    }
    Ok((Tensor::stack(&outs, 2)?, state))
}

// ---------------------------------------------------------------------------
// MLP + decoder layer
// ---------------------------------------------------------------------------

struct Mlp {
    gate_proj: BackboneLinear,
    up_proj: BackboneLinear,
    down_proj: BackboneLinear,
    act: Activation,
    cpu_fused_gate: bool,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder, projections: &mut ProjectionSource) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let gate_proj =
            projections.linear(hidden, intermediate, cfg.attention_bias, vb.pp("gate_proj"))?;
        let up_proj =
            projections.linear(hidden, intermediate, cfg.attention_bias, vb.pp("up_proj"))?;
        let down_proj =
            projections.linear(intermediate, hidden, cfg.attention_bias, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act: Activation::Silu,
            cpu_fused_gate: false,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.cpu_fused_gate {
            let gate = self.gate_proj.forward(x)?;
            let up = self.up_proj.forward(x)?;
            return self
                .down_proj
                .forward(&crate::gate_cpu::silu_mul(&gate, &up)?);
        }
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
    fn new(
        cfg: &Config,
        layer_type: LayerType,
        vb: VarBuilder,
        device: &Device,
        dtype: DType,
        projections: &mut ProjectionSource,
    ) -> Result<Self> {
        let input_layernorm =
            effective_norm_weight(vb.get(cfg.hidden_size, "input_layernorm.weight")?)?;
        let post_attention_layernorm =
            effective_norm_weight(vb.get(cfg.hidden_size, "post_attention_layernorm.weight")?)?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"), projections)?;
        let (linear_attn, self_attn) = match layer_type {
            LayerType::Linear => (
                Some(LinearAttn::new(
                    cfg,
                    vb.pp("linear_attn"),
                    device,
                    dtype,
                    projections,
                )?),
                None,
            ),
            LayerType::Full => (
                None,
                Some(Attention::new(cfg, vb.pp("self_attn"), projections)?),
            ),
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

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&mut LayerCache>,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let h = rms_norm_effective(x, &self.input_layernorm, self.eps)?;
        let h = if let Some(attn) = &self.linear_attn {
            attn.forward(&h, cache)?
        } else if let Some(attn) = &self.self_attn {
            attn.forward(&h, cos, sin, mask, cache)?
        } else {
            candle::bail!("decoder layer has neither linear nor full attention")
        };
        let h = h.broadcast_add(&residual)?;
        let normalized = rms_norm_effective(&h, &self.post_attention_layernorm, self.eps)?;
        let x = self.mlp.forward(&normalized)?.broadcast_add(&h)?;
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct LayerCache {
    pages: Option<PagedKv>,
    key: Option<Tensor>,
    value: Option<Tensor>,
    recurrent: Option<Tensor>,
    convolution: Option<Tensor>,
}

#[derive(Clone)]
struct ModelCache {
    tokens: usize,
    layers: Vec<LayerCache>,
}

impl ModelCache {
    fn branch_batch(&self, rows: usize) -> Result<Self> {
        let repeat = |tensor: &Option<Tensor>| {
            tensor
                .as_ref()
                .map(|tensor| Tensor::cat(&vec![tensor; rows], 0))
                .transpose()
        };
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let (key, value) = match &layer.pages {
                    Some(pages) => {
                        let (key, value) = pages.materialize()?;
                        (Some(key), Some(value))
                    }
                    None => (layer.key.clone(), layer.value.clone()),
                };
                Ok(LayerCache {
                    pages: None,
                    key: repeat(&key)?,
                    value: repeat(&value)?,
                    recurrent: repeat(&layer.recurrent)?,
                    convolution: repeat(&layer.convolution)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            tokens: self.tokens,
            layers,
        })
    }

    fn retention_bytes(&self, tokens: usize) -> Option<usize> {
        let mut bytes = tokens.checked_mul(8)?.checked_add(512)?;
        for layer in &self.layers {
            bytes = bytes.checked_add(512)?;
            if let Some(pages) = &layer.pages {
                bytes = bytes.checked_add(pages.retention_bytes()?)?;
            }
            for tensor in [
                &layer.key,
                &layer.value,
                &layer.recurrent,
                &layer.convolution,
            ]
            .into_iter()
            .flatten()
            {
                bytes = bytes.checked_add(
                    tensor
                        .elem_count()
                        .checked_mul(tensor.dtype().size_in_bytes())?,
                )?;
            }
        }
        Some(bytes)
    }

    fn compact(&self) -> Result<Self> {
        let copy = |tensor: &Option<Tensor>| {
            tensor
                .as_ref()
                .map(|t| t.force_contiguous().map(|t| t.detach()))
                .transpose()
        };
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                Ok(LayerCache {
                    pages: layer.pages.clone(),
                    key: copy(&layer.key)?,
                    value: copy(&layer.value)?,
                    recurrent: copy(&layer.recurrent)?,
                    convolution: copy(&layer.convolution)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            tokens: self.tokens,
            layers,
        })
    }
}

#[derive(Default)]
struct PrefixSnapshots {
    values: BTreeMap<Vec<u32>, (ModelCache, usize)>,
    fifo: VecDeque<Vec<u32>>,
    bytes: usize,
}

impl PrefixSnapshots {
    fn trim(&mut self, max_bytes: usize, incoming: usize) {
        while self.bytes > max_bytes.saturating_sub(incoming)
            || self.values.len() >= 16 && incoming > 0
        {
            let Some(key) = self.fifo.pop_front() else {
                break;
            };
            if let Some((_, bytes)) = self.values.remove(&key) {
                self.bytes -= bytes;
            }
        }
    }
}

#[derive(Clone)]
struct AttentionInputs {
    cos: Tensor,
    sin: Tensor,
    mask: Option<Tensor>,
}

#[derive(Default)]
struct AttentionInputCache {
    values: BTreeMap<(usize, usize), AttentionInputs>,
    fifo: VecDeque<(usize, usize)>,
    bytes: usize,
}

impl AttentionInputCache {
    const MAX_BYTES: usize = 4 * 1024 * 1024;
    const MAX_ENTRIES: usize = 32;
    fn insert(&mut self, key: (usize, usize), inputs: AttentionInputs) {
        let size = inputs.bytes();
        if size > Self::MAX_BYTES || self.values.contains_key(&key) {
            return;
        }
        while self.bytes + size > Self::MAX_BYTES || self.values.len() >= Self::MAX_ENTRIES {
            if let Some(key) = self.fifo.pop_front() {
                if let Some(previous) = self.values.remove(&key) {
                    self.bytes -= previous.bytes();
                }
            } else {
                break;
            }
        }
        self.values.insert(key, inputs);
        self.fifo.push_back(key);
        self.bytes += size;
    }
}

impl AttentionInputs {
    fn bytes(&self) -> usize {
        [&self.cos, &self.sin]
            .into_iter()
            .chain(self.mask.iter())
            .map(|tensor| tensor.elem_count() * tensor.dtype().size_in_bytes())
            .sum()
    }
}

pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: Tensor,
    rotary: RotaryEmbedding,
    eps: f32,
    device: Device,
    attention_inputs: Mutex<AttentionInputCache>,
    projection_chunk_rows: usize,
    fp32_attention: bool,
    attention_query_rows: usize,
    grouped_gqa: bool,
    direct_paged_attention: bool,
    kv_page_tokens: usize,
    runtime_lora_targets: usize,
    cpu_delta_rule: bool,
    cpu_causal_conv: bool,
    cpu_fused_gate: bool,
    #[cfg(feature = "cpu-blas")]
    cpu_blas: Option<Arc<crate::cpu_blas::Runtime>>,
}

impl Model {
    /// Build the text model from a `VarBuilder` rooted at the full weight
    /// prefix (`model.language_model.<module>` and `lm_head`).
    pub fn new(cfg: &Config, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        Self::new_with_projections(cfg, vb, device, dtype, &mut ProjectionSource::Dense)
    }

    fn new_with_projections(
        cfg: &Config,
        vb: VarBuilder,
        device: &Device,
        dtype: DType,
        projections: &mut ProjectionSource,
    ) -> Result<Self> {
        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("model.language_model.embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let lt = cfg.layer_types.get(i).copied().unwrap_or(LayerType::Linear);
            let layer = DecoderLayer::new(
                cfg,
                lt,
                vb.pp(format!("model.language_model.layers.{i}")),
                device,
                dtype,
                projections,
            )?;
            layers.push(layer);
        }
        projections.finish()?;
        let norm =
            effective_norm_weight(vb.get(cfg.hidden_size, "model.language_model.norm.weight")?)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary: RotaryEmbedding::new(cfg, device, dtype)?,
            eps: cfg.rms_norm_eps,
            device: device.clone(),
            attention_inputs: Mutex::new(AttentionInputCache::default()),
            projection_chunk_rows: 0,
            fp32_attention: false,
            attention_query_rows: 0,
            grouped_gqa: false,
            direct_paged_attention: false,
            kv_page_tokens: 0,
            runtime_lora_targets: 0,
            cpu_delta_rule: false,
            cpu_causal_conv: false,
            cpu_fused_gate: false,
            #[cfg(feature = "cpu-blas")]
            cpu_blas: None,
        })
    }

    fn set_projection_chunks(&mut self, rows: usize) {
        self.projection_chunk_rows = rows;
        self.visit_projections(|projection| projection.chunk_rows = rows);
    }

    fn visit_projections(&mut self, mut visit: impl FnMut(&mut BackboneLinear)) {
        for layer in &mut self.layers {
            for projection in [
                &mut layer.mlp.gate_proj,
                &mut layer.mlp.up_proj,
                &mut layer.mlp.down_proj,
            ] {
                visit(projection);
            }
            if let Some(attention) = &mut layer.linear_attn {
                for projection in [
                    &mut attention.in_proj_qkv,
                    &mut attention.in_proj_z,
                    &mut attention.in_proj_b,
                    &mut attention.in_proj_a,
                    &mut attention.out_proj,
                ] {
                    visit(projection);
                }
            }
            if let Some(attention) = &mut layer.self_attn {
                for projection in [
                    &mut attention.q_proj,
                    &mut attention.k_proj,
                    &mut attention.v_proj,
                    &mut attention.o_proj,
                ] {
                    visit(projection);
                }
            }
        }
    }

    #[cfg(feature = "cpu-blas")]
    pub(crate) fn set_cpu_blas_from_env(&mut self) -> CoreResult<()> {
        let Some(runtime) = crate::cpu_blas::configured()? else {
            return Ok(());
        };
        if !self.device.is_cpu() || self.embed_tokens.embeddings().dtype() != DType::F32 {
            return Err(Error::Unsupported(
                "OpenBLAS requires a CPU FP32 Qwen backbone".into(),
            ));
        }
        let mut compatible = true;
        self.visit_projections(|projection| {
            compatible &= match &projection.linear {
                Projection::Dense(linear) | Projection::RuntimeLora { base: linear, .. } => {
                    linear.weight().dtype() == DType::F32 && linear.weight().device().is_cpu()
                }
                #[cfg(feature = "quantization")]
                _ => false,
            };
        });
        if !compatible {
            return Err(Error::Unsupported(
                "OpenBLAS supports dense FP32 projections; packed weights are rejected".into(),
            ));
        }
        self.visit_projections(|projection| projection.blas = Some(runtime.clone()));
        self.cpu_blas = Some(runtime);
        Ok(())
    }

    #[cfg(feature = "cpu-blas")]
    pub(crate) fn record_cpu_blas(&self, extra: &mut BTreeMap<String, String>) {
        if let Some(runtime) = &self.cpu_blas {
            runtime.record(extra);
        }
    }

    fn set_fp32_attention(&mut self, enabled: bool) {
        self.fp32_attention = enabled;
        for layer in &mut self.layers {
            if let Some(attention) = &mut layer.self_attn {
                attention.fp32_compute = enabled;
            }
        }
    }

    pub(crate) fn set_attention_query_rows(&mut self, rows: usize) -> CoreResult<()> {
        if rows > 4096 || (rows > 0 && !self.device.is_cpu()) {
            return Err(Error::Unsupported(
                "attention query rows require CPU and 0..4096 rows".into(),
            ));
        }
        if rows > 0 && self.layers.iter().all(|layer| layer.self_attn.is_none()) {
            return Err(Error::Unsupported(
                "query blocking requires a full attention layer".into(),
            ));
        }
        self.attention_query_rows = rows;
        for layer in &mut self.layers {
            if let Some(attention) = &mut layer.self_attn {
                attention.query_rows = rows;
            }
        }
        self.attention_inputs = Mutex::new(AttentionInputCache::default());
        Ok(())
    }

    pub(crate) fn set_grouped_gqa(&mut self, enabled: bool) -> CoreResult<()> {
        if enabled
            && (!self.device.is_cpu() || self.layers.iter().all(|layer| layer.self_attn.is_none()))
        {
            return Err(Error::Unsupported(
                "grouped GQA requires CPU full attention layers".into(),
            ));
        }
        self.grouped_gqa = enabled;
        for layer in &mut self.layers {
            if let Some(attention) = &mut layer.self_attn {
                attention.grouped_gqa = enabled;
            }
        }
        Ok(())
    }

    pub(crate) fn set_cpu_delta_rule(&mut self, enabled: bool) {
        self.cpu_delta_rule = enabled;
        for layer in &mut self.layers {
            if let Some(attention) = &mut layer.linear_attn {
                attention.cpu_delta_rule = enabled;
            }
        }
    }

    pub(crate) fn set_cpu_fused_gate(&mut self, enabled: bool) {
        self.cpu_fused_gate = enabled;
        for layer in &mut self.layers {
            layer.mlp.cpu_fused_gate = enabled;
        }
    }

    pub(crate) fn set_cpu_causal_conv(&mut self, enabled: bool) {
        self.cpu_causal_conv = enabled;
        for layer in &mut self.layers {
            if let Some(attention) = &mut layer.linear_attn {
                attention.cpu_causal_conv = enabled;
            }
        }
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        self.forward_inner(ids, None)
    }

    fn empty_cache(&self) -> ModelCache {
        ModelCache {
            tokens: 0,
            layers: self
                .layers
                .iter()
                .map(|layer| LayerCache {
                    pages: (self.kv_page_tokens > 0 && layer.self_attn.is_some())
                        .then(|| PagedKv::new(self.kv_page_tokens)),
                    ..Default::default()
                })
                .collect(),
        }
    }

    fn forward_cached(&self, ids: &Tensor, cache: &mut ModelCache) -> Result<Tensor> {
        self.forward_inner(ids, Some(cache))
    }

    fn forward_inner(&self, ids: &Tensor, mut cache: Option<&mut ModelCache>) -> Result<Tensor> {
        let seq = ids.dims().last().copied().unwrap_or(0);
        let offset = cache.as_ref().map_or(0, |cache| cache.tokens);
        let inputs = self.attention_inputs(seq, offset)?;
        let mut hidden = self.embed_tokens.forward(ids)?; // [B, seq, hidden]
        for (index, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache.as_deref_mut().map(|cache| &mut cache.layers[index]);
            hidden = layer.forward(
                &hidden,
                &inputs.cos,
                &inputs.sin,
                inputs.mask.as_ref(),
                layer_cache,
            )?;
        }
        let hidden = rms_norm_effective(&hidden, &self.norm, self.eps)?;
        if let Some(cache) = cache {
            cache.tokens += seq;
        }
        Ok(hidden)
    }

    fn attention_inputs(&self, seq: usize, offset: usize) -> Result<AttentionInputs> {
        let key = (seq, offset);
        if let Ok(cache) = self.attention_inputs.lock() {
            if let Some(inputs) = cache.values.get(&key) {
                return Ok(inputs.clone());
            }
        }
        // Keys include the absolute offset, and this cache belongs to one
        // immutable model/device/dtype. No prompt, hidden or branch state is
        // retained. Build outside the lock; duplicate misses are harmless.
        let (cos, sin) = self.rotary.cos_sin(seq, offset, &self.device)?;
        let mask = if self.attention_query_rows == 0 {
            Some(causal_mask_at(seq, offset)?.to_device(&self.device)?)
        } else {
            None
        };
        let inputs = AttentionInputs { cos, sin, mask };
        if let Ok(mut cache) = self.attention_inputs.lock() {
            cache.insert(key, inputs.clone());
        }
        Ok(inputs)
    }
}

/// Causal suffix-to-prefix mask `[seq, offset + seq]` at absolute positions.
fn causal_mask_at(seq: usize, offset: usize) -> Result<Tensor> {
    let total = seq + offset;
    let mut data = vec![0.0f32; seq * total];
    for i in 0..seq {
        for j in 0..total {
            if offset + i < j {
                data[i * total + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (seq, total), &Device::Cpu)
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
            candle::Error::Msg(format!(
                "missing base weight `{base_key}` required for LoRA merge"
            ))
        })?;
        // Compute the LoRA delta in fp32 (candle's CPU matmul does not support
        // bf16) and only cast the merged result back to the target dtype.
        let delta = b.to_dtype(DType::F32)?.matmul(&a.to_dtype(DType::F32)?)?; // [out, in]
        let delta = delta.affine(lora_scale as f64, 0.0)?;
        let updated = base
            .to_dtype(DType::F32)?
            .broadcast_add(&delta)?
            .to_dtype(dtype)?;
        *base = updated;
    }
    Ok(())
}

/// Export the same CPU FP32 LoRA merge used by Kev inference as a new HF text
/// checkpoint for the pinned llama.cpp converter. No calibration is copied and
/// the source checkpoint is never written. The destination must not exist.
#[cfg(feature = "llamacpp")]
pub fn export_merged_hf(
    base_dir: &Path,
    adapter_dir: &Path,
    include_lm_head: bool,
    destination: &Path,
) -> CoreResult<usize> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(base_dir.join("config.json"))?)?;
    let mut text = config.get("text_config").unwrap_or(&config).clone();
    let model_type = text
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if !matches!(model_type, "qwen3_5_text" | "qwen3_5") {
        return Err(Error::Unsupported(
            "GGUF export requires dense Qwen3.5 text weights".into(),
        ));
    }
    let cfg = Config::from_value(&text).map_err(|e| Error::Package(e.to_string()))?;
    if cfg.attention_bias || cfg.linear_key_head_dim != cfg.linear_value_head_dim {
        return Err(Error::Unsupported("pinned GGUF profile requires bias-free projections and equal linear key/value head dimensions".into()));
    }
    let mut tensors =
        load_base_tensors_filtered(base_dir, &Device::Cpu, DType::F32, include_lm_head)?;
    let lora =
        candle::safetensors::load(adapter_dir.join("adapter_model.safetensors"), &Device::Cpu)
            .map_err(|e| Error::Package(e.to_string()))?;
    let (rank, alpha) = read_lora_hyperparams(&adapter_dir.join("adapter_config.json"))?;
    if rank == 0 {
        return Err(Error::Package(
            "GGUF export requires positive LoRA rank".into(),
        ));
    }
    merge_lora_into_map(&mut tensors, &lora, alpha / rank as f32, DType::F32)
        .map_err(|e| Error::Package(e.to_string()))?;
    for (name, tensor) in &tensors {
        if tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| Error::Package(e.to_string()))?
            .iter()
            .any(|x| !x.is_finite())
        {
            return Err(Error::Package(format!("non-finite exported weight {name}")));
        }
    }
    Model::new(
        &cfg,
        VarBuilder::from_tensors(tensors.clone(), DType::F32, &Device::Cpu),
        &Device::Cpu,
        DType::F32,
    )
    .map_err(|e| Error::Package(e.to_string()))?;
    if include_lm_head && tensors.contains_key("lm_head.bias") {
        return Err(Error::Unsupported(
            "pinned Qwen3.5 GGUF does not expose an LM-head bias".into(),
        ));
    }
    if include_lm_head && !tensors.contains_key("lm_head.weight") {
        return Err(Error::Package(
            "F3 export requires trained language-model weights".into(),
        ));
    }
    let tensors: HashMap<_, _> = tensors
        .into_iter()
        .map(|(name, tensor)| {
            let name = name
                .strip_prefix("model.language_model.")
                .map(|suffix| format!("model.{suffix}"))
                .unwrap_or(name);
            (name, tensor)
        })
        .collect();
    let bytes = tensors.values().map(|t| t.elem_count() * 4).sum();
    text["architectures"] = serde_json::json!(["Qwen3_5ForCausalLM"]);
    text["model_type"] = serde_json::json!("qwen3_5_text");
    // Kev has no vocabulary projection. Its unused auxiliary GGUF projection
    // shares embeddings; decision scores still use the external trained head.
    text["tie_word_embeddings"] = serde_json::json!(!include_lm_head);
    std::fs::create_dir(destination)?;
    candle::safetensors::save(&tensors, destination.join("model.safetensors"))
        .map_err(|e| Error::Package(e.to_string()))?;
    std::fs::write(
        destination.join("config.json"),
        serde_json::to_vec_pretty(&text)?,
    )?;
    Ok(bytes)
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
    model: Arc<Model>,
    head: Arc<Readout>,
    vocab_size: usize,
    input_vocab_size: usize,
    max_context: usize,
    dtype: String,
    device: Device,
    caches: BTreeMap<u64, ModelCache>,
    pending_prefills: BTreeMap<u64, (Vec<u32>, usize)>,
    prefixes: PrefixSnapshots,
    prefill_chunk_tokens: usize,
    base_weight_cache: bool,
}

enum Readout {
    LanguageModel(Linear),
    Pointer(PointerHead),
}

impl Qwen3_5Backend {
    /// Immutable CPU KV pages share full blocks across Kev forks. Attention
    /// materializes all original keys in order; this is not a paged kernel.
    pub fn with_kv_page_tokens(mut self, tokens: usize) -> CoreResult<Self> {
        if tokens != 0 && (!(16..=256).contains(&tokens) || !tokens.is_power_of_two()) {
            return Err(Error::Request(
                "KV page tokens must be 0 or a power of two in 16..256".into(),
            ));
        }
        if tokens > 0
            && (!self.device.is_cpu()
                || !matches!(self.head.as_ref(), Readout::Pointer(_))
                || self
                    .model
                    .layers
                    .iter()
                    .all(|layer| layer.self_attn.is_none()))
        {
            return Err(Error::Unsupported(
                "KV pages currently require CPU Kev with full attention".into(),
            ));
        }
        if tokens == 0 && self.model.direct_paged_attention {
            return Err(Error::Unsupported(
                "disable direct paged attention before KV pages".into(),
            ));
        }
        if tokens != self.model.kv_page_tokens {
            if !self.caches.is_empty()
                || !self.pending_prefills.is_empty()
                || !self.prefixes.values.is_empty()
            {
                return Err(Error::Unsupported(
                    "release retained/partial Qwen caches before changing KV pages".into(),
                ));
            }
            self.model_mut()?.kv_page_tokens = tokens;
        }
        Ok(self)
    }
    /// Read cached CPU FP32 Kev pages without full-KV concatenation. Softmax
    /// retains every causal key; page matmul/PV reductions require fresh gates.
    /// Configure pages and bounded query rows first, before handles/replicas.
    pub fn with_direct_paged_attention(mut self, enabled: bool) -> CoreResult<Self> {
        if enabled
            && (!self.device.is_cpu()
                || self.dtype != "fp32"
                || !matches!(self.head.as_ref(), Readout::Pointer(_))
                || self.model.kv_page_tokens == 0
                || self.model.attention_query_rows == 0)
        {
            return Err(Error::Unsupported(
                "direct paged attention requires CPU FP32 Kev, KV pages and bounded query rows"
                    .into(),
            ));
        }
        if enabled != self.model.direct_paged_attention {
            if !self.caches.is_empty()
                || !self.pending_prefills.is_empty()
                || !self.prefixes.values.is_empty()
            {
                return Err(Error::Unsupported(
                    "release retained/partial Qwen caches before changing direct paged attention"
                        .into(),
                ));
            }
            let model = self.model_mut()?;
            model.direct_paged_attention = enabled;
            for layer in &mut model.layers {
                if let Some(attention) = &mut layer.self_attn {
                    attention.direct_paged_attention = enabled;
                }
            }
        }
        Ok(self)
    }
    /// Group query heads by their existing K/V head without expanding K/V.
    /// Call shapes change; configure before caches/replicas and qualify anew.
    pub fn with_grouped_gqa(mut self, enabled: bool) -> CoreResult<Self> {
        if enabled != self.model.grouped_gqa {
            if !self.caches.is_empty() || !self.prefixes.values.is_empty() {
                return Err(Error::Unsupported(
                    "release retained Qwen caches before changing GQA execution".into(),
                ));
            }
            self.model_mut()?.set_grouped_gqa(enabled)?;
        }
        Ok(self)
    }
    /// Bound CPU attention score/mask rows; every causal key remains present.
    /// Reduction shapes change, so fresh conformance is required before serving.
    pub fn with_attention_query_rows(mut self, rows: usize) -> CoreResult<Self> {
        if rows == 0 && self.model.direct_paged_attention {
            return Err(Error::Unsupported(
                "disable direct paged attention before query blocks".into(),
            ));
        }
        if rows != self.model.attention_query_rows {
            if !self.caches.is_empty() || !self.prefixes.values.is_empty() {
                return Err(Error::Unsupported(
                    "release retained Qwen caches before changing attention query rows".into(),
                ));
            }
            self.model_mut()?.set_attention_query_rows(rows)?;
        }
        Ok(self)
    }
    /// Explicit immutable LP64 OpenBLAS profile, configured before any replica
    /// or retained/partial prefix. Default/uncompiled builds load no library.
    pub fn with_cpu_blas_from_env(self) -> CoreResult<Self> {
        if std::env::var_os("HUNCHO_CPU_BLAS_LIBRARY").is_none()
            && std::env::var_os("HUNCHO_CPU_BLAS_THREADS").is_none()
        {
            return Ok(self);
        }
        #[cfg(not(feature = "cpu-blas"))]
        {
            return Err(Error::Unsupported(
                "OpenBLAS requires --features cpu-blas".into(),
            ));
        }
        #[cfg(feature = "cpu-blas")]
        {
            if !self.caches.is_empty() || !self.prefixes.values.is_empty() {
                return Err(Error::Unsupported(
                    "configure OpenBLAS before creating Qwen caches".into(),
                ));
            }
            let mut backend = self;
            backend.model_mut()?.set_cpu_blas_from_env()?;
            Ok(backend)
        }
    }
    fn model_mut(&mut self) -> CoreResult<&mut Model> {
        Arc::get_mut(&mut self.model).ok_or_else(|| {
            Error::Unsupported("configure Qwen kernels before creating shared replicas".into())
        })
    }

    /// Bound the query length of native CPU Kev prefix attention. This changes
    /// projection/attention shapes, so paired and labeled qualification apply.
    /// It does not yield between jobs or implement a fair serving scheduler.
    pub fn with_prefill_chunk_tokens(mut self, tokens: usize) -> CoreResult<Self> {
        if tokens > 4096 {
            return Err(Error::Request(
                "prefill chunk tokens must be 0..4096".into(),
            ));
        }
        if tokens > 0
            && (!self.device.is_cpu() || !matches!(self.head.as_ref(), Readout::Pointer(_)))
        {
            return Err(Error::Unsupported(
                "chunked prefill currently supports CPU Kev only".into(),
            ));
        }
        if tokens != self.prefill_chunk_tokens
            && (!self.caches.is_empty() || !self.prefixes.values.is_empty())
        {
            return Err(Error::Unsupported(
                "release retained Qwen caches before changing prefill chunks".into(),
            ));
        }
        self.prefill_chunk_tokens = tokens;
        Ok(self)
    }
    /// Fuse CPU SiLU and gate multiplication with the original typed rounding.
    /// Configure before replicas or retained prefixes; qualification is required.
    pub fn with_cpu_fused_gate(mut self, enabled: bool) -> CoreResult<Self> {
        if enabled && !self.device.is_cpu() {
            return Err(Error::Unsupported("fused MLP gate is CPU-only".into()));
        }
        if enabled != self.model.cpu_fused_gate
            && (!self.caches.is_empty() || !self.prefixes.values.is_empty())
        {
            return Err(Error::Unsupported(
                "release retained Qwen caches before changing gate kernels".into(),
            ));
        }
        if enabled != self.model.cpu_fused_gate {
            self.model_mut()?.set_cpu_fused_gate(enabled);
        }
        Ok(self)
    }
    /// Optional CPU convolution buffers, with unchanged FP32 tap reduction.
    pub fn with_cpu_causal_conv(mut self, enabled: bool) -> CoreResult<Self> {
        if enabled && !self.device.is_cpu() {
            return Err(Error::Unsupported(
                "buffered causal convolution is CPU-only".into(),
            ));
        }
        if enabled != self.model.cpu_causal_conv
            && (!self.caches.is_empty() || !self.prefixes.values.is_empty())
        {
            return Err(Error::Unsupported(
                "release retained Qwen caches before changing convolution kernels".into(),
            ));
        }
        if enabled != self.model.cpu_causal_conv {
            self.model_mut()?.set_cpu_causal_conv(enabled);
        }
        Ok(self)
    }
    /// Optional CPU recurrence buffers; cache state remains immutable FP32.
    /// Requires qualification for the loaded model, dtype and CPU runtime.
    pub fn with_cpu_delta_rule(mut self, enabled: bool) -> CoreResult<Self> {
        if enabled && !self.device.is_cpu() {
            return Err(Error::Unsupported("buffered delta rule is CPU-only".into()));
        }
        if enabled != self.model.cpu_delta_rule
            && (!self.caches.is_empty() || !self.prefixes.values.is_empty())
        {
            return Err(Error::Unsupported(
                "release retained Qwen caches before changing delta-rule kernels".into(),
            ));
        }
        if enabled != self.model.cpu_delta_rule {
            self.model_mut()?.set_cpu_delta_rule(enabled);
        }
        Ok(self)
    }
    /// Experimental attention compute profile. Weights and retained KV keep
    /// their original dtype; only dense attention matmuls/softmax use FP32.
    /// Changing arithmetic requires qualification for this execution identity.
    pub fn with_fp32_attention(mut self, enabled: bool) -> CoreResult<Self> {
        if enabled != self.model.fp32_attention
            && (!self.caches.is_empty() || !self.prefixes.values.is_empty())
        {
            return Err(Error::Unsupported(
                "release retained Qwen caches before changing attention kernels".into(),
            ));
        }
        if enabled != self.model.fp32_attention {
            self.model_mut()?.set_fp32_attention(enabled);
        }
        Ok(self)
    }

    /// Experimental kernel profile; changing projection shapes requires
    /// qualification for the actual model/device/dtype before serving.
    pub fn with_projection_chunk_rows(mut self, rows: usize) -> CoreResult<Self> {
        if rows > 4096 {
            return Err(Error::Request(
                "projection chunk rows must be at most 4096 (zero disables)".into(),
            ));
        }
        if rows != self.model.projection_chunk_rows
            && (!self.caches.is_empty() || !self.prefixes.values.is_empty())
        {
            return Err(Error::Unsupported(
                "release retained Qwen caches before changing projection kernels".into(),
            ));
        }
        if rows != self.model.projection_chunk_rows {
            self.model_mut()?.set_projection_chunks(rows);
        }
        Ok(self)
    }

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
        Self::load_on_device(base_dir, adapter_dir, max_context, dtype, Device::Cpu)
    }

    /// Explicit device loader for F3. Stage conversion/LoRA merge on CPU and
    /// move both backbone and vocabulary head to the selected device.
    pub fn load_on_device(
        base_dir: &Path,
        adapter_dir: Option<&Path>,
        max_context: usize,
        dtype: impl Into<String>,
        device: Device,
    ) -> CoreResult<Self> {
        Self::load_with_head(
            base_dir,
            adapter_dir,
            None,
            max_context,
            dtype.into(),
            device,
            false,
            #[cfg(feature = "shared-base")]
            None,
        )
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
        Self::load_kev_on_device(
            base_dir,
            adapter_dir,
            head_path,
            max_context,
            dtype,
            Device::Cpu,
        )
    }

    /// Load Kev on the selected device, retaining FP32 pointer projections.
    pub fn load_kev_on_device(
        base_dir: &Path,
        adapter_dir: &Path,
        head_path: &Path,
        max_context: usize,
        dtype: impl Into<String>,
        device: Device,
    ) -> CoreResult<Self> {
        let dtype = dtype.into();
        if !matches!(dtype.as_str(), "fp32" | "fp16" | "f16") {
            return Err(Error::Unsupported(format!(
                "Kev's Candle backend supports fp32 or fp16, not `{dtype}`"
            )));
        }
        Self::load_with_head(
            base_dir,
            Some(adapter_dir),
            Some(head_path),
            max_context,
            dtype,
            device,
            false,
            #[cfg(feature = "shared-base")]
            None,
        )
    }

    /// Keep standard A/B adapter projections separate from immutable CPU FP32
    /// base weights. Different arithmetic requires fresh profile qualification.
    pub fn load_kev_runtime_lora(
        base_dir: &Path,
        adapter_dir: &Path,
        head_path: &Path,
        max_context: usize,
        dtype: impl Into<String>,
    ) -> CoreResult<Self> {
        Self::load_with_head(
            base_dir,
            Some(adapter_dir),
            Some(head_path),
            max_context,
            dtype.into(),
            Device::Cpu,
            true,
            #[cfg(feature = "shared-base")]
            None,
        )
    }

    /// CPU FP32 F3 backbone updates; the vocabulary projection is unchanged.
    /// Adapters targeting embeddings, norms or the LM head are rejected.
    pub fn load_runtime_lora(
        base_dir: &Path,
        adapter_dir: &Path,
        max_context: usize,
        dtype: impl Into<String>,
    ) -> CoreResult<Self> {
        Self::load_with_head(
            base_dir,
            Some(adapter_dir),
            None,
            max_context,
            dtype.into(),
            Device::Cpu,
            true,
            #[cfg(feature = "shared-base")]
            None,
        )
    }

    /// Share immutable CPU base tensors across independently merged adapters.
    /// Head, LoRA merge, activations and native cache handles remain isolated.
    #[cfg(feature = "shared-base")]
    pub fn load_kev_with_base_cache(
        base_dir: &Path,
        adapter_dir: &Path,
        head_path: &Path,
        max_context: usize,
        dtype: impl Into<String>,
        cache: &crate::shared_base::BaseWeightCache,
    ) -> CoreResult<Self> {
        let dtype = dtype.into();
        if !matches!(dtype.as_str(), "fp32" | "fp16" | "f16") {
            return Err(Error::Unsupported(format!(
                "Kev's Candle backend supports fp32 or fp16, not `{dtype}`"
            )));
        }
        Self::load_with_head(
            base_dir,
            Some(adapter_dir),
            Some(head_path),
            max_context,
            dtype,
            Device::Cpu,
            false,
            Some(cache),
        )
    }

    fn load_with_head(
        base_dir: &Path,
        adapter_dir: Option<&Path>,
        pointer_path: Option<&Path>,
        max_context: usize,
        dtype: String,
        device: Device,
        runtime_lora: bool,
        #[cfg(feature = "shared-base")] cache: Option<&crate::shared_base::BaseWeightCache>,
    ) -> CoreResult<Self> {
        if runtime_lora && (!device.is_cpu() || dtype != "fp32" || adapter_dir.is_none()) {
            return Err(Error::Unsupported(
                "runtime LoRA requires CPU FP32 Qwen with an adapter".into(),
            ));
        }
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
        log::info!(
            "loading Qwen3.5 on {} with {} weights",
            crate::device_label(&device),
            dtype_str,
        );

        let config_path = base_dir.join("config.json");
        let config = parse_config(&config_path)?;

        // Stage weights and merge LoRA on CPU. Turing cannot cast BF16
        // source tensors on CUDA, and staging avoids GPU merge temporaries.
        let weight_device = Device::Cpu;
        #[cfg(feature = "shared-base")]
        let base_weight_cache = match cache {
            Some(cache) => cache.enabled(),
            None => crate::shared_base::configured()?.is_some(),
        };
        #[cfg(not(feature = "shared-base"))]
        let base_weight_cache = false;
        #[cfg(feature = "clef")]
        if device.is_cuda() && dtype == DType::BF16 && !crate::device::supports_bf16(&device)? {
            return Err(Error::Unsupported(
                "Qwen BF16 execution requires a GPU with BF16 support; choose fp16 or fp32".into(),
            ));
        }
        #[cfg(feature = "shared-base")]
        let mut tensors = if let Some(cache) = cache {
            load_base_tensors_with_cache(
                base_dir,
                &weight_device,
                dtype,
                pointer_path.is_none(),
                cache,
            )?
        } else {
            load_base_tensors_filtered(base_dir, &weight_device, dtype, pointer_path.is_none())?
        };
        #[cfg(not(feature = "shared-base"))]
        let mut tensors =
            load_base_tensors_filtered(base_dir, &weight_device, dtype, pointer_path.is_none())?;
        if tensors.is_empty() {
            return Err(Error::Backend(format!(
                "no base-weight `*.safetensors` found in `{}`; the Qwen3.5 backend needs the \
                 full base weights (fetch the `{}` repo) in this directory",
                base_dir.display(),
                base_repo_hint(&config)
            )));
        }

        let mut projections = if runtime_lora {
            let rank = pointer_path
                .map(crate::kev::KevMetadata::load)
                .transpose()?
                .map(|meta| meta.lora_rank);
            ProjectionSource::RuntimeLora(runtime_lora::load(adapter_dir.unwrap(), &tensors, rank)?)
        } else {
            ProjectionSource::Dense
        };
        let runtime_lora_targets = if let ProjectionSource::RuntimeLora(adapters) = &projections {
            adapters.len()
        } else {
            0
        };
        if let Some(adapter_dir) = adapter_dir.filter(|_| !runtime_lora) {
            let adapter_path = adapter_dir.join("adapter_model.safetensors");
            let lora = candle::safetensors::load(&adapter_path, &weight_device).map_err(|e| {
                Error::Backend(
                    QwenError::Load(adapter_path.display().to_string(), e.to_string()).to_string(),
                )
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
                tensors.remove("lm_head.bias");
                Readout::Pointer(PointerHead::load(path, config.hidden_size, &device)?)
            }
            None => {
                let lm_head_w = tensors.remove("lm_head.weight").ok_or_else(|| {
                    Error::Backend("base weights are missing `lm_head.weight`".into())
                })?;
                let weight = lm_head_w.to_device(&device).map_err(|e| {
                    Error::Backend(format!("moving vocabulary weights to device: {e}"))
                })?;
                let bias = tensors
                    .remove("lm_head.bias")
                    .map(|bias| bias.to_device(&device))
                    .transpose()
                    .map_err(|e| {
                        Error::Backend(format!("moving vocabulary bias to device: {e}"))
                    })?;
                Readout::LanguageModel(Linear::new(weight, bias))
            }
        };
        let vocab_size = if pointer_path.is_some() {
            1
        } else {
            config.vocab_size
        };

        let tensors = tensors
            .into_iter()
            .map(|(name, tensor)| tensor.to_device(&device).map(|tensor| (name, tensor)))
            .collect::<candle::Result<HashMap<_, _>>>()
            .map_err(|e| Error::Backend(format!("moving Qwen3.5 weights to device: {e}")))?;
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);
        let mut model = Model::new_with_projections(&config, vb, &device, dtype, &mut projections)
            .map_err(|e| {
                Error::Backend(QwenError::Load("model".into(), e.to_string()).to_string())
            })?;
        model.runtime_lora_targets = runtime_lora_targets;

        Ok(Qwen3_5Backend {
            model: Arc::new(model),
            head: Arc::new(head),
            vocab_size,
            input_vocab_size: config.vocab_size,
            max_context,
            dtype: dtype_str,
            device,
            caches: BTreeMap::new(),
            pending_prefills: BTreeMap::new(),
            prefixes: PrefixSnapshots::default(),
            prefill_chunk_tokens: 0,
            base_weight_cache,
        })
    }
}

/// A short human hint for the base vocabulary/repo of the loaded config.
fn base_repo_hint(_cfg: &Config) -> &'static str {
    "Qwen/Qwen3.5-*"
}

fn parse_config(path: &Path) -> CoreResult<Config> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        Error::Backend(QwenError::Config(path.display().to_string(), e.to_string()).to_string())
    })?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        Error::Backend(QwenError::Config(path.display().to_string(), e.to_string()).to_string())
    })?;
    Config::from_value(&v).map_err(|e| {
        Error::Backend(QwenError::Config(path.display().to_string(), e.to_string()).to_string())
    })
}

/// Read `lora_alpha`/`r` from an `adapter_config.json`.
fn read_lora_hyperparams(path: &Path) -> CoreResult<(usize, f32)> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        Error::Backend(QwenError::Config(path.display().to_string(), e.to_string()).to_string())
    })?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        Error::Backend(QwenError::Config(path.display().to_string(), e.to_string()).to_string())
    })?;
    let r = v.get("r").and_then(|x| x.as_u64()).unwrap_or(16) as usize;
    let alpha = v.get("lora_alpha").and_then(|x| x.as_f64()).unwrap_or(32.0) as f32;
    Ok((r, alpha))
}

/// Load all `*.safetensors` files in a directory into a single tensor map,
/// converting to the requested dtype.
#[cfg(any(feature = "clef", test))]
pub(crate) fn load_base_tensors(
    dir: &Path,
    device: &Device,
    dtype: DType,
) -> CoreResult<HashMap<String, Tensor>> {
    load_base_tensors_filtered(dir, device, dtype, true)
}

fn load_base_tensors_filtered(
    dir: &Path,
    device: &Device,
    dtype: DType,
    include_lm_head: bool,
) -> CoreResult<HashMap<String, Tensor>> {
    #[cfg(feature = "shared-base")]
    if let Some(cache) = crate::shared_base::configured()? {
        return load_base_tensors_with_cache(dir, device, dtype, include_lm_head, cache);
    }
    #[cfg(not(feature = "shared-base"))]
    if std::env::var_os("HUNCHO_BASE_CACHE_BYTES").is_some_and(|value| value != "0") {
        return Err(Error::Unsupported(
            "HUNCHO_BASE_CACHE_BYTES requires feature shared-base".into(),
        ));
    }
    load_base_tensor_files(&base_tensor_files(dir)?, device, dtype, include_lm_head)
}

#[cfg(feature = "shared-base")]
fn load_base_tensors_with_cache(
    dir: &Path,
    device: &Device,
    dtype: DType,
    include_lm_head: bool,
    cache: &crate::shared_base::BaseWeightCache,
) -> CoreResult<HashMap<String, Tensor>> {
    if !device.is_cpu() {
        return Err(Error::Unsupported(
            "shared bases currently support CPU storage only".into(),
        ));
    }
    let files = base_tensor_files(dir)?;
    cache.load(&files, dtype, include_lm_head, || {
        load_base_tensor_files(&files, device, dtype, include_lm_head)
    })
}

fn base_tensor_files(dir: &Path) -> CoreResult<Vec<std::path::PathBuf>> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| {
            Error::Backend(QwenError::Load(dir.display().to_string(), e.to_string()).to_string())
        })?
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .filter(|p| {
            // The adapter sits next to the base weights in the package dir; do
            // not treat `adapter_model.safetensors` as a base shard.
            p.file_name()
                .map(|n| n != "adapter_model.safetensors" && n != "joint_head.safetensors")
                .unwrap_or(true)
        })
        .collect();
    entries.sort();
    Ok(entries)
}

fn load_base_tensor_files(
    entries: &[std::path::PathBuf],
    device: &Device,
    dtype: DType,
    include_lm_head: bool,
) -> CoreResult<HashMap<String, Tensor>> {
    let mut map = HashMap::new();
    for p in entries {
        // Model files must remain unchanged while loading. Map the shard so
        // excluded vision/MTP weights are never materialized on the device.
        let raw = unsafe { candle::safetensors::MmapedSafetensors::new(p) }.map_err(|e| {
            Error::Backend(QwenError::Load(p.display().to_string(), e.to_string()).to_string())
        })?;
        for (name, _) in raw.tensors() {
            // The Qwen3.5 base is a multimodal conditional-generation model; the
            // F3 adapter and this text backend only need the text backbone and
            // the LM head. Drop the vision tower and MTP tensors so we do not
            // hold the whole ~19 GB checkpoint in memory.
            let Some(k) = canonical_weight_name(&name) else {
                continue;
            };
            // Pointer inference never uses the vocabulary projection. Filter
            // before mmap materialization/casting, rather than loading a large
            // head only to remove it afterward.
            if !include_lm_head && k.starts_with("lm_head.") {
                continue;
            }
            let v = raw
                .load(&name, device)
                .and_then(|v| v.to_dtype(dtype))
                .map_err(|e| {
                    Error::Backend(QwenError::Load(k.clone(), e.to_string()).to_string())
                })?;
            if map.insert(k.clone(), v).is_some() {
                return Err(Error::Package(format!(
                    "duplicate canonical base tensor `{k}`"
                )));
            }
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
    fn replica(&self) -> CoreResult<Box<dyn Backend>> {
        if !self.device.is_cpu() {
            return Err(Error::Unsupported(
                "shared Qwen replicas currently support CPU only".into(),
            ));
        }
        Ok(Box::new(Self {
            model: self.model.clone(),
            head: self.head.clone(),
            vocab_size: self.vocab_size,
            input_vocab_size: self.input_vocab_size,
            max_context: self.max_context,
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            caches: BTreeMap::new(),
            pending_prefills: BTreeMap::new(),
            prefixes: PrefixSnapshots::default(),
            prefill_chunk_tokens: self.prefill_chunk_tokens,
            base_weight_cache: self.base_weight_cache,
        }))
    }

    fn id(&self) -> BackendId {
        BackendId::Candle
    }

    fn capabilities(&self) -> Capabilities {
        let mut extra = BTreeMap::from([
            ("device".into(), crate::device_label(&self.device)),
            ("native_execution".into(), "candle-qwen35-v1".into()),
        ]);
        if self.supports_fork_batch() {
            extra.insert(
                "cached_branch_batch".into(),
                "cpu-kev-equal-suffix-v1".into(),
            );
        }
        if self.supports_padded_fork_batch() {
            extra.insert(
                "cached_branch_padding".into(),
                "cpu-kev-causal-suffix-v1".into(),
            );
        }
        if self.base_weight_cache {
            extra.insert("base_weight_cache".into(), "content-checked-cpu-v1".into());
        }
        crate::cpu_profile::record(&mut extra);
        #[cfg(feature = "cpu-blas")]
        self.model.record_cpu_blas(&mut extra);
        if self.model.projection_chunk_rows > 0 {
            extra.insert(
                "projection_chunk_rows".into(),
                self.model.projection_chunk_rows.to_string(),
            );
        }
        if self.model.fp32_attention {
            extra.insert("attention_compute_dtype".into(), "fp32".into());
        }
        if self.model.attention_query_rows > 0 {
            extra.insert(
                "attention_query_rows".into(),
                self.model.attention_query_rows.to_string(),
            );
            extra.insert("attention_execution".into(), "cpu-query-blocks-v1".into());
        }
        if self.model.grouped_gqa {
            extra.insert("gqa_execution".into(), "cpu-grouped-queries-v1".into());
        }
        if self.model.runtime_lora_targets > 0 {
            extra.insert(
                "adapter_execution".into(),
                "cpu-fp32-runtime-lora-v1".into(),
            );
            extra.insert(
                "runtime_lora_targets".into(),
                self.model.runtime_lora_targets.to_string(),
            );
        }
        if self.model.direct_paged_attention {
            extra.insert("paged_attention".into(), "cpu-page-qk-pv-fp32-v1".into());
            extra.insert(
                "paged_attention_fallback".into(),
                "flat-independent-and-branch-batch-v1".into(),
            );
        }
        if self.model.kv_page_tokens > 0 {
            extra.insert(
                "kv_storage".into(),
                if self.model.direct_paged_attention {
                    "cpu-cow-pages-direct-v1"
                } else {
                    "cpu-cow-pages-materialize-v1"
                }
                .into(),
            );
            extra.insert(
                "kv_page_tokens".into(),
                self.model.kv_page_tokens.to_string(),
            );
        }
        if self.model.cpu_delta_rule {
            extra.insert("delta_rule_execution".into(), "cpu-buffered-v1".into());
        }
        if self.model.cpu_causal_conv {
            extra.insert("causal_conv_execution".into(), "cpu-buffered-v1".into());
        }
        if self.model.cpu_fused_gate {
            extra.insert("mlp_gate_execution".into(), "cpu-fused-silu-mul-v1".into());
        }
        if self.prefill_chunk_tokens > 0 {
            extra.insert(
                "prefill_chunk_tokens".into(),
                self.prefill_chunk_tokens.to_string(),
            );
        }
        #[cfg(feature = "quantization")]
        if let Some(scheme) = quantized::Scheme::from_dtype(&self.dtype) {
            extra.insert("weight_quantization".into(), scheme.profile().into());
            extra.insert("activation_dtype".into(), "fp32".into());
            extra.insert("recurrent_state_dtype".into(), "fp32".into());
            extra.insert("pointer_head_dtype".into(), "fp32".into());
            extra.insert("projection_kernel".into(), "candle-packed-cpu-v1".into());
        }
        if self.device.is_cuda() && matches!(self.head.as_ref(), Readout::LanguageModel(_)) {
            extra.insert("device_path".into(), "qwen-f3-cuda".into());
        }
        Capabilities {
            id: BackendId::Candle,
            dtype: self.dtype.clone(),
            max_context: self.max_context,
            supports_fork: matches!(self.head.as_ref(), Readout::Pointer(_)),
            supports_lora: true,
            families: vec![match self.head.as_ref() {
                Readout::Pointer(_) => Family::F2,
                _ => Family::F3,
            }],
            extra,
        }
    }

    fn forward(&mut self, input: ForwardInput) -> CoreResult<ForwardOutput> {
        if input
            .fork_from
            .is_some_and(|handle| self.pending_prefills.contains_key(&handle.id))
        {
            return Err(Error::Backend(
                "a partial prefill cannot produce question readouts".into(),
            ));
        }
        if input.retain_cache {
            return Err(Error::Unsupported(
                "use prefill to obtain an explicit cache handle".into(),
            ));
        }
        if input
            .tokens
            .iter()
            .any(|&token| token as usize >= self.input_vocab_size)
        {
            return Err(Error::Backend(
                "Qwen3.5 token ID is outside the vocabulary".into(),
            ));
        }
        let mut cache = input
            .fork_from
            .map(|handle| {
                self.caches
                    .get(&handle.id)
                    .cloned()
                    .ok_or_else(|| Error::Backend("unknown Qwen3.5 cache handle".into()))
            })
            .transpose()?;
        let prefix_len = cache.as_ref().map_or(0, |cache| cache.tokens);
        if input
            .positions
            .iter()
            .any(|&position| position >= input.tokens.len())
        {
            return Err(Error::Backend(
                "Qwen3.5 readout position is outside the token sequence".into(),
            ));
        }
        let sequence_len = prefix_len
            .checked_add(input.tokens.len())
            .ok_or_else(|| Error::Backend("sequence length overflow".into()))?;
        if sequence_len > self.max_context {
            return Err(Error::Backend(format!(
                "sequence length {} exceeds candle max_context {}",
                sequence_len, self.max_context
            )));
        }
        if matches!(self.head.as_ref(), Readout::LanguageModel(_)) {
            if let Some(codes) = &input.logit_codes {
                if codes.is_empty() || codes.iter().any(|&code| code as usize >= self.vocab_size) {
                    return Err(Error::Backend(
                        "requested vocabulary codes are empty or out of range".into(),
                    ));
                }
            }
        }
        if input.positions.is_empty() && cache.is_none() {
            return Ok(ForwardOutput::Logits {
                positions: Vec::new(),
                values: CoreTensor::zeros(vec![0, self.vocab_size]),
            });
        }
        if input.tokens.is_empty() {
            return Err(Error::Backend(
                "cached continuation requires nonempty suffix tokens".into(),
            ));
        }
        let candle =
            |e: candle::Error| Error::Backend(QwenError::Inference(e.to_string()).to_string());
        let ids = Tensor::new(input.tokens.as_slice(), &self.device)
            .map_err(&candle)?
            .unsqueeze(0)
            .map_err(&candle)?;
        let hidden = match &mut cache {
            Some(cache) => self.model.forward_cached(&ids, cache),
            None => self.model.forward(&ids),
        }
        .map_err(&candle)?;
        if input.positions.is_empty() {
            self.caches
                .insert(input.fork_from.unwrap().id, cache.unwrap());
            return Ok(ForwardOutput::Logits {
                positions: Vec::new(),
                values: CoreTensor::zeros(vec![0, self.vocab_size]),
            });
        }

        let output = self.readout(&hidden, &input)?;
        // Commit only after the complete forward/readout succeeds. A failed
        // branch remains at its previous offset and state.
        if let (Some(handle), Some(cache)) = (input.fork_from, cache) {
            self.caches.insert(handle.id, cache);
        }
        Ok(output)
    }

    fn supports_batch(&self) -> bool {
        true
    }

    fn forward_batch(&mut self, inputs: Vec<ForwardInput>) -> CoreResult<Vec<ForwardOutput>> {
        self.forward_independent_batch(inputs, false)
    }

    fn supports_fork_batch(&self) -> bool {
        self.device.is_cpu() && matches!(self.head.as_ref(), Readout::Pointer(_))
    }

    fn fork_batch_limits(&self) -> huncho_core::backend::BatchLimits {
        let mut limits = self.batch_limits();
        limits.max_rows = if self.supports_fork_batch() {
            limits
                .max_rows
                .min(63)
                .min(64usize.saturating_sub(self.caches.len()))
        } else {
            0
        };
        limits
    }

    fn forward_fork_batch(
        &mut self,
        parent: CacheHandle,
        inputs: Vec<ForwardInput>,
        work: &mut huncho_core::backend::ForkBatchWork,
    ) -> CoreResult<Vec<ForwardOutput>> {
        self.forward_cached_batch(parent, inputs, false, work)
    }

    fn supports_padded_fork_batch(&self) -> bool {
        self.supports_fork_batch()
    }

    fn forward_padded_fork_batch(
        &mut self,
        parent: CacheHandle,
        inputs: Vec<ForwardInput>,
        work: &mut huncho_core::backend::ForkBatchWork,
    ) -> CoreResult<Vec<ForwardOutput>> {
        self.forward_cached_batch(parent, inputs, true, work)
    }

    fn supports_padded_batch(&self) -> bool {
        self.device.is_cpu()
    }

    fn forward_padded_batch(
        &mut self,
        inputs: Vec<ForwardInput>,
    ) -> CoreResult<Vec<ForwardOutput>> {
        if !self.device.is_cpu() {
            return Err(Error::Unsupported(
                "Qwen padded batches currently support CPU only".into(),
            ));
        }
        self.forward_independent_batch(inputs, true)
    }

    fn fork(&mut self, handle: CacheHandle) -> CoreResult<CacheHandle> {
        if self.pending_prefills.contains_key(&handle.id) {
            return Err(Error::Backend("a partial prefill cannot be forked".into()));
        }
        self.check_cache_capacity()?;
        let cache = self
            .caches
            .get(&handle.id)
            .cloned()
            .ok_or_else(|| Error::Backend("unknown Qwen3.5 cache handle".into()))?;
        let fork = crate::next_cache_handle()?;
        self.caches.insert(fork.id, cache);
        Ok(fork)
    }

    fn prefill(&mut self, tokens: &[u32]) -> CoreResult<CacheHandle> {
        self.compute_prefix(tokens, &mut Default::default())
    }

    fn release_cache(&mut self, handle: CacheHandle) -> CoreResult<()> {
        self.pending_prefills.remove(&handle.id);
        self.caches
            .remove(&handle.id)
            .map(|_| ())
            .ok_or_else(|| Error::Backend("unknown Qwen3.5 cache handle".into()))
    }

    fn clear_prefix_cache(&mut self) -> CoreResult<()> {
        self.prefixes = PrefixSnapshots::default();
        Ok(())
    }

    fn prefill_cached(
        &mut self,
        tokens: &[u32],
        max_bytes: usize,
    ) -> CoreResult<huncho_core::backend::CachedPrefill> {
        self.cached_prefix(tokens, max_bytes, &mut Default::default())
    }

    fn prefill_cached_with_work(
        &mut self,
        tokens: &[u32],
        max_bytes: usize,
        work: &mut huncho_core::backend::PrefillWork,
    ) -> CoreResult<huncho_core::backend::CachedPrefill> {
        self.cached_prefix(tokens, max_bytes, work)
    }

    fn supports_resumable_prefill(&self) -> bool {
        self.prefill_chunk_tokens > 0
            && self.device.is_cpu()
            && matches!(self.head.as_ref(), Readout::Pointer(_))
    }

    fn begin_resumable_prefill(
        &mut self,
        tokens: &[u32],
        max_bytes: usize,
    ) -> CoreResult<huncho_core::backend::CachedPrefill> {
        if !self.supports_resumable_prefill() {
            return Err(Error::Unsupported(
                "resumable prefill requires CPU Kev with configured chunks".into(),
            ));
        }
        self.validate_prefix(tokens)?;
        self.check_cache_capacity()?;
        self.prefixes.trim(max_bytes, 0);
        let handle = crate::next_cache_handle()?;
        if max_bytes > 0 {
            if let Some((cache, _)) = self.prefixes.values.get(tokens) {
                self.caches.insert(handle.id, cache.clone());
                return Ok(huncho_core::backend::CachedPrefill { handle, hit: true });
            }
        }
        self.caches.insert(handle.id, self.model.empty_cache());
        self.pending_prefills
            .insert(handle.id, (tokens.to_vec(), max_bytes));
        Ok(huncho_core::backend::CachedPrefill { handle, hit: false })
    }

    fn advance_resumable_prefill(
        &mut self,
        handle: CacheHandle,
        work: &mut huncho_core::backend::PrefillWork,
    ) -> CoreResult<bool> {
        let (tokens, max_bytes) = self.pending_prefills.get(&handle.id).ok_or_else(|| {
            Error::Backend("unknown or completed resumable prefill handle".into())
        })?;
        let mut cache = self
            .caches
            .get(&handle.id)
            .ok_or_else(|| Error::Backend("unknown resumable cache handle".into()))?
            .clone();
        let offset = cache.tokens;
        let end = tokens.len().min(offset + self.prefill_chunk_tokens);
        let chunk = &tokens[offset..end];
        let ids = Tensor::new(chunk, &self.device)
            .and_then(|ids| ids.unsqueeze(0))
            .map_err(|e| Error::Backend(e.to_string()))?;
        work.forward_calls += 1;
        work.processed_tokens += chunk.len() as u64;
        if offset == self.prefill_chunk_tokens {
            work.chunked_prefills += 1;
        }
        self.model
            .forward_cached(&ids, &mut cache)
            .map_err(|e| Error::Backend(e.to_string()))?;
        self.caches.insert(handle.id, cache);
        let complete = end == tokens.len();
        let max_bytes = *max_bytes;
        if complete {
            let (tokens, _) = self.pending_prefills.remove(&handle.id).unwrap();
            self.retain_prefix(handle, &tokens, max_bytes);
        }
        Ok(complete)
    }
}

impl Qwen3_5Backend {
    fn compute_prefix(
        &mut self,
        tokens: &[u32],
        work: &mut huncho_core::backend::PrefillWork,
    ) -> CoreResult<CacheHandle> {
        if !matches!(self.head.as_ref(), Readout::Pointer(_)) {
            return Err(Error::Unsupported(
                "prefix caching is currently qualified only for the pointer readout".into(),
            ));
        }
        self.check_cache_capacity()?;
        self.validate_prefix(tokens)?;
        let mut cache = self.model.empty_cache();
        let chunk_size = if self.prefill_chunk_tokens == 0 {
            tokens.len()
        } else {
            self.prefill_chunk_tokens
        };
        for (index, chunk) in tokens.chunks(chunk_size).enumerate() {
            let ids = Tensor::new(chunk, &self.device)
                .and_then(|ids| ids.unsqueeze(0))
                .map_err(|e| Error::Backend(e.to_string()))?;
            work.forward_calls += 1;
            work.processed_tokens += chunk.len() as u64;
            if index == 1 {
                work.chunked_prefills += 1;
            }
            self.model
                .forward_cached(&ids, &mut cache)
                .map_err(|e| Error::Backend(e.to_string()))?;
        }
        let handle = crate::next_cache_handle()?;
        self.caches.insert(handle.id, cache);
        Ok(handle)
    }

    fn cached_prefix(
        &mut self,
        tokens: &[u32],
        max_bytes: usize,
        work: &mut huncho_core::backend::PrefillWork,
    ) -> CoreResult<huncho_core::backend::CachedPrefill> {
        self.prefixes.trim(max_bytes, 0);
        self.check_cache_capacity()?;
        if max_bytes > 0 {
            if let Some((cache, _)) = self.prefixes.values.get(tokens) {
                let handle = crate::next_cache_handle()?;
                self.caches.insert(handle.id, cache.clone());
                return Ok(huncho_core::backend::CachedPrefill { handle, hit: true });
            }
        }
        let handle = self.compute_prefix(tokens, work)?;
        self.retain_prefix(handle, tokens, max_bytes);
        Ok(huncho_core::backend::CachedPrefill { handle, hit: false })
    }

    fn validate_prefix(&self, tokens: &[u32]) -> CoreResult<()> {
        if tokens.is_empty()
            || tokens.len() > self.max_context
            || tokens
                .iter()
                .any(|&token| token as usize >= self.input_vocab_size)
        {
            return Err(Error::Backend(
                "prefill must be nonempty, contain valid token IDs and fit max_context".into(),
            ));
        }
        Ok(())
    }

    fn retain_prefix(&mut self, handle: CacheHandle, tokens: &[u32], max_bytes: usize) {
        // Concurrent resumable jobs can complete the same exact prefix.
        // Keep one immutable snapshot and charge it once.
        if self.prefixes.values.contains_key(tokens) {
            return;
        }
        let cache = &self.caches[&handle.id];
        if let Some(bytes) = cache
            .retention_bytes(tokens.len())
            .filter(|bytes| *bytes <= max_bytes)
        {
            // Make compact immutable copies before retention so small views
            // cannot retain larger temporary projection storage. Copy failures
            // leave the already-computed caller-owned prefix valid and uncached.
            if let Ok(snapshot) = cache.compact() {
                self.prefixes.trim(max_bytes, bytes);
                self.prefixes
                    .values
                    .insert(tokens.to_vec(), (snapshot, bytes));
                self.prefixes.fifo.push_back(tokens.to_vec());
                self.prefixes.bytes += bytes;
            }
        }
    }
}

impl Qwen3_5Backend {
    fn forward_cached_batch(
        &mut self,
        parent: CacheHandle,
        inputs: Vec<ForwardInput>,
        padded: bool,
        work: &mut huncho_core::backend::ForkBatchWork,
    ) -> CoreResult<Vec<ForwardOutput>> {
        if !self.supports_fork_batch() {
            return Err(Error::Unsupported(
                "cached-branch batching requires CPU Kev".into(),
            ));
        }
        let seq = inputs
            .iter()
            .map(|input| input.tokens.len())
            .max()
            .unwrap_or(0);
        let prefix = self
            .caches
            .get(&parent.id)
            .ok_or_else(|| Error::Backend("unknown Qwen3.5 cache handle".into()))?;
        if self.pending_prefills.contains_key(&parent.id)
            || prefix.tokens == 0
            || inputs.is_empty()
            || inputs.len() > 63
            || self.caches.len().saturating_add(inputs.len()) > 64
            || seq == 0
            || prefix
                .tokens
                .checked_add(seq)
                .map_or(true, |n| n > self.max_context)
            || inputs.iter().any(|input| {
                input.tokens.is_empty()
                    || (!padded && input.tokens.len() != seq)
                    || input.fork_from.is_some()
                    || input.retain_cache
                    || input.logit_codes.is_some()
                    || input.positions.is_empty()
                    || input.positions.iter().any(|&p| p >= input.tokens.len())
                    || input
                        .tokens
                        .iter()
                        .any(|&t| t as usize >= self.input_vocab_size)
            })
        {
            return Err(Error::Backend(
                "Kev branch batches require a complete parent, 1..=63 nonempty valid suffixes (equal lengths without padding) and available cache slots".into(),
            ));
        }
        let mut branches = Vec::with_capacity(inputs.len());
        let result = (|| {
            for _ in &inputs {
                branches.push(self.fork(parent)?);
                work.cache_forks += 1;
            }
            // Each row starts from the exact same immutable prefix, including
            // full KV, GDN recurrence and causal convolution. Tensor::cat owns
            // the private workspace; no prefix or published branch is mutated.
            let mut cache = self.caches[&parent.id]
                .branch_batch(inputs.len())
                .map_err(|e| Error::Backend(e.to_string()))?;
            let tokens: Vec<_> = inputs
                .iter()
                .flat_map(|input| {
                    input
                        .tokens
                        .iter()
                        .copied()
                        .chain(std::iter::repeat(0))
                        .take(seq)
                })
                .collect();
            let ids = Tensor::from_vec(tokens, (inputs.len(), seq), &self.device)
                .map_err(|e| Error::Backend(e.to_string()))?;
            work.forward_calls += 1;
            work.processed_tokens += (inputs.len() * seq) as u64;
            work.batch_calls += u64::from(inputs.len() > 1);
            let padding = inputs.len() * seq - inputs.iter().map(|i| i.tokens.len()).sum::<usize>();
            work.padded_tokens += padding as u64;
            work.padded_batch_calls += u64::from(padding > 0);
            let hidden = self
                .model
                .forward_cached(&ids, &mut cache)
                .map_err(|e| Error::Backend(e.to_string()))?;
            inputs
                .iter()
                .enumerate()
                .map(|(row, input)| {
                    let hidden = hidden
                        .narrow(0, row, 1)
                        .map_err(|e| Error::Backend(e.to_string()))?;
                    self.readout(&hidden, input)
                })
                .collect()
        })();
        // Cleanup must also happen after allocation/inference/readout failure.
        for branch in branches {
            self.caches.remove(&branch.id);
        }
        result
    }

    fn forward_independent_batch(
        &mut self,
        inputs: Vec<ForwardInput>,
        padded: bool,
    ) -> CoreResult<Vec<ForwardOutput>> {
        if inputs.is_empty() || inputs.len() > 64 {
            return Err(Error::Backend(
                "Qwen3.5 batch must contain 1..=64 independent rows".into(),
            ));
        }
        let seq = inputs.iter().map(|input| input.tokens.len()).max().unwrap();
        if seq == 0
            || seq > self.max_context
            || inputs.iter().any(|input| {
                input.tokens.is_empty()
                    || (!padded && input.tokens.len() != seq)
                    || input.fork_from.is_some()
                    || input.retain_cache
                    || input.positions.iter().any(|&p| p >= input.tokens.len())
                    || input
                        .tokens
                        .iter()
                        .any(|&token| token as usize >= self.input_vocab_size)
                    || (matches!(self.head.as_ref(), Readout::LanguageModel(_))
                        && input.logit_codes.as_ref().is_some_and(|codes| {
                            codes.is_empty() || codes.iter().any(|&c| c as usize >= self.vocab_size)
                        }))
            })
        {
            return Err(Error::Backend(
                "Qwen3.5 batches require valid independent rows without cache handles; ordinary batches require equal lengths"
                    .into(),
            ));
        }
        let mut tokens = Vec::with_capacity(seq * inputs.len());
        for input in &inputs {
            let end = tokens.len() + seq;
            tokens.extend_from_slice(&input.tokens);
            tokens.resize(end, 0);
        }
        let ids = Tensor::from_vec(tokens, (inputs.len(), seq), &self.device)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let hidden = self
            .model
            .forward(&ids)
            .map_err(|e| Error::Backend(e.to_string()))?;
        inputs
            .iter()
            .enumerate()
            .map(|(row, input)| {
                let hidden = hidden
                    .narrow(0, row, 1)
                    .map_err(|e| Error::Backend(e.to_string()))?;
                self.readout(&hidden, input)
            })
            .collect()
    }

    fn readout(&self, hidden: &Tensor, input: &ForwardInput) -> CoreResult<ForwardOutput> {
        if input.positions.is_empty() {
            return Ok(ForwardOutput::Logits {
                positions: Vec::new(),
                values: CoreTensor::zeros(vec![0, self.vocab_size]),
            });
        }
        let candle =
            |e: candle::Error| Error::Backend(QwenError::Inference(e.to_string()).to_string());
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
        let (logits, codes) = match (self.head.as_ref(), input.logit_codes.clone()) {
            (Readout::LanguageModel(head), Some(codes)) => {
                // The decision distribution only needs these vocabulary rows.
                // Preserve the trained projection/bias and native matmul dtype;
                // temperature and candidate softmax remain in core.
                let indices = Tensor::new(codes.as_slice(), &self.device).map_err(&candle)?;
                let weight = head.weight().index_select(&indices, 0).map_err(&candle)?;
                let bias = head
                    .bias()
                    .map(|bias| bias.index_select(&indices, 0))
                    .transpose()
                    .map_err(&candle)?;
                let logits = Linear::new(weight, bias)
                    .forward(&selected)
                    .map_err(&candle)?;
                (logits, Some(codes))
            }
            (Readout::LanguageModel(head), None) => {
                (head.forward(&selected).map_err(&candle)?, None)
            }
            (Readout::Pointer(head), _) => {
                let decide = hidden
                    .narrow(1, input.tokens.len() - 1, 1)
                    .and_then(|t| t.squeeze(0))
                    .map_err(&candle)?;
                (head.forward(&decide, &selected).map_err(&candle)?, None)
            }
        };
        let values = core_from_tensor(&logits.to_dtype(DType::F32).map_err(&candle)?)?;
        Ok(match codes {
            Some(codes) => ForwardOutput::SelectedLogits {
                positions: input.positions.clone(),
                codes,
                values,
            },
            None => ForwardOutput::Logits {
                positions: input.positions.clone(),
                values,
            },
        })
    }

    fn check_cache_capacity(&self) -> CoreResult<()> {
        if self.caches.len() >= 64 {
            Err(Error::Backend(
                "Qwen3.5 retained cache limit reached; release unused handles".into(),
            ))
        } else {
            Ok(())
        }
    }
}

/// Convert a 2-D candle tensor into a core tensor.
fn core_from_tensor(t: &Tensor) -> CoreResult<CoreTensor> {
    let dims = t.dims();
    let data = t
        .flatten_all()
        .and_then(|flat| flat.to_vec1::<f32>())
        .map_err(|e| Error::Backend(QwenError::Inference(e.to_string()).to_string()))?;
    CoreTensor::new(dims.to_vec(), data)
}

#[cfg(test)]
mod tests {
    #[test]
    fn failed_native_branch_inference_releases_every_temporary_handle() {
        use super::*;
        let root = Path::new("tests/fixtures/tiny_kev");
        let mut backend =
            Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, "fp32").unwrap();
        let parent = backend.prefill(&[1, 2]).unwrap();
        let before = backend.caches[&parent.id].tokens;
        let vocab = backend.input_vocab_size;
        // Pass admission then fail the actual native embedding lookup.
        backend.input_vocab_size += 1;
        let mut work = huncho_core::backend::ForkBatchWork::default();
        let invalid = ForwardInput::new(vec![vocab as u32, 1], vec![0, 1]);
        assert!(backend
            .forward_fork_batch(parent, vec![invalid; 2], &mut work)
            .is_err());
        assert_eq!(
            (work.cache_forks, work.forward_calls, work.batch_calls),
            (2, 1, 1)
        );
        assert_eq!(work.processed_tokens, 4);
        assert_eq!(backend.caches.len(), 1);
        assert_eq!(backend.caches[&parent.id].tokens, before);
        let invalid = ForwardInput::new(vec![vocab as u32, 1], vec![0, 1]);
        let mut padded_work = huncho_core::backend::ForkBatchWork::default();
        assert!(backend
            .forward_padded_fork_batch(
                parent,
                vec![invalid, ForwardInput::new(vec![1], vec![0])],
                &mut padded_work
            )
            .is_err());
        assert_eq!(
            (
                padded_work.cache_forks,
                padded_work.forward_calls,
                padded_work.padded_batch_calls,
                padded_work.padded_tokens
            ),
            (2, 1, 1, 1)
        );
        assert_eq!(backend.caches.len(), 1);
        assert_eq!(backend.caches[&parent.id].tokens, before);
        backend.input_vocab_size = vocab;
        backend
            .forward_fork_batch(
                parent,
                vec![ForwardInput::new(vec![3, 4], vec![0, 1]); 2],
                &mut Default::default(),
            )
            .unwrap();
        assert_eq!(backend.caches.len(), 1);
        backend.release_cache(parent).unwrap();
    }

    #[cfg(feature = "shared-base")]
    #[test]
    fn runtime_lora_shares_even_targeted_base_storage_and_survives_cache_eviction() {
        let root = Path::new("tests/fixtures/tiny_kev");
        let cache = crate::shared_base::BaseWeightCache::new(8 << 20);
        let load = || {
            Qwen3_5Backend::load_with_head(
                root,
                Some(root),
                Some(&root.join("head.pt")),
                512,
                "fp32".into(),
                Device::Cpu,
                true,
                Some(&cache),
            )
            .unwrap()
        };
        let mut a = load();
        let adapter = tempfile::tempdir().unwrap();
        std::fs::copy(
            root.join("adapter_config.json"),
            adapter.path().join("adapter_config.json"),
        )
        .unwrap();
        let weights =
            candle::safetensors::load(root.join("adapter_model.safetensors"), &Device::Cpu)
                .unwrap();
        let changed: HashMap<_, _> = weights
            .into_iter()
            .map(|(name, value)| {
                let value = if name.ends_with(".lora_B.weight") {
                    value.affine(-3., 0.).unwrap()
                } else {
                    value
                };
                (name, value)
            })
            .collect();
        candle::safetensors::save(&changed, adapter.path().join("adapter_model.safetensors"))
            .unwrap();
        let mut b = Qwen3_5Backend::load_with_head(
            root,
            Some(adapter.path()),
            Some(&root.join("head.pt")),
            512,
            "fp32".into(),
            Device::Cpu,
            true,
            Some(&cache),
        )
        .unwrap();
        assert!(!Arc::ptr_eq(&a.model, &b.model));
        let projection =
            |model: &Model| match &model.layers[1].self_attn.as_ref().unwrap().q_proj.linear {
                Projection::RuntimeLora { base, .. } => base.weight().clone(),
                _ => panic!("target must retain a runtime update"),
            };
        let (aw, bw) = (projection(&a.model), projection(&b.model));
        {
            let (first, _) = aw.storage_and_layout();
            let (second, _) = bw.storage_and_layout();
            assert!(
                std::ptr::eq(&*first, &*second),
                "targeted base storage must be shared"
            );
        }
        assert_eq!(cache.stats().unwrap().hits, 1);
        let input = ForwardInput::new(vec![1, 2, 3, 4], vec![0, 2, 3]);
        let original = a.forward(input.clone()).unwrap();
        let expected = b.forward(input.clone()).unwrap();
        assert_ne!(original.values().data(), expected.values().data());
        let parent = a.prefill(&[1, 2]).unwrap();
        assert!(b.fork(parent).is_err());
        cache.clear().unwrap();
        assert_eq!(cache.stats().unwrap().charged_bytes, 0);
        a.release_cache(parent).unwrap();
        drop(a);
        assert_eq!(
            expected.values().data(),
            b.forward(input).unwrap().values().data()
        );
        let (first, _) = aw.storage_and_layout();
        let (second, _) = bw.storage_and_layout();
        assert!(std::ptr::eq(&*first, &*second));
    }

    #[cfg(feature = "shared-base")]
    #[test]
    fn separate_adapter_models_keep_shared_untargeted_embedding_storage_after_base_eviction() {
        let root = Path::new("tests/fixtures/tiny_kev");
        let cache = crate::shared_base::BaseWeightCache::new(8 << 20);
        let a = Qwen3_5Backend::load_kev_with_base_cache(
            root,
            root,
            &root.join("head.pt"),
            512,
            "fp32",
            &cache,
        )
        .unwrap();
        let b = Qwen3_5Backend::load_kev_with_base_cache(
            root,
            root,
            &root.join("head.pt"),
            512,
            "fp32",
            &cache,
        )
        .unwrap();
        assert!(!Arc::ptr_eq(&a.model, &b.model));
        assert!(!Arc::ptr_eq(&a.head, &b.head));
        let (first, _) = a.model.embed_tokens.embeddings().storage_and_layout();
        let (second, _) = b.model.embed_tokens.embeddings().storage_and_layout();
        assert!(std::ptr::eq(&*first, &*second));
        assert_eq!(
            a.capabilities().extra["base_weight_cache"],
            "content-checked-cpu-v1"
        );
        assert_eq!(
            b.capabilities().extra["base_weight_cache"],
            "content-checked-cpu-v1"
        );
        cache.clear().unwrap();
        assert_eq!(cache.stats().unwrap().charged_bytes, 0);
        assert!(std::ptr::eq(&*first, &*second));
    }

    #[test]
    fn cpu_replicas_share_loaded_weights_but_never_copy_active_or_retained_state() {
        let root = Path::new("tests/fixtures/tiny_kev");
        let mut backend = Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, "fp32")
            .unwrap()
            .with_cpu_delta_rule(true)
            .unwrap()
            .with_cpu_causal_conv(true)
            .unwrap();
        let cached = backend.prefill_cached(&[1, 2, 3], 1 << 20).unwrap();
        let mut replica = backend.replica().unwrap();
        assert_eq!(Arc::strong_count(&backend.model), 2);
        assert_eq!(Arc::strong_count(&backend.head), 2);
        let mut continued = ForwardInput::new(vec![4, 5], vec![0, 1]);
        continued.fork_from = Some(cached.handle);
        assert!(replica.forward(continued).is_err());
        assert!(replica.release_cache(cached.handle).is_err());
        let fresh = replica.prefill_cached(&[1, 2, 3], 1 << 20).unwrap();
        assert!(!fresh.hit);
        replica.release_cache(fresh.handle).unwrap();
        let hit = backend.prefill_cached(&[1, 2, 3], 1 << 20).unwrap();
        assert!(hit.hit);
        backend.release_cache(hit.handle).unwrap();
        backend.release_cache(cached.handle).unwrap();
        backend.clear_prefix_cache().unwrap();
        assert!(backend.model_mut().is_err());
        drop(replica);
        assert!(backend.model_mut().is_ok());
    }

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
        m.insert(
            "model.language_model.embed_tokens.weight".into(),
            rand((cfg.vocab_size, h), device),
        );
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
            m.insert(
                format!("{p}.post_attention_layernorm.weight"),
                rand(h, device),
            );
            m.insert(
                format!("{p}.mlp.gate_proj.weight"),
                rand((inter, h), device),
            );
            m.insert(format!("{p}.mlp.up_proj.weight"), rand((inter, h), device));
            m.insert(
                format!("{p}.mlp.down_proj.weight"),
                rand((h, inter), device),
            );
            if i == 0 {
                m.insert(
                    format!("{p}.linear_attn.in_proj_qkv.weight"),
                    rand((conv_dim, h), device),
                );
                m.insert(
                    format!("{p}.linear_attn.in_proj_z.weight"),
                    rand((value_dim, h), device),
                );
                m.insert(
                    format!("{p}.linear_attn.in_proj_b.weight"),
                    rand((n_v, h), device),
                );
                m.insert(
                    format!("{p}.linear_attn.in_proj_a.weight"),
                    rand((n_v, h), device),
                );
                m.insert(
                    format!("{p}.linear_attn.out_proj.weight"),
                    rand((h, value_dim), device),
                );
                m.insert(
                    format!("{p}.linear_attn.conv1d.weight"),
                    rand((conv_dim, 1, cfg.linear_conv_kernel_dim), device),
                );
                m.insert(
                    format!("{p}.linear_attn.norm.weight"),
                    rand(cfg.linear_value_head_dim, device),
                );
                m.insert(format!("{p}.linear_attn.A_log"), rand(n_v, device));
                m.insert(format!("{p}.linear_attn.dt_bias"), rand(n_v, device));
            } else {
                m.insert(
                    format!("{p}.self_attn.q_proj.weight"),
                    rand((2 * heads * head_dim, h), device),
                );
                m.insert(
                    format!("{p}.self_attn.k_proj.weight"),
                    rand((kv_heads * head_dim, h), device),
                );
                m.insert(
                    format!("{p}.self_attn.v_proj.weight"),
                    rand((kv_heads * head_dim, h), device),
                );
                m.insert(
                    format!("{p}.self_attn.o_proj.weight"),
                    rand((h, heads * head_dim), device),
                );
                m.insert(
                    format!("{p}.self_attn.q_norm.weight"),
                    rand(head_dim, device),
                );
                m.insert(
                    format!("{p}.self_attn.k_norm.weight"),
                    rand(head_dim, device),
                );
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
    fn convolution_cache_tail_retains_only_its_visible_elements_on_cpu() {
        let root = Path::new("tests/fixtures/tiny_kev");
        for dtype in ["fp32", "fp16"] {
            let mut backend =
                Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype).unwrap();
            let handle = backend.prefill(&[1; 64]).unwrap();
            {
                let tail = backend.caches[&handle.id].layers[0]
                    .convolution
                    .as_ref()
                    .unwrap();
                assert_eq!(tail.dims()[2], 3);
                let (storage, layout) = tail.storage_and_layout();
                let allocated = match &*storage {
                    candle::Storage::Cpu(candle::CpuStorage::F32(data)) => data.len(),
                    candle::Storage::Cpu(candle::CpuStorage::F16(data)) => data.len(),
                    _ => panic!("expected CPU fp32/fp16 tail"),
                };
                assert_eq!(
                    allocated,
                    tail.elem_count(),
                    "retained tail must not own the complete prefix projection"
                );
                assert_eq!(layout.start_offset(), 0);
                assert!(layout.contiguous_offsets().is_some());
            }
            let fork = backend.fork(handle).unwrap();
            backend.release_cache(handle).unwrap();
            let mut suffix = ForwardInput::new(vec![2, 3], vec![0, 1]);
            suffix.fork_from = Some(fork);
            let cached = backend.forward(suffix).unwrap();
            let mut all = vec![1; 64];
            all.extend([2, 3]);
            let independent = backend
                .forward(ForwardInput::new(all, vec![64, 65]))
                .unwrap();
            for temperature in [0.75, 1., 2.40605] {
                let actual =
                    huncho_core::calibration::calibrate(cached.values().data(), temperature)
                        .unwrap();
                let expected =
                    huncho_core::calibration::calibrate(independent.values().data(), temperature)
                        .unwrap();
                assert_eq!(
                    huncho_core::calibration::argmax(&actual),
                    huncho_core::calibration::argmax(&expected)
                );
                assert!(actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| (a - b).abs() <= 1e-4));
            }
            backend.release_cache(fork).unwrap();
        }
    }

    #[test]
    fn forward_produces_finite_hidden() {
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let model = build_model(&cfg, &device);
        let ids = Tensor::new(&[1u32, 2, 3, 4][..], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let hidden = model.forward(&ids).unwrap();
        assert_eq!(hidden.dims(), &[1, 4, 16]);
        let v3 = hidden.to_vec3::<f32>().unwrap();
        assert!(v3.iter().flatten().flatten().all(|x| x.is_finite()));
    }

    #[test]
    fn attention_input_reuse_is_offset_aware_and_bounded() {
        let cfg = tiny_cfg();
        let model = build_model(&cfg, &Device::Cpu);
        let initial = model.attention_inputs(3, 0).unwrap();
        let repeated = model.attention_inputs(3, 0).unwrap();
        assert_eq!(
            initial.mask.as_ref().unwrap().id(),
            repeated.mask.as_ref().unwrap().id()
        );
        assert_eq!(initial.cos.id(), repeated.cos.id());
        let offset = model.attention_inputs(3, 2).unwrap();
        assert_eq!(offset.mask.as_ref().unwrap().dims(), &[3, 5]);
        assert_ne!(
            initial.cos.to_vec2::<f32>().unwrap()[0],
            offset.cos.to_vec2::<f32>().unwrap()[0]
        );
        let mask = offset.mask.as_ref().unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(mask[0][2], 0.);
        assert_eq!(mask[0][3], f32::NEG_INFINITY);
        for seq in 1..=40 {
            model.attention_inputs(seq, 0).unwrap();
        }
        {
            let cache = model.attention_inputs.lock().unwrap();
            assert_eq!(cache.values.len(), AttentionInputCache::MAX_ENTRIES);
            assert!(cache.bytes <= AttentionInputCache::MAX_BYTES);
            assert!(!cache.values.contains_key(&(1, 0)));
        }
        let large = model.attention_inputs(1100, 0).unwrap();
        assert!(large.bytes() > AttentionInputCache::MAX_BYTES);
        assert!(!model
            .attention_inputs
            .lock()
            .unwrap()
            .values
            .contains_key(&(1100, 0)));
    }

    #[test]
    fn query_profile_removes_quadratic_setup_and_invalidates_old_masks() {
        let mut model = build_model(&tiny_cfg(), &Device::Cpu);
        assert!(model.attention_inputs(3, 0).unwrap().mask.is_some());
        model.set_attention_query_rows(7).unwrap();
        assert!(model.attention_inputs.lock().unwrap().values.is_empty());
        for (seq, offset) in [(3, 0), (1100, 0), (1100, 37)] {
            let inputs = model.attention_inputs(seq, offset).unwrap();
            assert!(inputs.mask.is_none());
            assert_eq!(
                inputs.bytes(),
                (inputs.cos.elem_count() + inputs.sin.elem_count()) * 4
            );
        }
        model.set_attention_query_rows(0).unwrap();
        assert!(model.attention_inputs.lock().unwrap().values.is_empty());
        assert_eq!(
            model.attention_inputs(3, 2).unwrap().mask.unwrap().dims(),
            [3, 5]
        );
    }

    #[test]
    fn forward_is_causal() {
        // Changing later tokens must not alter position-0 hidden of either the
        // DeltaNet (layer 0) or the masked full-attention (layer 1) path.
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let model = build_model(&cfg, &device);
        let ids_a = Tensor::new(&[5u32, 6, 7][..], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let ids_b = Tensor::new(&[5u32, 10, 11][..], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let ha = model
            .forward(&ids_a)
            .unwrap()
            .index_select(&Tensor::new(&[0u32], &device).unwrap(), 1)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        let hb = model
            .forward(&ids_b)
            .unwrap()
            .index_select(&Tensor::new(&[0u32], &device).unwrap(), 1)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        for (x, y) in ha[0][0].iter().zip(hb[0][0].iter()) {
            assert!(
                (x - y).abs() < 1e-4,
                "position-0 hidden changed: {x} vs {y}"
            );
        }
    }

    #[test]
    fn direct_paged_cached_execution_never_materializes_complete_kv() {
        let root = Path::new("tests/fixtures/tiny_kev");
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
        let row = &fixture["cases"][0]["rows"][0];
        let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
        let positions: Vec<usize> = serde_json::from_value(row["positions"].clone()).unwrap();
        let prefix = row["prefix_len"].as_u64().unwrap() as usize;
        let mut results = Vec::new();
        for direct in [false, true] {
            let mut backend =
                Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, "fp32")
                    .unwrap()
                    .with_kv_page_tokens(16)
                    .unwrap()
                    .with_attention_query_rows(7)
                    .unwrap()
                    .with_direct_paged_attention(direct)
                    .unwrap();
            kv_pages::reset_path_counts();
            let parent = backend.prefill(&tokens[..prefix]).unwrap();
            let branch = backend.fork(parent).unwrap();
            let mut input = ForwardInput::new(
                tokens[prefix..].to_vec(),
                positions.iter().map(|p| p - prefix).collect(),
            );
            input.fork_from = Some(branch);
            let output = backend.forward(input).unwrap();
            let counts = kv_pages::path_counts();
            assert_eq!(counts, if direct { (0, 2) } else { (2, 0) });
            for layer in &backend.caches[&branch.id].layers {
                if layer.pages.is_some() {
                    assert!(layer.key.is_none() && layer.value.is_none());
                }
            }
            results.push(output.values().data().to_vec());
            backend.release_cache(branch).unwrap();
            backend.release_cache(parent).unwrap();
        }
        for temperature in [0.75, 1., 2.40605] {
            let a = huncho_core::calibration::calibrate(&results[0], temperature).unwrap();
            let b = huncho_core::calibration::calibrate(&results[1], temperature).unwrap();
            assert!(a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= 1e-4));
        }
    }

    #[test]
    fn backend_returns_logits_shape() {
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let weights = tiny_weights(&cfg, &device);
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &device);
        let model = Model::new(&cfg, vb, &device, DType::F32).unwrap();
        let lm_head = linear_b(
            cfg.hidden_size,
            cfg.vocab_size,
            false,
            VarBuilder::from_tensors(weights, DType::F32, &device).pp("lm_head"),
        )
        .unwrap();

        let mut backend = Qwen3_5Backend {
            model: Arc::new(model),
            head: Arc::new(Readout::LanguageModel(lm_head)),
            vocab_size: cfg.vocab_size,
            input_vocab_size: cfg.vocab_size,
            max_context: 32,
            dtype: "fp32".into(),
            device,
            caches: BTreeMap::new(),
            pending_prefills: BTreeMap::new(),
            prefixes: PrefixSnapshots::default(),
            prefill_chunk_tokens: 0,
            base_weight_cache: false,
        };
        let out = backend
            .forward(ForwardInput::new(vec![1, 2, 3, 4], vec![3]))
            .unwrap();
        assert_eq!(out.values().shape(), &[1, 32]);
        assert!(out.values().data().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn lo_merge_changes_weights() {
        let device = Device::Cpu;
        let mut map = HashMap::new();
        map.insert(
            "model.language_model.layers.0.mlp.gate_proj.weight".into(),
            Tensor::zeros((8, 4), DType::F32, &device).unwrap(),
        );
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
        let merged = map["model.language_model.layers.0.mlp.gate_proj.weight"]
            .to_vec2::<f32>()
            .unwrap();
        assert!(
            merged.iter().flatten().all(|x| (*x - 4.0).abs() < 1e-4),
            "expected merged=4, got {merged:?}"
        );
    }

    #[test]
    fn pointer_loading_filters_unused_vocabulary_weights_before_materialization() {
        let dir = tempfile::tempdir().unwrap();
        let weights: HashMap<String, Tensor> = HashMap::from([
            (
                "lm_head.weight".into(),
                Tensor::zeros((32, 16), DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "lm_head.bias".into(),
                Tensor::zeros(32, DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "model.language_model.norm.weight".into(),
                Tensor::zeros(16, DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "model.visual.norm.weight".into(),
                Tensor::zeros(16, DType::F32, &Device::Cpu).unwrap(),
            ),
        ]);
        candle::safetensors::save(&weights, dir.path().join("model.safetensors")).unwrap();
        let pointer =
            load_base_tensors_filtered(dir.path(), &Device::Cpu, DType::F32, false).unwrap();
        assert_eq!(pointer.len(), 1);
        assert!(pointer.contains_key("model.language_model.norm.weight"));
        let lm = load_base_tensors(dir.path(), &Device::Cpu, DType::F32).unwrap();
        assert_eq!(lm.len(), 3);
        assert!(lm.contains_key("lm_head.weight"));
    }

    #[test]
    fn projection_chunks_preserve_order_bias_dtype_and_partial_rows() {
        for dtype in [DType::F32, DType::F16] {
            let device = Device::Cpu;
            let weight = Tensor::new(
                &[[0.5f32, 0.25, -0.5, 1.0], [-0.25, 0.5, 1.0, 0.25]],
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let bias = Tensor::new(&[0.5f32, -0.25], &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let linear = Linear::new(weight, Some(bias));
            let input = Tensor::from_vec(
                (0..48).map(|v| v as f32 / 4.0).collect(),
                (3, 4, 4),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let expected = linear
                .forward(&input)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut projection = BackboneLinear::from(linear);
            for rows in [0, 1, 5, 12, 64] {
                projection.chunk_rows = rows;
                let output = projection.forward(&input).unwrap();
                assert_eq!(output.dims(), &[3, 4, 2]);
                assert_eq!(output.dtype(), dtype);
                assert_eq!(
                    output
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                    expected
                );
            }
        }
    }

    /// Diagnostic only: retain evidence for a failed full-model precision gate,
    /// without exposing tracing in serving or treating finite outputs as a pass.
    #[cfg(feature = "clef")]
    #[test]
    #[ignore = "requires explicit pinned model/suite paths and qualification hardware"]
    fn trace_kev_prefix_precision() {
        use huncho_core::calibration::calibrate;
        use huncho_core::conformance::load_suite;
        use huncho_core::manifest::ModelManifest;
        use huncho_core::prompt::formatter_for;
        use huncho_core::tokenizer::HfTokenizer;

        fn difference(left: &Tensor, right: &Tensor) -> serde_json::Value {
            assert_eq!(left.dims(), right.dims());
            let host = |tensor: &Tensor| {
                tensor
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            let (left, right) = (host(left), host(right));
            let mut peak = 0.0f64;
            let mut square_sum = 0.0f64;
            let mut changed = 0usize;
            for (&left, &right) in left.iter().zip(&right) {
                assert!(left.is_finite() && right.is_finite());
                let delta = (left as f64 - right as f64).abs();
                peak = peak.max(delta);
                square_sum += delta * delta;
                changed += usize::from(left != right);
            }
            serde_json::json!({"max_abs": peak, "rms": (square_sum / left.len() as f64).sqrt(),
                "changed_elements": changed, "elements": left.len()})
        }

        let required = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("set {name}"));
        let base = required("HUNCHO_TRACE_BASE");
        let package = required("HUNCHO_TRACE_PACKAGE");
        let suite_path = required("HUNCHO_TRACE_GOLDEN");
        let output = required("HUNCHO_TRACE_OUTPUT");
        let dtype = std::env::var("HUNCHO_TRACE_DTYPE").unwrap_or_else(|_| "fp16".into());
        let case_id = std::env::var("HUNCHO_TRACE_CASE").unwrap_or_else(|_| "short".into());
        let chunk_rows = std::env::var("HUNCHO_TRACE_PROJECTION_CHUNK_ROWS")
            .unwrap_or_else(|_| "0".into())
            .parse::<usize>()
            .unwrap();
        let fp32_attention = std::env::var("HUNCHO_TRACE_ATTENTION_FP32")
            .unwrap_or_else(|_| "false".into())
            .parse::<bool>()
            .unwrap();
        let package = Path::new(&package);
        let manifest = ModelManifest::load(package.join("huncho-model.json")).unwrap();
        assert_eq!(manifest.family, Family::F2);
        assert_eq!(manifest.prompt_contract.template, "kev-v1");
        let tokenizer = HfTokenizer::from_file(
            package.join(
                manifest
                    .backbone
                    .tokenizer
                    .as_ref()
                    .expect("pinned tokenizer"),
            ),
        )
        .unwrap();
        let formatter = formatter_for(&manifest);
        let suite = load_suite(&suite_path).unwrap();
        let case = suite
            .cases
            .iter()
            .find(|case| case.id == case_id)
            .expect("case ID");
        let device = crate::device::device_from_env().unwrap();
        let mut backend = Qwen3_5Backend::load_kev_on_device(
            Path::new(&base),
            package,
            &package.join(&manifest.head.weights),
            manifest.backbone.max_context,
            &dtype,
            device.clone(),
        )
        .unwrap();
        backend = backend
            .with_projection_chunk_rows(chunk_rows)
            .unwrap()
            .with_fp32_attention(fp32_attention)
            .unwrap();
        let temperature = manifest.calibration.resolve("candle", &dtype).temperature;
        let mut rows = Vec::new();
        for (qid, question) in &case.request.questions {
            let prompt = formatter
                .build(&case.request.state, question, &tokenizer)
                .unwrap();
            let prefix_len = prompt.prefix_len;
            let suffix_len = prompt.tokens.len() - prefix_len;
            assert!(prefix_len > 0 && suffix_len > 0);
            assert!(prompt
                .candidates
                .iter()
                .all(|candidate| candidate.position >= prefix_len));
            let ids = |tokens: &[u32]| Tensor::new(tokens, &device).unwrap().unsqueeze(0).unwrap();
            let model = &backend.model;
            let full_inputs = model.attention_inputs(prompt.tokens.len(), 0).unwrap();
            let prefix_inputs = model.attention_inputs(prefix_len, 0).unwrap();
            let suffix_inputs = model.attention_inputs(suffix_len, prefix_len).unwrap();
            let mut full = model.embed_tokens.forward(&ids(&prompt.tokens)).unwrap();
            let mut prefix = model
                .embed_tokens
                .forward(&ids(&prompt.tokens[..prefix_len]))
                .unwrap();
            let mut suffix = model
                .embed_tokens
                .forward(&ids(&prompt.tokens[prefix_len..]))
                .unwrap();
            let mut layers = Vec::new();
            for (index, layer) in model.layers.iter().enumerate() {
                let normalized =
                    rms_norm_effective(&full, &layer.input_layernorm, layer.eps).unwrap();
                let projection = match (&layer.linear_attn, &layer.self_attn) {
                    (Some(attention), _) => &attention.in_proj_qkv,
                    (_, Some(attention)) => &attention.q_proj,
                    _ => panic!("missing attention"),
                };
                // Hold input values fixed: this isolates projection differences
                // caused by split GEMM shape/layout from accumulated layer drift.
                let projected = projection.forward(&normalized).unwrap();
                let split_projection = Tensor::cat(
                    &[
                        projection
                            .forward(
                                &normalized
                                    .narrow(1, 0, prefix_len)
                                    .unwrap()
                                    .contiguous()
                                    .unwrap(),
                            )
                            .unwrap(),
                        projection
                            .forward(
                                &normalized
                                    .narrow(1, prefix_len, suffix_len)
                                    .unwrap()
                                    .contiguous()
                                    .unwrap(),
                            )
                            .unwrap(),
                    ],
                    1,
                )
                .unwrap();
                let projection_delta = difference(&projected, &split_projection);
                let mut cache = LayerCache::default();
                full = layer
                    .forward(
                        &full,
                        &full_inputs.cos,
                        &full_inputs.sin,
                        full_inputs.mask.as_ref(),
                        None,
                    )
                    .unwrap();
                prefix = layer
                    .forward(
                        &prefix,
                        &prefix_inputs.cos,
                        &prefix_inputs.sin,
                        prefix_inputs.mask.as_ref(),
                        Some(&mut cache),
                    )
                    .unwrap();
                suffix = layer
                    .forward(
                        &suffix,
                        &suffix_inputs.cos,
                        &suffix_inputs.sin,
                        suffix_inputs.mask.as_ref(),
                        Some(&mut cache),
                    )
                    .unwrap();
                layers.push(serde_json::json!({"layer": index,
                    "kind": if layer.linear_attn.is_some() {"linear"} else {"full"},
                    "fixed_input_projection": projection_delta,
                    "prefix_hidden": difference(&full.narrow(1, 0, prefix_len).unwrap(), &prefix),
                    "suffix_hidden": difference(&full.narrow(1, prefix_len, suffix_len).unwrap(), &suffix)}));
            }
            full = rms_norm_effective(&full, &model.norm, model.eps).unwrap();
            suffix = rms_norm_effective(&suffix, &model.norm, model.eps).unwrap();
            let input = ForwardInput::new(
                prompt.tokens.clone(),
                prompt.candidates.iter().map(|c| c.position).collect(),
            );
            let mut continuation = ForwardInput::new(
                prompt.tokens[prefix_len..].to_vec(),
                prompt
                    .candidates
                    .iter()
                    .map(|c| c.position - prefix_len)
                    .collect(),
            );
            let full_probs = calibrate(
                backend.readout(&full, &input).unwrap().values().data(),
                temperature,
            )
            .unwrap();
            let suffix_probs = calibrate(
                backend
                    .readout(&suffix, &continuation)
                    .unwrap()
                    .values()
                    .data(),
                temperature,
            )
            .unwrap();
            let actual_full =
                calibrate(backend.forward(input).unwrap().values().data(), temperature).unwrap();
            let parent = backend.prefill(&prompt.tokens[..prefix_len]).unwrap();
            let branch = backend.fork(parent).unwrap();
            continuation.fork_from = Some(branch);
            let actual_suffix = calibrate(
                backend.forward(continuation).unwrap().values().data(),
                temperature,
            )
            .unwrap();
            backend.release_cache(branch).unwrap();
            backend.release_cache(parent).unwrap();
            for (manual, actual) in [(&full_probs, &actual_full), (&suffix_probs, &actual_suffix)] {
                assert_eq!(manual.len(), actual.len());
                assert!(
                    manual
                        .iter()
                        .zip(actual)
                        .all(|(a, b)| (a - b).abs() <= 1e-6),
                    "trace must reproduce the actual native path"
                );
            }
            let delta = actual_full
                .iter()
                .zip(&actual_suffix)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            rows.push(serde_json::json!({"question_id": qid, "prefix_tokens": prefix_len, "suffix_tokens": suffix_len,
                "labels": prompt.candidates.iter().map(|c| &c.label).collect::<Vec<_>>(), "layers": layers,
                "independent_probabilities": actual_full, "prefix_probabilities": actual_suffix,
                "max_probability_delta": delta, "within_probability_delta_gate": delta <= 1e-4}));
        }
        let report = serde_json::json!({"diagnostic_only": true, "model": manifest.name,
            "adapter": manifest.adapter, "backbone_source": manifest.backbone.source,
            "device": crate::device_label(&device), "dtype": dtype, "temperature": temperature,
            "execution_metadata": backend.capabilities().extra,
            "case_id": case_id, "rows": rows, "scope": "layer drift and fixed-input split-projection comparison; no release qualification"});
        std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}

//! Cloudflare's trained evidence routing and joint schema scoring layers.
use candle::{DType, IndexOp, Module, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, Embedding, LayerNorm, Linear, VarBuilder,
};
use huncho_core::prompt::clef::EncodedRecord;
use std::collections::BTreeMap;

#[derive(serde::Deserialize)]
pub(super) struct HeadConfig {
    hidden_size: usize,
    width: usize,
    routing_layers: usize,
    layers: usize,
    heads: usize,
    feedforward: usize,
}
impl HeadConfig {
    pub fn validate(&self, hidden: usize) -> huncho_core::error::Result<()> {
        if self.hidden_size != hidden
            || self.width == 0
            || self.heads == 0
            || !self.width.is_multiple_of(self.heads)
            || self.feedforward == 0
        {
            return Err(huncho_core::error::Error::Package(
                "invalid Clef head dimensions or backbone/head hidden-size mismatch".into(),
            ));
        }
        Ok(())
    }
}

/// PyTorch MultiheadAttention stores Q/K/V together even for cross attention.
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    width: usize,
}
impl Attention {
    fn load(cfg: &HeadConfig, vb: VarBuilder) -> candle::Result<Self> {
        let w = cfg.width;
        let weight = vb.get((3 * w, w), "in_proj_weight")?;
        let bias = vb.get(3 * w, "in_proj_bias")?;
        let projection = |offset| -> candle::Result<Linear> {
            Ok(Linear::new(
                weight.narrow(0, offset, w)?.contiguous()?,
                Some(bias.narrow(0, offset, w)?.contiguous()?),
            ))
        };
        Ok(Self {
            q: projection(0)?,
            k: projection(w)?,
            v: projection(2 * w)?,
            out: linear(w, w, vb.pp("out_proj"))?,
            heads: cfg.heads,
            width: w,
        })
    }
    fn forward(&self, query: &Tensor, memory: &Tensor) -> candle::Result<Tensor> {
        let head_dim = self.width / self.heads;
        let reshape = |x: Tensor| -> candle::Result<Tensor> {
            let n = x.dim(0)?;
            x.reshape((n, self.heads, head_dim))?
                .transpose(0, 1)?
                .contiguous()
        };
        let q = reshape(self.q.forward(query)?)?;
        let k = reshape(self.k.forward(memory)?)?;
        let v = reshape(self.v.forward(memory)?)?;
        let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? / (head_dim as f64).sqrt())?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let output = weights
            .matmul(&v)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((query.dim(0)?, self.width))?;
        self.out.forward(&output)
    }
}

struct FeedForward {
    first: Linear,
    last: Linear,
}
impl FeedForward {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        self.last.forward(&self.first.forward(x)?.gelu_erf()?)
    }
}
struct EvidenceLayer {
    query_norm: LayerNorm,
    memory_norm: LayerNorm,
    attention: Attention,
    ff_norm: LayerNorm,
    ff: FeedForward,
}
impl EvidenceLayer {
    fn load(cfg: &HeadConfig, vb: VarBuilder) -> candle::Result<Self> {
        Ok(Self {
            query_norm: layer_norm(cfg.width, 1e-5, vb.pp("query_norm"))?,
            memory_norm: layer_norm(cfg.width, 1e-5, vb.pp("memory_norm"))?,
            attention: Attention::load(cfg, vb.pp("attention"))?,
            ff_norm: layer_norm(cfg.width, 1e-5, vb.pp("feedforward_norm"))?,
            ff: FeedForward {
                first: linear(cfg.width, cfg.feedforward, vb.pp("feedforward.0"))?,
                last: linear(cfg.feedforward, cfg.width, vb.pp("feedforward.3"))?,
            },
        })
    }
    fn forward(&self, q: &Tensor, memory: &Tensor) -> candle::Result<Tensor> {
        let q = (q + self.attention.forward(
            &self.query_norm.forward(q)?,
            &self.memory_norm.forward(memory)?,
        )?)?;
        &q + self.ff.forward(&self.ff_norm.forward(&q)?)?
    }
}
struct DecoderLayer {
    self_attn: Attention,
    cross_attn: Attention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    ff: FeedForward,
}
impl DecoderLayer {
    fn load(cfg: &HeadConfig, vb: VarBuilder) -> candle::Result<Self> {
        Ok(Self {
            self_attn: Attention::load(cfg, vb.pp("self_attn"))?,
            cross_attn: Attention::load(cfg, vb.pp("multihead_attn"))?,
            norm1: layer_norm(cfg.width, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(cfg.width, 1e-5, vb.pp("norm2"))?,
            norm3: layer_norm(cfg.width, 1e-5, vb.pp("norm3"))?,
            ff: FeedForward {
                first: linear(cfg.width, cfg.feedforward, vb.pp("linear1"))?,
                last: linear(cfg.feedforward, cfg.width, vb.pp("linear2"))?,
            },
        })
    }
    fn forward(&self, x: &Tensor, memory: &Tensor) -> candle::Result<Tensor> {
        let normalized = self.norm1.forward(x)?;
        let x = (x + self.self_attn.forward(&normalized, &normalized)?)?;
        let x = (&x + self.cross_attn.forward(&self.norm2.forward(&x)?, memory)?)?;
        &x + self.ff.forward(&self.norm3.forward(&x)?)?
    }
}

pub(super) struct JointHead {
    hidden_norm: LayerNorm,
    memory: Linear,
    question: Linear,
    option_question: Linear,
    global: Linear,
    option_context: Linear,
    option_lexical: Linear,
    types: Embedding,
    evidence: Vec<EvidenceLayer>,
    summary_norm: LayerNorm,
    layers: Vec<DecoderLayer>,
    field_norm: LayerNorm,
    option_norm: LayerNorm,
    residual: FeedForward,
    prior_scale: f64,
    joint_scale: f64,
    gate: f64,
}
impl JointHead {
    pub fn load(cfg: &HeadConfig, vb: VarBuilder) -> candle::Result<Self> {
        let project = |name| linear_no_bias(cfg.hidden_size, cfg.width, vb.pp(name));
        let scalar = |name| vb.get((), name)?.to_dtype(DType::F32)?.to_scalar::<f32>();
        Ok(Self {
            hidden_norm: layer_norm(cfg.hidden_size, 1e-5, vb.pp("hidden_norm"))?,
            memory: project("memory_projection")?,
            question: project("question_projection")?,
            option_question: project("option_question_projection")?,
            global: project("global_projection")?,
            option_context: project("option_context_projection")?,
            option_lexical: project("option_lexical_projection")?,
            types: embedding(3, cfg.width, vb.pp("type_embedding"))?,
            evidence: (0..cfg.routing_layers)
                .map(|i| EvidenceLayer::load(cfg, vb.pp(format!("evidence_layers.{i}"))))
                .collect::<candle::Result<_>>()?,
            summary_norm: layer_norm(cfg.width, 1e-5, vb.pp("option_summary_norm"))?,
            layers: (0..cfg.layers)
                .map(|i| DecoderLayer::load(cfg, vb.pp(format!("layers.{i}"))))
                .collect::<candle::Result<_>>()?,
            field_norm: layer_norm(cfg.width, 1e-5, vb.pp("field_norm"))?,
            option_norm: layer_norm(cfg.width, 1e-5, vb.pp("option_norm"))?,
            residual: FeedForward {
                first: linear(4 * cfg.width, cfg.width, vb.pp("residual_scorer.0"))?,
                last: linear(cfg.width, 1, vb.pp("residual_scorer.3"))?,
            },
            prior_scale: (scalar("prior_logit_scale")? as f64).min(100f64.ln()).exp(),
            joint_scale: (scalar("joint_logit_scale")? as f64).min(100f64.ln()).exp(),
            gate: 1. / (1. + (-(scalar("residual_gate")? as f64)).exp()),
        })
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        lexical_weight: &Tensor,
        record: &EncodedRecord,
        vectorized: bool,
        grouped_pooling: bool,
    ) -> candle::Result<Vec<Vec<f32>>> {
        let hidden = self.hidden_norm.forward(hidden)?;
        let memory = self.memory.forward(&hidden)?;
        let global = hidden.i(hidden.dim(0)? - 1)?;
        let span_mean = |span: (usize, usize)| hidden.narrow(0, span.0, span.1 - span.0)?.mean(0);
        let questions = Tensor::stack(
            &record
                .questions
                .iter()
                .map(|q| span_mean(q.question_span))
                .collect::<candle::Result<Vec<_>>>()?,
            0,
        )?;
        let grouped_lexical = if grouped_pooling {
            Some(lexical_means(lexical_weight, record, hidden.dtype())?)
        } else {
            None
        };
        let mut option_offset = 0;
        let mut lexical = Vec::new();
        let mut option_queries = Vec::new();
        let mut contexts = Vec::new();
        for (i, q) in record.questions.iter().enumerate() {
            let context = Tensor::stack(
                &q.option_spans
                    .iter()
                    .copied()
                    .map(span_mean)
                    .collect::<candle::Result<Vec<_>>>()?,
                0,
            )?;
            let vectors = if let Some(lexical) = &grouped_lexical {
                lexical.narrow(0, option_offset, q.option_ids.len())?
            } else {
                let mut vectors = Vec::new();
                for &(start, end) in &q.option_spans {
                    let ids = Tensor::new(&record.input_ids[start..end], hidden.device())?;
                    // Cast just the gathered rows; keep the full vocabulary in its
                    // compact backbone dtype and never compute vocabulary logits.
                    vectors.push(
                        lexical_weight
                            .index_select(&ids, 0)?
                            .to_dtype(hidden.dtype())?
                            .mean(0)?,
                    );
                }
                Tensor::stack(&vectors, 0)?
            };
            option_offset += q.option_ids.len();
            if vectorized {
                contexts.push(context);
            } else {
                option_queries.push(
                    (self.option_context.forward(&context)?
                        + self.option_lexical.forward(&vectors)?)?
                    .broadcast_add(
                        &self
                            .option_question
                            .forward(&questions.i(i)?.unsqueeze(0)?)?,
                    )?,
                );
            }
            lexical.push(vectors);
        }
        let question_ids = if vectorized {
            let ids: Vec<_> = record
                .questions
                .iter()
                .enumerate()
                .flat_map(|(i, question)| {
                    std::iter::repeat(i as u32).take(question.option_ids.len())
                })
                .collect();
            Some(Tensor::new(ids.as_slice(), hidden.device())?)
        } else {
            None
        };
        let mut routed = if let Some(question_ids) = &question_ids {
            (self.option_context.forward(&Tensor::cat(&contexts, 0)?)?
                + self.option_lexical.forward(&Tensor::cat(&lexical, 0)?)?)?
            .broadcast_add(
                &self
                    .option_question
                    .forward(&questions)?
                    .index_select(question_ids, 0)?,
            )?
        } else {
            Tensor::cat(&option_queries, 0)?
        };
        for layer in &self.evidence {
            routed = layer.forward(&routed, &memory)?;
        }
        let base = self.question.forward(&questions)?;
        let counts: Vec<_> = record
            .questions
            .iter()
            .map(|q| q.option_ids.len())
            .collect();
        let summaries = if grouped_pooling {
            Some(grouped_summaries(&routed, &base, &counts)?)
        } else {
            None
        };
        let mut offset = 0;
        let mut split = Vec::new();
        let mut scalar_summaries = Vec::new();
        for (i, q) in record.questions.iter().enumerate() {
            let options = routed.narrow(0, offset, q.option_ids.len())?;
            offset += q.option_ids.len();
            if !grouped_pooling {
                let weights = candle_nn::ops::softmax(
                    &(options.matmul(&base.i(i)?.unsqueeze(1)?)?.squeeze(1)?
                        / (base.dim(1)? as f64).sqrt())?,
                    0,
                )?;
                scalar_summaries.push(options.broadcast_mul(&weights.unsqueeze(1)?)?.sum(0)?);
            }
            split.push(options);
        }
        let summaries = match summaries {
            Some(summaries) => summaries,
            None => Tensor::stack(&scalar_summaries, 0)?,
        };
        let type_ids: Vec<_> = record.questions.iter().map(|q| q.question_type).collect();
        let type_ids = Tensor::new(type_ids.as_slice(), hidden.device())?;
        let mut fields = ((base + self.summary_norm.forward(&summaries)?)?
            .broadcast_add(&self.global.forward(&global.unsqueeze(0)?)?)?
            + self.types.forward(&type_ids)?)?;
        for layer in &self.layers {
            fields = layer.forward(&fields, &memory)?;
        }
        fields = self.field_norm.forward(&fields)?;
        let flat_logits = if let Some(question_ids) = &question_ids {
            self.vector_scores(
                &questions,
                &global,
                &Tensor::cat(&lexical, 0)?,
                &routed,
                &fields,
                question_ids,
            )?
        } else {
            let mut logits = Vec::new();
            for (i, (lexical, routed)) in lexical.iter().zip(split).enumerate() {
                let anchor = normalize(&(questions.i(i)? + &global)?, 1e-12)?;
                let prior = (normalize(lexical, 1e-12)?
                    .matmul(&anchor.unsqueeze(1)?)?
                    .squeeze(1)?
                    * self.prior_scale)?;
                let options = self.option_norm.forward(&routed)?;
                let field = fields.i(i)?.unsqueeze(0)?.broadcast_as(options.shape())?;
                let product = (&field * &options)?;
                let delta = (&field - &options)?.abs()?;
                let cosine =
                    (normalize(&field, 1e-8)? * normalize(&options, 1e-8)?)?.sum(D::Minus1)?;
                let features = Tensor::cat(&[&field, &options, &product, &delta], D::Minus1)?;
                let residual = self.residual.forward(&features)?.squeeze(1)?;
                let joint = ((cosine * self.joint_scale)? + residual)?;
                logits.push((prior + (joint * self.gate)?)?.to_dtype(DType::F32)?);
            }
            Tensor::cat(&logits, 0)?
        };
        // Transfer all question logits once. Both execution modes retain
        // original option order; the scalar mode retains original arithmetic.
        let values = flat_logits.to_vec1::<f32>()?;
        let mut offset = 0;
        Ok(record
            .questions
            .iter()
            .map(|question| {
                let end = offset + question.option_ids.len();
                let row = values[offset..end].to_vec();
                offset = end;
                row
            })
            .collect())
    }

    /// Group per-row operations without changing spans, routing, field order,
    /// normalization axes or learned scales. GEMM shapes/reduction kernels can
    /// change rounding, so this remains a separately qualified execution mode.
    fn vector_scores(
        &self,
        questions: &Tensor,
        global: &Tensor,
        lexical: &Tensor,
        routed: &Tensor,
        fields: &Tensor,
        question_ids: &Tensor,
    ) -> candle::Result<Tensor> {
        let anchors =
            normalize(&questions.broadcast_add(global)?, 1e-12)?.index_select(question_ids, 0)?;
        let prior = ((normalize(lexical, 1e-12)? * anchors)?.sum(D::Minus1)? * self.prior_scale)?;
        let options = self.option_norm.forward(routed)?;
        let fields = fields.index_select(question_ids, 0)?;
        let product = (&fields * &options)?;
        let delta = (&fields - &options)?.abs()?;
        let cosine = (normalize(&fields, 1e-8)? * normalize(&options, 1e-8)?)?.sum(D::Minus1)?;
        let features = Tensor::cat(&[&fields, &options, &product, &delta], D::Minus1)?;
        let residual = self.residual.forward(&features)?.squeeze(1)?;
        let joint = ((cosine * self.joint_scale)? + residual)?;
        (prior + (joint * self.gate)?)?.to_dtype(DType::F32)
    }
}

/// Gather/cast option lexical rows once, preserving every original token and
/// per-span mean reduction. No full vocabulary projection or host copy.
fn lexical_means(weight: &Tensor, record: &EncodedRecord, dtype: DType) -> candle::Result<Tensor> {
    let mut ids = Vec::new();
    let mut lengths = Vec::new();
    for question in &record.questions {
        for &(start, end) in &question.option_spans {
            if start >= end || end > record.input_ids.len() {
                candle::bail!("invalid lexical option span")
            }
            ids.extend_from_slice(&record.input_ids[start..end]);
            lengths.push(end - start);
        }
    }
    let ids = Tensor::new(ids.as_slice(), weight.device())?;
    let gathered = weight.index_select(&ids, 0)?.to_dtype(dtype)?;
    let mut means = Vec::with_capacity(lengths.len());
    let mut offset = 0;
    for length in lengths {
        means.push(gathered.narrow(0, offset, length)?.mean(0)?);
        offset += length;
    }
    Tensor::stack(&means, 0)
}

/// Bucket by exact option count. Each BMM/softmax/reduction remains local to a
/// question; no padding, candidate masking or cross-request joint schema.
fn grouped_summaries(options: &Tensor, base: &Tensor, counts: &[usize]) -> candle::Result<Tensor> {
    let mut buckets = BTreeMap::<usize, Vec<(usize, usize)>>::new();
    let mut offset = 0;
    for (index, &count) in counts.iter().enumerate() {
        if count == 0 {
            candle::bail!("empty joint option group")
        }
        buckets.entry(count).or_default().push((index, offset));
        offset += count;
    }
    if offset != options.dim(0)? || base.dim(0)? != counts.len() {
        candle::bail!("joint summary shape mismatch")
    }
    let mut summaries = vec![None; counts.len()];
    for (count, rows) in buckets {
        let block = Tensor::stack(
            &rows
                .iter()
                .map(|&(_, offset)| options.narrow(0, offset, count))
                .collect::<candle::Result<Vec<_>>>()?,
            0,
        )?;
        let anchors = Tensor::stack(
            &rows
                .iter()
                .map(|&(index, _)| base.i(index))
                .collect::<candle::Result<Vec<_>>>()?,
            0,
        )?;
        let scores =
            (block.matmul(&anchors.unsqueeze(2)?)?.squeeze(2)? / (base.dim(1)? as f64).sqrt())?;
        let weights = candle_nn::ops::softmax(&scores, 1)?;
        let pooled = block.broadcast_mul(&weights.unsqueeze(2)?)?.sum(1)?;
        for (row, (index, _)) in rows.iter().enumerate() {
            summaries[*index] = Some(pooled.i(row)?);
        }
    }
    Tensor::stack(
        &summaries
            .into_iter()
            .map(|summary| {
                summary.ok_or_else(|| candle::Error::Msg("missing joint summary".into()))
            })
            .collect::<candle::Result<Vec<_>>>()?,
        0,
    )
}

fn normalize(x: &Tensor, eps: f64) -> candle::Result<Tensor> {
    let f32_x = x.to_dtype(DType::F32)?;
    let norm = f32_x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(eps, f64::INFINITY)?;
    f32_x.broadcast_div(&norm)?.to_dtype(x.dtype())
}

//! The backend abstraction (PRD §6.1).
//!
//! Every backend implements [`Backend`]. The core never assumes a backend
//! supports custom attention masks; instead it models F2 fan-out via
//! [`Backend::fork`] over a shared prefill cache.

use std::collections::BTreeMap;

use crate::contract::SystemOneRequest;
use crate::error::Result;
use crate::manifest::{BackendId, Family};
use crate::tensor::Tensor;

/// A handle to a prefilled cache (KV + any recurrent state) that can be forked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheHandle {
    pub id: u64,
}

/// Raw option logits from a model that encodes and scores a whole request.
/// Question and option ids are explicit so backend ordering cannot change answers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RequestOutput {
    pub logits: BTreeMap<String, BTreeMap<String, f32>>,
    /// Tokens in the shared prompt, counted once per request.
    pub input_tokens: u64,
}

/// One complete joint schema. Batching must never split its questions.
#[derive(Clone, Copy)]
pub struct RequestBatchInput<'a> {
    pub request: &'a SystemOneRequest,
    pub max_context: usize,
}

/// Attempted native work for a group of whole requests, including failures.
/// Token usage in each output excludes padding; physical work includes it.
#[derive(Debug, Default)]
pub struct RequestBatchWork {
    pub forward_calls: u64,
    pub processed_tokens: u64,
    pub batch_calls: u64,
    pub padded_batch_calls: u64,
    pub padded_tokens: u64,
    pub prepared_questions: u64,
}

/// The input to a single [`Backend::forward`] call.
///
/// Processes one sequence per call. Concurrent calls are not a batch; a future
/// scheduler (CORE-07) needs an explicit backend batching interface.
#[derive(Debug, Clone)]
pub struct ForwardInput {
    /// Token ids for the sequence; only new suffix tokens when `fork_from` is set.
    pub tokens: Vec<u32>,
    /// Positions (0-based into `tokens`) to read features/logits from. Must be
    /// sorted ascending; each row of the output corresponds to one position.
    pub positions: Vec<usize>,
    /// Optional sorted, unique vocabulary codes needed by a candidate-logit
    /// head. Backends may return only these columns using `SelectedLogits`, or
    /// ignore the hint and return the full vocabulary as before.
    pub logit_codes: Option<Vec<u32>>,
    /// The typed-question index for the Laya decision head type embedding
    /// (`0`=choice, `1`=score, `2`=noul). Unused by backends that do not
    /// implement the typed option-marker head.
    pub qtype: u32,
    /// Legacy retention hint. Use `Backend::prefill` to obtain an explicit handle.
    pub retain_cache: bool,
    /// Continue this isolated branch with suffix tokens. Positions are relative
    /// to the supplied suffix, and the backend must account for prefix offsets.
    pub fork_from: Option<CacheHandle>,
}

/// A fresh caller-owned prefix handle, optionally cloned from an immutable
/// retained snapshot. The caller releases it just like a normal prefill handle.
pub struct CachedPrefill {
    pub handle: CacheHandle,
    pub hit: bool,
}

/// Physical prefix work, including attempted calls before a failed prefill.
/// Retained snapshot hits submit zero work. Split prefixes account for every
/// chunk while logical API usage remains the complete original prompt.
#[derive(Debug, Default)]
pub struct PrefillWork {
    pub forward_calls: u64,
    pub processed_tokens: u64,
    pub chunked_prefills: u64,
}

/// Actual isolated forks and native suffix work, including failed attempts.
/// Prefix tokens are not recomputed. Caller token budgets must nevertheless
/// charge each complete context to bound the private batched KV workspace.
#[derive(Debug, Default)]
pub struct ForkBatchWork {
    pub cache_forks: u64,
    pub forward_calls: u64,
    pub processed_tokens: u64,
    pub batch_calls: u64,
    pub padded_batch_calls: u64,
    pub padded_tokens: u64,
}

/// Hard native batch allocation limits, in addition to the caller token budget.
/// Each input is conservatively charged `positions.len() + 1` readout rows,
/// including a possible final decision row. Oversized singletons run independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchLimits {
    pub max_rows: usize,
    pub max_readouts: Option<usize>,
}
impl Default for BatchLimits {
    fn default() -> Self {
        Self {
            max_rows: 64,
            max_readouts: None,
        }
    }
}

impl ForwardInput {
    pub fn new(tokens: Vec<u32>, positions: Vec<usize>) -> Self {
        ForwardInput {
            tokens,
            positions,
            logit_codes: None,
            qtype: 0,
            retain_cache: false,
            fork_from: None,
        }
    }

    /// Builder-style setter for the typed-question index.
    pub fn with_qtype(mut self, qtype: u32) -> Self {
        self.qtype = qtype;
        self
    }

    /// Request raw logits for these codes without changing candidate order.
    pub fn with_logit_codes(mut self, mut codes: Vec<u32>) -> Self {
        codes.sort_unstable();
        codes.dedup();
        self.logit_codes = Some(codes);
        self
    }
}

/// The output of a [`Backend::forward`] call.
#[derive(Debug, Clone)]
pub enum ForwardOutput {
    /// Hidden states at the requested positions. Shape `[n_positions, dim]`.
    Features {
        positions: Vec<usize>,
        values: Tensor,
    },
    /// Logits over the vocabulary at the requested positions.
    /// Shape `[n_positions, vocab_size]`.
    Logits {
        positions: Vec<usize>,
        values: Tensor,
    },
    /// Raw vocabulary logits for explicitly identified columns only.
    /// Shape `[n_positions, codes.len()]`; no activation or calibration applied.
    SelectedLogits {
        positions: Vec<usize>,
        codes: Vec<u32>,
        values: Tensor,
    },
}

impl ForwardOutput {
    /// The positions this output was produced for, in order.
    pub fn positions(&self) -> &[usize] {
        match self {
            ForwardOutput::Features { positions, .. } => positions,
            ForwardOutput::Logits { positions, .. } => positions,
            ForwardOutput::SelectedLogits { positions, .. } => positions,
        }
    }

    /// The raw values tensor.
    pub fn values(&self) -> &Tensor {
        match self {
            ForwardOutput::Features { values, .. } => values,
            ForwardOutput::Logits { values, .. } => values,
            ForwardOutput::SelectedLogits { values, .. } => values,
        }
    }
}

/// The capabilities reported by a backend.
#[derive(Debug, Clone, Default)]
pub struct Capabilities {
    /// Backend identifier.
    pub id: BackendId,
    /// The dtype the backend was loaded as (e.g. `fp32`, `fp16`, `int8`).
    pub dtype: String,
    /// Maximum context length supported by the loaded model.
    pub max_context: usize,
    /// Whether KV/recurrent-state fork is supported (required for F2).
    pub supports_fork: bool,
    /// Whether multi-LoRA serving is supported.
    pub supports_lora: bool,
    /// The families this backend can serve.
    pub families: Vec<Family>,
    /// Extra backend-specific capabilities. `device`, when present, is a
    /// human-readable description of the loaded execution device.
    pub extra: BTreeMap<String, String>,
}

/// A loaded model instance. Constructed once per `(model, backend, dtype)` and
/// reused across requests.
pub trait Backend: Send + Sync {
    /// An independent execution context over identical immutable model weights.
    /// No live handles or retained prefixes may transfer into the new context.
    /// Unsupported backends fail explicitly; this is distinct from KV `fork`.
    fn replica(&self) -> Result<Box<dyn Backend>> {
        Err(crate::error::Error::Unsupported(
            "this backend does not support independent shared-weight replicas".into(),
        ))
    }
    /// The backend identifier.
    fn id(&self) -> BackendId;

    /// The capabilities of the loaded model.
    fn capabilities(&self) -> Capabilities;

    /// Run a forward pass, returning features or logits at the requested
    /// positions. Callers must not assume arbitrary attention masks are
    /// supported; fan-out is expressed via `fork_from`.
    fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput>;

    /// Whether independent, equal-length sequences can share one native
    /// backbone call. This does not imply padding or cached-branch batching.
    fn supports_batch(&self) -> bool {
        false
    }

    fn batch_limits(&self) -> BatchLimits {
        BatchLimits::default()
    }

    /// Run up to 64 independent equal-length sequences, returning one output
    /// per input in the same order. Each row retains its own positions, codes,
    /// and question type; no cross-row attention is permitted.
    fn forward_batch(&mut self, _inputs: Vec<ForwardInput>) -> Result<Vec<ForwardOutput>> {
        Err(crate::error::Error::Unsupported(
            "backend does not expose native batching".into(),
        ))
    }

    /// Whether equal-length suffixes of one immutable parent can share a
    /// native backbone call. Independent batching does not imply this support.
    fn supports_fork_batch(&self) -> bool {
        false
    }

    /// Fork one complete parent into private rows and evaluate nonempty,
    /// equal-length suffixes. Positions are relative to each suffix. No input
    /// may retain state or supply another handle. Return owned readouts in input
    /// order, leave the parent unchanged, and release all temporary branches on
    /// success or failure. This does not publish continuation handles.
    fn forward_fork_batch(
        &mut self,
        _parent: CacheHandle,
        _inputs: Vec<ForwardInput>,
        _work: &mut ForkBatchWork,
    ) -> Result<Vec<ForwardOutput>> {
        Err(crate::error::Error::Unsupported(
            "backend does not expose native cached-branch batching".into(),
        ))
    }

    /// Whether independent mixed-length rows can be right-padded without
    /// changing valid readouts. Bidirectional/joint heads must exclude padding.
    fn supports_padded_batch(&self) -> bool {
        false
    }

    fn supports_padded_fork_batch(&self) -> bool {
        false
    }

    /// Same parent/ownership contract as `forward_fork_batch`, with mixed-length
    /// suffixes. Right-padding must be invisible to original markers and final
    /// decisions. No padded suffix continuation state may survive the call.
    fn forward_padded_fork_batch(
        &mut self,
        _parent: CacheHandle,
        _inputs: Vec<ForwardInput>,
        _work: &mut ForkBatchWork,
    ) -> Result<Vec<ForwardOutput>> {
        Err(crate::error::Error::Unsupported(
            "backend does not expose padded cached-branch batching".into(),
        ))
    }

    /// At most 64 independent nonempty rows, with original relative readouts.
    /// The backend pads to the maximum length; all padding must be fully
    /// invisible to valid outputs and each final decision uses its real length.
    fn forward_padded_batch(&mut self, _inputs: Vec<ForwardInput>) -> Result<Vec<ForwardOutput>> {
        Err(crate::error::Error::Unsupported(
            "backend does not expose mixed-length padded batching".into(),
        ))
    }

    /// Whether this backend collates complete joint schemas in a native forward.
    fn supports_request_batch(&self) -> bool {
        false
    }

    /// Whether whole schemas with different lengths can share a native call.
    /// Original-length joint heads must exclude padding completely.
    fn supports_padded_request_batch(&self) -> bool {
        false
    }

    /// Prepare every complete request before inference, collate equal lengths
    /// (or bounded padding), and return outputs in the original request order.
    /// Charge B*max_length against budget and at most max_padding_percent of
    /// that rectangle to padding. An oversized singleton stays independent.
    /// Unsupported backends must fail; serial fallback is not native batching.
    fn forward_request_batch(
        &mut self,
        _inputs: &[RequestBatchInput<'_>],
        _budget: usize,
        _max_padding_percent: usize,
        _work: &mut RequestBatchWork,
    ) -> Result<Vec<RequestOutput>> {
        Err(crate::error::Error::Unsupported(
            "backend does not expose whole-request native batching".into(),
        ))
    }

    fn forward_request(
        &mut self,
        _request: &SystemOneRequest,
        _max_context: usize,
    ) -> Result<RequestOutput> {
        Err(crate::error::Error::Unsupported(
            "backend does not support whole-request inference".into(),
        ))
    }

    /// Fork a prefilled cache, isolating it for one question branch (required
    /// for F2). Backends that do not support forking return
    /// [`crate::error::Error::Unsupported`].
    fn fork(&mut self, handle: CacheHandle) -> Result<CacheHandle>;

    /// Prefill a sequence and return a cache handle without reading outputs.
    fn prefill(&mut self, _tokens: &[u32]) -> Result<CacheHandle> {
        Err(crate::error::Error::Unsupported(
            "backend does not expose a prefill handle".into(),
        ))
    }

    /// Optional exact cross-request prefix retention under a charged-byte
    /// budget. The default computes a fresh prefix and never claims a hit.
    fn prefill_cached(&mut self, tokens: &[u32], _max_bytes: usize) -> Result<CachedPrefill> {
        self.prefill(tokens)
            .map(|handle| CachedPrefill { handle, hit: false })
    }

    /// Default single-call accounting; backends with chunked prefills override
    /// this method and update work before each native attempt.
    fn prefill_cached_with_work(
        &mut self,
        tokens: &[u32],
        max_bytes: usize,
        work: &mut PrefillWork,
    ) -> Result<CachedPrefill> {
        work.forward_calls += 1;
        work.processed_tokens += tokens.len() as u64;
        let result = self.prefill_cached(tokens, max_bytes);
        if result.as_ref().is_ok_and(|cached| cached.hit) {
            work.forward_calls -= 1;
            work.processed_tokens -= tokens.len() as u64;
        }
        result
    }

    /// Whether a caller can pause an F2 prefix between bounded native calls.
    /// A partial handle must never be forked or used for question readouts.
    fn supports_resumable_prefill(&self) -> bool {
        false
    }

    /// Validate a complete prefix and allocate an empty private cache, or clone
    /// an exact retained snapshot. This submits no model work. The caller owns
    /// the handle immediately and must release it on cancellation or failure.
    fn begin_resumable_prefill(
        &mut self,
        _tokens: &[u32],
        _max_bytes: usize,
    ) -> Result<CachedPrefill> {
        Err(crate::error::Error::Unsupported(
            "backend does not expose resumable prefill".into(),
        ))
    }

    /// Submit at most one configured prefix chunk. Return true only once the
    /// complete prefix is usable. Work includes native attempts that fail;
    /// cache state must commit atomically after a successful chunk.
    fn advance_resumable_prefill(
        &mut self,
        _handle: CacheHandle,
        _work: &mut PrefillWork,
    ) -> Result<bool> {
        Err(crate::error::Error::Unsupported(
            "backend does not expose resumable prefill".into(),
        ))
    }

    /// Clear retained snapshots without invalidating active caller-owned forks.
    fn clear_prefix_cache(&mut self) -> Result<()> {
        Ok(())
    }

    /// Release a prefix or branch after use. Handles must belong to this loaded
    /// backend; releasing a parent must not invalidate its immutable forks.
    fn release_cache(&mut self, _handle: CacheHandle) -> Result<()> {
        Err(crate::error::Error::Unsupported(
            "backend does not expose cache release".into(),
        ))
    }
}

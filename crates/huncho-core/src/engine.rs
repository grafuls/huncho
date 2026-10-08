//! The model engine: ties the manifest, backend, tokenizer, head, prompt
//! builder, and calibration layer into the evaluation pipeline.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::backend::{
    Backend, CacheHandle, ForwardInput, RequestBatchInput, RequestBatchWork, RequestOutput,
};
use crate::cache_key;
use crate::calibration::{self, bucket_size, confidence};
use crate::contract::{Answer, Question, StateValue, SystemOneRequest, SystemOneResponse, Usage};
use crate::error::{Error, Result};
use crate::head::{self, HeadParams};
use crate::manifest::{BackendId, CalibrationEntry, Family, ModelManifest};
use crate::prompt::{formatter_for, BuiltPrompt, Candidate, CandidateKind, PromptFormatter};
use crate::prompt_cache::PromptCache;
use crate::response_cache::ResponseCache;
#[cfg(test)]
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;

mod batching;
#[cfg(feature = "external-scores")]
mod external_scores;
mod fork_batch;
mod resumable;
use batching::padded_groups;
#[cfg(feature = "external-scores")]
pub use external_scores::{ExternalEvaluation, MarkerReadout};
pub use resumable::ResumableEvaluation;

/// Options controlling a single evaluation.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct EvalOptions {
    /// Include engine-specific extras in the response (API-05). Off by default.
    pub extensions: bool,
    /// Override the model's max context (for testing). `None` uses manifest.
    pub max_context: Option<usize>,
    /// Use the legacy F3 duplicated-position/full-vocabulary readout when
    /// comparing probabilities and performance against the optimized path.
    pub reference_readout: bool,
    /// Opt in to request-local native Kev prefix reuse after qualifying this
    /// model/device with conformance. Independent forwards remain the default.
    pub prefix_cache: bool,
    /// Charged bytes for exact native cross-request prefix snapshots. Zero
    /// disables retention; positive values require qualified prefix fan-out.
    pub persistent_prefix_bytes: usize,
    /// Opt-in native batches, bounded by physical input tokens. F5 collates
    /// whole schemas across requests; other families collate questions.
    /// CPU Kev can combine equal-length question batches with prefix reuse;
    /// that profile charges complete contexts to bound KV workspaces.
    pub max_batch_tokens: Option<usize>,
    /// Maximum padded positions as a percentage of physical batch positions.
    /// Zero keeps exact lengths. 1..=100 requires native padded support.
    pub max_batch_padding_percent: usize,
    /// Prepare all prompts before submitting model work. F1–F4 can prepare
    /// while another request executes; F5 encodes a whole group inside its
    /// backend before the first forward, without splitting any schema.
    pub prepare_all: bool,
    /// CPU Kev only: release execution capacity after each prefix chunk and
    /// question. Requires prefix reuse and a configured backend chunk size.
    pub cooperative_prefill: bool,
}

/// Physical work submitted to the backend, separate from logical wire usage.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EvalStats {
    pub forward_calls: u64,
    pub prefill_calls: u64,
    /// Prefixes actually submitted across more than one native prefill call.
    pub chunked_prefills: u64,
    pub processed_tokens: u64,
    pub reused_prefix_tokens: u64,
    pub cache_forks: u64,
    pub batch_calls: u64,
    /// Native batches of multiple suffix rows from one immutable parent.
    pub fork_batch_calls: u64,
    pub padded_batch_calls: u64,
    pub padded_tokens: u64,
    /// Native batches containing sequences from more than one request.
    pub cross_request_batches: u64,
    pub persistent_prefix_hits: u64,
    /// Exact successful responses reused; a hit submits no physical work.
    pub result_cache_hits: u64,
    /// Prepared F1–F4 prompts reused; model forwards still execute normally.
    pub prompt_cache_hits: u64,
    /// Questions prepared ahead of execution, excluding retained whole results.
    pub prepared_questions: u64,
    /// Completed prefix chunks that leave more prefix work for a later step.
    pub prefill_yields: u64,
    /// Qualification/serving switches between distinct active prefix jobs.
    pub prefill_interleaves: u64,
}

impl EvalStats {
    fn accumulate_request_batch(&mut self, work: &RequestBatchWork) {
        self.forward_calls += work.forward_calls;
        self.processed_tokens += work.processed_tokens;
        self.batch_calls += work.batch_calls;
        self.cross_request_batches += work.batch_calls;
        self.padded_batch_calls += work.padded_batch_calls;
        self.padded_tokens += work.padded_tokens;
        self.prepared_questions += work.prepared_questions;
    }

    pub fn accumulate(&mut self, work: &Self) {
        self.forward_calls += work.forward_calls;
        self.prefill_calls += work.prefill_calls;
        self.chunked_prefills += work.chunked_prefills;
        self.processed_tokens += work.processed_tokens;
        self.reused_prefix_tokens += work.reused_prefix_tokens;
        self.cache_forks += work.cache_forks;
        self.batch_calls += work.batch_calls;
        self.fork_batch_calls += work.fork_batch_calls;
        self.padded_batch_calls += work.padded_batch_calls;
        self.padded_tokens += work.padded_tokens;
        self.cross_request_batches += work.cross_request_batches;
        self.persistent_prefix_hits += work.persistent_prefix_hits;
        self.result_cache_hits += work.result_cache_hits;
        self.prompt_cache_hits += work.prompt_cache_hits;
        self.prepared_questions += work.prepared_questions;
        self.prefill_yields += work.prefill_yields;
        self.prefill_interleaves += work.prefill_interleaves;
    }
}

/// Opaque, single-use evaluation prepared for one immutable engine/replica group.
/// Request, options and prompts cannot be altered or transferred to another model.
pub struct PreparedEvaluation {
    owner: Arc<()>,
    request: SystemOneRequest,
    options: EvalOptions,
    kind: PreparedKind,
    result_key: Option<Vec<u8>>,
    stats: EvalStats,
}

enum PreparedKind {
    Cached(SystemOneResponse),
    Prompts(Vec<BuiltPrompt>),
    Joint,
}

struct RequestPrefix<'a> {
    backend: &'a Mutex<Box<dyn Backend>>,
    handle: CacheHandle,
    tokens: Vec<u32>,
}

impl Drop for RequestPrefix<'_> {
    fn drop(&mut self) {
        if let Ok(mut backend) = self.backend.lock() {
            let _ = backend.release_cache(self.handle);
        }
    }
}

/// A loaded, serving-ready model.
pub struct Engine {
    manifest: Arc<ModelManifest>,
    /// The backend. Wrapped in a [`Mutex`] so a shared [`Engine`] can drive it
    /// through `&self`; this is the synchronization point for the (v1) in-process
    /// scheduler, which serializes forwards per model.
    backend: Arc<Mutex<Box<dyn Backend>>>,
    tokenizer: Arc<dyn Tokenizer>,
    formatter: Arc<dyn PromptFormatter>,
    head: Arc<HeadParams>,
    backend_id: BackendId,
    dtype: String,
    /// Actual execution device reported by the loaded backend.
    device: String,
    /// Loaded backend kernel/execution metadata, separate from the dtype label.
    execution_metadata: BTreeMap<String, String>,
    /// Resolved calibration entry for this backend+dtype.
    calibration: CalibrationEntry,
    supports_fork: bool,
    supports_batch: bool,
    supports_fork_batch: bool,
    supports_padded_batch: bool,
    batch_limits: crate::backend::BatchLimits,
    supports_resumable_prefill: bool,
    response_cache: Option<Arc<Mutex<ResponseCache>>>,
    prompt_cache: Option<Arc<Mutex<PromptCache>>>,
    preparation_identity: Arc<()>,
}

impl Engine {
    /// Build an engine from its parts.
    pub fn new(
        manifest: ModelManifest,
        backend: Box<dyn Backend>,
        tokenizer: Box<dyn Tokenizer>,
        head: HeadParams,
        backend_id: BackendId,
        dtype: impl Into<String>,
    ) -> Result<Engine> {
        let dtype = dtype.into();
        let calibration = manifest
            .calibration
            .resolve(&backend_id.to_string(), &dtype);
        let mut capabilities = backend.capabilities();
        let supports_fork = capabilities.supports_fork;
        let supports_fork_batch = backend.supports_fork_batch();
        let supports_batch = if manifest.family == Family::F5 {
            backend.supports_request_batch()
        } else {
            backend.supports_batch()
        };
        let supports_padded_batch = if manifest.family == Family::F5 {
            backend.supports_padded_request_batch()
        } else {
            backend.supports_padded_batch()
        };
        let batch_limits = backend.batch_limits();
        if !(1..=64).contains(&batch_limits.max_rows) || batch_limits.max_readouts == Some(0) {
            return Err(Error::Backend("invalid native batch limits".into()));
        }
        let supports_resumable_prefill = backend.supports_resumable_prefill();
        let device = capabilities
            .extra
            .remove("device")
            .unwrap_or_else(|| "unknown device".into());
        Ok(Engine {
            formatter: formatter_for(&manifest).into(),
            manifest: Arc::new(manifest),
            backend: Arc::new(Mutex::new(backend)),
            tokenizer: tokenizer.into(),
            head: Arc::new(head),
            backend_id,
            dtype,
            device,
            execution_metadata: capabilities.extra,
            calibration,
            supports_fork,
            supports_batch,
            supports_fork_batch,
            supports_padded_batch,
            batch_limits,
            supports_resumable_prefill,
            response_cache: None,
            prompt_cache: None,
            preparation_identity: Arc::new(()),
        })
    }

    /// Share this immutable model's preprocessing, heads, calibration and exact
    /// caches, with an independently locked backend and fresh native state.
    /// Prepared packets can move within this replica group, never to a model
    /// created separately. Serving qualification is still the caller's duty.
    pub fn replica(&self) -> Result<Self> {
        let original = self
            .backend
            .lock()
            .map_err(|_| Error::Backend("backend lock poisoned".into()))?;
        let backend = original.replica()?;
        let expected = original.capabilities();
        let actual = backend.capabilities();
        if backend.id() != original.id()
            || actual.id != expected.id
            || actual.dtype != expected.dtype
            || actual.extra != expected.extra
            || actual.families != expected.families
            || actual.max_context != expected.max_context
            || actual.supports_fork != expected.supports_fork
            || actual.supports_lora != expected.supports_lora
            || backend.supports_batch() != original.supports_batch()
            || backend.supports_fork_batch() != original.supports_fork_batch()
            || backend.supports_padded_batch() != original.supports_padded_batch()
            || backend.supports_request_batch() != original.supports_request_batch()
            || backend.supports_padded_request_batch() != original.supports_padded_request_batch()
            || backend.batch_limits() != original.batch_limits()
            || backend.supports_resumable_prefill() != original.supports_resumable_prefill()
        {
            return Err(Error::Backend(
                "replica changed the model execution identity".into(),
            ));
        }
        Ok(Self {
            manifest: self.manifest.clone(),
            backend: Arc::new(Mutex::new(backend)),
            tokenizer: self.tokenizer.clone(),
            formatter: self.formatter.clone(),
            head: self.head.clone(),
            backend_id: self.backend_id,
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            execution_metadata: self.execution_metadata.clone(),
            calibration: self.calibration.clone(),
            supports_fork: self.supports_fork,
            supports_batch: self.supports_batch,
            supports_fork_batch: self.supports_fork_batch,
            supports_padded_batch: self.supports_padded_batch,
            batch_limits: self.batch_limits,
            supports_resumable_prefill: self.supports_resumable_prefill,
            response_cache: self.response_cache.clone(),
            prompt_cache: self.prompt_cache.clone(),
            preparation_identity: self.preparation_identity.clone(),
        })
    }

    /// Opt in to exact whole-request response reuse for this immutable engine.
    /// Zero (default) disables retention. The FIFO has a charged-byte budget
    /// and 1,024-entry bound; oversized responses and errors are not cached.
    /// Cached data includes input text and is held only for this engine's life.
    pub fn with_result_cache(mut self, max_bytes: usize) -> Self {
        self.response_cache =
            (max_bytes > 0).then(|| Arc::new(Mutex::new(ResponseCache::new(max_bytes))));
        self
    }

    /// Reuse exact prepared F1–F4 prompts within this immutable engine. Zero
    /// disables retention and cache-key/lock overhead. F5 owns its preparation.
    /// A charged-byte budget and 1,024-entry FIFO bound retained inputs/tokens.
    pub fn with_prompt_cache(mut self, max_bytes: usize) -> Self {
        self.prompt_cache = (max_bytes > 0 && self.family() != Family::F5)
            .then(|| Arc::new(Mutex::new(PromptCache::new(max_bytes))));
        self
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn family(&self) -> Family {
        self.manifest.family
    }

    pub fn backend_id(&self) -> BackendId {
        self.backend_id
    }

    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    pub fn execution_metadata(&self) -> &BTreeMap<String, String> {
        &self.execution_metadata
    }

    pub fn calibration(&self) -> &CalibrationEntry {
        &self.calibration
    }

    /// Qualification must break probability ties in the same candidate order
    /// as inference, including formatter-specific yes/no conventions.
    pub(crate) fn candidate_labels(
        &self,
        state: &StateValue,
        question: &Question,
    ) -> Result<Vec<String>> {
        let candidates = if self.family() == Family::F5 {
            joint_candidates(question)
        } else {
            self.formatter
                .build(state, question, self.tokenizer.as_ref())?
                .candidates
        };
        Ok(candidates
            .into_iter()
            .map(|candidate| candidate.label)
            .collect())
    }

    pub fn supports_prefix_cache(&self) -> bool {
        self.family() == Family::F2
            && self.manifest.prompt_contract.template == "kev-v1"
            && self.supports_fork
    }

    pub fn supports_batch(&self) -> bool {
        self.supports_batch
    }

    pub fn supports_fork_batch(&self) -> bool {
        self.supports_prefix_cache() && self.supports_fork_batch
    }

    pub fn supports_padded_batch(&self) -> bool {
        self.supports_padded_batch
    }

    pub fn supports_resumable_prefill(&self) -> bool {
        self.supports_prefix_cache() && self.supports_resumable_prefill && self.device == "CPU"
    }

    /// Qualification starts retained-prefix tests from fresh model state.
    /// Active immutable branch handles are not released by this operation.
    pub fn clear_prefix_cache(&self) -> Result<()> {
        self.backend
            .lock()
            .map_err(|_| Error::Backend("backend lock poisoned".into()))?
            .clear_prefix_cache()
    }

    /// Evaluate a request and produce a response.
    pub fn eval(&self, req: &SystemOneRequest, opts: &EvalOptions) -> Result<SystemOneResponse> {
        self.eval_with_stats(req, opts, &mut EvalStats::default())
    }

    /// Prepare owned inputs without acquiring the backend lock. Successful
    /// preparation freezes all prompts/options for later, identical model work.
    /// Errors submit no forwards. F5 keeps tokenization in its joint backend.
    pub fn prepare_eval_with_stats(
        &self,
        request: SystemOneRequest,
        options: EvalOptions,
        stats: &mut EvalStats,
    ) -> Result<PreparedEvaluation> {
        self.prepare_eval(request, options, stats, true)
    }

    /// Fresh owned preparation for qualification, bypassing all retention.
    pub fn prepare_eval_uncached_with_stats(
        &self,
        request: SystemOneRequest,
        options: EvalOptions,
        stats: &mut EvalStats,
    ) -> Result<PreparedEvaluation> {
        self.prepare_eval(request, options, stats, false)
    }

    fn prepare_eval(
        &self,
        request: SystemOneRequest,
        options: EvalOptions,
        stats: &mut EvalStats,
        reuse: bool,
    ) -> Result<PreparedEvaluation> {
        *stats = EvalStats::default();
        request.validate()?;
        self.validate_options(&options)?;
        let key = self
            .response_cache
            .as_ref()
            .filter(|_| reuse)
            .map(|_| cache_key::request(&request, &options))
            .transpose()?;
        let cached = if let (Some(cache), Some(key)) = (&self.response_cache, &key) {
            cache.lock().ok().and_then(|cache| cache.get(key))
        } else {
            None
        };
        let kind = if let Some(response) = cached {
            stats.result_cache_hits = 1;
            PreparedKind::Cached(response)
        } else if self.family() == Family::F5 {
            PreparedKind::Joint
        } else {
            PreparedKind::Prompts(self.prepare_prompts(&request, &options, stats, reuse)?)
        };
        Ok(PreparedEvaluation {
            owner: self.preparation_identity.clone(),
            request,
            options,
            kind,
            result_key: key,
            stats: stats.clone(),
        })
    }

    /// Consume preparation on its originating engine. Backend call order,
    /// shapes, heads and calibration are identical to the corresponding eval.
    pub fn eval_prepared_with_stats(
        &self,
        prepared: PreparedEvaluation,
        stats: &mut EvalStats,
    ) -> Result<SystemOneResponse> {
        *stats = EvalStats::default();
        if prepared.options.cooperative_prefill {
            let preparation = prepared.stats.clone();
            let mut cursor = self.begin_resumable_evaluation(prepared)?;
            stats.accumulate(&preparation);
            loop {
                let mut step = EvalStats::default();
                let result = self.advance_resumable_evaluation(&mut cursor, &mut step);
                stats.accumulate(&step);
                if let Some(response) = result? {
                    return Ok(response);
                }
            }
        }
        if !Arc::ptr_eq(&self.preparation_identity, &prepared.owner) {
            return Err(Error::Request(
                "prepared evaluation belongs to a different engine".into(),
            ));
        }
        *stats = prepared.stats;
        let prompts = match prepared.kind {
            PreparedKind::Cached(response) => return Ok(response),
            PreparedKind::Prompts(prompts) => Some(prompts),
            PreparedKind::Joint => None,
        };
        let response = self.eval_inputs_with_stats(
            &prepared.request,
            &prepared.options,
            stats,
            true,
            prompts,
        )?;
        if let (Some(key), Some(cache)) = (prepared.result_key, &self.response_cache) {
            if let Ok(mut cache) = cache.lock() {
                cache.insert(key, &response);
            }
        }
        Ok(response)
    }

    /// Collate opaque preparations from this engine into bounded native batches.
    /// F5 batches complete schemas; other families batch questions.
    /// Each request keeps its own IDs, usage and
    /// extensions; prompt boundaries and trained heads remain unchanged.
    /// A backend failure fails this entire collated group.
    pub fn eval_prepared_batch_with_stats(
        &self,
        packets: Vec<PreparedEvaluation>,
        budget: usize,
        stats: &mut EvalStats,
    ) -> Result<Vec<SystemOneResponse>> {
        *stats = EvalStats::default();
        if packets.is_empty() || packets.len() > 64 || budget == 0 {
            return Err(Error::Request(
                "collation requires 1–64 requests and a positive token budget".into(),
            ));
        }
        // Check the complete group before any work or retained responses escape.
        let mut options = packets[0].options.clone();
        options.extensions = false;
        for packet in &packets {
            if !Arc::ptr_eq(&self.preparation_identity, &packet.owner) {
                return Err(Error::Request(
                    "prepared evaluation belongs to a different engine".into(),
                ));
            }
            let mut other = packet.options.clone();
            other.extensions = false;
            if serde_json::to_vec(&other)? != serde_json::to_vec(&options)?
                || other.prefix_cache
                || other.max_batch_tokens != Some(budget)
            {
                return Err(Error::Request(
                    "collated requests require identical execution options and token budgets without prefix reuse".into(),
                ));
            }
        }
        if self.family() == Family::F5 && self.supports_batch() {
            return self.eval_prepared_joint_batch(packets, budget, stats);
        }
        if !self.supports_batch() {
            let mut responses = Vec::with_capacity(packets.len());
            for packet in packets {
                let mut work = EvalStats::default();
                let result = self.eval_prepared_with_stats(packet, &mut work);
                stats.accumulate(&work);
                responses.push(result?);
            }
            return Ok(responses);
        }
        struct Scatter {
            index: usize,
            model: String,
            keys: Vec<(String, String)>,
            tokens: u64,
            extensions: bool,
            result_key: Option<Vec<u8>>,
        }
        let mut responses: Vec<Option<SystemOneResponse>> = vec![None; packets.len()];
        let mut scatter = Vec::new();
        let mut prompts = Vec::new();
        let mut origins = Vec::new();
        // Prepared prompts already contain each original state's tokens. The
        // synthetic IDs only address the internal scatter map, never a prompt.
        let mut combined = SystemOneRequest {
            state: StateValue::from("prepared requests"),
            model: self.manifest.name.clone(),
            questions: Default::default(),
        };
        for (index, packet) in packets.into_iter().enumerate() {
            stats.accumulate(&packet.stats);
            match packet.kind {
                PreparedKind::Cached(response) => responses[index] = Some(response),
                PreparedKind::Joint => {
                    unreachable!("whole schemas are dispatched before question collation")
                }
                PreparedKind::Prompts(prepared) => {
                    let tokens = prepared.iter().map(|p| p.token_len() as u64).sum();
                    let mut keys = Vec::with_capacity(prepared.len());
                    for (question_index, (id, question)) in
                        packet.request.questions.into_iter().enumerate()
                    {
                        let key = format!("{index}:{question_index}");
                        combined.questions.insert(key.clone(), question);
                        keys.push((key, id));
                        origins.push(index);
                    }
                    prompts.extend(prepared);
                    options.extensions |= packet.options.extensions;
                    scatter.push(Scatter {
                        index,
                        model: packet.request.model,
                        keys,
                        tokens,
                        extensions: packet.options.extensions,
                        result_key: packet.result_key,
                    });
                }
            }
        }
        if !scatter.is_empty() {
            let mut combined_response = self.eval_batched(
                &combined,
                &options,
                stats,
                budget,
                false,
                Some((prompts, Some(&origins))),
            )?;
            let mut logits = combined_response
                .extensions
                .take()
                .and_then(|extensions| extensions.raw_logits)
                .unwrap_or_default();
            for request in scatter {
                let mut answers = BTreeMap::new();
                let mut raw = BTreeMap::new();
                for (key, original) in request.keys {
                    answers.insert(
                        original.clone(),
                        combined_response
                            .answers
                            .remove(&key)
                            .ok_or_else(|| Error::Backend("collation omitted a question".into()))?,
                    );
                    if request.extensions {
                        raw.insert(
                            original,
                            logits.remove(&key).ok_or_else(|| {
                                Error::Backend("collation omitted raw logits".into())
                            })?,
                        );
                    }
                }
                let mut response =
                    SystemOneResponse::new(request.model, answers, Usage::new(request.tokens));
                if request.extensions {
                    response.extensions = Some(self.extensions(raw));
                }
                if let (Some(key), Some(cache)) = (request.result_key, &self.response_cache) {
                    if let Ok(mut cache) = cache.lock() {
                        cache.insert(key, &response);
                    }
                }
                responses[request.index] = Some(response);
            }
        }
        responses
            .into_iter()
            .map(|response| {
                response.ok_or_else(|| Error::Backend("collation omitted a request".into()))
            })
            .collect()
    }

    pub fn eval_with_stats(
        &self,
        req: &SystemOneRequest,
        opts: &EvalOptions,
        stats: &mut EvalStats,
    ) -> Result<SystemOneResponse> {
        *stats = EvalStats::default();
        let key = if self.response_cache.is_some() {
            // Preserve question/option order and all execution/extension options.
            // Artifact, tokenizer, head and temperature identity are engine-local.
            Some(cache_key::request(req, opts)?)
        } else {
            None
        };
        if let (Some(key), Some(cache)) = (&key, &self.response_cache) {
            if let Ok(cache) = cache.lock() {
                if let Some(response) = cache.get(key) {
                    stats.result_cache_hits = 1;
                    return Ok(response);
                }
            }
        }
        let response = self.eval_inner_with_stats(req, opts, stats, true)?;
        if let (Some(key), Some(cache)) = (key, &self.response_cache) {
            if let Ok(mut cache) = cache.lock() {
                cache.insert(key, &response);
            }
        }
        Ok(response)
    }

    /// Execute fresh prompt preparation and model work regardless of configured
    /// retention. Qualification must use this path so cached work cannot mask
    /// drift or make physical fork/batch coverage vacuous.
    pub fn eval_uncached_with_stats(
        &self,
        req: &SystemOneRequest,
        opts: &EvalOptions,
        stats: &mut EvalStats,
    ) -> Result<SystemOneResponse> {
        self.eval_inner_with_stats(req, opts, stats, false)
    }

    fn eval_inner_with_stats(
        &self,
        req: &SystemOneRequest,
        opts: &EvalOptions,
        stats: &mut EvalStats,
        reuse_prompts: bool,
    ) -> Result<SystemOneResponse> {
        *stats = EvalStats::default();
        if opts.cooperative_prefill {
            let prepared = self.prepare_eval(req.clone(), opts.clone(), stats, reuse_prompts)?;
            return self.eval_prepared_with_stats(prepared, stats);
        }
        let prompts = if opts.prepare_all && self.family() != Family::F5 {
            req.validate()?;
            self.validate_options(opts)?;
            Some(self.prepare_prompts(req, opts, stats, reuse_prompts)?)
        } else {
            None
        };
        self.eval_inputs_with_stats(req, opts, stats, reuse_prompts, prompts)
    }

    fn validate_options(&self, opts: &EvalOptions) -> Result<()> {
        if (opts.persistent_prefix_bytes > 0 && !opts.prefix_cache)
            || opts.max_batch_tokens == Some(0)
            || (opts.cooperative_prefill && !opts.prefix_cache)
            || opts.max_batch_padding_percent > 100
            || (opts.max_batch_padding_percent > 0 && opts.max_batch_tokens.is_none())
        {
            return Err(Error::Request(
                "batch token budget must be positive, persistent prefixes require prefix reuse, and padding percent 0..100 requires batching"
                    .into(),
            ));
        }
        if opts.prefix_cache
            && opts.max_batch_tokens.is_some()
            && (!self.supports_fork_batch()
                || opts.cooperative_prefill
                || opts.max_batch_padding_percent > 0)
        {
            return Err(Error::Unsupported(
                "cached-branch batching requires CPU Kev, equal lengths and no cooperative scheduling".into(),
            ));
        }
        if opts.max_batch_padding_percent > 0 && !self.supports_padded_batch() {
            return Err(Error::Unsupported(
                "padded batching requires a supported CPU backend with original-length readouts"
                    .into(),
            ));
        }
        Ok(())
    }

    fn prepare_prompts(
        &self,
        req: &SystemOneRequest,
        opts: &EvalOptions,
        stats: &mut EvalStats,
        reuse: bool,
    ) -> Result<Vec<BuiltPrompt>> {
        let max_context = opts
            .max_context
            .unwrap_or(self.manifest.backbone.max_context);
        let mut prompts = Vec::with_capacity(req.questions.len());
        for (id, question) in &req.questions {
            let prompt = self.build_prompt(&req.state, question, stats, reuse)?;
            if prompt.token_len() > max_context
                || prompt.candidates.len() > self.manifest.prompt_contract.max_options
            {
                return Err(Error::Request(format!(
                    "question `{id}` exceeds model input budgets"
                )));
            }
            stats.prepared_questions += 1;
            prompts.push(prompt);
        }
        Ok(prompts)
    }

    fn eval_inputs_with_stats(
        &self,
        req: &SystemOneRequest,
        opts: &EvalOptions,
        stats: &mut EvalStats,
        reuse_prompts: bool,
        prepared: Option<Vec<BuiltPrompt>>,
    ) -> Result<SystemOneResponse> {
        req.validate()?;
        self.validate_options(opts)?;

        if self.family() == Family::F5 {
            return self.eval_joint(req, opts, stats);
        }
        if let Some(budget) = opts.max_batch_tokens.filter(|_| opts.prefix_cache) {
            return self.eval_fork_batched(req, opts, stats, budget, reuse_prompts, prepared);
        }
        if let Some(budget) = opts.max_batch_tokens.filter(|_| self.supports_batch) {
            return self.eval_batched(
                req,
                opts,
                stats,
                budget,
                reuse_prompts,
                prepared.map(|prompts| (prompts, None)),
            );
        }

        let max_context = opts
            .max_context
            .unwrap_or(self.manifest.backbone.max_context);
        let max_options = self.manifest.prompt_contract.max_options;

        let mut answers: BTreeMap<String, Answer> = BTreeMap::new();
        let mut raw_logits: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let mut total_tokens = 0u64;
        let use_prefix =
            opts.prefix_cache && self.supports_prefix_cache() && req.questions.len() > 1;
        let mut prefix: Option<RequestPrefix<'_>> = None;
        let mut prepared = prepared.map(Vec::into_iter);

        // Iterate questions deterministically.
        for (id, question) in &req.questions {
            let mut prompt =
                self.next_prompt(&mut prepared, &req.state, question, stats, reuse_prompts)?;

            // CORE-08: enforce model input budgets; never silently truncate.
            let n_tokens = prompt.token_len();
            if n_tokens > max_context {
                return Err(Error::Request(format!(
                    "question `{id}` requires {n_tokens} tokens but the model context is {max_context}"
                )));
            }
            let n_options = prompt.candidates.len();
            if n_options > max_options {
                return Err(Error::Request(format!(
                    "question `{id}` has {n_options} options; the model supports at most {max_options}"
                )));
            }
            total_tokens += n_tokens as u64;

            // Run the backend (v1 serializes forwards per model).
            let mut positions: Vec<usize> = prompt.candidates.iter().map(|c| c.position).collect();
            let reference_f3 = self.family() == Family::F3 && opts.reference_readout;
            if !reference_f3 {
                positions.sort_unstable();
                positions.dedup();
            }
            let mut backend = self
                .backend
                .lock()
                .map_err(|_| Error::Backend("backend lock poisoned".into()))?;
            let prefix_len = prompt.prefix_len;
            let eligible = use_prefix
                && prefix_len > 0
                && prefix_len < prompt.tokens.len()
                && positions.iter().all(|&position| position >= prefix_len);
            if eligible && prefix.is_none() {
                let mut prefill_work = crate::backend::PrefillWork::default();
                let result = backend.prefill_cached_with_work(
                    &prompt.tokens[..prefix_len],
                    opts.persistent_prefix_bytes,
                    &mut prefill_work,
                );
                stats.prefill_calls += prefill_work.forward_calls;
                stats.processed_tokens += prefill_work.processed_tokens;
                stats.chunked_prefills += prefill_work.chunked_prefills;
                let cached = result?;
                if cached.hit {
                    stats.persistent_prefix_hits += 1;
                }
                let handle = cached.handle;
                prefix = Some(RequestPrefix {
                    backend: &self.backend,
                    handle,
                    tokens: prompt.tokens[..prefix_len].to_vec(),
                });
            }
            let branch = if eligible
                && prefix
                    .as_ref()
                    .is_some_and(|prefix| prefix.tokens == prompt.tokens[..prefix_len])
            {
                let handle = backend.fork(prefix.as_ref().unwrap().handle)?;
                stats.cache_forks += 1;
                stats.reused_prefix_tokens += prefix_len as u64;
                prompt.tokens = prompt.tokens.split_off(prefix_len);
                for position in &mut positions {
                    *position -= prefix_len;
                }
                for candidate in &mut prompt.candidates {
                    candidate.position -= prefix_len;
                }
                Some(handle)
            } else {
                None
            };
            let mut input = ForwardInput::new(prompt.tokens, positions).with_qtype(prompt.qtype);
            input.fork_from = branch;
            if self.family() == Family::F3 && !reference_f3 {
                input =
                    input.with_logit_codes(prompt.candidates.iter().map(|c| c.code_id).collect());
            }
            stats.forward_calls += 1;
            stats.processed_tokens += input.tokens.len() as u64;
            let output = backend.forward(input);
            let released = branch
                .map(|handle| backend.release_cache(handle))
                .transpose();
            drop(backend);
            let output = output?;
            released?;

            // Head -> candidate logits.
            let logits = head::candidate_logits(
                self.manifest.family,
                self.manifest.head.kind,
                &output,
                &prompt.candidates,
                &self.head,
            )?;

            // Calibration: temperature + softmax.
            let temperature = self.temperature_for(question, n_options);
            let probabilities = calibration::calibrate(&logits, temperature)?;

            if opts.extensions {
                raw_logits.insert(id.clone(), logits);
            }

            let answer = self.build_answer(question, &prompt.candidates, &probabilities)?;
            answers.insert(id.clone(), answer);
        }

        let mut response =
            SystemOneResponse::new(req.model.clone(), answers, Usage::new(total_tokens));
        if opts.extensions {
            response.extensions = Some(self.extensions(raw_logits));
        }
        Ok(response)
    }

    fn eval_batched(
        &self,
        req: &SystemOneRequest,
        opts: &EvalOptions,
        stats: &mut EvalStats,
        budget: usize,
        reuse_prompts: bool,
        preparation: Option<(Vec<BuiltPrompt>, Option<&[usize]>)>,
    ) -> Result<SystemOneResponse> {
        let (prepared, origins) = match preparation {
            Some((prompts, origins)) => (Some(prompts), origins),
            None => (None, None),
        };
        if opts.max_batch_padding_percent > 0 && !self.supports_padded_batch() {
            return Err(Error::Unsupported(
                "padded batching requires a supported CPU backend with original-length readouts"
                    .into(),
            ));
        }
        let max_context = opts
            .max_context
            .unwrap_or(self.manifest.backbone.max_context);
        let mut jobs = Vec::with_capacity(req.questions.len());
        let mut buckets: BTreeMap<usize, Vec<(usize, ForwardInput)>> = BTreeMap::new();
        let mut total_tokens = 0;
        let mut prepared = prepared.map(Vec::into_iter);
        for (id, question) in &req.questions {
            let mut prompt =
                self.next_prompt(&mut prepared, &req.state, question, stats, reuse_prompts)?;
            let length = prompt.token_len();
            if length > max_context
                || prompt.candidates.len() > self.manifest.prompt_contract.max_options
            {
                return Err(Error::Request(format!(
                    "question `{id}` exceeds model input budgets"
                )));
            }
            total_tokens += length as u64;
            let reference = self.family() == Family::F3 && opts.reference_readout;
            let mut positions: Vec<_> = prompt.candidates.iter().map(|c| c.position).collect();
            if !reference {
                positions.sort_unstable();
                positions.dedup();
            }
            let mut input = ForwardInput::new(std::mem::take(&mut prompt.tokens), positions)
                .with_qtype(prompt.qtype);
            if self.family() == Family::F3 && !reference {
                input =
                    input.with_logit_codes(prompt.candidates.iter().map(|c| c.code_id).collect());
            }
            buckets.entry(length).or_default().push((jobs.len(), input));
            jobs.push((id, question, prompt));
        }
        let mut outputs: Vec<Option<crate::backend::ForwardOutput>> = vec![None; jobs.len()];
        let groups = padded_groups(
            buckets.into_values().flatten().collect(),
            budget,
            opts.max_batch_padding_percent,
            self.batch_limits,
        );
        for group in groups {
            // An oversized singleton still runs independently, never silently
            // truncates. Exact-length bucketing needs no padding/mask changes.
            let length = group
                .iter()
                .map(|(_, input)| input.tokens.len())
                .max()
                .unwrap();
            let logical = group
                .iter()
                .map(|(_, input)| input.tokens.len())
                .sum::<usize>();
            let cross_request = origins.is_some_and(|origins| {
                group
                    .iter()
                    .any(|(index, _)| origins[*index] != origins[group[0].0])
            });
            let (indices, inputs): (Vec<_>, Vec<_>) = group.into_iter().unzip();
            let count = inputs.len();
            stats.forward_calls += 1;
            stats.processed_tokens += (length * count) as u64;
            let padding = length * count - logical;
            stats.padded_tokens += padding as u64;
            let mut backend = self
                .backend
                .lock()
                .map_err(|_| Error::Backend("backend lock poisoned".into()))?;
            let results = if count == 1 {
                vec![backend.forward(inputs.into_iter().next().unwrap())?]
            } else {
                stats.batch_calls += 1;
                stats.cross_request_batches += u64::from(cross_request);
                if padding > 0 {
                    stats.padded_batch_calls += 1;
                    backend.forward_padded_batch(inputs)?
                } else {
                    backend.forward_batch(inputs)?
                }
            };
            if results.len() != count {
                return Err(Error::Backend(
                    "batch backend returned the wrong number of sequences".into(),
                ));
            }
            for (index, result) in indices.into_iter().zip(results) {
                outputs[index] = Some(result);
            }
        }
        let mut answers = BTreeMap::new();
        let mut raw_logits = BTreeMap::new();
        for ((id, question, prompt), output) in jobs.into_iter().zip(outputs) {
            let output =
                output.ok_or_else(|| Error::Backend("batch backend omitted a sequence".into()))?;
            let logits = head::candidate_logits(
                self.family(),
                self.manifest.head.kind,
                &output,
                &prompt.candidates,
                &self.head,
            )?;
            let probabilities = calibration::calibrate(
                &logits,
                self.temperature_for(question, prompt.candidates.len()),
            )?;
            answers.insert(
                id.clone(),
                self.build_answer(question, &prompt.candidates, &probabilities)?,
            );
            if opts.extensions {
                raw_logits.insert(id.clone(), logits);
            }
        }
        let mut response =
            SystemOneResponse::new(req.model.clone(), answers, Usage::new(total_tokens));
        if opts.extensions {
            response.extensions = Some(self.extensions(raw_logits));
        }
        Ok(response)
    }

    fn build_prompt(
        &self,
        state: &StateValue,
        question: &Question,
        stats: &mut EvalStats,
        reuse: bool,
    ) -> Result<BuiltPrompt> {
        let cache = self.prompt_cache.as_ref().filter(|_| reuse);
        let key = cache
            .map(|_| cache_key::prompt(state, question))
            .transpose()?;
        if let (Some(cache), Some(key)) = (cache, &key) {
            if let Ok(cache) = cache.lock() {
                if let Some(prompt) = cache.get(key) {
                    stats.prompt_cache_hits += 1;
                    return Ok(prompt);
                }
            }
        }
        // Concurrent misses may prepare twice; never serialize tokenization
        // behind a cache lock, and never retain formatting/tokenization errors.
        let prompt = self
            .formatter
            .build(state, question, self.tokenizer.as_ref())?;
        if let (Some(cache), Some(key)) = (cache, key) {
            if let Ok(mut cache) = cache.lock() {
                cache.insert(key, &prompt);
            }
        }
        Ok(prompt)
    }

    fn next_prompt(
        &self,
        prepared: &mut Option<std::vec::IntoIter<BuiltPrompt>>,
        state: &StateValue,
        question: &Question,
        stats: &mut EvalStats,
        reuse: bool,
    ) -> Result<BuiltPrompt> {
        match prepared {
            Some(prompts) => prompts
                .next()
                .ok_or_else(|| Error::Backend("prepared prompt missing".into())),
            None => self.build_prompt(state, question, stats, reuse),
        }
    }

    fn joint_candidates_checked(
        &self,
        req: &SystemOneRequest,
    ) -> Result<BTreeMap<String, Vec<Candidate>>> {
        req.questions
            .iter()
            .map(|(id, question)| {
                let values = joint_candidates(question);
                if values.is_empty() || values.len() > self.manifest.prompt_contract.max_options {
                    return Err(Error::Request(format!(
                        "question `{id}` has an unsupported number of options"
                    )));
                }
                Ok((id.clone(), values))
            })
            .collect()
    }

    fn joint_max_context(&self, opts: &EvalOptions) -> usize {
        opts.max_context
            .unwrap_or(self.manifest.backbone.max_context)
            .min(self.manifest.backbone.max_context)
    }

    fn eval_prepared_joint_batch(
        &self,
        packets: Vec<PreparedEvaluation>,
        budget: usize,
        stats: &mut EvalStats,
    ) -> Result<Vec<SystemOneResponse>> {
        let percent = packets[0].options.max_batch_padding_percent;
        let mut responses = vec![None; packets.len()];
        let mut jobs = Vec::new();
        for (index, packet) in packets.into_iter().enumerate() {
            stats.accumulate(&packet.stats);
            match &packet.kind {
                PreparedKind::Cached(response) => responses[index] = Some(response.clone()),
                PreparedKind::Joint => {
                    packet.request.validate()?;
                    let candidates = self.joint_candidates_checked(&packet.request)?;
                    jobs.push((index, packet, candidates));
                }
                PreparedKind::Prompts(_) => {
                    return Err(Error::Request(
                        "joint batches require whole-schema preparations".into(),
                    ))
                }
            }
        }
        if !jobs.is_empty() {
            let inputs: Vec<_> = jobs
                .iter()
                .map(|(_, packet, _)| RequestBatchInput {
                    request: &packet.request,
                    max_context: self.joint_max_context(&packet.options),
                })
                .collect();
            let mut work = RequestBatchWork::default();
            let result = self
                .backend
                .lock()
                .map_err(|_| Error::Backend("backend lock poisoned".into()))?
                .forward_request_batch(&inputs, budget, percent, &mut work);
            stats.accumulate_request_batch(&work);
            let outputs = result?;
            let logical: u64 = outputs
                .iter()
                .map(|output| output.input_tokens)
                .try_fold(0u64, |a, b| a.checked_add(b))
                .ok_or_else(|| Error::Backend("joint batch token count overflow".into()))?;
            if outputs.len() != jobs.len()
                || work.forward_calls == 0
                || work.forward_calls > jobs.len() as u64
                || work.batch_calls > work.forward_calls
                || work.batch_calls > jobs.len() as u64 / 2
                || work.padded_batch_calls > work.batch_calls
                || (work.padded_tokens > 0) != (work.padded_batch_calls > 0)
                || work.prepared_questions
                    != jobs
                        .iter()
                        .map(|(_, packet, _)| packet.request.questions.len() as u64)
                        .sum::<u64>()
                || logical.checked_add(work.padded_tokens) != Some(work.processed_tokens)
            {
                return Err(Error::Backend(
                    "joint batch returned inconsistent outputs or physical work".into(),
                ));
            }
            let completed = jobs
                .into_iter()
                .zip(outputs)
                .map(|((index, packet, candidates), output)| {
                    let response =
                        self.joint_response(&packet.request, &packet.options, output, &candidates)?;
                    Ok((index, packet.result_key, response))
                })
                .collect::<Result<Vec<_>>>()?;
            // Publish retained results only after every raw output has passed
            // schema, usage, finite-score and shared calibration checks.
            for (index, key, response) in completed {
                if let (Some(key), Some(cache)) = (key, &self.response_cache) {
                    if let Ok(mut cache) = cache.lock() {
                        cache.insert(key, &response);
                    }
                }
                responses[index] = Some(response);
            }
        }
        responses
            .into_iter()
            .map(|response| {
                response.ok_or_else(|| Error::Backend("joint batch omitted a response".into()))
            })
            .collect()
    }

    /// Joint-schema models own tokenization and score every question together.
    fn eval_joint(
        &self,
        req: &SystemOneRequest,
        opts: &EvalOptions,
        stats: &mut EvalStats,
    ) -> Result<SystemOneResponse> {
        let max_context = self.joint_max_context(opts);
        let candidates = self.joint_candidates_checked(req)?;
        let mut backend = self
            .backend
            .lock()
            .map_err(|_| Error::Backend("backend lock poisoned".into()))?;
        let output = if opts.prepare_all && self.supports_batch() {
            let mut work = RequestBatchWork::default();
            let result = backend.forward_request_batch(
                &[RequestBatchInput {
                    request: req,
                    max_context,
                }],
                opts.max_batch_tokens.unwrap_or(usize::MAX),
                opts.max_batch_padding_percent,
                &mut work,
            );
            stats.accumulate_request_batch(&work);
            let mut outputs = result?;
            if outputs.len() != 1
                || work.forward_calls != 1
                || work.batch_calls != 0
                || work.padded_batch_calls != 0
                || work.padded_tokens != 0
                || outputs[0].input_tokens != work.processed_tokens
            {
                return Err(Error::Backend(
                    "joint singleton returned inconsistent outputs or work".into(),
                ));
            }
            outputs.remove(0)
        } else {
            stats.forward_calls += 1;
            let output = backend.forward_request(req, max_context)?;
            stats.processed_tokens += output.input_tokens;
            output
        };
        drop(backend);
        self.joint_response(req, opts, output, &candidates)
    }

    fn joint_response(
        &self,
        req: &SystemOneRequest,
        opts: &EvalOptions,
        output: RequestOutput,
        candidates: &BTreeMap<String, Vec<Candidate>>,
    ) -> Result<SystemOneResponse> {
        let max_context = self.joint_max_context(opts);
        if output.input_tokens == 0 || output.input_tokens > max_context as u64 {
            return Err(Error::Backend(
                "joint backend returned invalid token usage".into(),
            ));
        }
        if output.logits.len() != req.questions.len() {
            return Err(Error::Backend(
                "joint backend returned the wrong number of questions".into(),
            ));
        }
        let mut answers = BTreeMap::new();
        let mut raw_logits = BTreeMap::new();
        for (id, question) in &req.questions {
            let options = &candidates[id];
            let scores = output
                .logits
                .get(id)
                .ok_or_else(|| Error::Backend(format!("joint backend omitted question `{id}`")))?;
            if scores.len() != options.len() {
                return Err(Error::Backend(format!(
                    "joint backend returned the wrong options for `{id}`"
                )));
            }
            let logits = options
                .iter()
                .map(|c| {
                    let label = match (question, c.label.as_str()) {
                        (Question::Noul { .. }, "yes") => "true",
                        (Question::Noul { .. }, "no") => "false",
                        _ => &c.label,
                    };
                    let value = scores.get(label).copied().ok_or_else(|| {
                        Error::Backend(format!("joint backend omitted option `{label}` for `{id}`"))
                    })?;
                    if !value.is_finite() {
                        return Err(Error::Backend(format!(
                            "joint backend returned non-finite logits for `{id}`"
                        )));
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<_>>>()?;
            let probabilities =
                calibration::calibrate(&logits, self.temperature_for(question, options.len()))?;
            answers.insert(
                id.clone(),
                self.build_answer(question, options, &probabilities)?,
            );
            if opts.extensions {
                raw_logits.insert(id.clone(), logits);
            }
        }
        let mut response =
            SystemOneResponse::new(req.model.clone(), answers, Usage::new(output.input_tokens));
        if opts.extensions {
            response.extensions = Some(self.extensions(raw_logits));
        }
        Ok(response)
    }

    /// Resolve the temperature for a question (Laya uses per-type base temps plus
    /// per `{type}:{bucket}` overrides; F4 uses per-type temps; others use the default).
    fn temperature_for(&self, question: &Question, n_options: usize) -> f32 {
        let type_name = question.type_name();
        // Laya-style per-option-count override, keyed `{type}:{bucket}`.
        if let Some(tbo) = &self.calibration.temperature_by_options {
            let bucket = format!("{}:{}", type_name, bucket_size(n_options));
            if let Some(t) = tbo.get(&bucket) {
                return *t;
            }
        }
        if let Some(per_type) = &self.calibration.per_type_temperatures {
            if let Some(t) = per_type.get(type_name) {
                return *t;
            }
        }
        self.calibration.temperature
    }

    fn build_answer(
        &self,
        question: &Question,
        candidates: &[Candidate],
        probabilities: &[f32],
    ) -> Result<Answer> {
        if candidates.len() != probabilities.len() {
            return Err(Error::Calibration(format!(
                "candidate/probability count mismatch: {} vs {}",
                candidates.len(),
                probabilities.len()
            )));
        }
        let conf = if self.manifest.prompt_contract.template == "kev-v1"
            && matches!(question, Question::Score { .. })
        {
            calibration::confidence_score(probabilities)
        } else {
            confidence(probabilities, &self.calibration.confidence)
        };
        match question {
            Question::Choice { .. } => {
                let mut best = 0usize;
                for (i, &p) in probabilities.iter().enumerate() {
                    if p > probabilities[best] {
                        best = i;
                    }
                }
                let mut probs_map = BTreeMap::new();
                for c in candidates {
                    probs_map.insert(c.label.clone(), probabilities[c.index]);
                }
                Ok(Answer::Choice {
                    choice: candidates[best].label.clone(),
                    probabilities: probs_map,
                    confidence: conf,
                })
            }
            Question::Score { .. } => {
                let mut probs_map = BTreeMap::new();
                let mut legend = BTreeMap::new();
                let mut score = 0.0f32;
                for c in candidates {
                    let key = c.label.clone(); // level index as string
                    probs_map.insert(key.clone(), probabilities[c.index]);
                    legend.insert(key, c.description.clone().unwrap_or_else(|| "".into()));
                    score += c.index as f32 * probabilities[c.index];
                }
                Ok(Answer::Score {
                    probabilities: probs_map,
                    score,
                    legend,
                    confidence: conf,
                })
            }
            Question::Noul { .. } => {
                // Formatter-specific candidate order; return the probability of yes.
                let p_yes = candidates
                    .iter()
                    .position(|c| c.label == "yes")
                    .map(|i| probabilities[i])
                    .unwrap_or(0.0);
                Ok(Answer::Noul { noul: p_yes })
            }
        }
    }

    fn extensions(&self, raw_logits: BTreeMap<String, Vec<f32>>) -> crate::contract::Extensions {
        crate::contract::Extensions {
            backend: Some(self.backend_id.to_string()),
            dtype: Some(self.dtype.clone()),
            calibration_status: Some(format!("{:?}", self.calibration.status).to_lowercase()),
            confidence_definition: Some(self.calibration.confidence.to_string()),
            prompt_contract_hash: Some(self.manifest.prompt_contract.contract_hash.clone()),
            raw_logits: if raw_logits.is_empty() {
                None
            } else {
                Some(raw_logits)
            },
        }
    }
}

fn joint_candidates(question: &Question) -> Vec<Candidate> {
    let labels: Vec<(String, Option<String>, CandidateKind)> = match question {
        Question::Choice { criteria, .. } => criteria
            .keys()
            .map(|label| (label.clone(), None, CandidateKind::Option))
            .collect(),
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(index, level)| {
                (
                    index.to_string(),
                    Some(
                        level
                            .as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| level.to_string()),
                    ),
                    CandidateKind::Level,
                )
            })
            .collect(),
        Question::Noul { .. } => vec![
            ("yes".into(), None, CandidateKind::YesNo),
            ("no".into(), None, CandidateKind::YesNo),
        ],
    };
    labels
        .into_iter()
        .enumerate()
        .map(|(index, (label, description, kind))| Candidate {
            kind,
            position: 0,
            code_id: 0,
            label,
            description,
            index,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, CacheHandle, Capabilities, ForwardOutput};
    use crate::manifest::{
        Backbone, BackboneSource, CalibrationConfig, CalibrationEntry, CalibrationStatus,
        ConfidenceDef, HeadConfig, HeadKind, PromptContract, Reference,
    };
    use crate::tokenizer::SimpleTokenizer;

    /// A deterministic backend that returns hidden states (Features) so the
    /// feature-projection head (F1) can be exercised without real weights.
    struct TestBackend;

    impl Backend for TestBackend {
        fn replica(&self) -> Result<Box<dyn Backend>> {
            Ok(Box::new(Self))
        }
        fn id(&self) -> BackendId {
            BackendId::Onnx
        }
        fn capabilities(&self) -> Capabilities {
            let mut c = Capabilities::default();
            c.id = BackendId::Onnx;
            c.dtype = "fp32".into();
            c.max_context = 4096;
            c.families = vec![Family::F1];
            c
        }
        fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
            // Produce a sparse deterministic hidden vector per position derived
            // from the token id at that position.
            let n = input.positions.len();
            let hidden = 64usize;
            let mut data = vec![0.0f32; n * hidden];
            for (row, &pos) in input.positions.iter().enumerate() {
                let tok = input.tokens.get(pos).copied().unwrap_or(0);
                let col = (tok as usize) % hidden;
                data[row * hidden + col] = tok as f32 + 0.5;
            }
            let values = Tensor::new(vec![n, hidden], data).unwrap();
            Ok(ForwardOutput::Features {
                positions: input.positions,
                values,
            })
        }
        fn fork(&mut self, handle: CacheHandle) -> Result<CacheHandle> {
            Ok(handle)
        }
    }

    fn manifest() -> ModelManifest {
        let m = ModelManifest {
            schema_version: crate::manifest::MANIFEST_SCHEMA_VERSION.into(),
            name: "test".into(),
            family: Family::F1,
            backbone: Backbone {
                source: BackboneSource::Hf {
                    repo: "x".into(),
                    revision: "y".into(),
                },
                artifacts: Default::default(),
                hidden_size: 1024,
                max_context: 4096,
                tokenizer: None,
            },
            adapter: None,
            f3: None,
            head: HeadConfig {
                kind: HeadKind::OptionMarker,
                weights: "head.safetensors".into(),
                width: 1,
                pointer_offset: None,
            },
            prompt_contract: PromptContract {
                template: "laya-v1".into(),
                option_marker_tokens: vec!["<option:0>".into()],
                state_budget: 3072,
                head_budget: 512,
                max_options: 255,
                contract_hash: "abc".into(),
                max_len: 512,
                head_max_len: 192,
            },
            calibration: CalibrationConfig {
                default: CalibrationEntry {
                    temperature: 1.0,
                    per_type_temperatures: None,
                    temperature_by_options: None,
                    confidence: ConfidenceDef::Peak,
                    status: CalibrationStatus::Fit,
                },
                entries: Default::default(),
                eval_set_hash: None,
            },
            reference: Some(Reference {
                family_impl: "x".into(),
                revision: "y".into(),
                golden: "g.json".into(),
            }),
            capabilities: Default::default(),
        };
        m
    }

    fn engine() -> Engine {
        let tk: Box<dyn Tokenizer> = Box::new(SimpleTokenizer::new(32768));
        Engine::new(
            manifest(),
            Box::new(TestBackend),
            tk,
            HeadParams::default(),
            BackendId::Onnx,
            "fp32",
        )
        .unwrap()
    }

    #[test]
    fn replicas_share_immutable_preparation_and_bounded_exact_caches_only_within_the_group() {
        let primary = engine()
            .with_prompt_cache(1 << 20)
            .with_result_cache(1 << 20);
        let replica = primary.replica().unwrap();
        assert!(Arc::ptr_eq(&primary.manifest, &replica.manifest));
        assert!(Arc::ptr_eq(&primary.tokenizer, &replica.tokenizer));
        assert!(Arc::ptr_eq(&primary.head, &replica.head));
        assert!(Arc::ptr_eq(
            primary.response_cache.as_ref().unwrap(),
            replica.response_cache.as_ref().unwrap()
        ));
        let request = req();
        let opts = EvalOptions {
            extensions: true,
            prepare_all: true,
            ..Default::default()
        };
        let expected = primary
            .eval_uncached_with_stats(&request, &opts, &mut Default::default())
            .unwrap();
        let prepared = primary
            .prepare_eval_with_stats(request.clone(), opts.clone(), &mut Default::default())
            .unwrap();
        let mut work = EvalStats::default();
        let result = replica
            .eval_prepared_with_stats(prepared, &mut work)
            .unwrap();
        assert_eq!(
            serde_json::to_value(&result).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        assert_eq!(work.forward_calls, request.questions.len() as u64);
        primary.eval_with_stats(&request, &opts, &mut work).unwrap();
        assert_eq!(work.result_cache_hits, 1);
        assert_eq!(work.forward_calls, 0);
        let independent = engine();
        let packet = primary
            .prepare_eval_with_stats(request, opts, &mut Default::default())
            .unwrap();
        assert!(independent
            .eval_prepared_with_stats(packet, &mut work)
            .is_err());
        assert_eq!(work.forward_calls, 0);
    }

    #[test]
    fn exact_result_reuse_preserves_all_fields_and_submits_no_work() {
        let engine = engine()
            .with_result_cache(1024 * 1024)
            .with_prompt_cache(1024 * 1024);
        let opts = EvalOptions {
            extensions: true,
            ..Default::default()
        };
        let request = req();
        let mut stats = EvalStats::default();
        let first = engine.eval_with_stats(&request, &opts, &mut stats).unwrap();
        assert_eq!(stats.forward_calls, 3);
        assert_eq!(stats.result_cache_hits, 0);
        let expected = serde_json::to_vec(&first).unwrap();
        let mut second = engine.eval_with_stats(&request, &opts, &mut stats).unwrap();
        assert_eq!(serde_json::to_vec(&second).unwrap(), expected);
        assert_eq!(stats.result_cache_hits, 1);
        assert_eq!(stats.forward_calls, 0);
        assert_eq!(stats.processed_tokens, 0);
        assert_eq!(stats.prompt_cache_hits, 0);
        assert!(second.usage.input_tokens > 0);
        second.model.clear();
        second
            .extensions
            .as_mut()
            .unwrap()
            .raw_logits
            .as_mut()
            .unwrap()
            .clear();
        let third = engine.eval(&request, &opts).unwrap();
        assert_eq!(serde_json::to_vec(&third).unwrap(), expected);
        let uncached = engine
            .eval_uncached_with_stats(&request, &opts, &mut stats)
            .unwrap();
        assert_eq!(serde_json::to_vec(&uncached).unwrap(), expected);
        assert_eq!(stats.result_cache_hits, 0);
        assert_eq!(stats.forward_calls, 3);
        assert_eq!(stats.prompt_cache_hits, 0);
    }

    #[test]
    fn prepared_prompt_reuse_preserves_model_work_options_and_field_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        struct CountingFormatter(Arc<AtomicUsize>);
        impl PromptFormatter for CountingFormatter {
            fn family(&self) -> Family {
                Family::F1
            }
            fn build(
                &self,
                state: &StateValue,
                question: &Question,
                tokenizer: &dyn Tokenizer,
            ) -> Result<BuiltPrompt> {
                self.0.fetch_add(1, Ordering::SeqCst);
                crate::prompt::default_formatter(Family::F1).build(state, question, tokenizer)
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let mut engine = engine().with_prompt_cache(1024 * 1024);
        engine.formatter = Arc::new(CountingFormatter(calls.clone()));
        let opts = EvalOptions {
            extensions: true,
            ..Default::default()
        };
        let request = req();
        let mut stats = EvalStats::default();
        let first = engine.eval_with_stats(&request, &opts, &mut stats).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(stats.prompt_cache_hits, 0);
        let second = engine.eval_with_stats(&request, &opts, &mut stats).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(stats.prompt_cache_hits, 3);
        assert_eq!(stats.result_cache_hits, 0);
        assert_eq!(stats.forward_calls, 3);
        assert_eq!(stats.processed_tokens, first.usage.input_tokens);
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        let independent = engine
            .eval_uncached_with_stats(&request, &opts, &mut stats)
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 6);
        assert_eq!(stats.prompt_cache_hits, 0);
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&independent).unwrap()
        );
        // IDs, question order, model alias and extensions do not alter the
        // per-question prompt. Output fields still follow the current request.
        let mut reordered = request.clone();
        reordered.model = "alias".into();
        reordered.questions.reverse();
        let question = reordered.questions.shift_remove("department").unwrap();
        reordered.questions.insert("renamed".into(), question);
        let response = engine
            .eval_with_stats(&reordered, &EvalOptions::default(), &mut stats)
            .unwrap();
        assert_eq!(stats.prompt_cache_hits, 3);
        assert_eq!(response.model, "alias");
        assert!(response.answers.contains_key("renamed"));
        assert!(!response.answers.contains_key("department"));
        assert!(response.extensions.is_none());
        // Candidate order does affect the prepared prompt.
        if let Question::Choice { criteria, .. } = &mut reordered.questions["renamed"] {
            criteria.reverse();
        }
        engine
            .eval_with_stats(&reordered, &opts, &mut stats)
            .unwrap();
        assert_eq!(stats.prompt_cache_hits, 2);
        assert_eq!(calls.load(Ordering::SeqCst), 7);
        reordered.state = "different state".into();
        engine
            .eval_with_stats(&reordered, &opts, &mut stats)
            .unwrap();
        assert_eq!(stats.prompt_cache_hits, 0);
        assert_eq!(calls.load(Ordering::SeqCst), 10);
        // Cached prompts never bypass current context/validation constraints.
        assert!(engine
            .eval_with_stats(
                &request,
                &EvalOptions {
                    max_context: Some(1),
                    ..opts.clone()
                },
                &mut stats
            )
            .is_err());
        assert_eq!(stats.forward_calls, 0);
        let mut invalid = request.clone();
        invalid.model.clear();
        assert!(engine.eval_with_stats(&invalid, &opts, &mut stats).is_err());
        assert_eq!(stats.prompt_cache_hits, 0);
        // Retention belongs to this engine and zero disables an existing cache.
        let mut other = self::engine().with_prompt_cache(1024 * 1024);
        other.formatter = Arc::new(CountingFormatter(calls.clone()));
        other.eval_with_stats(&request, &opts, &mut stats).unwrap();
        assert_eq!(stats.prompt_cache_hits, 0);
        let engine = engine.with_prompt_cache(0);
        engine.eval_with_stats(&request, &opts, &mut stats).unwrap();
        assert_eq!(stats.prompt_cache_hits, 0);
    }

    #[test]
    fn prepared_evaluation_freezes_inputs_and_is_bound_to_its_engine() {
        let engine = engine().with_prompt_cache(1024 * 1024);
        let mut request = req();
        let mut opts = EvalOptions {
            extensions: true,
            prepare_all: true,
            ..Default::default()
        };
        let expected = engine
            .eval_uncached_with_stats(
                &request,
                &EvalOptions {
                    prepare_all: false,
                    ..opts.clone()
                },
                &mut Default::default(),
            )
            .unwrap();
        let mut stats = EvalStats::default();
        let prepared = engine
            .prepare_eval_with_stats(request.clone(), opts.clone(), &mut stats)
            .unwrap();
        assert_eq!(stats.prepared_questions, 3);
        assert_eq!(stats.forward_calls, 0);
        request.state = "caller changed its copy".into();
        request.questions.reverse();
        opts.extensions = false;
        let response = engine
            .eval_prepared_with_stats(prepared, &mut stats)
            .unwrap();
        assert_eq!(stats.forward_calls, 3);
        assert_eq!(stats.prepared_questions, 3);
        assert_eq!(
            serde_json::to_vec(&response).unwrap(),
            serde_json::to_vec(&expected).unwrap()
        );
        let prepared = engine
            .prepare_eval_with_stats(req(), opts.clone(), &mut stats)
            .unwrap();
        let other = self::engine();
        assert!(other
            .eval_prepared_with_stats(prepared, &mut stats)
            .is_err());
        assert_eq!(stats.forward_calls, 0);
        let mut invalid = req();
        let first = engine
            .formatter
            .build(
                &invalid.state,
                &invalid.questions["department"],
                engine.tokenizer.as_ref(),
            )
            .unwrap()
            .token_len();
        if let Question::Noul { instructions, .. } = &mut invalid.questions["is_refund"] {
            *instructions = serde_json::json!("long ".repeat(200)).into();
        }
        assert!(engine
            .prepare_eval_with_stats(
                invalid,
                EvalOptions {
                    max_context: Some(first),
                    ..opts
                },
                &mut stats
            )
            .is_err());
        assert_eq!(stats.prepared_questions, 1);
        assert_eq!(stats.forward_calls, 0);
    }

    #[test]
    fn prepared_result_retention_and_qualification_have_nonvacuous_work() {
        let engine = engine()
            .with_result_cache(1024 * 1024)
            .with_prompt_cache(1024 * 1024);
        let request = req();
        let opts = EvalOptions {
            prepare_all: true,
            extensions: true,
            ..Default::default()
        };
        let mut stats = EvalStats::default();
        let prepared = engine
            .prepare_eval_with_stats(request.clone(), opts.clone(), &mut stats)
            .unwrap();
        let response = engine
            .eval_prepared_with_stats(prepared, &mut stats)
            .unwrap();
        assert_eq!(stats.prepared_questions, 3);
        assert_eq!(stats.forward_calls, 3);
        let prepared = engine
            .prepare_eval_with_stats(request.clone(), opts.clone(), &mut stats)
            .unwrap();
        assert_eq!(stats.result_cache_hits, 1);
        assert_eq!(stats.prepared_questions, 0);
        let cached = engine
            .eval_prepared_with_stats(prepared, &mut stats)
            .unwrap();
        assert_eq!(
            serde_json::to_vec(&response).unwrap(),
            serde_json::to_vec(&cached).unwrap()
        );
        assert_eq!(stats.forward_calls, 0);
        assert_eq!(stats.result_cache_hits, 1);
        let fresh = engine
            .eval_uncached_with_stats(&request, &opts, &mut stats)
            .unwrap();
        assert_eq!(
            serde_json::to_vec(&response).unwrap(),
            serde_json::to_vec(&fresh).unwrap()
        );
        assert_eq!(stats.forward_calls, 3);
        assert_eq!(stats.prepared_questions, 3);
        assert_eq!(stats.prompt_cache_hits, 0);
        assert_eq!(stats.result_cache_hits, 0);
    }

    #[test]
    fn prompt_preparation_errors_and_oversize_entries_are_not_retained() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        struct TransientFormatter(Arc<AtomicUsize>);
        impl PromptFormatter for TransientFormatter {
            fn family(&self) -> Family {
                Family::F1
            }
            fn build(
                &self,
                state: &StateValue,
                question: &Question,
                tokenizer: &dyn Tokenizer,
            ) -> Result<BuiltPrompt> {
                if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err(Error::Request("transient preparation error".into()));
                }
                crate::prompt::default_formatter(Family::F1).build(state, question, tokenizer)
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let mut engine = engine().with_prompt_cache(1024 * 1024);
        engine.formatter = Arc::new(TransientFormatter(calls.clone()));
        let request = req();
        let mut stats = EvalStats::default();
        assert!(engine
            .eval_with_stats(&request, &Default::default(), &mut stats)
            .is_err());
        assert_eq!(stats.forward_calls, 0);
        engine
            .eval_with_stats(&request, &Default::default(), &mut stats)
            .unwrap();
        assert_eq!(stats.prompt_cache_hits, 0);
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        engine
            .eval_with_stats(&request, &Default::default(), &mut stats)
            .unwrap();
        assert_eq!(stats.prompt_cache_hits, 3);
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        let engine = engine.with_prompt_cache(1);
        for _ in 0..2 {
            engine
                .eval_with_stats(&request, &Default::default(), &mut stats)
                .unwrap();
            assert_eq!(stats.prompt_cache_hits, 0);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn result_reuse_keys_include_order_inputs_and_execution_options() {
        let engine = engine().with_result_cache(1024 * 1024);
        let request = req();
        let opts = EvalOptions::default();
        engine.eval(&request, &opts).unwrap();
        let mut alternatives = vec![request.clone(); 5];
        alternatives[0].state = "different state".into();
        alternatives[1].model = "alias".into();
        alternatives[2].questions.reverse();
        if let Question::Choice { criteria, .. } = &mut alternatives[3].questions["department"] {
            criteria.reverse();
        }
        if let Question::Noul { instructions, .. } = &mut alternatives[4].questions["is_refund"] {
            *instructions = serde_json::json!("different instructions").into();
        }
        let mut stats = EvalStats::default();
        for request in alternatives {
            engine.eval_with_stats(&request, &opts, &mut stats).unwrap();
            assert_eq!(stats.result_cache_hits, 0);
            assert_eq!(stats.forward_calls, 3);
        }
        for opts in [
            EvalOptions {
                extensions: true,
                ..Default::default()
            },
            EvalOptions {
                max_context: Some(1024),
                ..Default::default()
            },
            EvalOptions {
                reference_readout: true,
                ..Default::default()
            },
            EvalOptions {
                prefix_cache: true,
                ..Default::default()
            },
            EvalOptions {
                max_batch_tokens: Some(1024),
                ..Default::default()
            },
            EvalOptions {
                prepare_all: true,
                ..Default::default()
            },
        ] {
            engine.eval_with_stats(&request, &opts, &mut stats).unwrap();
            assert_eq!(stats.result_cache_hits, 0);
            assert_eq!(stats.forward_calls, 3);
        }
        // A separate loaded engine cannot hit the first one's entries.
        let other = self::engine().with_result_cache(1024 * 1024);
        other
            .eval_with_stats(&request, &EvalOptions::default(), &mut stats)
            .unwrap();
        assert_eq!(stats.result_cache_hits, 0);
        // Failed requests never produce entries or overwrite a valid result.
        let invalid = EvalOptions {
            max_context: Some(1),
            ..Default::default()
        };
        for _ in 0..2 {
            assert!(engine
                .eval_with_stats(&request, &invalid, &mut stats)
                .is_err());
            assert_eq!(stats.result_cache_hits, 0);
        }
        engine
            .eval_with_stats(&request, &EvalOptions::default(), &mut stats)
            .unwrap();
        assert_eq!(stats.result_cache_hits, 1);
    }

    #[test]
    fn result_reuse_distinguishes_absent_and_explicit_null_descriptions() {
        struct PositionBackend;
        impl Backend for PositionBackend {
            fn id(&self) -> BackendId {
                BackendId::Onnx
            }
            fn capabilities(&self) -> Capabilities {
                TestBackend.capabilities()
            }
            fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
                // An extra description token changes later candidate context,
                // hence its calibrated probability as well as logical usage.
                let data = input
                    .positions
                    .iter()
                    .map(|&pos| pos as f32 / 10.)
                    .collect();
                Ok(ForwardOutput::Features {
                    values: Tensor::new(vec![input.positions.len(), 1], data)?,
                    positions: input.positions,
                })
            }
            fn fork(&mut self, _: CacheHandle) -> Result<CacheHandle> {
                Err(Error::Unsupported("test backend has no cache".into()))
            }
        }
        let mut manifest = manifest();
        manifest.prompt_contract.template = "generic".into();
        let engine = Engine::new(
            manifest,
            Box::new(PositionBackend),
            Box::new(SimpleTokenizer::new(32768)),
            HeadParams::scalar_linear(1, vec![1.0], 0.0).unwrap(),
            BackendId::Onnx,
            "fp32",
        )
        .unwrap()
        .with_result_cache(1024 * 1024)
        .with_prompt_cache(1024 * 1024);
        let absent: SystemOneRequest = serde_json::from_value(serde_json::json!({
            "model": "test", "state": "input",
            "questions": {"q": {"type": "choice", "instructions": "choose",
                "criteria": {"first": null, "second": "description"}}}
        }))
        .unwrap();
        let mut explicit_null = absent.clone();
        let Question::Choice { criteria, .. } = &mut explicit_null.questions["q"] else {
            unreachable!()
        };
        criteria["first"] = Some(serde_json::Value::Null);
        // Wire serialization loses the Option discriminant for library callers.
        assert_eq!(
            serde_json::to_vec(&absent).unwrap(),
            serde_json::to_vec(&explicit_null).unwrap()
        );
        let opts = EvalOptions::default();
        let first = engine.eval(&absent, &opts).unwrap();
        let mut stats = EvalStats::default();
        let second = engine
            .eval_with_stats(&explicit_null, &opts, &mut stats)
            .unwrap();
        assert_eq!(stats.result_cache_hits, 0);
        assert_eq!(stats.forward_calls, 1);
        assert_eq!(stats.prompt_cache_hits, 0);
        assert_ne!(first.usage.input_tokens, second.usage.input_tokens);
        assert_ne!(
            serde_json::to_vec(&first.answers).unwrap(),
            serde_json::to_vec(&second.answers).unwrap()
        );
        let independent = engine
            .eval_uncached_with_stats(&explicit_null, &opts, &mut stats)
            .unwrap();
        assert_eq!(
            serde_json::to_vec(&second).unwrap(),
            serde_json::to_vec(&independent).unwrap()
        );
        for request in [&absent, &explicit_null] {
            engine.eval_with_stats(request, &opts, &mut stats).unwrap();
            assert_eq!(stats.result_cache_hits, 1);
            assert_eq!(stats.forward_calls, 0);
        }
    }

    #[test]
    fn conformance_bypasses_populated_result_cache() {
        use crate::conformance::{run_suite, GoldenCase, GoldenSuite};
        let engine = engine()
            .with_result_cache(1024 * 1024)
            .with_prompt_cache(1024 * 1024);
        let request = req();
        let response = engine.eval(&request, &EvalOptions::default()).unwrap();
        let case = GoldenCase {
            id: "same request".into(),
            request,
            expected: response
                .answers
                .iter()
                .map(|(id, answer)| {
                    let probabilities = match answer {
                        Answer::Choice { probabilities, .. }
                        | Answer::Score { probabilities, .. } => probabilities.clone(),
                        Answer::Noul { noul } => [("yes".into(), *noul), ("no".into(), 1. - noul)]
                            .into_iter()
                            .collect(),
                    };
                    (id.clone(), probabilities)
                })
                .collect(),
            targets: BTreeMap::new(),
        };
        let suite = GoldenSuite {
            schema_version: "1.0".into(),
            family: "F1".into(),
            hash: None,
            cases: vec![case.clone(), case],
        };
        let report = run_suite(&engine, &suite, &Default::default()).unwrap();
        assert!(report.passed);
        assert_eq!(report.work.forward_calls, 6);
        assert_eq!(report.work.result_cache_hits, 0);
        assert_eq!(report.work.prompt_cache_hits, 0);
    }

    fn req() -> SystemOneRequest {
        let json = serde_json::json!({
            "state": "The customer wants a refund because the shoes are too small.",
            "model": "test",
            "questions": {
                "department": { "type": "choice", "instructions": "Which team?", "criteria": { "returns": "money back", "billing": "charge issue" } },
                "is_refund": { "type": "noul", "instructions": "The customer is requesting a refund" },
                "severity": { "type": "score", "instructions": "severity?", "criteria": ["low","mid","high"] }
            }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn eval_returns_typed_answers() {
        let e = engine();
        let resp = e.eval(&req(), &EvalOptions::default()).unwrap();
        assert_eq!(resp.answers.len(), 3);
        assert!(matches!(resp.answers["department"], Answer::Choice { .. }));
        assert!(matches!(resp.answers["is_refund"], Answer::Noul { .. }));
        assert!(matches!(resp.answers["severity"], Answer::Score { .. }));
        assert!(resp.extensions.is_none());
    }

    #[test]
    fn extensions_off_by_default() {
        let e = engine();
        let resp = e.eval(&req(), &EvalOptions::default()).unwrap();
        assert!(resp.extensions.is_none());
        let resp2 = e
            .eval(
                &req(),
                &EvalOptions {
                    extensions: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(resp2.extensions.is_some());
        let ext = resp2.extensions.unwrap();
        assert_eq!(ext.backend.as_deref(), Some("onnx"));
        assert!(ext.raw_logits.is_some());
    }

    #[test]
    fn score_distribution_sums_to_one() {
        let e = engine();
        let resp = e.eval(&req(), &EvalOptions::default()).unwrap();
        if let Answer::Score { probabilities, .. } = &resp.answers["severity"] {
            let s: f32 = probabilities.values().sum();
            assert!((s - 1.0).abs() < 1e-4, "sum={s}");
        } else {
            panic!("expected score answer");
        }
    }

    #[test]
    fn rejects_oversized_input() {
        let e = engine();
        let r = req();
        // Force a context that is too small.
        let resp = e.eval(
            &r,
            &EvalOptions {
                max_context: Some(4),
                ..Default::default()
            },
        );
        assert!(resp.is_err());
    }

    #[test]
    fn batch_buckets_preserve_typed_answers_and_enforce_token_budgets() {
        use std::sync::Arc;
        struct BatchBackend {
            groups: Arc<Mutex<Vec<(usize, usize)>>>,
            omit: bool,
        }
        impl Backend for BatchBackend {
            fn id(&self) -> BackendId {
                BackendId::Onnx
            }
            fn capabilities(&self) -> Capabilities {
                TestBackend.capabilities()
            }
            fn supports_batch(&self) -> bool {
                true
            }
            fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
                self.groups.lock().unwrap().push((1, input.tokens.len()));
                TestBackend.forward(input)
            }
            fn forward_batch(&mut self, inputs: Vec<ForwardInput>) -> Result<Vec<ForwardOutput>> {
                assert!(inputs
                    .iter()
                    .all(|input| input.tokens.len() == inputs[0].tokens.len()));
                self.groups
                    .lock()
                    .unwrap()
                    .push((inputs.len(), inputs[0].tokens.len()));
                let mut outputs = inputs
                    .into_iter()
                    .map(|input| TestBackend.forward(input))
                    .collect::<Result<Vec<_>>>()?;
                if self.omit {
                    outputs.pop();
                }
                Ok(outputs)
            }
            fn fork(&mut self, _: CacheHandle) -> Result<CacheHandle> {
                unreachable!()
            }
        }
        let mut request = req();
        let original = request.questions.clone();
        request.questions.clear();
        for copy in 0..5 {
            for (id, question) in &original {
                request
                    .questions
                    .insert(format!("{copy}-{id}"), question.clone());
            }
        }
        let reference = engine()
            .eval(
                &request,
                &EvalOptions {
                    extensions: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let groups = Arc::new(Mutex::new(Vec::new()));
        let make_engine = |omit| {
            Engine::new(
                manifest(),
                Box::new(BatchBackend {
                    groups: groups.clone(),
                    omit,
                }),
                Box::new(SimpleTokenizer::new(32768)),
                HeadParams::default(),
                BackendId::Onnx,
                "fp32",
            )
            .unwrap()
        };
        let engine = make_engine(false).with_prompt_cache(1024 * 1024);
        let longest = original
            .values()
            .map(|question| {
                engine
                    .formatter
                    .build(&request.state, question, engine.tokenizer.as_ref())
                    .unwrap()
                    .tokens
                    .len()
            })
            .max()
            .unwrap();
        let budget = longest * 2;
        let opts = EvalOptions {
            extensions: true,
            max_batch_tokens: Some(budget),
            prepare_all: true,
            ..Default::default()
        };
        let mut stats = EvalStats::default();
        let batched = engine.eval_with_stats(&request, &opts, &mut stats).unwrap();
        assert_eq!(
            serde_json::to_value(&batched).unwrap(),
            serde_json::to_value(&reference).unwrap()
        );
        let observed = groups.lock().unwrap();
        assert!(observed
            .iter()
            .all(|(rows, length)| rows * length <= budget));
        assert_eq!(
            stats.batch_calls,
            observed.iter().filter(|(rows, _)| *rows > 1).count() as u64
        );
        assert_eq!(stats.forward_calls, observed.len() as u64);
        assert!(stats.batch_calls > 0);
        assert!(stats.forward_calls < request.questions.len() as u64);
        assert_eq!(stats.processed_tokens, reference.usage.input_tokens);
        assert_eq!(stats.prompt_cache_hits, 12);
        assert_eq!(stats.prepared_questions, 15);
        drop(observed);
        // Distinct states with colliding question IDs and mixed extensions
        // must scatter to exactly the independently calibrated response.
        let requests: Vec<_> = (0..6)
            .map(|index| {
                let mut request = req();
                request.state = StateValue::from(format!("distinct input {index}"));
                request.model = format!("alias-{index}");
                request
            })
            .collect();
        let options: Vec<_> = (0..6)
            .map(|index| EvalOptions {
                extensions: index % 2 == 0,
                ..opts.clone()
            })
            .collect();
        let expected: Vec<_> = requests
            .iter()
            .zip(&options)
            .map(|(request, options)| {
                engine
                    .eval_uncached_with_stats(
                        request,
                        &EvalOptions {
                            max_batch_tokens: None,
                            ..options.clone()
                        },
                        &mut Default::default(),
                    )
                    .unwrap()
            })
            .collect();
        let packets = requests
            .iter()
            .zip(&options)
            .map(|(request, options)| {
                engine
                    .prepare_eval_uncached_with_stats(
                        request.clone(),
                        options.clone(),
                        &mut Default::default(),
                    )
                    .unwrap()
            })
            .collect();
        groups.lock().unwrap().clear();
        let collated = engine
            .eval_prepared_batch_with_stats(packets, budget, &mut stats)
            .unwrap();
        assert_eq!(
            serde_json::to_vec(&collated).unwrap(),
            serde_json::to_vec(&expected).unwrap()
        );
        assert!(stats.cross_request_batches > 0);
        assert_eq!(
            stats.processed_tokens,
            expected.iter().map(|r| r.usage.input_tokens).sum::<u64>()
        );
        assert_eq!(stats.result_cache_hits, 0);
        assert_eq!(stats.prompt_cache_hits, 0);
        assert_eq!(stats.prepared_questions, 18);
        assert!(groups
            .lock()
            .unwrap()
            .iter()
            .all(|(rows, length)| rows * length <= budget || *rows == 1));
        let foreign = make_engine(false)
            .prepare_eval_with_stats(req(), opts.clone(), &mut Default::default())
            .unwrap();
        groups.lock().unwrap().clear();
        assert!(engine
            .eval_prepared_batch_with_stats(vec![foreign], budget, &mut stats)
            .is_err());
        assert!(groups.lock().unwrap().is_empty());

        let suite = crate::conformance::GoldenSuite {
            schema_version: "1.0".into(),
            family: "F1".into(),
            hash: None,
            cases: requests
                .iter()
                .zip(&expected)
                .enumerate()
                .map(
                    |(index, (request, response))| crate::conformance::GoldenCase {
                        id: index.to_string(),
                        request: request.clone(),
                        targets: Default::default(),
                        expected: response
                            .answers
                            .iter()
                            .map(|(id, answer)| {
                                (
                                    id.clone(),
                                    crate::conformance::answer_probabilities(answer).unwrap(),
                                )
                            })
                            .collect(),
                    },
                )
                .collect(),
        };
        let report = crate::conformance::run_suite_with_cross_request_batches(
            &engine,
            &suite,
            &Default::default(),
            &opts,
            6,
        )
        .unwrap();
        assert!(report.passed);
        assert!(report.work.cross_request_batches > 0);
        assert_eq!(report.cross_request_max_requests, Some(6));
        let tiny = EvalOptions {
            max_batch_tokens: Some(1),
            ..opts.clone()
        };
        assert!(crate::conformance::run_suite_with_cross_request_batches(
            &engine,
            &suite,
            &Default::default(),
            &tiny,
            6,
        )
        .is_err());
        let repeated = engine.eval_with_stats(&request, &opts, &mut stats).unwrap();
        assert_eq!(
            serde_json::to_vec(&repeated).unwrap(),
            serde_json::to_vec(&batched).unwrap()
        );
        assert_eq!(stats.prompt_cache_hits, 15);
        assert!(stats.batch_calls > 0);
        assert!(make_engine(true).eval(&request, &opts).is_err());
        let singleton = EvalOptions {
            max_batch_tokens: Some(1),
            ..opts.clone()
        };
        engine
            .eval_with_stats(&request, &singleton, &mut stats)
            .unwrap();
        assert_eq!(stats.batch_calls, 0);
        assert_eq!(stats.forward_calls, request.questions.len() as u64);
        assert!(engine
            .eval(
                &request,
                &EvalOptions {
                    max_batch_tokens: Some(0),
                    ..Default::default()
                }
            )
            .is_err());
        assert!(engine
            .eval(
                &request,
                &EvalOptions {
                    prefix_cache: true,
                    ..opts.clone()
                }
            )
            .is_err());
        groups.lock().unwrap().clear();
        assert!(engine
            .eval(
                &request,
                &EvalOptions {
                    max_context: Some(1),
                    ..opts
                }
            )
            .is_err());
        assert!(groups.lock().unwrap().is_empty());
    }

    #[test]
    fn f3_compact_and_legacy_readouts_have_identical_typed_probabilities() {
        use std::sync::Arc;
        struct ReadoutBackend {
            compact: bool,
            inputs: Arc<Mutex<Vec<ForwardInput>>>,
        }
        impl Backend for ReadoutBackend {
            fn id(&self) -> BackendId {
                BackendId::Candle
            }
            fn capabilities(&self) -> Capabilities {
                Capabilities::default()
            }
            fn fork(&mut self, _: CacheHandle) -> Result<CacheHandle> {
                Err(Error::Unsupported("test".into()))
            }
            fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
                if input.logit_codes.is_some() {
                    assert_eq!(
                        input.positions.len(),
                        1,
                        "shared prediction position must be projected once"
                    );
                }
                self.inputs.lock().unwrap().push(input.clone());
                let rows = input.positions.len();
                let codes = input.logit_codes.unwrap_or_default();
                if self.compact {
                    let data = codes.iter().map(|&code| code as f32 / 10.0).collect();
                    Ok(ForwardOutput::SelectedLogits {
                        positions: input.positions,
                        values: Tensor::new(vec![1, codes.len()], data)?,
                        codes,
                    })
                } else {
                    Ok(ForwardOutput::Logits {
                        positions: input.positions,
                        values: Tensor::new(
                            vec![rows, 32],
                            (0..rows * 32)
                                .map(|code| (code % 32) as f32 / 10.0)
                                .collect(),
                        )?,
                    })
                }
            }
        }
        let mut m = manifest();
        m.family = Family::F3;
        m.head.kind = HeadKind::CandidateLogit;
        m.prompt_contract.template = "nimble-v1".into();
        m.f3 = Some(crate::manifest::F3Config {
            candidate_codes: vec!["A".into(), "B".into(), "C".into()],
            candidate_token_ids: vec![29, 7, 18],
            system_prompt: "Classify".into(),
            prompt_code_sha256: "fixture".into(),
            max_input_tokens: 4096,
        });
        m.calibration.default.temperature = 2.40605;
        let inputs = Arc::new(Mutex::new(Vec::new()));
        let make = |compact| {
            Engine::new(
                m.clone(),
                Box::new(ReadoutBackend {
                    compact,
                    inputs: inputs.clone(),
                }),
                Box::new(SimpleTokenizer::new(32768)),
                HeadParams::default(),
                BackendId::Candle,
                "fp32",
            )
            .unwrap()
        };
        let optimized = make(true)
            .eval(
                &req(),
                &EvalOptions {
                    extensions: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let legacy = make(false)
            .eval(
                &req(),
                &EvalOptions {
                    extensions: true,
                    reference_readout: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            serde_json::to_value(optimized.answers).unwrap(),
            serde_json::to_value(legacy.answers).unwrap()
        );
        assert_eq!(optimized.usage.input_tokens, legacy.usage.input_tokens);
        assert_eq!(
            optimized.extensions.unwrap().raw_logits,
            legacy.extensions.unwrap().raw_logits
        );
        assert_eq!(inputs.lock().unwrap().len(), 6);
    }
}

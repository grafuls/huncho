//! Owned CPU Kev evaluations paused only at native-call boundaries.

use super::*;
use std::collections::VecDeque;

/// Single-use, context-bound request state. Dropping it releases a partial or
/// complete prefix. It never holds the backend mutex between steps.
pub struct ResumableEvaluation {
    backend: Arc<Mutex<Box<dyn Backend>>>,
    model: String,
    options: EvalOptions,
    questions: VecDeque<(String, Question, BuiltPrompt)>,
    answers: BTreeMap<String, Answer>,
    raw_logits: BTreeMap<String, Vec<f32>>,
    total_tokens: u64,
    prefix: Option<CacheHandle>,
    prefix_tokens: Vec<u32>,
    prefix_ready: bool,
    use_prefix: bool,
    result_key: Option<Vec<u8>>,
    cached: Option<SystemOneResponse>,
    finished: bool,
}

impl ResumableEvaluation {
    fn release(&mut self) {
        if let Some(handle) = self.prefix.take() {
            if let Ok(mut backend) = self.backend.lock() {
                let _ = backend.release_cache(handle);
            }
        }
    }
}

impl Drop for ResumableEvaluation {
    fn drop(&mut self) {
        self.release();
    }
}

impl Engine {
    /// Consume a validated preparation without submitting model work. Every
    /// step must run on this exact backend context, including after a pause.
    pub fn begin_resumable_evaluation(
        &self,
        prepared: PreparedEvaluation,
    ) -> Result<ResumableEvaluation> {
        if !Arc::ptr_eq(&self.preparation_identity, &prepared.owner) {
            return Err(Error::Request(
                "prepared evaluation belongs to a different engine".into(),
            ));
        }
        if !prepared.options.cooperative_prefill
            || !prepared.options.prefix_cache
            || prepared.options.max_batch_tokens.is_some()
            || !self.supports_resumable_prefill()
        {
            return Err(Error::Unsupported(
                "cooperative prefill requires CPU Kev, prefix reuse and configured chunks".into(),
            ));
        }
        let (prompts, cached) = match prepared.kind {
            PreparedKind::Prompts(prompts) => (prompts, None),
            PreparedKind::Cached(response) => (Vec::new(), Some(response)),
            PreparedKind::Joint => {
                return Err(Error::Unsupported(
                    "joint requests cannot use resumable Kev prefill".into(),
                ))
            }
        };
        let total_tokens = prompts.iter().map(|prompt| prompt.token_len() as u64).sum();
        let use_prefix = prepared.request.questions.len() > 1;
        let questions = prepared
            .request
            .questions
            .into_iter()
            .zip(prompts)
            .map(|((id, question), prompt)| (id, question, prompt))
            .collect();
        Ok(ResumableEvaluation {
            backend: self.backend.clone(),
            model: prepared.request.model,
            options: prepared.options,
            questions,
            answers: BTreeMap::new(),
            raw_logits: BTreeMap::new(),
            total_tokens,
            prefix: None,
            prefix_tokens: Vec::new(),
            prefix_ready: false,
            use_prefix,
            result_key: prepared.result_key,
            cached,
            finished: false,
        })
    }

    /// Execute one prefix chunk or one complete question, then release the
    /// backend lock. `None` is a scheduling boundary; `Some` finishes the job.
    /// Stats describe this step only, including failed native attempts.
    pub fn advance_resumable_evaluation(
        &self,
        cursor: &mut ResumableEvaluation,
        stats: &mut EvalStats,
    ) -> Result<Option<SystemOneResponse>> {
        *stats = EvalStats::default();
        if !Arc::ptr_eq(&self.backend, &cursor.backend) {
            return Err(Error::Request(
                "resumable evaluation belongs to a different backend context".into(),
            ));
        }
        if cursor.finished {
            return Err(Error::Request(
                "resumable evaluation is already finished".into(),
            ));
        }
        let result = self.resumable_step(cursor, stats);
        if result.is_err() || result.as_ref().is_ok_and(Option::is_some) {
            cursor.finished = true;
            cursor.release();
        }
        result
    }

    fn resumable_step(
        &self,
        cursor: &mut ResumableEvaluation,
        stats: &mut EvalStats,
    ) -> Result<Option<SystemOneResponse>> {
        if let Some(response) = cursor.cached.take() {
            return Ok(Some(response));
        }
        if let Some((_, _, prompt)) = cursor.questions.front() {
            let prefix_len = prompt.prefix_len;
            let eligible = cursor.use_prefix
                && prefix_len > 0
                && prefix_len < prompt.tokens.len()
                && prompt
                    .candidates
                    .iter()
                    .all(|candidate| candidate.position >= prefix_len);
            if eligible && cursor.prefix.is_none() {
                let cached = self
                    .backend
                    .lock()
                    .map_err(|_| Error::Backend("backend lock poisoned".into()))?
                    .begin_resumable_prefill(
                        &prompt.tokens[..prefix_len],
                        cursor.options.persistent_prefix_bytes,
                    )?;
                cursor.prefix = Some(cached.handle);
                cursor.prefix_tokens = prompt.tokens[..prefix_len].to_vec();
                cursor.prefix_ready = cached.hit;
                stats.persistent_prefix_hits += u64::from(cached.hit);
            }
            if let Some(handle) = cursor.prefix.filter(|_| !cursor.prefix_ready) {
                let mut work = crate::backend::PrefillWork::default();
                let result = self
                    .backend
                    .lock()
                    .map_err(|_| Error::Backend("backend lock poisoned".into()))?
                    .advance_resumable_prefill(handle, &mut work);
                stats.prefill_calls += work.forward_calls;
                stats.processed_tokens += work.processed_tokens;
                stats.chunked_prefills += work.chunked_prefills;
                cursor.prefix_ready = result?;
                stats.prefill_yields += u64::from(!cursor.prefix_ready);
                return Ok(None);
            }
        }
        if let Some((id, question, mut prompt)) = cursor.questions.pop_front() {
            let n_options = prompt.candidates.len();
            let mut positions: Vec<_> = prompt
                .candidates
                .iter()
                .map(|candidate| candidate.position)
                .collect();
            positions.sort_unstable();
            positions.dedup();
            let prefix_len = prompt.prefix_len;
            let eligible = cursor.prefix_ready
                && prefix_len == cursor.prefix_tokens.len()
                && prefix_len < prompt.tokens.len()
                && prompt.tokens[..prefix_len] == cursor.prefix_tokens
                && positions.iter().all(|&position| position >= prefix_len);
            let output = {
                let mut backend = self
                    .backend
                    .lock()
                    .map_err(|_| Error::Backend("backend lock poisoned".into()))?;
                let branch = if eligible {
                    let branch = backend.fork(cursor.prefix.unwrap())?;
                    stats.cache_forks += 1;
                    stats.reused_prefix_tokens += prefix_len as u64;
                    prompt.tokens = prompt.tokens.split_off(prefix_len);
                    for position in &mut positions {
                        *position -= prefix_len;
                    }
                    for candidate in &mut prompt.candidates {
                        candidate.position -= prefix_len;
                    }
                    Some(branch)
                } else {
                    None
                };
                let mut input =
                    ForwardInput::new(prompt.tokens, positions).with_qtype(prompt.qtype);
                input.fork_from = branch;
                stats.forward_calls += 1;
                stats.processed_tokens += input.tokens.len() as u64;
                let result = backend.forward(input);
                let released = branch
                    .map(|handle| backend.release_cache(handle))
                    .transpose();
                let output = result?;
                released?;
                output
            };
            let logits = head::candidate_logits(
                self.family(),
                self.manifest.head.kind,
                &output,
                &prompt.candidates,
                &self.head,
            )?;
            let probabilities =
                calibration::calibrate(&logits, self.temperature_for(&question, n_options))?;
            let answer = self.build_answer(&question, &prompt.candidates, &probabilities)?;
            cursor.answers.insert(id.clone(), answer);
            if cursor.options.extensions {
                cursor.raw_logits.insert(id, logits);
            }
        }
        if !cursor.questions.is_empty() {
            return Ok(None);
        }
        let mut response = SystemOneResponse::new(
            cursor.model.clone(),
            std::mem::take(&mut cursor.answers),
            Usage::new(cursor.total_tokens),
        );
        if cursor.options.extensions {
            response.extensions = Some(self.extensions(std::mem::take(&mut cursor.raw_logits)));
        }
        if let (Some(key), Some(cache)) = (cursor.result_key.take(), &self.response_cache) {
            if let Ok(mut cache) = cache.lock() {
                cache.insert(key, &response);
            }
        }
        Ok(Some(response))
    }
}

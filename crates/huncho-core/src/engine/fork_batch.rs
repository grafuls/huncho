//! Whole-question CPU Kev suffix batches from one immutable state prefix.
use super::batching::padded_groups_with_prefix;
use super::*;
use crate::backend::{BatchLimits, ForkBatchWork, PrefillWork};

impl Engine {
    pub(super) fn eval_fork_batched(
        &self,
        req: &SystemOneRequest,
        opts: &EvalOptions,
        stats: &mut EvalStats,
        budget: usize,
        reuse_prompts: bool,
        prepared: Option<Vec<BuiltPrompt>>,
    ) -> Result<SystemOneResponse> {
        // Validate every original context before allocating any native state.
        let prompts = match prepared {
            Some(prompts) => prompts,
            None => self.prepare_prompts(req, opts, stats, reuse_prompts)?,
        };
        let max_context = opts
            .max_context
            .unwrap_or(self.manifest.backbone.max_context);
        if prompts.len() != req.questions.len()
            || prompts.iter().any(|p| {
                p.token_len() > max_context
                    || p.candidates.len() > self.manifest.prompt_contract.max_options
            })
        {
            return Err(Error::Request(
                "cached-batch prompts exceed model input budgets".into(),
            ));
        }
        let eligible = |p: &BuiltPrompt| {
            req.questions.len() > 1
                && p.prefix_len > 0
                && p.prefix_len < p.tokens.len()
                && p.candidates.iter().all(|c| c.position >= p.prefix_len)
        };
        let prefix_tokens = prompts
            .iter()
            .find(|p| eligible(p))
            .map(|p| p.tokens[..p.prefix_len].to_vec());
        let prefix_len = prefix_tokens.as_ref().map_or(0, Vec::len);
        let total_tokens = prompts.iter().map(|p| p.token_len() as u64).sum();
        let mut jobs = Vec::with_capacity(prompts.len());
        let mut buckets: BTreeMap<(bool, usize), Vec<(usize, ForwardInput)>> = BTreeMap::new();
        for ((id, question), mut prompt) in req.questions.iter().zip(prompts) {
            let cached = eligible(&prompt)
                && prefix_tokens
                    .as_ref()
                    .is_some_and(|tokens| *tokens == prompt.tokens[..prompt.prefix_len]);
            let length = prompt.token_len();
            let mut positions: Vec<_> = prompt.candidates.iter().map(|c| c.position).collect();
            positions.sort_unstable();
            positions.dedup();
            let mut input = ForwardInput::new(std::mem::take(&mut prompt.tokens), positions)
                .with_qtype(prompt.qtype);
            if cached {
                input.tokens = input.tokens.split_off(prefix_len);
                for position in &mut input.positions {
                    *position -= prefix_len;
                }
                for candidate in &mut prompt.candidates {
                    candidate.position -= prefix_len;
                }
            }
            buckets
                .entry((cached, length))
                .or_default()
                .push((jobs.len(), input));
            jobs.push((id, question, prompt));
        }
        // Create the RAII owner outside the backend guard's scope so errors
        // never release a parent while its mutex is still held.
        let prefix = match prefix_tokens {
            Some(tokens) => {
                let cached = {
                    let mut backend = self
                        .backend
                        .lock()
                        .map_err(|_| Error::Backend("backend lock poisoned".into()))?;
                    let mut work = PrefillWork::default();
                    let result = backend.prefill_cached_with_work(
                        &tokens,
                        opts.persistent_prefix_bytes,
                        &mut work,
                    );
                    stats.prefill_calls += work.forward_calls;
                    stats.processed_tokens += work.processed_tokens;
                    stats.chunked_prefills += work.chunked_prefills;
                    result
                }?;
                stats.persistent_prefix_hits += u64::from(cached.hit);
                Some(RequestPrefix {
                    backend: &self.backend,
                    handle: cached.handle,
                    tokens,
                })
            }
            None => None,
        };
        let mut outputs = vec![None; jobs.len()];
        let mut separated: BTreeMap<bool, Vec<(usize, ForwardInput)>> = BTreeMap::new();
        for ((cached, _), bucket) in buckets {
            separated.entry(cached).or_default().extend(bucket);
        }
        for (cached, bucket) in separated {
            // Reserve one handle for the parent. Charge full B*(prefix+suffix)
            // including retained prefix KV: materialization is not zero-cost.
            let limits = BatchLimits {
                max_rows: self.batch_limits.max_rows.min(if cached { 63 } else { 64 }),
                ..self.batch_limits
            };
            for group in padded_groups_with_prefix(
                bucket,
                budget,
                opts.max_batch_padding_percent,
                limits,
                if cached { prefix_len } else { 0 },
            ) {
                let (indices, mut inputs): (Vec<_>, Vec<_>) = group.into_iter().unzip();
                let count = inputs.len();
                let physical = count * inputs.iter().map(|i| i.tokens.len()).max().unwrap();
                let padding = physical - inputs.iter().map(|i| i.tokens.len()).sum::<usize>();
                let result = {
                    let mut backend = self
                        .backend
                        .lock()
                        .map_err(|_| Error::Backend("backend lock poisoned".into()))?;
                    if cached {
                        let mut work = ForkBatchWork::default();
                        let parent = prefix.as_ref().unwrap().handle;
                        let result = if padding > 0 {
                            backend.forward_padded_fork_batch(parent, inputs, &mut work)
                        } else {
                            backend.forward_fork_batch(parent, inputs, &mut work)
                        };
                        stats.forward_calls += work.forward_calls;
                        stats.processed_tokens += work.processed_tokens;
                        stats.batch_calls += work.batch_calls;
                        stats.fork_batch_calls += work.batch_calls;
                        stats.padded_batch_calls += work.padded_batch_calls;
                        stats.fork_padded_batch_calls += work.padded_batch_calls;
                        stats.padded_tokens += work.padded_tokens;
                        stats.cache_forks += work.cache_forks;
                        stats.reused_prefix_tokens += work.cache_forks * prefix_len as u64;
                        result
                    } else {
                        stats.forward_calls += 1;
                        stats.processed_tokens += physical as u64;
                        stats.padded_tokens += padding as u64;
                        stats.padded_batch_calls += u64::from(padding > 0);
                        if count == 1 {
                            backend.forward(inputs.pop().unwrap()).map(|out| vec![out])
                        } else {
                            stats.batch_calls += 1;
                            if padding > 0 {
                                backend.forward_padded_batch(inputs)
                            } else {
                                backend.forward_batch(inputs)
                            }
                        }
                    }
                }?;
                if result.len() != count {
                    return Err(Error::Backend(
                        "branch batch returned the wrong number of rows".into(),
                    ));
                }
                for (index, output) in indices.into_iter().zip(result) {
                    outputs[index] = Some(output);
                }
            }
        }
        let mut answers = BTreeMap::new();
        let mut raw_logits = BTreeMap::new();
        for ((id, question, prompt), output) in jobs.into_iter().zip(outputs) {
            let output =
                output.ok_or_else(|| Error::Backend("branch batch omitted a row".into()))?;
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
}

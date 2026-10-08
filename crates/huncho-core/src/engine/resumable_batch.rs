//! Cooperative CPU Kev question groups. Only a bounded native call runs per step.
use super::*;
use crate::backend::{BatchLimits, ForkBatchWork, PrefillWork};
use crate::engine::batching::padded_groups_with_prefix;

pub(super) struct ResumableBatch {
    cached: bool,
    jobs: Vec<(String, Question, BuiltPrompt)>,
    inputs: Vec<ForwardInput>,
}

fn split_batch(
    batch: ResumableBatch,
    budget: usize,
    padding: usize,
    limits: BatchLimits,
    prefix: usize,
) -> VecDeque<ResumableBatch> {
    let mut jobs: Vec<_> = batch.jobs.into_iter().map(Some).collect();
    padded_groups_with_prefix(
        batch.inputs.into_iter().enumerate().collect(),
        budget,
        padding,
        limits,
        prefix,
    )
    .into_iter()
    .map(|group| {
        let (indices, inputs): (Vec<_>, Vec<_>) = group.into_iter().unzip();
        ResumableBatch {
            cached: batch.cached,
            jobs: indices
                .into_iter()
                .map(|i| jobs[i].take().unwrap())
                .collect(),
            inputs,
        }
    })
    .collect()
}

impl Engine {
    pub(super) fn prepare_resumable_batches(
        &self,
        questions: VecDeque<(String, Question, BuiltPrompt)>,
        options: &EvalOptions,
    ) -> Result<(VecDeque<ResumableBatch>, Vec<u32>)> {
        let eligible = |p: &BuiltPrompt| {
            questions.len() > 1
                && p.prefix_len > 0
                && p.prefix_len < p.tokens.len()
                && p.candidates.iter().all(|c| c.position >= p.prefix_len)
        };
        let prefix = questions
            .iter()
            .find(|(_, _, p)| eligible(p))
            .map(|(_, _, p)| p.tokens[..p.prefix_len].to_vec())
            .unwrap_or_default();
        let count = questions.len();
        let mut cached = ResumableBatch {
            cached: true,
            jobs: Vec::new(),
            inputs: Vec::new(),
        };
        let mut independent = ResumableBatch {
            cached: false,
            jobs: Vec::new(),
            inputs: Vec::new(),
        };
        for (id, question, mut prompt) in questions {
            let use_prefix = count > 1
                && !prefix.is_empty()
                && prompt.prefix_len == prefix.len()
                && prefix.len() < prompt.tokens.len()
                && prompt.tokens[..prefix.len()] == prefix
                && prompt.candidates.iter().all(|c| c.position >= prefix.len());
            let mut positions: Vec<_> = prompt.candidates.iter().map(|c| c.position).collect();
            positions.sort_unstable();
            positions.dedup();
            let mut tokens = std::mem::take(&mut prompt.tokens);
            if use_prefix {
                tokens = tokens.split_off(prefix.len());
                for p in &mut positions {
                    *p -= prefix.len();
                }
                for c in &mut prompt.candidates {
                    c.position -= prefix.len();
                }
            }
            let input = ForwardInput::new(tokens, positions).with_qtype(prompt.qtype);
            let group = if use_prefix {
                &mut cached
            } else {
                &mut independent
            };
            group.jobs.push((id, question, prompt));
            group.inputs.push(input);
        }
        let mut groups = VecDeque::new();
        for group in [cached, independent] {
            // Keep immutable job/input pairs together while sorting by length.
            let mut pairs: Vec<_> = group.jobs.into_iter().zip(group.inputs).collect();
            pairs.sort_by_key(|(_, input)| input.tokens.len());
            let (jobs, inputs) = pairs.into_iter().unzip();
            let limits = BatchLimits {
                max_rows: self
                    .batch_limits
                    .max_rows
                    .min(if group.cached { 63 } else { 64 }),
                ..self.batch_limits
            };
            groups.extend(split_batch(
                ResumableBatch {
                    cached: group.cached,
                    jobs,
                    inputs,
                },
                options.max_batch_tokens.unwrap(),
                options.max_batch_padding_percent,
                limits,
                if group.cached { prefix.len() } else { 0 },
            ));
        }
        Ok((groups, prefix))
    }

    pub(super) fn resumable_batch_step(
        &self,
        cursor: &mut ResumableEvaluation,
        stats: &mut EvalStats,
    ) -> Result<Option<SystemOneResponse>> {
        if !cursor.prefix_tokens.is_empty() && !cursor.prefix_ready {
            let mut backend = self
                .backend
                .lock()
                .map_err(|_| Error::Backend("backend lock poisoned".into()))?;
            if cursor.prefix.is_none() {
                let cached = backend.begin_resumable_prefill(
                    &cursor.prefix_tokens,
                    cursor.options.persistent_prefix_bytes,
                )?;
                cursor.prefix = Some(cached.handle);
                cursor.prefix_ready = cached.hit;
                stats.persistent_prefix_hits += u64::from(cached.hit);
            }
            if !cursor.prefix_ready {
                let mut work = PrefillWork::default();
                let result = backend.advance_resumable_prefill(cursor.prefix.unwrap(), &mut work);
                stats.prefill_calls += work.forward_calls;
                stats.processed_tokens += work.processed_tokens;
                stats.chunked_prefills += work.chunked_prefills;
                cursor.prefix_ready = result?;
                stats.prefill_yields += u64::from(!cursor.prefix_ready);
                return Ok(None);
            }
        }
        if let Some(batch) = cursor.batches.as_mut().unwrap().pop_front() {
            let (jobs, outputs) = {
                let mut backend = self
                    .backend
                    .lock()
                    .map_err(|_| Error::Backend("backend lock poisoned".into()))?;
                let limits = if batch.cached {
                    backend.fork_batch_limits()
                } else {
                    backend.batch_limits()
                };
                if limits.max_rows == 0 {
                    return Err(Error::Backend(
                        "no cache child rows available for cooperative group".into(),
                    ));
                }
                // Other jobs can own partial/complete parents. Repartition under
                // the same backend lock that submits work; never overcommit them.
                let prefix = if batch.cached {
                    cursor.prefix_tokens.len()
                } else {
                    0
                };
                let mut groups = split_batch(
                    batch,
                    cursor.options.max_batch_tokens.unwrap(),
                    cursor.options.max_batch_padding_percent,
                    limits,
                    prefix,
                );
                let mut batch = groups.pop_front().unwrap();
                for group in groups.into_iter().rev() {
                    cursor.batches.as_mut().unwrap().push_front(group);
                }
                let count = batch.inputs.len();
                let physical = count * batch.inputs.iter().map(|i| i.tokens.len()).max().unwrap();
                let padding = physical - batch.inputs.iter().map(|i| i.tokens.len()).sum::<usize>();
                let outputs = if batch.cached {
                    let mut work = ForkBatchWork::default();
                    let result = if padding > 0 {
                        backend.forward_padded_fork_batch(
                            cursor.prefix.unwrap(),
                            batch.inputs,
                            &mut work,
                        )
                    } else {
                        backend.forward_fork_batch(cursor.prefix.unwrap(), batch.inputs, &mut work)
                    };
                    stats.forward_calls += work.forward_calls;
                    stats.processed_tokens += work.processed_tokens;
                    stats.batch_calls += work.batch_calls;
                    stats.fork_batch_calls += work.batch_calls;
                    stats.padded_batch_calls += work.padded_batch_calls;
                    stats.fork_padded_batch_calls += work.padded_batch_calls;
                    stats.padded_tokens += work.padded_tokens;
                    stats.cache_forks += work.cache_forks;
                    stats.reused_prefix_tokens +=
                        work.cache_forks * cursor.prefix_tokens.len() as u64;
                    result?
                } else {
                    stats.forward_calls += 1;
                    stats.processed_tokens += physical as u64;
                    stats.padded_tokens += padding as u64;
                    stats.padded_batch_calls += u64::from(padding > 0);
                    stats.batch_calls += u64::from(count > 1);
                    if count == 1 {
                        vec![backend.forward(batch.inputs.pop().unwrap())?]
                    } else if padding > 0 {
                        backend.forward_padded_batch(batch.inputs)?
                    } else {
                        backend.forward_batch(batch.inputs)?
                    }
                };
                if outputs.len() != count {
                    return Err(Error::Backend("cooperative batch omitted a row".into()));
                }
                (batch.jobs, outputs)
            };
            for ((id, question, prompt), output) in jobs.into_iter().zip(outputs) {
                let mut logits = head::candidate_logits(
                    self.family(),
                    self.manifest.head.kind,
                    &output,
                    &prompt.candidates,
                    &self.head,
                )?;
                let probabilities = calibration::calibrate_readout(
                    &mut logits,
                    self.temperature_for(&question, prompt.candidates.len()),
                    cursor.options.extensions,
                )?;
                cursor.answers.insert(
                    id.clone(),
                    self.build_answer(&question, &prompt.candidates, &probabilities)?,
                );
                if cursor.options.extensions {
                    cursor.raw_logits.insert(id, logits);
                }
            }
        }
        if cursor.batches.as_ref().unwrap().is_empty() {
            self.finish_resumable_response(cursor)
        } else {
            Ok(None)
        }
    }
}

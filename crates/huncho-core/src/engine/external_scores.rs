//! Async runtimes own the forward; the core owns prompts and calibrated wire
//! answers. This ABI requires an integrated scalar option head, never features.
use super::*;
use crate::manifest::HeadKind;

#[derive(Debug, serde::Serialize)]
pub struct MarkerReadout {
    pub tokens: Vec<u32>,
    pub positions: Vec<usize>,
    pub qtype: u32,
}

impl batching::BatchShape for &MarkerReadout {
    fn token_len(&self) -> usize {
        self.tokens.len()
    }
    fn readout_rows(&self) -> usize {
        // Keep the native planner's conservative final-decision row charge.
        self.positions.len().saturating_add(1)
    }
}

/// Immutable CPU external graph collation profile; no prefix/retention policy.
#[derive(Debug, Clone, Copy)]
pub struct MarkerBatchProfile {
    pub max_batch_tokens: usize,
    pub max_batch_padding_percent: usize,
}
impl MarkerBatchProfile {
    pub fn validate(self) -> Result<()> {
        if !(1..=65536).contains(&self.max_batch_tokens) || self.max_batch_padding_percent > 100 {
            return Err(Error::Request(
                "external marker batches require 1..65536 tokens and padding percent 0..100".into(),
            ));
        }
        Ok(())
    }
}

/// Original readout indices in one tensor call, with its physical row length.
#[derive(Debug, serde::Serialize)]
pub struct MarkerBatch {
    pub readouts: Vec<usize>,
    pub sequence: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput};
    use crate::head::HeadParams;
    use crate::tensor::Tensor;
    use crate::tokenizer::SimpleTokenizer;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn values(tokens: &[u32], positions: &[usize], qtype: u32) -> Vec<f32> {
        positions
            .iter()
            .map(|&p| (tokens[p] as f32 * 0.01 + p as f32 * 0.17) * (qtype + 1) as f32 - 0.3)
            .collect()
    }
    struct Scalar(Arc<AtomicUsize>);
    impl Backend for Scalar {
        fn id(&self) -> BackendId {
            BackendId::Onnx
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                id: BackendId::Onnx,
                dtype: "fp32".into(),
                families: vec![Family::F1, Family::F2],
                max_context: 512,
                ..Default::default()
            }
        }
        fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(ForwardOutput::Features {
                values: Tensor::new(
                    vec![input.positions.len(), 1],
                    values(&input.tokens, &input.positions, input.qtype),
                )?,
                positions: input.positions,
            })
        }
        fn fork(&mut self, _: CacheHandle) -> Result<CacheHandle> {
            unreachable!()
        }
    }
    fn engine(family: Family) -> (Engine, Arc<AtomicUsize>) {
        let mut manifest: ModelManifest = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/mock-model/huncho-model.json"
        )))
        .unwrap();
        manifest.name = "external-fixture".into();
        manifest.family = family;
        manifest.head.kind = if family == Family::F1 {
            HeadKind::OptionMarker
        } else {
            HeadKind::Pointer
        };
        manifest.calibration.entries.clear();
        manifest.calibration.default.per_type_temperatures = Some(BTreeMap::from([
            ("choice".into(), 1.1),
            ("score".into(), 0.8),
            ("noul".into(), 2.4),
        ]));
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Engine::new(
                manifest,
                Box::new(Scalar(calls.clone())),
                Box::new(SimpleTokenizer::new(4096)),
                HeadParams::scalar_linear(1, vec![1.0], 0.0).unwrap(),
                BackendId::Onnx,
                "fp32",
            )
            .unwrap(),
            calls,
        )
    }
    fn request() -> SystemOneRequest {
        serde_json::from_value(serde_json::json!({"model":"external-fixture", "state":{"text":"a refund"}, "questions":{
            "z_choice":{"type":"choice", "instructions":"Team?", "criteria":{"shipping":null,"billing":"Charges","returns":"Refunds"}},
            "a_noul":{"type":"noul", "instructions":"Urgent?"},
            "m_score":{"type":"score", "instructions":"Priority?", "criteria":["low","medium","high"]}
        }})).unwrap()
    }
    fn scores(plan: &ExternalEvaluation) -> Vec<Vec<f32>> {
        plan.readouts()
            .iter()
            .map(|input| values(&input.tokens, &input.positions, input.qtype))
            .collect()
    }
    #[test]
    fn external_conformance_requires_complete_observed_vectors_and_fresh_work() {
        use crate::conformance::{
            answer_probabilities, run_external_marker_suite, GoldenCase, GoldenSuite,
        };
        let (engine, calls) = engine(Family::F1);
        let request = request();
        let expected = engine.eval(&request, &EvalOptions::default()).unwrap();
        let plan = engine
            .prepare_external_markers(request.clone(), EvalOptions::default())
            .unwrap();
        let raw = scores(&plan);
        let (response, work) = engine.finish_external_markers(plan, raw).unwrap();
        let mut suite = GoldenSuite {
            schema_version: "1.0".into(),
            family: "F1".into(),
            hash: None,
            cases: vec![GoldenCase {
                id: "synthetic-gate-fixture".into(),
                request,
                expected: expected
                    .answers
                    .iter()
                    .map(|(id, answer)| (id.clone(), answer_probabilities(answer).unwrap()))
                    .collect(),
                targets: BTreeMap::from([
                    ("a_noul".into(), "yes".into()),
                    ("m_score".into(), "1".into()),
                    ("z_choice".into(), "billing".into()),
                ]),
            }],
        };
        let report =
            run_external_marker_suite(&engine, &suite, &[response.clone()], work.clone()).unwrap();
        assert!(report.passed);
        assert_eq!(report.outcome_calibration.unwrap().questions, 3);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "qualification must never replay a native forward"
        );
        assert!(run_external_marker_suite(&engine, &suite, &[], work.clone()).is_err());
        let mut fake = work.clone();
        fake.result_cache_hits = 1;
        assert!(run_external_marker_suite(&engine, &suite, &[response.clone()], fake).is_err());
        let mut fake = work.clone();
        fake.forward_calls -= 1;
        assert!(run_external_marker_suite(&engine, &suite, &[response.clone()], fake).is_err());
        let mut wrong = response.clone();
        wrong.model = "alias".into();
        assert!(run_external_marker_suite(&engine, &suite, &[wrong], work.clone()).is_err());
        let mut wrong = response.clone();
        wrong.usage.input_tokens += 1;
        assert!(run_external_marker_suite(&engine, &suite, &[wrong], work.clone()).is_err());
        suite.cases[0].targets.remove("a_noul");
        assert!(
            run_external_marker_suite(&engine, &suite, &[response.clone()], work.clone()).is_err()
        );
        suite.cases[0].targets.clear();
        assert!(run_external_marker_suite(&engine, &suite, &[response], work).is_err());
    }
    #[test]
    fn integrated_marker_scores_preserve_native_typed_calibration_answers_and_usage() {
        let (engine, calls) = engine(Family::F1);
        let options = EvalOptions {
            extensions: true,
            ..Default::default()
        };
        let expected = engine.eval(&request(), &options).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        let plan = engine.prepare_external_markers(request(), options).unwrap();
        let raw = scores(&plan);
        let (actual, work) = engine.finish_external_markers(plan, raw).unwrap();
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
        assert_eq!(work.forward_calls, 3);
        assert_eq!(work.prepared_questions, 3);
        assert!(work.processed_tokens > 0);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "external preparation/finishing must never execute the backend"
        );
    }
    #[test]
    fn external_batches_keep_logical_usage_and_share_fixed_labeled_and_paired_gates() {
        use crate::conformance::{
            answer_probabilities, run_external_marker_batched_suite, GoldenCase, GoldenSuite,
        };
        let (engine, calls) = engine(Family::F1);
        let request = request();
        let expected = engine.eval(&request, &EvalOptions::default()).unwrap();
        let suite = GoldenSuite {
            schema_version: "1.0".into(),
            family: "F1".into(),
            hash: None,
            cases: vec![GoldenCase {
                id: "synthetic-batch-gate-fixture".into(),
                request: request.clone(),
                expected: expected
                    .answers
                    .iter()
                    .map(|(id, answer)| (id.clone(), answer_probabilities(answer).unwrap()))
                    .collect(),
                targets: BTreeMap::from([
                    ("a_noul".into(), "yes".into()),
                    ("m_score".into(), "1".into()),
                    ("z_choice".into(), "billing".into()),
                ]),
            }],
        };
        let profile = MarkerBatchProfile {
            max_batch_tokens: 4096,
            max_batch_padding_percent: 100,
        };
        let plan = engine
            .prepare_external_markers(request.clone(), EvalOptions::default())
            .unwrap();
        let groups = plan.marker_batches(profile).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].readouts.len(), 3);
        let raw = scores(&plan);
        let (response, work) = engine
            .finish_external_marker_batches(plan, profile, raw)
            .unwrap();
        assert_eq!(
            serde_json::to_value(&response).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        assert_eq!(work.forward_calls, 1);
        assert_eq!(work.batch_calls, 1);
        assert_eq!(work.padded_batch_calls, 1);
        assert!(work.padded_tokens > 0);
        assert_eq!(
            work.processed_tokens - work.padded_tokens,
            response.usage.input_tokens
        );
        let plan = engine
            .prepare_external_markers(request.clone(), EvalOptions::default())
            .unwrap();
        let raw = scores(&plan);
        let (independent, independent_work) = engine.finish_external_markers(plan, raw).unwrap();
        let qualify = |responses: &[SystemOneResponse],
                       work: EvalStats,
                       independent: &[SystemOneResponse],
                       scalar: EvalStats| {
            run_external_marker_batched_suite(
                &engine,
                &suite,
                responses,
                work,
                independent,
                scalar,
                profile,
            )
        };
        let report = qualify(
            &[response.clone()],
            work.clone(),
            &[independent.clone()],
            independent_work.clone(),
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.optimization_parity.unwrap().max_prob_delta, 0.);
        assert_eq!(report.max_batch_padding_percent, 100);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "external qualification must never replay a native backend"
        );
        let mut fake = work.clone();
        fake.processed_tokens -= 1;
        assert!(qualify(
            &[response.clone()],
            fake,
            &[independent.clone()],
            independent_work.clone()
        )
        .is_err());
        assert!(qualify(
            &[response.clone()],
            work.clone(),
            &[],
            independent_work.clone()
        )
        .is_err());
        let mut wrong = response.clone();
        wrong.usage.input_tokens += 1;
        assert!(qualify(
            &[wrong],
            work.clone(),
            &[independent.clone()],
            independent_work.clone()
        )
        .is_err());
        // A change below the external 1e-3 limit must still fail paired 1e-4.
        let mut changed = response;
        if let Answer::Choice { probabilities, .. } = changed.answers.get_mut("z_choice").unwrap() {
            *probabilities.get_mut("shipping").unwrap() += 0.0002;
            *probabilities.get_mut("billing").unwrap() -= 0.0002;
        }
        assert!(
            !qualify(&[changed], work, &[independent], independent_work)
                .unwrap()
                .passed
        );
    }

    #[test]
    fn external_marker_groups_bound_padding_rows_markers_and_vacuous_profiles() {
        let (engine, _) = engine(Family::F1);
        let mut request = request();
        let question = request.questions["z_choice"].clone();
        request.questions.clear();
        for i in 0..129 {
            request.questions.insert(format!("q{i}"), question.clone());
        }
        let plan = engine
            .prepare_external_markers(request, EvalOptions::default())
            .unwrap();
        let profile = MarkerBatchProfile {
            max_batch_tokens: 65536,
            max_batch_padding_percent: 0,
        };
        let groups = plan.marker_batches(profile).unwrap();
        assert_eq!(
            groups
                .iter()
                .map(|group| group.readouts.len())
                .collect::<Vec<_>>(),
            [64, 64, 1]
        );
        let mut indices: Vec<_> = groups
            .into_iter()
            .flat_map(|group| group.readouts)
            .collect();
        indices.sort_unstable();
        assert_eq!(indices, (0..129).collect::<Vec<_>>());
        for (tokens, padding) in [(0, 0), (65537, 0), (1, 101)] {
            assert!(plan
                .marker_batches(MarkerBatchProfile {
                    max_batch_tokens: tokens,
                    max_batch_padding_percent: padding,
                })
                .is_err());
        }
        let work = plan
            .marker_work(Some(MarkerBatchProfile {
                max_batch_tokens: 1,
                max_batch_padding_percent: 0,
            }))
            .unwrap();
        assert_eq!(work.forward_calls, 129);
        assert_eq!(work.batch_calls, 0);
        assert_eq!(work.padded_tokens, 0);
    }
    #[test]
    fn malformed_foreign_nonfinite_and_unsupported_external_inputs_fail_closed() {
        let (one, calls) = engine(Family::F1);
        let (two, _) = engine(Family::F1);
        let plan = one
            .prepare_external_markers(request(), EvalOptions::default())
            .unwrap();
        let raw = scores(&plan);
        assert!(two.finish_external_markers(plan, raw).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let plan = one
                .prepare_external_markers(request(), EvalOptions::default())
                .unwrap();
            let mut raw = scores(&plan);
            raw[0][0] = bad;
            assert!(one.finish_external_markers(plan, raw).is_err());
        }
        let plan = one
            .prepare_external_markers(request(), EvalOptions::default())
            .unwrap();
        assert!(one.finish_external_markers(plan, vec![]).is_err());
        let plan = one
            .prepare_external_markers(request(), EvalOptions::default())
            .unwrap();
        let mut raw = scores(&plan);
        raw[0].pop();
        assert!(one.finish_external_markers(plan, raw).is_err());
        let mut alias = request();
        alias.model = "alias".into();
        assert!(one
            .prepare_external_markers(alias, EvalOptions::default())
            .is_err());
        for options in [
            EvalOptions {
                prefix_cache: true,
                ..Default::default()
            },
            EvalOptions {
                max_context: Some(usize::MAX),
                ..Default::default()
            },
            EvalOptions {
                max_context: Some(1),
                ..Default::default()
            },
        ] {
            assert!(one.prepare_external_markers(request(), options).is_err());
        }
        let (pointer, _) = engine(Family::F2);
        assert!(pointer
            .prepare_external_markers(request(), EvalOptions::default())
            .is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}

/// Immutable, single-use inputs bound to one immutable engine group and its original request.
/// Returned scalar rows must correspond exactly to `readouts()` in order.
pub struct ExternalEvaluation {
    owner: Arc<()>,
    request: SystemOneRequest,
    options: EvalOptions,
    prompts: Vec<BuiltPrompt>,
    readouts: Vec<MarkerReadout>,
}

impl ExternalEvaluation {
    pub fn readouts(&self) -> &[MarkerReadout] {
        &self.readouts
    }

    pub fn marker_batches(&self, profile: MarkerBatchProfile) -> Result<Vec<MarkerBatch>> {
        profile.validate()?;
        let mut inputs: Vec<_> = self.readouts.iter().enumerate().collect();
        inputs.sort_by_key(|(_, input)| input.tokens.len());
        Ok(batching::padded_groups(
            inputs,
            profile.max_batch_tokens,
            profile.max_batch_padding_percent,
            crate::backend::BatchLimits {
                max_rows: 64,
                max_readouts: Some(8192),
            },
        )
        .into_iter()
        .map(|group| MarkerBatch {
            sequence: group
                .iter()
                .map(|(_, input)| input.tokens.len())
                .max()
                .unwrap(),
            readouts: group.into_iter().map(|(i, _)| i).collect(),
        })
        .collect())
    }

    /// Expected physical calls/rectangles, derived from the immutable plan.
    /// The async caller still owns actual graph execution/provenance.
    pub fn marker_work(&self, profile: Option<MarkerBatchProfile>) -> Result<EvalStats> {
        let groups = match profile {
            Some(profile) => self.marker_batches(profile)?,
            None => self
                .readouts
                .iter()
                .enumerate()
                .map(|(i, input)| MarkerBatch {
                    readouts: vec![i],
                    sequence: input.tokens.len(),
                })
                .collect(),
        };
        let mut work = EvalStats {
            prepared_questions: self.readouts.len() as u64,
            ..Default::default()
        };
        for group in groups {
            let physical = group
                .sequence
                .checked_mul(group.readouts.len())
                .ok_or_else(|| {
                    Error::Request("external marker rectangle overflows token counters".into())
                })?;
            let logical: usize = group
                .readouts
                .iter()
                .map(|&i| self.readouts[i].tokens.len())
                .sum();
            let padding = physical - logical;
            work.forward_calls += 1;
            work.batch_calls += u64::from(group.readouts.len() > 1);
            work.padded_batch_calls += u64::from(padding > 0);
            work.processed_tokens += physical as u64;
            work.padded_tokens += padding as u64;
        }
        Ok(work)
    }
}

impl Engine {
    /// Consume raw per-question scores from the declared actual tensor groups.
    /// Wire usage stays logical; physical counters include every padded slot.
    pub fn finish_external_marker_batches(
        &self,
        plan: ExternalEvaluation,
        profile: MarkerBatchProfile,
        scores: Vec<Vec<f32>>,
    ) -> Result<(SystemOneResponse, EvalStats)> {
        let work = plan.marker_work(Some(profile))?;
        let (response, _) = self.finish_external_markers(plan, scores)?;
        Ok((response, work))
    }

    /// Prepare F1 integrated-head inputs without any backend call or lock.
    /// The external runtime must supply raw scalar head logits at each marker.
    /// Scheduling/retention options are deliberately outside this first ABI.
    pub fn prepare_external_markers(
        &self,
        request: SystemOneRequest,
        options: EvalOptions,
    ) -> Result<ExternalEvaluation> {
        request.validate()?;
        if self.family() != Family::F1
            || self.manifest.head.kind != HeadKind::OptionMarker
            || self.manifest.head.width != 1
        {
            return Err(Error::Unsupported(
                "external marker scores support F1 option heads only".into(),
            ));
        }
        if request.model != self.manifest.name {
            return Err(Error::Request(
                "external request model must match the immutable package name".into(),
            ));
        }
        if options.reference_readout
            || options.prefix_cache
            || options.persistent_prefix_bytes > 0
            || options.max_batch_tokens.is_some()
            || options.max_batch_padding_percent > 0
            || options.prepare_all
            || options.cooperative_prefill
        {
            return Err(Error::Unsupported(
                "external marker ABI accepts extensions/context bounds only".into(),
            ));
        }
        let max_context = options
            .max_context
            .unwrap_or(self.manifest.backbone.max_context);
        if max_context == 0 || max_context > self.manifest.backbone.max_context {
            return Err(Error::Request(
                "external context must fit the declared model budget".into(),
            ));
        }
        let mut prompts = Vec::with_capacity(request.questions.len());
        let mut readouts = Vec::with_capacity(request.questions.len());
        for (id, question) in &request.questions {
            // Uncached preparation keeps qualification fresh and cannot reuse
            // a previous wire response in place of actual external inference.
            let mut prompt =
                self.build_prompt(&request.state, question, &mut EvalStats::default(), false)?;
            if prompt.tokens.is_empty()
                || prompt.tokens.len() > max_context
                || prompt.candidates.is_empty()
                || prompt.candidates.len() > self.manifest.prompt_contract.max_options
            {
                return Err(Error::Request(format!(
                    "question `{id}` exceeds external model input budgets"
                )));
            }
            let mut positions: Vec<_> = prompt
                .candidates
                .iter()
                .map(|candidate| candidate.position)
                .collect();
            positions.sort_unstable();
            positions.dedup();
            if positions.len() != prompt.candidates.len()
                || positions
                    .iter()
                    .any(|&position| position >= prompt.tokens.len())
                || (self.manifest.prompt_contract.template == "laya-v1"
                    && positions.iter().any(|&position| {
                        self.tokenizer.mask_token_id() != prompt.tokens.get(position).copied()
                    }))
            {
                return Err(Error::Request(
                    "external marker outside its token sequence".into(),
                ));
            }
            readouts.push(MarkerReadout {
                tokens: std::mem::take(&mut prompt.tokens),
                positions,
                qtype: prompt.qtype,
            });
            prompts.push(prompt);
        }
        Ok(ExternalEvaluation {
            owner: self.preparation_identity.clone(),
            request,
            options,
            prompts,
            readouts,
        })
    }

    /// Consume a frozen plan and raw graph-integrated scalar scores. No feature
    /// projection, vocabulary pooling, decode or alternative calibration occurs.
    pub fn finish_external_markers(
        &self,
        plan: ExternalEvaluation,
        scores: Vec<Vec<f32>>,
    ) -> Result<(SystemOneResponse, EvalStats)> {
        if !Arc::ptr_eq(&self.preparation_identity, &plan.owner) {
            return Err(Error::Request(
                "external plan belongs to another engine".into(),
            ));
        }
        if scores.len() != plan.readouts.len()
            || scores.iter().zip(&plan.readouts).any(|(scores, input)| {
                scores.len() != input.positions.len()
                    || scores.iter().any(|score| !score.is_finite())
            })
        {
            return Err(Error::Backend(
                "external scalar scores must match every marker and be finite".into(),
            ));
        }
        let mut answers = BTreeMap::new();
        let mut raw_logits = BTreeMap::new();
        let mut stats = EvalStats::default();
        for (((id, question), prompt), (input, scores)) in plan
            .request
            .questions
            .iter()
            .zip(plan.prompts)
            .zip(plan.readouts.into_iter().zip(scores))
        {
            let mut logits = prompt
                .candidates
                .iter()
                .map(|candidate| {
                    let row = input
                        .positions
                        .binary_search(&candidate.position)
                        .map_err(|_| {
                            Error::Backend("external result omitted a candidate marker".into())
                        })?;
                    Ok(scores[row])
                })
                .collect::<Result<Vec<_>>>()?;
            let probabilities = calibration::calibrate_readout(
                &mut logits,
                self.temperature_for(question, prompt.candidates.len()),
                plan.options.extensions,
            )?;
            answers.insert(
                id.clone(),
                self.build_answer(question, &prompt.candidates, &probabilities)?,
            );
            if plan.options.extensions {
                raw_logits.insert(id.clone(), logits);
            }
            stats.forward_calls += 1;
            stats.processed_tokens += input.tokens.len() as u64;
            stats.prepared_questions += 1;
        }
        let mut response = SystemOneResponse::new(
            plan.request.model,
            answers,
            Usage::new(stats.processed_tokens),
        );
        if plan.options.extensions {
            response.extensions = Some(self.extensions(raw_logits));
        }
        Ok((response, stats))
    }
}

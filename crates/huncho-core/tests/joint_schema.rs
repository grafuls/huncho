use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use huncho_core::backend::{
    Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput, RequestOutput,
};
use huncho_core::contract::{Answer, SystemOneRequest};
use huncho_core::engine::{Engine, EvalOptions};
use huncho_core::error::Result;
use huncho_core::head::HeadParams;
use huncho_core::manifest::{BackendId, Family, ModelManifest};
use huncho_core::tokenizer::SimpleTokenizer;
use serde_json::json;

struct JointBackend {
    calls: Arc<AtomicUsize>,
    output: RequestOutput,
    check_order: bool,
}

impl Backend for JointBackend {
    fn id(&self) -> BackendId {
        BackendId::Clef
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.id(),
            families: vec![Family::F5],
            ..Default::default()
        }
    }
    fn forward(&mut self, _: ForwardInput) -> Result<ForwardOutput> {
        panic!("must not split the joint request")
    }
    fn fork(&mut self, _: CacheHandle) -> Result<CacheHandle> {
        panic!("no KV cache")
    }
    fn forward_request(
        &mut self,
        request: &SystemOneRequest,
        max_context: usize,
    ) -> Result<RequestOutput> {
        assert_eq!(request.questions.len(), 3);
        if self.check_order {
            assert_eq!(
                request
                    .questions
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                vec!["c", "a", "b"]
            );
        }
        assert_eq!(max_context, 512);
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.output.clone())
    }
}

fn request() -> SystemOneRequest {
    serde_json::from_value(json!({
        "model": "clef", "state": {"total": 123}, "questions": {
            "c": {"type": "noul", "instructions": "urgent?"},
            "a": {"type": "choice", "instructions": "team", "criteria": {"z": "first", "b": "second"}},
            "b": {"type": "score", "instructions": "severity", "criteria": ["low", "high"]}
        }
    })).unwrap()
}

fn output() -> RequestOutput {
    serde_json::from_value(json!({
        "input_tokens": 42,
        "logits": {"a": {"b": 0.0, "z": 0.0}, "b": {"0": 0.0, "1": 2.1972246}, "c": {"false": 0.0, "true": 2.1972246}}
    })).unwrap()
}

fn engine(output: RequestOutput) -> (Engine, Arc<AtomicUsize>) {
    engine_with_order_check(output, true)
}

fn engine_with_order_check(output: RequestOutput, check_order: bool) -> (Engine, Arc<AtomicUsize>) {
    let manifest: ModelManifest = serde_json::from_value(json!({
        "schema_version": "1.0", "name": "clef", "family": "F5",
        "backbone": {"source": {"kind": "local", "path": "."}, "hidden_size": 16, "max_context": 512},
        "head": {"kind": "joint-schema", "weights": "joint_head.safetensors"},
        "prompt_contract": {"template": "clef-native-v1", "state_budget": 512, "head_budget": 512, "max_options": 255, "contract_hash": "fixture"},
        "calibration": {"default": {"temperature": 2.0, "confidence": "max-probability", "status": "pending"}}
    })).unwrap();
    manifest.validate().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = Engine::new(
        manifest,
        Box::new(JointBackend {
            calls: calls.clone(),
            output,
            check_order,
        }),
        Box::new(SimpleTokenizer::new(32768)),
        HeadParams::default(),
        BackendId::Clef,
        "bf16",
    )
    .unwrap()
    .with_prompt_cache(1024 * 1024);
    (engine, calls)
}

#[test]
fn joint_result_reuse_requires_the_complete_ordered_schema() {
    let (engine, calls) = engine_with_order_check(output(), false);
    let engine = engine.with_result_cache(1024 * 1024);
    let opts = EvalOptions {
        extensions: true,
        ..Default::default()
    };
    let mut request = request();
    let original = engine.eval(&request, &opts).unwrap();
    let cached = engine.eval(&request, &opts).unwrap();
    assert_eq!(
        serde_json::to_vec(&original).unwrap(),
        serde_json::to_vec(&cached).unwrap()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    request.questions.reverse();
    engine.eval(&request, &opts).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    if let huncho_core::contract::Question::Choice { criteria, .. } = &mut request.questions["a"] {
        criteria.reverse();
    }
    let changed = engine.eval(&request, &opts).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert!(matches!(&changed.answers["a"], Answer::Choice { choice, .. } if choice == "b"));
    let repeated = engine.eval(&request, &opts).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        serde_json::to_vec(&changed).unwrap(),
        serde_json::to_vec(&repeated).unwrap()
    );
}

#[test]
fn joint_forward_preserves_labels_types_calibration_and_usage() {
    let (engine, calls) = engine(output());
    let response = engine
        .eval(
            &request(),
            &EvalOptions {
                extensions: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(response.usage.input_tokens, 42); // Once, not once per question.
    assert_eq!(response.usage.output_tokens, 0);
    match &response.answers["a"] {
        Answer::Choice {
            choice, confidence, ..
        } => {
            assert_eq!(choice, "z"); // Reference tie-breaking follows caller order.
            assert_eq!(*confidence, 0.5); // Raw max probability, not Jev peak confidence.
        }
        _ => panic!("choice"),
    }
    match &response.answers["b"] {
        Answer::Score {
            score,
            confidence,
            legend,
            ..
        } => {
            assert!((*score - 0.75).abs() < 1e-6);
            assert!((*confidence - 0.75).abs() < 1e-6);
            assert_eq!(legend["1"], "high");
        }
        _ => panic!("score"),
    }
    assert!(matches!(response.answers["c"], Answer::Noul { noul } if (noul - 0.75).abs() < 1e-6));
    let extensions = response.extensions.unwrap();
    assert_eq!(extensions.dtype.as_deref(), Some("bf16"));
    assert_eq!(
        extensions.confidence_definition.as_deref(),
        Some("max-probability")
    );
    assert_eq!(extensions.raw_logits.unwrap()["a"], vec![0.0, 0.0]);
}

#[test]
fn rejects_incomplete_mislabelled_and_nonfinite_backend_outputs() {
    let mut bad = Vec::new();
    let mut missing = output();
    missing.logits.remove("c");
    bad.push(missing);
    let mut wrong_id = output();
    let scores = wrong_id.logits.remove("a").unwrap();
    wrong_id.logits.insert("unknown".into(), scores);
    bad.push(wrong_id);
    let mut wrong_option = output();
    wrong_option.logits.get_mut("a").unwrap().remove("z");
    bad.push(wrong_option);
    let mut renamed = output();
    renamed
        .logits
        .get_mut("a")
        .unwrap()
        .insert("unknown".into(), 0.0);
    renamed.logits.get_mut("a").unwrap().remove("z");
    bad.push(renamed);
    let mut nonfinite = output();
    nonfinite
        .logits
        .get_mut("a")
        .unwrap()
        .insert("z".into(), f32::NAN);
    bad.push(nonfinite);
    let mut tokens = output();
    tokens.input_tokens = 513;
    bad.push(tokens);
    for output in bad {
        assert!(engine(output)
            .0
            .eval(&request(), &EvalOptions::default())
            .is_err());
    }
}

// Malformed native reports must fail before any newly produced result is cached.
struct BatchJointBackend {
    fault: Arc<AtomicUsize>,
}
impl Backend for BatchJointBackend {
    fn id(&self) -> BackendId {
        BackendId::Clef
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.id(),
            families: vec![Family::F5],
            max_context: 512,
            ..Default::default()
        }
    }
    fn forward(&mut self, _: ForwardInput) -> Result<ForwardOutput> {
        panic!("whole schemas only")
    }
    fn fork(&mut self, _: CacheHandle) -> Result<CacheHandle> {
        panic!("no prefix cache")
    }
    fn forward_request(&mut self, _: &SystemOneRequest, _: usize) -> Result<RequestOutput> {
        Ok(output())
    }
    fn supports_request_batch(&self) -> bool {
        true
    }
    fn forward_request_batch(
        &mut self,
        inputs: &[huncho_core::backend::RequestBatchInput<'_>],
        _: usize,
        _: usize,
        work: &mut huncho_core::backend::RequestBatchWork,
    ) -> Result<Vec<RequestOutput>> {
        *work = huncho_core::backend::RequestBatchWork {
            forward_calls: 1,
            processed_tokens: 42 * inputs.len() as u64,
            batch_calls: u64::from(inputs.len() > 1),
            prepared_questions: inputs
                .iter()
                .map(|i| i.request.questions.len() as u64)
                .sum(),
            ..Default::default()
        };
        let mut outputs = vec![output(); inputs.len()];
        match self.fault.load(Ordering::SeqCst) {
            1 => {
                outputs.pop();
            }
            2 => {
                work.processed_tokens += 1;
            }
            3 => {
                outputs[1]
                    .logits
                    .get_mut("a")
                    .unwrap()
                    .insert("z".into(), f32::NAN);
            }
            4 => {
                work.batch_calls = 2;
            }
            5 => {
                work.padded_tokens = 1;
                work.processed_tokens += 1;
            }
            6 => {
                work.prepared_questions = 0;
            }
            7 => {
                outputs[1].logits.remove("b");
            }
            _ => {}
        }
        Ok(outputs)
    }
}
#[test]
fn malformed_native_joint_groups_never_publish_partial_results() {
    use huncho_core::engine::EvalStats;
    let (reference, _) = engine(output());
    for fault_code in 1..=7 {
        let fault = Arc::new(AtomicUsize::new(fault_code));
        let engine = Engine::new(
            reference.manifest().clone(),
            Box::new(BatchJointBackend {
                fault: fault.clone(),
            }),
            Box::new(SimpleTokenizer::new(32768)),
            HeadParams::default(),
            BackendId::Clef,
            "bf16",
        )
        .unwrap()
        .with_result_cache(1 << 20);
        let options = EvalOptions {
            prepare_all: true,
            max_batch_tokens: Some(256),
            extensions: true,
            ..Default::default()
        };
        let mut second = request();
        second.state = huncho_core::contract::StateValue::from("another complete request");
        let requests = [request(), second];
        let prepare = || {
            requests
                .iter()
                .cloned()
                .map(|r| {
                    engine
                        .prepare_eval_with_stats(r, options.clone(), &mut Default::default())
                        .unwrap()
                })
                .collect()
        };
        let mut work = EvalStats::default();
        assert!(
            engine
                .eval_prepared_batch_with_stats(prepare(), 256, &mut work)
                .is_err(),
            "fault {fault_code}"
        );
        assert_eq!(work.forward_calls, 1);
        fault.store(0, Ordering::SeqCst);
        let answers = engine
            .eval_prepared_batch_with_stats(prepare(), 256, &mut work)
            .unwrap();
        assert_eq!(answers.len(), 2);
        assert_eq!(
            work.result_cache_hits, 0,
            "partial cache escaped fault {fault_code}"
        );
        assert_eq!(work.forward_calls, 1);
        assert_eq!(work.prepared_questions, 6);
        let cached = engine
            .eval_prepared_batch_with_stats(prepare(), 256, &mut work)
            .unwrap();
        assert_eq!(work.result_cache_hits, 2);
        assert_eq!(work.forward_calls, 0);
        assert_eq!(
            serde_json::to_value(cached).unwrap(),
            serde_json::to_value(answers).unwrap()
        );
    }
}

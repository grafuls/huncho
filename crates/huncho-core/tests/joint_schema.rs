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
        assert_eq!(
            request
                .questions
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["c", "a", "b"]
        );
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
        }),
        Box::new(SimpleTokenizer::new(32768)),
        HeadParams::default(),
        BackendId::Clef,
        "bf16",
    )
    .unwrap();
    (engine, calls)
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

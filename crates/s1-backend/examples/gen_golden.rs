//! Generate golden conformance vectors for the offline mock reference (CONF-01).
//!
//! The mock backend is the offline reference for the conformance harness. This
//! tool runs a fixed set of requests through the mock engine and writes the
//! resulting probability vectors to a `golden.json` suite that `s1 conform`
//! and the integration test later reproduce.
//!
//! Usage:
//!   cargo run -p s1-backend --example gen_golden \
//!     [manifest] [output]
//!
//! Defaults: `examples/mock-model/s1-model.json` -> `examples/mock-model/golden.json`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use s1_backend::MockBackend;
use s1_core::backend::Backend;
use s1_core::conformance::{GoldenCase, GoldenSuite};
use s1_core::contract::{Answer, SystemOneRequest};
use s1_core::engine::{Engine, EvalOptions};
use s1_core::head::HeadParams;
use s1_core::manifest::{BackendId, ModelManifest};
use s1_core::tokenizer::{SimpleTokenizer, Tokenizer};

fn expected_from_answer(answer: &Answer) -> BTreeMap<String, f32> {
    match answer {
        Answer::Choice { probabilities, .. } => probabilities.clone(),
        Answer::Score { probabilities, .. } => probabilities.clone(),
        Answer::Noul { noul } => {
            let mut m = BTreeMap::new();
            m.insert("yes".to_string(), *noul);
            m.insert("no".to_string(), 1.0 - noul);
            m
        }
    }
}

fn build_engine(manifest_path: &std::path::Path) -> anyhow::Result<Engine> {
    let manifest = ModelManifest::load(manifest_path)?;
    let backend: Box<dyn Backend> = Box::new(
        MockBackend::with_vocab(4096)
            .with_backend(BackendId::Onnx)
            .with_dtype("fp32"),
    );
    let tokenizer: Box<dyn Tokenizer> = Box::new(SimpleTokenizer::new(32768));
    let engine = Engine::new(
        manifest,
        backend,
        tokenizer,
        HeadParams::default(),
        BackendId::Onnx,
        "fp32",
    )?;
    Ok(engine)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let manifest_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/mock-model/s1-model.json"));
    let output_path = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/mock-model/golden.json"));

    let engine = build_engine(&manifest_path)?;
    let family = engine.family().to_string();

    // A fixed set of conformance inputs. These exercise choice, noul, and score
    // questions, both standalone and packed into a single request.
    let cases: Vec<(&str, serde_json::Value)> = vec![
        (
            "choice-short-state",
            serde_json::json!({
                "state": "The customer wants a refund because the shoes are too small.",
                "model": "mock-laya",
                "questions": {
                    "department": {
                        "type": "choice",
                        "instructions": "Which team handles this?",
                        "criteria": { "returns": "The customer wants money back", "billing": "Charge problem" }
                    }
                }
            }),
        ),
        (
            "choice-structured-state",
            serde_json::json!({
                "state": { "ticket": "A seller is disputing a shipment delay", "priority": "high" },
                "model": "mock-laya",
                "questions": {
                    "resolution": {
                        "type": "choice",
                        "instructions": "What should the agent do first?",
                        "criteria": {
                            "refund": "Issue a refund immediately",
                            "escalate": "Escalate to a supervisor",
                            "inspect": "Inspect the package first"
                        }
                    }
                }
            }),
        ),
        (
            "noul-single",
            serde_json::json!({
                "state": "A user asks whether their account was charged twice.",
                "model": "mock-laya",
                "questions": {
                    "is_refund": {
                        "type": "noul",
                        "instructions": "Is the customer requesting a refund?",
                        "criteria": { "true": "Asks for money back", "false": "Does not ask" }
                    }
                }
            }),
        ),
        (
            "score-many-levels",
            serde_json::json!({
                "state": "The server has been down for three hours with no ETA.",
                "model": "mock-laya",
                "questions": {
                    "severity": {
                        "type": "score",
                        "instructions": "Rate the severity of the incident",
                        "criteria": ["Low", "Moderate", "High", "Critical"]
                    }
                }
            }),
        ),
        (
            "packed-multi-question",
            serde_json::json!({
                "state": "A support ticket about a billing error on the premium plan.",
                "model": "mock-laya",
                "questions": {
                    "department": {
                        "type": "choice",
                        "instructions": "Which team handles this?",
                        "criteria": { "returns": "Refund request", "billing": "Charging issue" }
                    },
                    "is_refund": {
                        "type": "noul",
                        "instructions": "Is the customer requesting a refund?",
                        "criteria": { "true": "Requests money back", "false": "Not a refund" }
                    },
                    "severity": {
                        "type": "score",
                        "instructions": "Rate severity",
                        "criteria": ["Low", "Moderate", "High"]
                    }
                }
            }),
        ),
    ];

    let mut golden_cases = Vec::new();
    for (id, json) in cases {
        let request: SystemOneRequest = serde_json::from_value(json)?;
        request.validate()?;
        let response = engine.eval(&request, &EvalOptions::default())?;
        let mut expected = BTreeMap::new();
        for (qid, answer) in &response.answers {
            expected.insert(qid.clone(), expected_from_answer(answer));
        }
        golden_cases.push(GoldenCase {
            id: id.to_string(),
            request,
            expected,
        });
    }

    let suite = GoldenSuite {
        schema_version: "1.0".to_string(),
        family,
        hash: Some("mock-reference-hash-v1".to_string()),
        cases: golden_cases,
    };

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(&suite)?;
    std::fs::write(&output_path, bytes)?;
    println!(
        "wrote {} cases to {}",
        suite.cases.len(),
        output_path.display()
    );
    Ok(())
}

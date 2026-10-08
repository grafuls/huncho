//! Offline conformance integration test for the mock reference (CONF-02/03).
//!
//! The mock backend is the offline reference for the conformance harness. This
//! test builds a serving engine from the example mock model package, runs the
//! checked-in golden vectors, and asserts that the engine reproduces them
//! within tolerance. This is the CI gate that would otherwise be driven by a
//! real reference implementation.

use std::path::PathBuf;

use huncho_backend::MockBackend;
use huncho_core::backend::Backend;
use huncho_core::conformance::{self, ConformanceThresholds};
use huncho_core::engine::Engine;
use huncho_core::head::HeadParams;
use huncho_core::manifest::{BackendId, ModelManifest};
use huncho_core::tokenizer::{SimpleTokenizer, Tokenizer};

fn workspace_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn build_mock_engine(manifest_path: &std::path::Path) -> Engine {
    let manifest = ModelManifest::load(manifest_path).expect("load example manifest");
    let backend: Box<dyn Backend> = Box::new(
        MockBackend::with_vocab(4096)
            .with_backend(BackendId::Onnx)
            .with_dtype("fp32"),
    );
    let tokenizer: Box<dyn Tokenizer> = Box::new(SimpleTokenizer::new(32768));
    Engine::new(
        manifest,
        backend,
        tokenizer,
        HeadParams::default(),
        BackendId::Onnx,
        "fp32",
    )
    .expect("build mock engine")
}

#[test]
fn mock_reproduces_golden_suite() {
    let manifest_path = workspace_dir().join("examples/mock-model/huncho-model.json");
    let golden_path = workspace_dir().join("examples/mock-model/golden.json");

    let engine = build_mock_engine(&manifest_path);
    let suite = conformance::load_suite(&golden_path).expect("load golden suite");
    let report = conformance::run_suite(&engine, &suite, &ConformanceThresholds::default())
        .expect("run suite");

    assert!(
        report.passed,
        "mock conformance failed: max_delta={}, argmax={}, ece={}",
        report.max_prob_delta, report.argmax_agreement, report.ece
    );
    assert!(
        report.max_prob_delta < 1e-6,
        "expected near-identical deterministic probabilities, got {}",
        report.max_prob_delta
    );
    assert_eq!(report.argmax_agreement, 1.0);
}

#[test]
fn mock_deterministic_across_runs() {
    // Re-running the suite must produce bit-identical probabilities (the mock is
    // deterministic on a given platform), which is what makes it a stable
    // offline reference.
    let manifest_path = workspace_dir().join("examples/mock-model/huncho-model.json");
    let golden_path = workspace_dir().join("examples/mock-model/golden.json");
    let engine = build_mock_engine(&manifest_path);

    let suite = conformance::load_suite(&golden_path).unwrap();
    let r1 = conformance::run_suite(&engine, &suite, &ConformanceThresholds::default()).unwrap();
    let r2 = conformance::run_suite(&engine, &suite, &ConformanceThresholds::default()).unwrap();
    assert_eq!(r1.max_prob_delta, r2.max_prob_delta);
    assert_eq!(r1.ece, r2.ece);
}

#[test]
fn labeled_suite_reports_outcome_calibration_without_changing_fidelity() {
    let engine = build_mock_engine(&workspace_dir().join("examples/mock-model/huncho-model.json"));
    let mut suite =
        conformance::load_suite(workspace_dir().join("examples/mock-model/golden.json")).unwrap();
    let original =
        conformance::run_suite(&engine, &suite, &ConformanceThresholds::default()).unwrap();
    assert!(original.outcome_calibration.is_none());
    let mut questions = 0;
    for case in &mut suite.cases {
        for (qid, distribution) in &case.expected {
            // Observed outcomes deliberately disagree with the predicted mode.
            let target = distribution
                .iter()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
                .clone();
            case.targets.insert(qid.clone(), target);
            questions += 1;
        }
    }
    let report =
        conformance::run_suite(&engine, &suite, &ConformanceThresholds::default()).unwrap();
    assert!(report.passed);
    assert_eq!(report.max_prob_delta, original.max_prob_delta);
    let outcomes = report.outcome_calibration.unwrap();
    assert_eq!(outcomes.questions, questions);
    assert!(outcomes.reference_ece > 0.1);
    assert!((outcomes.reference_ece - outcomes.backend_ece).abs() < 1e-6);
    assert!((outcomes.reference_brier - outcomes.backend_brier).abs() < 1e-6);
}

#[test]
fn malformed_or_incomplete_golden_suites_cannot_pass() {
    let engine = build_mock_engine(&workspace_dir().join("examples/mock-model/huncho-model.json"));
    let suite =
        conformance::load_suite(workspace_dir().join("examples/mock-model/golden.json")).unwrap();
    let check = |suite| {
        assert!(conformance::run_suite(&engine, &suite, &ConformanceThresholds::default()).is_err())
    };
    let mut empty = suite.clone();
    empty.cases.clear();
    check(empty);
    let mut missing = suite.clone();
    missing.cases[0].expected.pop_first();
    check(missing);
    for invalid in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        let mut bad = suite.clone();
        *bad.cases[0]
            .expected
            .values_mut()
            .next()
            .unwrap()
            .values_mut()
            .next()
            .unwrap() = invalid;
        check(bad);
    }
    let mut bad = suite.clone();
    for value in bad.cases[0]
        .expected
        .values_mut()
        .next()
        .unwrap()
        .values_mut()
    {
        *value = 0.0;
    }
    check(bad);
    let mut mislabeled = suite.clone();
    let distribution = mislabeled.cases[0].expected.values_mut().next().unwrap();
    let (_, probability) = distribution.pop_first().unwrap();
    distribution.insert("unknown label".into(), probability);
    check(mislabeled);
    let mut partial_targets = suite.clone();
    let question = partial_targets.cases[0]
        .expected
        .keys()
        .next()
        .unwrap()
        .clone();
    partial_targets.cases[0]
        .targets
        .insert(question, "unknown label".into());
    check(partial_targets);
}

#[test]
fn argmax_gate_and_outcome_metrics_follow_candidate_order_on_ties() {
    let manifest =
        ModelManifest::load(workspace_dir().join("examples/mock-model/huncho-model.json")).unwrap();
    let engine = Engine::new(
        manifest,
        Box::new(MockBackend::with_vocab(512)),
        Box::new(SimpleTokenizer::new(32768)),
        HeadParams::scalar_linear(512, vec![0.0; 512], 0.0).unwrap(),
        BackendId::Onnx,
        "fp32",
    )
    .unwrap();
    for (first, second) in [("a", "z"), ("z", "a")] {
        let request: huncho_core::contract::SystemOneRequest = serde_json::from_str(
            &format!(r#"{{"model":"mock-laya","state":"tie","questions":{{"q":{{"type":"choice","instructions":"pick","criteria":{{"{first}":null,"{second}":null}}}}}}}}"#),
        ).unwrap();
        let response = engine.eval(&request, &Default::default()).unwrap();
        assert!(
            matches!(&response.answers["q"], huncho_core::contract::Answer::Choice { choice, .. } if choice == first)
        );
        let mut suite = conformance::GoldenSuite {
            schema_version: "1.0".into(),
            family: "F1".into(),
            hash: None,
            cases: vec![conformance::GoldenCase {
                id: "tie".into(),
                request,
                expected: [(
                    "q".into(),
                    [(first.into(), 0.499999), (second.into(), 0.500001)].into(),
                )]
                .into(),
                targets: [("q".into(), first.into())].into(),
            }],
        };
        let report = conformance::run_suite(&engine, &suite, &Default::default()).unwrap();
        assert!(report.max_prob_delta < 1e-3);
        assert_eq!(report.argmax_agreement, 0.0);
        assert!(
            !report.passed,
            "tiny delta must not mask a different selected label"
        );
        for probability in suite.cases[0].expected.get_mut("q").unwrap().values_mut() {
            *probability = 0.5;
        }
        let report = conformance::run_suite(&engine, &suite, &Default::default()).unwrap();
        assert!(report.passed);
        let outcomes = report.outcome_calibration.unwrap();
        assert_eq!(outcomes.backend_ece, 0.5);
        assert_eq!(outcomes.reference_ece, 0.5);
    }
}

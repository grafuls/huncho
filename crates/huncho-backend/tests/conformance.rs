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
    let report =
        conformance::run_suite(&engine, &suite, &ConformanceThresholds::default()).expect("run suite");

    assert!(
        report.passed,
        "mock conformance failed: max_delta={}, argmax={}, ece={}",
        report.max_prob_delta,
        report.argmax_agreement,
        report.ece
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

//! End-to-end conformance (CONF-02) for the ONNX backend (BE-01).
//!
//! This drives a real ONNX Runtime session through the full engine pipeline
//! (prompt -> head -> calibration -> golden comparison) and asserts it
//! reproduces the offline mock reference within the default thresholds.
//!
//! The golden suite and manifest live in `examples/mock-model/`. The committed
//! `mock-model.onnx` artifact (hidden=512) reproduces the mock backend's
//! deterministic hidden-state rule exactly, so the ONNX engine must agree with
//! the reference probabilities (max delta 0, argmax 1.0).
//!
//! Requires `--features onnx`:
//!   cargo test -p huncho-backend --features onnx --test onnx_conformance

#![cfg(feature = "onnx")]

use huncho_backend::OnnxBackend;
use huncho_core::backend::Backend;
use huncho_core::conformance;
use huncho_core::engine::Engine;
use huncho_core::head::HeadParams;
use huncho_core::manifest::{BackendId, ModelManifest};
use huncho_core::tokenizer::{SimpleTokenizer, Tokenizer};

#[test]
fn onnx_backend_passes_reference_golden() {
    let manifest_path = "../../examples/mock-model/huncho-model.json";
    let artifact_path = "../../examples/mock-model/mock-model.onnx";
    let golden_path = "../../examples/mock-model/golden.json";

    let manifest = ModelManifest::load(manifest_path).unwrap();

    let backend: Box<dyn Backend> = Box::new(
        OnnxBackend::load(
            artifact_path,
            manifest.backbone.hidden_size,
            manifest.backbone.max_context,
            "fp32",
        )
        .unwrap(),
    );
    let tokenizer: Box<dyn Tokenizer> = Box::new(SimpleTokenizer::new(32768));

    let engine = Engine::new(
        manifest,
        backend,
        tokenizer,
        HeadParams::default(),
        BackendId::Onnx,
        "fp32",
    )
    .unwrap();

    let suite = conformance::load_suite(golden_path).unwrap();
    let report =
        conformance::run_suite(&engine, &suite, &conformance::ConformanceThresholds::default())
            .unwrap();

    assert_eq!(report.backend, "onnx");
    assert!(
        report.passed,
        "ONNX backend failed reference conformance: max_delta={} argmax={} ece={}",
        report.max_prob_delta,
        report.argmax_agreement,
        report.ece,
    );
    assert!(
        report.max_prob_delta <= 1e-3,
        "max prob delta {:.6} exceeds 1e-3",
        report.max_prob_delta
    );
}

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

use huncho_backend::onnx::OnnxOptions;
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
    let report = conformance::run_suite(
        &engine,
        &suite,
        &conformance::ConformanceThresholds::default(),
    )
    .unwrap();

    assert_eq!(report.backend, "onnx");
    assert!(
        report.passed,
        "ONNX backend failed reference conformance: max_delta={} argmax={} ece={}",
        report.max_prob_delta, report.argmax_agreement, report.ece,
    );
    assert!(
        report.max_prob_delta <= 1e-3,
        "max prob delta {:.6} exceeds 1e-3",
        report.max_prob_delta
    );
}

#[test]
fn graph_side_readout_and_bound_buffer_pass_unchanged_probability_goldens() {
    let suite = conformance::load_suite("../../examples/mock-model/golden.json").unwrap();
    let manifest = ModelManifest::load("../../examples/mock-model/huncho-model.json").unwrap();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tiny_mock_readout.onnx"
    );
    for bytes in [0, 1024 * 1024] {
        let engine = Engine::new(
            manifest.clone(),
            Box::new(
                OnnxBackend::load_with_options(
                    path,
                    manifest.backbone.hidden_size,
                    manifest.backbone.max_context,
                    "fp32",
                    OnnxOptions {
                        compact_readout: true,
                        output_buffer_bytes: bytes,
                        ..Default::default()
                    },
                )
                .unwrap(),
            ),
            Box::new(SimpleTokenizer::new(32768)),
            HeadParams::default(),
            BackendId::Onnx,
            "fp32",
        )
        .unwrap();
        let report = conformance::run_suite(&engine, &suite, &Default::default()).unwrap();
        assert!(report.passed);
        assert_eq!(report.max_prob_delta, 0.);
        assert_eq!(report.argmax_agreement, 1.);
        assert_eq!(report.execution_metadata["onnx_readout"], "gather-v1");
    }
}

#[test]
fn native_onnx_batches_qualify_per_request_and_cross_request_work() {
    use huncho_core::engine::EvalOptions;
    let mut suite = conformance::load_suite("../../examples/mock-model/golden.json").unwrap();
    let mut case = suite.cases[0].clone();
    let question = case.request.questions.values().next().unwrap().clone();
    let expected = case.expected.values().next().unwrap().clone();
    case.request
        .questions
        .insert("duplicate_prompt".into(), question);
    case.expected.insert("duplicate_prompt".into(), expected);
    suite.cases = vec![case.clone(), case];
    suite.cases[1].id = "second-request".into();
    let manifest = ModelManifest::load("../../examples/mock-model/huncho-model.json").unwrap();
    let engine = Engine::new(
        manifest.clone(),
        Box::new(
            OnnxBackend::load_with_options(
                "../../examples/mock-model/mock-model.onnx",
                manifest.backbone.hidden_size,
                manifest.backbone.max_context,
                "fp32",
                OnnxOptions {
                    native_batch: true,
                    output_buffer_bytes: 1024 * 1024,
                    ..Default::default()
                },
            )
            .unwrap(),
        ),
        Box::new(SimpleTokenizer::new(32768)),
        HeadParams::default(),
        BackendId::Onnx,
        "fp32",
    )
    .unwrap()
    .with_prompt_cache(1024 * 1024)
    .with_result_cache(1024 * 1024);
    let options = EvalOptions {
        max_batch_tokens: Some(4096),
        prepare_all: true,
        ..Default::default()
    };
    let per_request =
        conformance::run_suite_with_options(&engine, &suite, &Default::default(), &options)
            .unwrap();
    let cross_request = conformance::run_suite_with_cross_request_batches(
        &engine,
        &suite,
        &Default::default(),
        &options,
        2,
    )
    .unwrap();
    for report in [&per_request, &cross_request] {
        assert!(report.passed, "{report:?}");
        assert_eq!(report.max_prob_delta, 0.);
        assert_eq!(report.argmax_agreement, 1.);
        assert_eq!(
            report.optimization_parity.as_ref().unwrap().max_prob_delta,
            0.
        );
        assert!(report.work.batch_calls > 0);
        assert_eq!(report.work.prepared_questions, 4);
        assert_eq!(report.work.result_cache_hits, 0);
        assert_eq!(report.work.prompt_cache_hits, 0);
        assert_eq!(
            report.execution_metadata["onnx_native_batch"],
            "equal-length-v1"
        );
    }
    assert!(cross_request.work.cross_request_batches > 0);
    assert!(cross_request.work.forward_calls < per_request.work.forward_calls);
    assert_eq!(
        cross_request.work.processed_tokens,
        per_request.work.processed_tokens
    );
}

#[cfg(feature = "onnx-shared")]
#[test]
fn actual_shared_cpu_contexts_pass_unchanged_probability_goldens_concurrently() {
    let manifest = ModelManifest::load("../../examples/mock-model/huncho-model.json").unwrap();
    let suite = conformance::load_suite("../../examples/mock-model/golden.json").unwrap();
    let primary = Engine::new(
        manifest.clone(),
        Box::new(
            OnnxBackend::load_with_options(
                "../../examples/mock-model/mock-model.onnx",
                manifest.backbone.hidden_size,
                manifest.backbone.max_context,
                "fp32",
                OnnxOptions {
                    shared_initializers: true,
                    intra_threads: 1,
                    ..Default::default()
                },
            )
            .unwrap(),
        ),
        Box::new(SimpleTokenizer::new(32768)),
        HeadParams::default(),
        BackendId::Onnx,
        "fp32",
    )
    .unwrap();
    let contexts = vec![
        primary.replica().unwrap(),
        primary.replica().unwrap(),
        primary,
    ];
    std::thread::scope(|scope| {
        let jobs: Vec<_> = contexts
            .iter()
            .map(|engine| {
                scope.spawn(|| {
                    let report =
                        conformance::run_suite(engine, &suite, &Default::default()).unwrap();
                    assert!(report.passed);
                    assert_eq!(report.max_prob_delta, 0.);
                    assert_eq!(report.argmax_agreement, 1.);
                    assert_eq!(
                        report.execution_metadata["onnx_initializer_residency"],
                        "immutable-cpu-v1"
                    );
                    assert!(report.work.forward_calls > 0);
                })
            })
            .collect();
        for job in jobs {
            job.join().unwrap();
        }
    });
}

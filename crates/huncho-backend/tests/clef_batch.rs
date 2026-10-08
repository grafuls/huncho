//! Whole-schema CPU batches, compared to frozen independent PyTorch fixtures.
#![cfg(feature = "clef")]
use huncho_backend::ClefBackend;
use huncho_core::{
    backend::{Backend, RequestBatchInput, RequestBatchWork, RequestOutput},
    conformance::{
        run_suite_with_cross_request_batches, run_suite_with_options, GoldenCase, GoldenSuite,
    },
    contract::SystemOneRequest,
    engine::{Engine, EvalOptions, EvalStats},
    manifest::{BackendId, ModelManifest},
    tokenizer::SimpleTokenizer,
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_clef")
}
fn frozen() -> Value {
    serde_json::from_slice(&std::fs::read(root().join("golden.json")).unwrap()).unwrap()
}
fn manifest() -> ModelManifest {
    ModelManifest::load(root().join("huncho-model.json")).unwrap()
}
fn load(dtype: &str, grouped: bool) -> ClefBackend {
    ClefBackend::load(&root(), &manifest(), dtype, candle::Device::Cpu)
        .unwrap()
        .with_vectorized_head(grouped)
        .with_grouped_pooling(grouped)
        .unwrap()
}
fn requests(reference: &Value) -> Vec<SystemOneRequest> {
    reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| serde_json::from_value(c["request"].clone()).unwrap())
        .collect()
}
fn raw_parity(a: &RequestOutput, b: &RequestOutput, raw_tolerance: f32) {
    assert_eq!(a.input_tokens, b.input_tokens);
    assert_eq!(
        a.logits.keys().collect::<Vec<_>>(),
        b.logits.keys().collect::<Vec<_>>()
    );
    for (q, values) in &a.logits {
        assert_eq!(
            values.keys().collect::<Vec<_>>(),
            b.logits[q].keys().collect::<Vec<_>>()
        );
        for (id, a) in values {
            let expected = b.logits[q][id];
            assert!(
                a.is_finite() && (a - expected).abs() <= raw_tolerance,
                "{q}/{id}: {a} versus {expected}, delta {}",
                (a - expected).abs()
            );
        }
        for temperature in [0.75, 1., 2.40605] {
            let a = huncho_core::calibration::calibrate(
                &values.values().copied().collect::<Vec<_>>(),
                temperature,
            )
            .unwrap();
            let b = huncho_core::calibration::calibrate(
                &b.logits[q].values().copied().collect::<Vec<_>>(),
                temperature,
            )
            .unwrap();
            assert_eq!(
                huncho_core::calibration::argmax(&a),
                huncho_core::calibration::argmax(&b)
            );
            assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 1e-4));
        }
    }
}
#[test]
fn native_complete_schemas_keep_original_joint_head_lengths_and_physical_rectangles() {
    let reference = frozen();
    let requests = requests(&reference);
    let inputs: Vec<_> = requests
        .iter()
        .map(|request| RequestBatchInput {
            request,
            max_context: 4096,
        })
        .collect();
    for (dtype, grouped) in [("fp32", false), ("fp16", false), ("fp32", true)] {
        let mut backend = load(dtype, grouped);
        assert!(backend.supports_request_batch() && backend.supports_padded_request_batch());
        assert!(!backend.supports_batch()); // Questions cannot become rows.
        let independent: Vec<_> = requests
            .iter()
            .map(|r| backend.forward_request(r, 4096).unwrap())
            .collect();
        for percent in [0, 10] {
            let mut work = RequestBatchWork::default();
            let outputs = backend
                .forward_request_batch(&inputs, 3000, percent, &mut work)
                .unwrap();
            assert_eq!(work.forward_calls, if percent == 0 { 2 } else { 1 });
            assert_eq!(work.batch_calls, 1);
            assert_eq!(work.prepared_questions, 9);
            let logical: u64 = independent.iter().map(|o| o.input_tokens).sum();
            assert_eq!(work.processed_tokens, logical + work.padded_tokens);
            assert_eq!(work.padded_batch_calls, u64::from(percent > 0));
            assert_eq!(work.padded_tokens, if percent == 0 { 0 } else { 226 });
            assert!(work.padded_tokens * 100 <= work.processed_tokens * percent as u64);
            for (index, (actual, expected)) in outputs.iter().zip(&independent).enumerate() {
                // Match the existing raw dtype fixture bound. Probability
                // delta/argmax gates above remain fixed for both precisions.
                raw_parity(actual, expected, if dtype == "fp32" { 3e-5 } else { 0.005 });
                for (q, values) in &actual.logits {
                    for (label, value) in values {
                        let frozen = reference["cases"][index]["logits"][q][label]
                            .as_f64()
                            .unwrap() as f32;
                        assert!(
                            (value - frozen).abs() <= if dtype == "fp32" { 3e-5 } else { 0.005 }
                        );
                    }
                }
            }
        }
    }
}

fn suite() -> GoldenSuite {
    let reference = frozen();
    let cases = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let request: SystemOneRequest =
                serde_json::from_value(case["request"].clone()).unwrap();
            let mut expected: BTreeMap<String, BTreeMap<String, f32>> =
                serde_json::from_value(case["probabilities"].clone()).unwrap();
            let mut targets = BTreeMap::new();
            for (id, q) in &request.questions {
                if q.type_name() == "noul" {
                    let probabilities = &expected[id];
                    expected.insert(
                        id.clone(),
                        BTreeMap::from([
                            ("no".into(), probabilities["false"]),
                            ("yes".into(), probabilities["true"]),
                        ]),
                    );
                }
                targets.insert(id.clone(), expected[id].keys().next().unwrap().clone());
            }
            GoldenCase {
                id: index.to_string(),
                request,
                expected,
                targets,
            }
        })
        .collect();
    GoldenSuite {
        schema_version: "1.0".into(),
        family: "F5".into(),
        hash: None,
        cases,
    }
}
fn make_engine() -> Engine {
    Engine::new(
        manifest(),
        Box::new(load("fp32", false)),
        Box::new(SimpleTokenizer::new(512)),
        Default::default(),
        BackendId::Clef,
        "fp32",
    )
    .unwrap()
    .with_result_cache(1024 * 1024)
}
#[test]
fn opaque_whole_request_collation_preserves_wire_cache_ownership_and_fresh_fixed_gates() {
    let engine = make_engine();
    let suite = suite();
    let options = EvalOptions {
        max_batch_tokens: Some(3000),
        max_batch_padding_percent: 10,
        prepare_all: true,
        extensions: true,
        ..Default::default()
    };
    let report =
        run_suite_with_cross_request_batches(&engine, &suite, &Default::default(), &options, 3)
            .unwrap();
    assert!(report.passed, "{report:?}");
    assert_eq!(report.work.cross_request_batches, 1);
    assert_eq!(report.work.forward_calls, 1);
    assert_eq!(report.work.prepared_questions, 9);
    assert_eq!(report.work.padded_tokens, 226);
    assert_eq!(report.outcome_calibration.unwrap().questions, 9);
    assert!(report.optimization_parity.unwrap().max_prob_delta <= 1e-4);
    assert_eq!(report.work.result_cache_hits, 0);
    // A one-request run cannot establish actual cross-request batching.
    assert!(run_suite_with_options(&engine, &suite, &Default::default(), &options).is_err());
    let mut incomplete = suite.clone();
    incomplete.cases[0].targets.clear();
    assert!(run_suite_with_cross_request_batches(
        &engine,
        &incomplete,
        &Default::default(),
        &options,
        3
    )
    .is_err());
    let mut drifted = suite.clone();
    drifted.cases[0].expected.insert(
        "a_urgent".into(),
        BTreeMap::from([("yes".into(), 0.999), ("no".into(), 0.001)]),
    );
    assert!(
        !run_suite_with_cross_request_batches(&engine, &drifted, &Default::default(), &options, 3)
            .unwrap()
            .passed
    );
    let prepare = |engine: &Engine| {
        suite
            .cases
            .iter()
            .map(|case| {
                engine
                    .prepare_eval_with_stats(
                        case.request.clone(),
                        options.clone(),
                        &mut Default::default(),
                    )
                    .unwrap()
            })
            .collect()
    };
    let mut work = EvalStats::default();
    let first = engine
        .eval_prepared_batch_with_stats(prepare(&engine), 3000, &mut work)
        .unwrap();
    assert_eq!(work.forward_calls, 1);
    for (output, case) in first.iter().zip(&suite.cases) {
        let original = engine
            .eval(
                &case.request,
                &EvalOptions {
                    extensions: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(output.usage.input_tokens, original.usage.input_tokens);
        assert_eq!(output.usage.output_tokens, original.usage.output_tokens);
        assert_eq!(output.model, original.model);
        assert_eq!(
            output.answers.keys().collect::<Vec<_>>(),
            original.answers.keys().collect::<Vec<_>>()
        );
    }
    let cached = engine
        .eval_prepared_batch_with_stats(prepare(&engine), 3000, &mut work)
        .unwrap();
    assert_eq!(work.forward_calls, 0);
    assert_eq!(work.result_cache_hits, 3);
    assert_eq!(
        serde_json::to_value(cached).unwrap(),
        serde_json::to_value(&first).unwrap()
    );
    let other = make_engine(); // Separate owner cannot submit even cached packets.
    assert!(other
        .eval_prepared_batch_with_stats(prepare(&engine), 3000, &mut work)
        .is_err());
    assert_eq!(work.forward_calls, 0);
    let replica = engine.replica().unwrap();
    assert!(replica
        .eval_prepared_batch_with_stats(prepare(&engine), 3000, &mut work)
        .is_ok());
}

#[test]
fn native_schema_validation_fails_before_any_forward_and_oversized_singletons_stay_intact() {
    let reference = frozen();
    let requests = requests(&reference);
    let mut backend = load("fp32", false);
    let valid = RequestBatchInput {
        request: &requests[0],
        max_context: 4096,
    };
    for (rows, budget, percent) in [
        (vec![], 3000, 0),
        (vec![valid; 65], 3000, 0),
        (vec![valid], 0, 0),
        (vec![valid], 3000, 101),
        (
            vec![
                valid,
                RequestBatchInput {
                    request: &requests[1],
                    max_context: 1,
                },
            ],
            3000,
            10,
        ),
    ] {
        let mut work = RequestBatchWork::default();
        assert!(backend
            .forward_request_batch(&rows, budget, percent, &mut work)
            .is_err());
        assert_eq!(work.forward_calls, 0);
        assert_eq!(work.processed_tokens, 0);
    }
    let mut work = RequestBatchWork::default();
    let output = backend
        .forward_request_batch(&[valid], 1, 0, &mut work)
        .unwrap();
    assert_eq!(work.forward_calls, 1);
    assert_eq!(work.batch_calls, 0);
    assert_eq!(work.processed_tokens, 714);
    assert_eq!(work.padded_tokens, 0);
    let independent = backend.forward_request(&requests[0], 4096).unwrap();
    raw_parity(&output[0], &independent, 3e-5);
}

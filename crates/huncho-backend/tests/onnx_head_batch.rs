//! Strict raw F1 batches preserve frozen independent CPU fixture distributions.
#![cfg(feature = "onnx")]
use huncho_backend::{onnx::OnnxOptions, OnnxBackend};
use huncho_core::{
    backend::{Backend, CacheHandle, ForwardInput, ForwardOutput},
    calibration::{argmax, calibrate},
};
use serde_json::Value;
use std::path::{Path, PathBuf};
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/integrated_f1")
}
fn frozen() -> Value {
    serde_json::from_slice(&std::fs::read(root().join("reference.json")).unwrap()).unwrap()
}
fn input(row: &Value) -> ForwardInput {
    ForwardInput::new(
        serde_json::from_value(row["tokens"].clone()).unwrap(),
        serde_json::from_value(row["positions"].clone()).unwrap(),
    )
    .with_qtype(row["qtype"].as_u64().unwrap() as u32)
}
fn load(graph: &str, budget: usize, shared: bool) -> OnnxBackend {
    OnnxBackend::load_with_options(
        root().join(graph),
        4,
        128,
        "fp32",
        OnnxOptions {
            integrated_head: true,
            native_batch: true,
            output_buffer_bytes: budget,
            shared_initializers: shared,
            intra_threads: 1,
            ..Default::default()
        },
    )
    .unwrap()
}
fn parity(actual: &[f32], expected: &[f32], qtype: u32) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-6),
        "{actual:?} vs {expected:?}"
    );
    if actual.is_empty() {
        return;
    }
    let t = [1.1, 0.8, 2.4][qtype as usize];
    let (a, b) = (
        calibrate(actual, t).unwrap(),
        calibrate(expected, t).unwrap(),
    );
    assert_eq!(argmax(&a), argmax(&b));
    assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 1e-4));
}
#[test]
fn native_raw_scores_keep_types_markers_mask_lengths_and_owned_outputs() {
    let frozen = frozen();
    for shared in if cfg!(feature = "onnx-shared") {
        vec![false, true]
    } else {
        vec![false]
    } {
        for budget in [0, 4096] {
            let mut masked = load("batch-masked.onnx", budget, shared);
            let mut plain = load("batch.onnx", budget, shared);
            assert!(masked.supports_batch() && masked.supports_padded_batch());
            assert!(!plain.supports_padded_batch());
            assert_eq!(
                masked.capabilities().extra["onnx_integrated_head"],
                "graph-integrated-f1-batch-v1"
            );
            for (case, rows) in frozen["readouts"].as_array().unwrap().iter().enumerate() {
                let inputs: Vec<_> = rows.as_array().unwrap().iter().map(input).collect();
                let baselines: Vec<_> = inputs
                    .iter()
                    .map(|input| masked.forward(input.clone()).unwrap())
                    .collect();
                let outputs = masked.forward_padded_batch(inputs.clone()).unwrap();
                for (row, ((output, baseline), input)) in
                    outputs.iter().zip(&baselines).zip(&inputs).enumerate()
                {
                    assert!(matches!(output, ForwardOutput::Logits { .. }));
                    assert_eq!(output.positions(), input.positions);
                    parity(
                        output.values().data(),
                        baseline.values().data(),
                        input.qtype,
                    );
                    let id = ["z_choice", "m_score", "a_noul"][input.qtype as usize];
                    let expected: Vec<f32> = serde_json::from_value(
                        frozen["responses"][case]["extensions"]["raw_logits"][id].clone(),
                    )
                    .unwrap();
                    parity(output.values().data(), &expected, input.qtype);
                    assert_eq!(outputs[row].values().shape(), &[input.positions.len(), 1]);
                }
                let saved = outputs[0].values().data().to_vec();
                let mut reordered = inputs[0].clone();
                reordered.positions = vec![
                    inputs[0].positions[1],
                    inputs[0].positions[0],
                    inputs[0].positions[1],
                ];
                let mut empty = inputs[1].clone();
                empty.positions.clear();
                let another = masked
                    .forward_padded_batch(vec![reordered.clone(), empty])
                    .unwrap();
                let scalar = masked.forward(reordered).unwrap();
                parity(
                    another[0].values().data(),
                    scalar.values().data(),
                    inputs[0].qtype,
                );
                assert_eq!(another[1].values().shape(), &[0, 1]);
                assert_eq!(outputs[0].values().data(), saved);
                // Equal lengths, different types and marker counts, including duplicates.
                let mut equal = vec![inputs[0].clone(); 3];
                for (index, input) in equal.iter_mut().enumerate() {
                    input.qtype = index as u32;
                }
                equal[1].positions.reverse();
                equal[2].positions.truncate(1);
                let independent: Vec<_> = equal
                    .iter()
                    .map(|i| plain.forward(i.clone()).unwrap())
                    .collect();
                let group = plain.forward_batch(equal.clone()).unwrap();
                for ((out, baseline), input) in group.iter().zip(&independent).zip(equal) {
                    parity(out.values().data(), baseline.values().data(), input.qtype);
                }
                if inputs
                    .iter()
                    .all(|i| i.tokens.len() == inputs[0].tokens.len())
                {
                    let group = plain.forward_batch(inputs.clone()).unwrap();
                    for ((out, expected), input) in group.iter().zip(&baselines).zip(&inputs) {
                        parity(out.values().data(), expected.values().data(), input.qtype);
                    }
                } else {
                    assert!(plain.forward_batch(inputs.clone()).is_err());
                }
                assert!(plain.forward_padded_batch(inputs).is_err());
            }
            if budget > 0 {
                assert!(masked.output_buffer_reuses() > 0);
            }
            assert!(masked.retained_output_bytes() <= budget);
            if shared {
                let mut replica = masked.replica().unwrap();
                let input = input(&frozen["readouts"][0][0]);
                let expected = masked.forward(input.clone()).unwrap();
                drop(masked);
                for out in replica.forward_batch(vec![input.clone(); 2]).unwrap() {
                    parity(out.values().data(), expected.values().data(), input.qtype);
                }
            }
        }
    }
}
#[test]
fn native_head_batches_reject_wrong_abis_invalid_types_counts_and_nonfinite_scores() {
    for (graph, native) in [("model.onnx", true), ("batch.onnx", false)] {
        assert!(OnnxBackend::load_with_options(
            root().join(graph),
            4,
            128,
            "fp32",
            OnnxOptions {
                integrated_head: true,
                native_batch: native,
                ..Default::default()
            }
        )
        .is_err());
    }
    let mut backend = load("batch-masked.onnx", 4096, false);
    let input = input(&frozen()["readouts"][0][0]);
    let expected = backend.forward(input.clone()).unwrap();
    let mut invalids = vec![
        vec![],
        vec![input.clone(); 65],
        vec![ForwardInput::new(vec![], vec![])],
        vec![ForwardInput::new(vec![1; 129], vec![0])],
        vec![ForwardInput::new(vec![1], vec![1])],
    ];
    for kind in 0..5 {
        let mut bad = input.clone();
        match kind {
            0 => bad.qtype = 3,
            1 => bad.retain_cache = true,
            2 => bad.fork_from = Some(CacheHandle { id: 1 }),
            3 => bad.logit_codes = Some(vec![0]),
            _ => bad.positions = vec![0; 8193],
        }
        invalids.push(vec![input.clone(), bad]);
    }
    for invalid in invalids {
        assert!(backend.forward_padded_batch(invalid).is_err());
        parity(
            backend.forward(input.clone()).unwrap().values().data(),
            expected.values().data(),
            input.qtype,
        );
    }
    let mut bad = load("batch-nonfinite.onnx", 4096, false);
    assert!(bad.forward_batch(vec![input; 2]).is_err());
}

#[cfg(feature = "clef")]
#[test]
fn native_head_engine_qualifies_frozen_typed_requests_and_cross_request_padding() {
    use huncho_core::{
        conformance::{run_suite_with_cross_request_batches, run_suite_with_options, GoldenSuite},
        engine::{Engine, EvalOptions},
        head::HeadParams,
        manifest::{BackendId, ModelManifest},
        tokenizer::HfTokenizer,
    };
    let manifest = ModelManifest::load(root().join("huncho-model.json")).unwrap();
    let suite: GoldenSuite =
        serde_json::from_slice(&std::fs::read(root().join("golden.json")).unwrap()).unwrap();
    let engine = Engine::new(
        manifest,
        Box::new(load("batch-masked.onnx", 4096, false)),
        Box::new(HfTokenizer::from_file_unbounded(root().join("tokenizer.json")).unwrap()),
        HeadParams::scalar_linear(1, vec![1.], 0.).unwrap(),
        BackendId::Onnx,
        "fp32",
    )
    .unwrap();
    let options = EvalOptions {
        max_batch_tokens: Some(4096),
        max_batch_padding_percent: 50,
        ..Default::default()
    };
    let report = run_suite_with_options(&engine, &suite, &Default::default(), &options).unwrap();
    assert!(report.passed, "{report:?}");
    assert!(report.work.padded_batch_calls > 0 && report.work.padded_tokens > 0);
    assert!(report.optimization_parity.unwrap().max_prob_delta <= 1e-4);
    let mut options = options;
    options.prepare_all = true;
    let report =
        run_suite_with_cross_request_batches(&engine, &suite, &Default::default(), &options, 4)
            .unwrap();
    assert!(report.passed, "{report:?}");
    assert!(report.work.cross_request_batches > 0);
    assert_eq!(report.outcome_calibration.unwrap().questions, 12);
}

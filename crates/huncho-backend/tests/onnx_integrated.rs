//! Actual integrated F1 CPU graphs, with frozen independent synthetic scores.
#![cfg(feature = "onnx")]
use huncho_backend::{
    onnx::{OnnxExecutionProvider, OnnxOptions},
    OnnxBackend,
};
use huncho_core::backend::{Backend, CacheHandle, ForwardInput, ForwardOutput};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/integrated_f1")
}
fn load(file: &str, budget: usize) -> OnnxBackend {
    OnnxBackend::load_with_options(
        root().join(file),
        4,
        128,
        "fp32",
        OnnxOptions {
            integrated_head: true,
            output_buffer_bytes: budget,
            intra_threads: 1,
            ..Default::default()
        },
    )
    .unwrap()
}
fn frozen() -> Value {
    serde_json::from_slice(&std::fs::read(root().join("reference.json")).unwrap()).unwrap()
}
fn input(value: &Value) -> ForwardInput {
    ForwardInput::new(
        serde_json::from_value(value["tokens"].clone()).unwrap(),
        serde_json::from_value(value["positions"].clone()).unwrap(),
    )
    .with_qtype(value["qtype"].as_u64().unwrap() as u32)
}

#[test]
fn raw_typed_graph_scores_match_independent_native_frozen_reference_and_reuse_owned_outputs() {
    let reference = frozen();
    for graph in ["model.onnx", "masked.onnx"] {
        for budget in [0, 64] {
            let mut backend = load(graph, budget);
            assert!(!backend.supports_batch());
            assert_eq!(
                backend.capabilities().extra["onnx_integrated_head"],
                "graph-integrated-f1-v1"
            );
            let mut saved = None;
            for (index, readouts) in reference["readouts"].as_array().unwrap().iter().enumerate() {
                for readout in readouts.as_array().unwrap() {
                    let result = backend.forward(input(readout)).unwrap();
                    assert!(matches!(result, ForwardOutput::Logits { .. }));
                    assert_eq!(result.values().shape(), &[result.positions().len(), 1]);
                    let qtype = readout["qtype"].as_u64().unwrap();
                    let id = match qtype {
                        0 => "z_choice",
                        1 => "m_score",
                        2 => "a_noul",
                        _ => unreachable!(),
                    };
                    let expected: Vec<f32> = serde_json::from_value(
                        reference["responses"][index]["extensions"]["raw_logits"][id].clone(),
                    )
                    .unwrap();
                    assert!(result
                        .values()
                        .data()
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-6));
                    if saved.is_none() {
                        saved = Some(result);
                    }
                }
            }
            let first = &reference["readouts"][0][0];
            let expected = backend.forward(input(first)).unwrap();
            let mut reordered = input(first);
            reordered.positions = vec![
                first["positions"][1].as_u64().unwrap() as usize,
                first["positions"][0].as_u64().unwrap() as usize,
                first["positions"][1].as_u64().unwrap() as usize,
            ];
            let actual = backend.forward(reordered.clone()).unwrap();
            for (row, &position) in reordered.positions.iter().enumerate() {
                let index = expected
                    .positions()
                    .iter()
                    .position(|&p| p == position)
                    .unwrap();
                assert_eq!(
                    actual.values().row(row).unwrap(),
                    expected.values().row(index).unwrap()
                );
            }
            let previous = saved.unwrap();
            assert_eq!(
                previous.values().data(),
                expected.values().data(),
                "returned data must own its snapshot across output-buffer reuse"
            );
            let mut empty = input(first);
            empty.positions.clear();
            assert_eq!(backend.forward(empty).unwrap().values().shape(), &[0, 1]);
            if budget > 0 {
                assert!(backend.output_buffer_reuses() > 0);
                assert!(backend.retained_output_bytes() <= budget);
            } else {
                assert_eq!(backend.retained_output_bytes(), 0);
            }
        }
    }
}

#[test]
fn integrated_contract_rejects_unsupported_profiles_bad_graphs_and_inputs_without_gpu_probes() {
    for options in [
        OnnxOptions {
            integrated_head: true,
            compact_readout: true,
            ..Default::default()
        },
        OnnxOptions {
            integrated_head: true,
            execution_provider: OnnxExecutionProvider::Cuda { device: 0 },
            ..Default::default()
        },
    ] {
        assert!(
            OnnxBackend::load_with_options("absent.onnx", 4, 128, "fp32", options)
                .err()
                .unwrap()
                .to_string()
                .contains("CPU fp32")
        );
    }
    assert!(OnnxBackend::load_with_options(
        "absent.onnx",
        4,
        128,
        "fp16",
        OnnxOptions {
            integrated_head: true,
            ..Default::default()
        }
    )
    .is_err());
    for graph in [
        root().join("wrong-output.onnx"),
        root().parent().unwrap().join("tiny_encoder.onnx"),
    ] {
        assert!(OnnxBackend::load_with_options(
            graph,
            4,
            128,
            "fp32",
            OnnxOptions {
                integrated_head: true,
                ..Default::default()
            }
        )
        .is_err());
    }
    let mut backend = load("model.onnx", 64);
    let first = input(&frozen()["readouts"][0][0]);
    let mut invalids = Vec::new();
    let mut bad = first.clone();
    bad.positions = vec![bad.tokens.len()];
    invalids.push(bad);
    let mut bad = first.clone();
    bad.qtype = 3;
    invalids.push(bad);
    let mut bad = first.clone();
    bad.retain_cache = true;
    invalids.push(bad);
    let mut bad = first.clone();
    bad.fork_from = Some(CacheHandle { id: 1 });
    invalids.push(bad);
    let mut bad = first.clone();
    bad.logit_codes = Some(vec![0]);
    invalids.push(bad);
    let mut bad = first.clone();
    bad.tokens.resize(129, 0);
    invalids.push(bad);
    invalids.push(ForwardInput::new(vec![], vec![]));
    for bad in invalids {
        assert!(backend.forward(bad).is_err());
    }
    assert!(backend
        .forward_batch(vec![first.clone(), first.clone()])
        .is_err());
    let mut nonfinite = load("nonfinite.onnx", 64);
    assert!(nonfinite
        .forward(first.clone())
        .err()
        .unwrap()
        .to_string()
        .contains("nonfinite"));
    assert!(backend.forward(first).is_ok());
}

#[cfg(feature = "clef")]
#[test]
fn actual_native_typed_calibration_passes_fixed_synthetic_vectors_and_rejects_drift() {
    use huncho_core::{
        conformance::{run_suite, GoldenSuite},
        engine::{Engine, EvalOptions},
        manifest::{BackendId, ModelManifest},
        tokenizer::HfTokenizer,
    };
    let manifest: ModelManifest =
        serde_json::from_slice(&std::fs::read(root().join("huncho-model.json")).unwrap()).unwrap();
    let suite: GoldenSuite =
        serde_json::from_slice(&std::fs::read(root().join("golden.json")).unwrap()).unwrap();
    let engine = Engine::new(
        manifest,
        Box::new(load("model.onnx", 64)),
        Box::new(HfTokenizer::from_file(root().join("tokenizer.json")).unwrap()),
        Default::default(),
        BackendId::Onnx,
        "fp32",
    )
    .unwrap();
    let report = run_suite(&engine, &suite, &Default::default()).unwrap();
    assert!(report.passed && report.max_prob_delta <= 1e-6);
    assert_eq!(report.outcome_calibration.unwrap().questions, 12);
    let reference = frozen();
    for (index, case) in suite.cases.iter().enumerate() {
        let actual = engine
            .eval(
                &case.request,
                &EvalOptions {
                    extensions: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let expected = &reference["responses"][index];
        assert_eq!(
            actual.usage.input_tokens,
            expected["usage"]["input_tokens"].as_u64().unwrap()
        );
        assert_eq!(actual.usage.output_tokens, 0);
        assert_eq!(actual.answers.len(), 3);
        for (id, values) in actual.extensions.unwrap().raw_logits.unwrap() {
            let expected: Vec<f32> =
                serde_json::from_value(expected["extensions"]["raw_logits"][&id].clone()).unwrap();
            assert!(values
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() <= 1e-6));
        }
    }
    let mut incomplete = suite.clone();
    incomplete.cases[0].targets.remove("a_noul");
    assert!(run_suite(&engine, &incomplete, &Default::default()).is_err());
    let mut drifted = suite.clone();
    drifted.cases[0].expected.insert(
        "z_choice".into(),
        std::collections::BTreeMap::from([
            ("shipping".into(), 0.999),
            ("billing".into(), 0.0005),
            ("returns".into(), 0.0005),
        ]),
    );
    assert!(
        !run_suite(&engine, &drifted, &Default::default())
            .unwrap()
            .passed
    );
}

#[cfg(feature = "onnx-shared")]
#[test]
fn integrated_shared_replicas_keep_raw_scores_and_independent_buffers_after_primary_drop() {
    let dir = tempfile::tempdir().unwrap();
    let graph = dir.path().join("model.onnx");
    std::fs::copy(root().join("masked.onnx"), &graph).unwrap();
    let primary = OnnxBackend::load_with_options(
        &graph,
        4,
        128,
        "fp32",
        OnnxOptions {
            integrated_head: true,
            shared_initializers: true,
            output_buffer_bytes: 64,
            intra_threads: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let identity = primary.capabilities().extra;
    assert!(
        identity["onnx_shared_initializer_bytes"]
            .parse::<usize>()
            .unwrap()
            > 0
    );
    let replicas = (0..3)
        .map(|_| primary.replica().unwrap())
        .collect::<Vec<_>>();
    std::fs::write(&graph, b"source replaced after snapshot").unwrap();
    drop(primary);
    let handles = replicas
        .into_iter()
        .map(|mut replica| {
            let identity = identity.clone();
            std::thread::spawn(move || {
                assert_eq!(replica.capabilities().extra, identity);
                let reference = frozen();
                let mut first = None;
                for readouts in reference["readouts"].as_array().unwrap() {
                    for readout in readouts.as_array().unwrap() {
                        let output = replica.forward(input(readout)).unwrap();
                        assert!(output.values().data().iter().all(|v| v.is_finite()));
                        if first.is_none() {
                            first = Some(output);
                        }
                    }
                }
                let next = replica
                    .forward(input(&reference["readouts"][0][0]))
                    .unwrap();
                assert_eq!(first.unwrap().values().data(), next.values().data());
                next.values().data().to_vec()
            })
        })
        .collect::<Vec<_>>();
    let outputs = handles
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect::<Vec<_>>();
    assert!(outputs.iter().all(|v| v == &outputs[0]));
}

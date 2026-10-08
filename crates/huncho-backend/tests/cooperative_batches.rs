//! Actual CPU Kev scheduling over unchanged independent fixture probabilities.
//! Labels exercise gate plumbing only, never released outcome acceptance.
#![cfg(feature = "clef")]
use huncho_backend::Qwen3_5Backend;
use huncho_core::{
    backend::Backend,
    conformance::{GoldenCase, GoldenSuite},
    engine::{Engine, EvalOptions, EvalStats},
    manifest::{BackendId, ModelManifest},
    tokenizer::HfTokenizer,
};
use std::{collections::BTreeMap, path::Path};

fn fixture() -> (ModelManifest, GoldenSuite) {
    let root = Path::new("tests/fixtures/tiny_kev");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(
        &std::fs::read("../../examples/mock-model/huncho-model.json").unwrap(),
    )
    .unwrap();
    value["name"] = serde_json::json!("tiny-kev");
    value["family"] = serde_json::json!("F2");
    value["head"] = serde_json::json!({"kind":"pointer","weights":"head.pt","width":4});
    value["prompt_contract"]["template"] = serde_json::json!("kev-v1");
    value["prompt_contract"]["state_budget"] = serde_json::json!(512);
    value["prompt_contract"]["head_budget"] = serde_json::json!(512);
    value["calibration"] = serde_json::json!({"default":{"temperature":2.40605,"confidence":"peak","status":"fit"},"entries":{}});
    let manifest: ModelManifest = serde_json::from_value(value).unwrap();
    let formatter = huncho_core::prompt::formatter_for(&manifest);
    let tokenizer = HfTokenizer::from_file_unbounded(root.join("tokenizer.json")).unwrap();
    let cases = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, case)| {
            let request: huncho_core::contract::SystemOneRequest =
                serde_json::from_value(case["request"].clone()).unwrap();
            let mut expected: BTreeMap<String, BTreeMap<String, f32>> = BTreeMap::new();
            for ((id, question), row) in request
                .questions
                .iter()
                .zip(case["rows"].as_array().unwrap())
            {
                let prompt = formatter
                    .build(&request.state, question, &tokenizer)
                    .unwrap();
                let probabilities: Vec<f32> =
                    serde_json::from_value(row["probabilities"].clone()).unwrap();
                expected.insert(
                    id.clone(),
                    prompt
                        .candidates
                        .into_iter()
                        .map(|c| c.label)
                        .zip(probabilities)
                        .collect(),
                );
            }
            let targets = expected
                .iter()
                .map(|(id, p)| (id.clone(), p.keys().next().unwrap().clone()))
                .collect();
            GoldenCase {
                id: i.to_string(),
                request,
                expected,
                targets,
            }
        })
        .collect();
    (
        manifest,
        GoldenSuite {
            schema_version: "1.0".into(),
            family: "F2".into(),
            hash: None,
            cases,
        },
    )
}
fn backend(dtype: &str, profile: usize) -> Qwen3_5Backend {
    let root = Path::new("tests/fixtures/tiny_kev");
    let backend = if profile == 4 {
        Qwen3_5Backend::load_kev_runtime_lora(root, root, &root.join("head.pt"), 512, dtype)
    } else {
        Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype)
    }
    .unwrap();
    let mut backend = backend.with_prefill_chunk_tokens(3).unwrap();
    if profile > 0 {
        backend = backend
            .with_attention_query_rows(7)
            .unwrap()
            .with_grouped_gqa(true)
            .unwrap()
            .with_cpu_delta_rule(true)
            .unwrap()
            .with_cpu_causal_conv(true)
            .unwrap()
            .with_cpu_fused_gate(true)
            .unwrap();
    }
    if profile >= 2 {
        backend = backend.with_kv_page_tokens(16).unwrap();
    }
    if profile == 3 {
        backend = backend.with_direct_paged_attention(true).unwrap();
    }
    backend
}
fn engine(manifest: ModelManifest, backend: Qwen3_5Backend, dtype: &str) -> Engine {
    Engine::new(
        manifest,
        Box::new(backend),
        Box::new(
            HfTokenizer::from_file_unbounded("tests/fixtures/tiny_kev/tokenizer.json").unwrap(),
        ),
        Default::default(),
        BackendId::Candle,
        dtype,
    )
    .unwrap()
}
fn options(padding: usize) -> EvalOptions {
    EvalOptions {
        prefix_cache: true,
        cooperative_prefill: true,
        prepare_all: true,
        max_batch_tokens: Some(4096),
        max_batch_padding_percent: padding,
        extensions: true,
        ..Default::default()
    }
}

#[test]
fn native_equal_and_padded_cooperative_groups_keep_frozen_probabilities_and_fresh_gates() {
    let (manifest, original) = fixture();
    for dtype in ["fp32", "fp16"] {
        for profile in 0..=4 {
            if dtype == "fp16" && profile >= 3 {
                continue;
            }
            for padding in [0, 25] {
                let engine = engine(manifest.clone(), backend(dtype, profile), dtype);
                let mut suite = original.clone();
                if padding == 0 {
                    for case in &mut suite.cases {
                        let duplicates: Vec<_> = case
                            .request
                            .questions
                            .iter()
                            .map(|(id, q)| (id.clone(), q.clone()))
                            .collect();
                        for (id, question) in duplicates {
                            let duplicate = format!("{id}-duplicate");
                            case.request.questions.insert(duplicate.clone(), question);
                            case.expected
                                .insert(duplicate.clone(), case.expected[&id].clone());
                            case.targets.insert(duplicate, case.targets[&id].clone());
                        }
                    }
                }
                let opts = options(padding);
                let report = engine.qualify_for_serving(&suite, &opts, None).unwrap();
                assert!(report.passed, "{dtype}/{profile}/{padding}: {report:?}");
                assert!(report.work.prefill_yields > 0 && report.work.prefill_interleaves > 0);
                assert!(report.work.fork_batch_calls > 0);
                if padding > 0 {
                    assert!(
                        report.work.fork_padded_batch_calls > 0 && report.work.padded_tokens > 0
                    );
                }
                assert!(report.optimization_parity.unwrap().max_prob_delta <= 1e-4);
                let mut work = EvalStats::default();
                let actual = engine
                    .eval_for_serving_with_stats(&suite.cases[0].request, &opts, &mut work)
                    .unwrap();
                let baseline = engine
                    .eval(
                        &suite.cases[0].request,
                        &EvalOptions {
                            extensions: true,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(actual.usage.input_tokens, baseline.usage.input_tokens);
                assert!(work.fork_batch_calls > 0 && work.cache_forks > 0);
                let mut too_small = opts.clone();
                too_small.max_batch_tokens = Some(1);
                assert!(engine
                    .qualify_for_serving(&suite, &too_small, None)
                    .unwrap_err()
                    .to_string()
                    .contains("actual native batch"));
                assert!(engine.require_serving_qualification(&opts, None).is_err());
            }
        }
    }
}

#[test]
fn live_partial_parents_shrink_native_groups_without_leaks_or_probability_drift() {
    let (manifest, suite) = fixture();
    let opts = options(25);
    for occupied in [0, 61, 62] {
        let mut backend = backend("fp32", 2);
        for _ in 0..occupied {
            backend.begin_resumable_prefill(&[1, 2, 3, 4], 0).unwrap();
        }
        assert_eq!(
            backend.fork_batch_limits().max_rows,
            (64usize - occupied).min(63)
        );
        let engine = engine(manifest.clone(), backend, "fp32");
        let request = &suite.cases[0].request;
        // More than the handle cap: dropping partial request parents must
        // leave all pre-existing partial jobs alive and reclaim only this job.
        for _ in 0..67 {
            let prepared = engine
                .prepare_eval_uncached_with_stats(
                    request.clone(),
                    opts.clone(),
                    &mut Default::default(),
                )
                .unwrap();
            let mut cursor = engine.begin_resumable_evaluation(prepared).unwrap();
            assert!(engine
                .advance_resumable_evaluation(&mut cursor, &mut Default::default())
                .unwrap()
                .is_none());
            drop(cursor);
        }
        let mut work = EvalStats::default();
        let actual = engine.eval_with_stats(request, &opts, &mut work).unwrap();
        assert_eq!(work.cache_forks, request.questions.len() as u64);
        if occupied == 62 {
            assert_eq!(work.fork_batch_calls, 0);
            assert_eq!(work.forward_calls, request.questions.len() as u64);
        } else {
            assert!(work.fork_batch_calls > 0);
        }
        let expected = engine.eval(request, &EvalOptions::default()).unwrap();
        for (id, answer) in &actual.answers {
            let a = serde_json::to_value(answer).unwrap();
            let b = serde_json::to_value(&expected.answers[id]).unwrap();
            let probabilities = |value: &serde_json::Value| -> Vec<f64> {
                if value.get("probabilities").is_some() {
                    value["probabilities"]
                        .as_object()
                        .unwrap()
                        .values()
                        .map(|p| p.as_f64().unwrap())
                        .collect()
                } else {
                    vec![value["noul"].as_f64().unwrap()]
                }
            };
            assert!(probabilities(&a)
                .iter()
                .zip(probabilities(&b))
                .all(|(a, b)| (a - b).abs() <= 1e-4));
        }
    }
}

#[test]
fn large_qualification_cohorts_leave_live_room_for_actual_native_groups() {
    let (manifest, mut suite) = fixture();
    let first = suite.cases[0].clone();
    suite.cases = (0..65)
        .map(|i| {
            let mut case = first.clone();
            case.id = format!("frozen-copy-{i}");
            case
        })
        .collect();
    let engine = engine(manifest, backend("fp32", 2), "fp32");
    let report = engine
        .qualify_for_serving(&suite, &options(25), None)
        .unwrap();
    assert!(report.passed, "{report:?}");
    assert!(report.work.fork_batch_calls >= 65);
    assert!(report.work.prefill_interleaves > 0);
    assert_eq!(
        report.outcome_calibration.unwrap().questions,
        65 * first.request.questions.len()
    );
    assert!(report.optimization_parity.unwrap().max_prob_delta <= 1e-4);
}

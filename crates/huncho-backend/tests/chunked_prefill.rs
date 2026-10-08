//! CPU chunk boundaries advance KV, recurrence and convolution together.
#![cfg(feature = "candle")]
use huncho_backend::Qwen3_5Backend;
use huncho_core::backend::{Backend, ForwardInput, PrefillWork};
use huncho_core::calibration::{argmax, calibrate};
use std::path::Path;

#[test]
fn chunked_cpu_prefill_preserves_original_probabilities_and_accounts_for_actual_calls() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for dtype in ["fp32", "fp16"] {
        let load =
            || Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype).unwrap();
        assert!(load().with_prefill_chunk_tokens(4097).is_err());
        for chunk_size in [0, 1, 3, 16, 4096] {
            let mut baseline = load();
            let mut backend = load()
                .with_cpu_delta_rule(true)
                .unwrap()
                .with_cpu_causal_conv(true)
                .unwrap()
                .with_prefill_chunk_tokens(chunk_size)
                .unwrap();
            assert_eq!(
                backend.capabilities().extra.get("prefill_chunk_tokens"),
                (chunk_size > 0).then(|| chunk_size.to_string()).as_ref()
            );
            let mut invalid_work = PrefillWork::default();
            assert!(backend
                .prefill_cached_with_work(&[u32::MAX], 1 << 20, &mut invalid_work)
                .is_err());
            assert_eq!(invalid_work.forward_calls, 0);
            assert_eq!(invalid_work.processed_tokens, 0);
            for case in golden["cases"].as_array().unwrap() {
                let rows = case["rows"].as_array().unwrap();
                let first: Vec<u32> = serde_json::from_value(rows[0]["tokens"].clone()).unwrap();
                let prefix = rows[0]["prefix_len"].as_u64().unwrap() as usize;
                let mut work = PrefillWork::default();
                let cached = backend
                    .prefill_cached_with_work(&first[..prefix], 1 << 20, &mut work)
                    .unwrap();
                let calls = if chunk_size == 0 {
                    1
                } else {
                    prefix.div_ceil(chunk_size)
                };
                assert!(!cached.hit);
                assert_eq!(work.forward_calls, calls as u64);
                assert_eq!(work.processed_tokens, prefix as u64);
                assert_eq!(work.chunked_prefills, u64::from(calls > 1));
                backend.release_cache(cached.handle).unwrap();
                let mut hit_work = PrefillWork::default();
                let hit = backend
                    .prefill_cached_with_work(&first[..prefix], 1 << 20, &mut hit_work)
                    .unwrap();
                assert!(hit.hit);
                assert_eq!(hit_work.forward_calls, 0);
                assert_eq!(hit_work.processed_tokens, 0);
                assert_eq!(hit_work.chunked_prefills, 0);
                for row in rows {
                    let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
                    let positions: Vec<usize> =
                        serde_json::from_value(row["positions"].clone()).unwrap();
                    let expected = baseline
                        .forward(ForwardInput::new(tokens.clone(), positions.clone()))
                        .unwrap();
                    let fork = backend.fork(hit.handle).unwrap();
                    let mut suffix = ForwardInput::new(
                        tokens[prefix..].to_vec(),
                        positions.iter().map(|p| p - prefix).collect(),
                    );
                    suffix.fork_from = Some(fork);
                    let actual = backend.forward(suffix).unwrap();
                    for temperature in [0.75, 1.0, 2.40605] {
                        let a = calibrate(actual.values().data(), temperature).unwrap();
                        let b = calibrate(expected.values().data(), temperature).unwrap();
                        assert_eq!(argmax(&a), argmax(&b));
                        assert!(
                            a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= 1e-4),
                            "{dtype}/{chunk_size}: {a:?} vs {b:?}"
                        );
                    }
                    backend.release_cache(fork).unwrap();
                }
                backend.release_cache(hit.handle).unwrap();
                backend.clear_prefix_cache().unwrap();
            }
            let retained = backend.prefill_cached(&[1, 2, 3], 1 << 20).unwrap().handle;
            backend.release_cache(retained).unwrap();
            assert!(backend
                .with_prefill_chunk_tokens(if chunk_size == 0 { 1 } else { 0 })
                .is_err());
        }
    }
}

#[cfg(feature = "clef")]
#[test]
fn engine_qualification_requires_actual_chunking_and_keeps_original_goldens_and_usage() {
    use huncho_core::conformance::{GoldenCase, GoldenSuite};
    use huncho_core::engine::{Engine, EvalOptions, EvalStats};
    use huncho_core::manifest::{BackendId, ModelManifest};
    use huncho_core::tokenizer::HfTokenizer;
    use std::collections::BTreeMap;
    let root = Path::new("tests/fixtures/tiny_kev");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let mut json: serde_json::Value = serde_json::from_slice(
        &std::fs::read("../../examples/mock-model/huncho-model.json").unwrap(),
    )
    .unwrap();
    json["name"] = serde_json::json!("tiny-kev");
    json["family"] = serde_json::json!("F2");
    json["head"] = serde_json::json!({"kind":"pointer","weights":"head.pt","width":4});
    json["prompt_contract"]["template"] = serde_json::json!("kev-v1");
    json["prompt_contract"]["state_budget"] = serde_json::json!(512);
    json["prompt_contract"]["head_budget"] = serde_json::json!(512);
    json["calibration"] = serde_json::json!({"default":{"temperature":2.40605,"confidence":"peak","status":"fit"},"entries":{}});
    let manifest: ModelManifest = serde_json::from_value(json).unwrap();
    let formatter = huncho_core::prompt::formatter_for(&manifest);
    let tokenizer = HfTokenizer::from_file_unbounded(root.join("tokenizer.json")).unwrap();
    let mut cases = Vec::new();
    for (index, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        let request: huncho_core::contract::SystemOneRequest =
            serde_json::from_value(case["request"].clone()).unwrap();
        let mut expected = BTreeMap::new();
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
        cases.push(GoldenCase {
            id: index.to_string(),
            request,
            expected,
            targets: Default::default(),
        });
    }
    let suite = GoldenSuite {
        schema_version: "1.0".into(),
        family: "F2".into(),
        hash: None,
        cases,
    };
    for chunk_size in [3, 4096] {
        let backend = Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, "fp32")
            .unwrap()
            .with_prefill_chunk_tokens(chunk_size)
            .unwrap();
        let engine = Engine::new(
            manifest.clone(),
            Box::new(backend),
            Box::new(HfTokenizer::from_file_unbounded(root.join("tokenizer.json")).unwrap()),
            Default::default(),
            BackendId::Candle,
            "fp32",
        )
        .unwrap();
        let options = EvalOptions {
            prefix_cache: true,
            ..Default::default()
        };
        let report = huncho_core::conformance::run_suite_with_options(
            &engine,
            &suite,
            &Default::default(),
            &options,
        );
        if chunk_size == 4096 {
            assert!(report
                .unwrap_err()
                .to_string()
                .contains("actual prefix split"));
            continue;
        }
        let report = report.unwrap();
        assert!(report.passed, "{report:?}");
        assert!(report.work.chunked_prefills > 0);
        assert!(report.work.prefill_calls > report.work.chunked_prefills);
        assert!(report.optimization_parity.unwrap().max_prob_delta <= 1e-4);
        let request = &suite.cases[0].request;
        let (mut base_work, mut optimized_work) = (EvalStats::default(), EvalStats::default());
        let baseline = engine
            .eval_with_stats(request, &EvalOptions::default(), &mut base_work)
            .unwrap();
        let optimized = engine
            .eval_with_stats(request, &options, &mut optimized_work)
            .unwrap();
        assert_eq!(baseline.usage.input_tokens, optimized.usage.input_tokens);
        assert!(optimized_work.processed_tokens < base_work.processed_tokens);
        assert!(optimized_work.chunked_prefills > 0);

        let cooperative = EvalOptions {
            prefix_cache: true,
            cooperative_prefill: true,
            prepare_all: true,
            extensions: true,
            ..Default::default()
        };
        let report = huncho_core::conformance::run_suite_with_options(
            &engine,
            &suite,
            &Default::default(),
            &cooperative,
        )
        .unwrap();
        assert!(report.passed, "{report:?}");
        assert!(report.cooperative_prefill);
        assert!(report.work.prefill_yields > 0);
        assert!(report.work.prefill_interleaves > 0);
        assert!(report.optimization_parity.unwrap().max_prob_delta <= 1e-4);
        let retained_options = EvalOptions {
            persistent_prefix_bytes: 1 << 20,
            ..cooperative.clone()
        };
        let retained_report = huncho_core::conformance::run_suite_with_options(
            &engine,
            &suite,
            &Default::default(),
            &retained_options,
        )
        .unwrap();
        assert!(retained_report.passed, "{retained_report:?}");
        assert!(retained_report.work.persistent_prefix_hits > 0);
        let mut singleton = suite.clone();
        singleton.cases.truncate(1);
        assert!(huncho_core::conformance::run_suite_with_options(
            &engine,
            &singleton,
            &Default::default(),
            &cooperative
        )
        .unwrap_err()
        .to_string()
        .contains("interleaved"));
        let mut ordinary = cooperative.clone();
        ordinary.cooperative_prefill = false;
        let serial = engine.eval(request, &ordinary).unwrap();
        let mut work = EvalStats::default();
        let resumed = engine
            .eval_with_stats(request, &cooperative, &mut work)
            .unwrap();
        assert_eq!(
            serde_json::to_vec(&serial).unwrap(),
            serde_json::to_vec(&resumed).unwrap()
        );
        assert!(work.prefill_yields > 0);
        let replica = engine.replica().unwrap();
        for _ in 0..67 {
            let prepared = engine
                .prepare_eval_uncached_with_stats(
                    request.clone(),
                    cooperative.clone(),
                    &mut Default::default(),
                )
                .unwrap();
            let mut cursor = engine.begin_resumable_evaluation(prepared).unwrap();
            assert!(replica
                .advance_resumable_evaluation(&mut cursor, &mut Default::default())
                .is_err());
            assert!(engine
                .advance_resumable_evaluation(&mut cursor, &mut Default::default())
                .unwrap()
                .is_none());
            // Drop an incomplete prefix repeatedly; no handles may leak.
            drop(cursor);
        }
        assert_eq!(
            serde_json::to_vec(&serial).unwrap(),
            serde_json::to_vec(&engine.eval(request, &cooperative).unwrap()).unwrap()
        );
    }
}

#[test]
fn resumable_native_prefixes_interleave_without_changing_chunk_arithmetic() {
    let root = Path::new("tests/fixtures/tiny_kev");
    for dtype in ["fp32", "fp16"] {
        let load = || {
            Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype)
                .unwrap()
                .with_prefill_chunk_tokens(2)
                .unwrap()
        };
        let mut serial = load();
        let mut resumed = load();
        let prefixes = [vec![1, 2, 3, 4, 5], vec![5, 4, 3, 2, 1]];
        let handles: Vec<_> = prefixes
            .iter()
            .map(|tokens| {
                resumed
                    .begin_resumable_prefill(tokens, 1 << 20)
                    .unwrap()
                    .handle
            })
            .collect();
        let mut replica = resumed.replica().unwrap();
        let mut invalid = PrefillWork::default();
        assert!(replica
            .advance_resumable_prefill(handles[0], &mut invalid)
            .is_err());
        assert_eq!(invalid.forward_calls, 0);
        for handle in &handles {
            assert!(resumed.fork(*handle).is_err());
            let mut readout = ForwardInput::new(vec![1], vec![0]);
            readout.fork_from = Some(*handle);
            assert!(resumed.forward(readout).is_err());
        }
        let mut work = PrefillWork::default();
        for round in 0..3 {
            for handle in &handles {
                assert_eq!(
                    resumed
                        .advance_resumable_prefill(*handle, &mut work)
                        .unwrap(),
                    round == 2
                );
            }
        }
        assert_eq!(work.forward_calls, 6);
        assert_eq!(work.processed_tokens, 10);
        assert_eq!(work.chunked_prefills, 2);
        for (tokens, handle) in prefixes.iter().zip(handles) {
            let original = serial.prefill(tokens).unwrap();
            let mut input = ForwardInput::new(vec![3, 4], vec![0, 1]);
            input.fork_from = Some(original);
            let expected = serial.forward(input.clone()).unwrap();
            input.fork_from = Some(handle);
            let actual = resumed.forward(input).unwrap();
            assert_eq!(
                actual
                    .values()
                    .data()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .values()
                    .data()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
            resumed.release_cache(handle).unwrap();
            serial.release_cache(original).unwrap();
            let hit = resumed.begin_resumable_prefill(tokens, 1 << 20).unwrap();
            assert!(hit.hit);
            assert!(resumed
                .advance_resumable_prefill(hit.handle, &mut PrefillWork::default())
                .is_err());
            resumed.release_cache(hit.handle).unwrap();
        }
        let mut parents = Vec::new();
        for _ in 0..64 {
            parents.push(
                resumed
                    .begin_resumable_prefill(&[1, 2, 3], 0)
                    .unwrap()
                    .handle,
            );
        }
        assert!(resumed.begin_resumable_prefill(&[1], 0).is_err());
        for parent in parents {
            resumed.release_cache(parent).unwrap();
        }
        assert!(resumed.begin_resumable_prefill(&[], 0).is_err());
        assert!(resumed.begin_resumable_prefill(&[u32::MAX], 0).is_err());
        let parent = resumed.begin_resumable_prefill(&[1], 0).unwrap();
        resumed.release_cache(parent.handle).unwrap();
    }
}

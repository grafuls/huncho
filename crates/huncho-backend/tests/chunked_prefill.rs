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
    }
}

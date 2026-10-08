//! Right padding must not select a padded decision row or affect valid causal outputs.
#![cfg(feature = "candle")]
use huncho_backend::Qwen3_5Backend;
use huncho_core::{
    backend::{Backend, ForwardInput},
    calibration::{argmax, calibrate},
};
use std::path::Path;

fn inputs() -> Vec<ForwardInput> {
    [5, 13, 47]
        .into_iter()
        .enumerate()
        .map(|(row, n)| {
            ForwardInput::new(
                (0..n)
                    .map(|i| ((i * 19 + row * 7 + 1) % 380) as u32)
                    .collect(),
                vec![n - 1, 1, n - 1],
            )
        })
        .collect()
}

#[test]
fn native_cpu_pointer_padding_preserves_final_valid_decisions_and_isolation() {
    let root = Path::new("tests/fixtures/tiny_kev");
    for dtype in ["fp32", "fp16"] {
        let mut backend =
            Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype).unwrap();
        assert!(backend.supports_padded_batch());
        let inputs = inputs();
        let reference: Vec<_> = inputs
            .iter()
            .map(|i| backend.forward(i.clone()).unwrap())
            .collect();
        let padded = backend.forward_padded_batch(inputs.clone()).unwrap();
        for (a, b) in padded.iter().zip(&reference) {
            assert_eq!(a.positions(), b.positions());
            let a = calibrate(a.values().data(), 2.40605).unwrap();
            let b = calibrate(b.values().data(), 2.40605).unwrap();
            assert_eq!(argmax(&a), argmax(&b));
            assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 1e-4));
        }
        let mut wrong = inputs[0].clone();
        wrong.tokens.resize(47, 0);
        let wrong = backend.forward(wrong).unwrap();
        assert!(
            wrong
                .values()
                .data()
                .iter()
                .zip(padded[0].values().data())
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "fixture must detect use of a padded final decision row"
        );
        let mut replica = backend.replica().unwrap();
        let replay = replica.forward_padded_batch(inputs.clone()).unwrap();
        assert!(padded
            .iter()
            .zip(replay)
            .all(|(a, b)| a.values().data() == b.values().data()));
        assert!(backend.forward_batch(inputs.clone()).is_err());
        let mut invalid = inputs.clone();
        invalid[0].positions = vec![5];
        assert!(backend.forward_padded_batch(invalid).is_err());
        let parent = backend.prefill(&[1, 2, 3]).unwrap();
        let mut invalid = inputs.clone();
        invalid[0].fork_from = Some(parent);
        assert!(backend.forward_padded_batch(invalid).is_err());
        backend.release_cache(parent).unwrap();
        assert!(backend
            .forward_padded_batch(vec![ForwardInput::new(vec![], vec![])])
            .is_err());
        assert!(backend
            .forward_padded_batch(vec![inputs[0].clone(); 65])
            .is_err());
        assert_eq!(
            reference[0].values().data(),
            backend.forward(inputs[0].clone()).unwrap().values().data()
        );
    }
}

#[test]
fn native_cpu_candidate_padding_preserves_selected_columns_and_positions() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let temp = tempfile::tempdir().unwrap();
    std::fs::copy(root.join("config.json"), temp.path().join("config.json")).unwrap();
    let mut tensors =
        candle::safetensors::load(root.join("model.safetensors"), &candle::Device::Cpu).unwrap();
    let embedding = tensors
        .iter()
        .find(|(name, _)| name.ends_with("embed_tokens.weight"))
        .unwrap()
        .1
        .clone();
    tensors.insert("lm_head.weight".into(), embedding);
    candle::safetensors::save(&tensors, temp.path().join("model.safetensors")).unwrap();
    for dtype in ["fp32", "fp16"] {
        let mut backend = Qwen3_5Backend::load(temp.path(), Some(root), 512, dtype).unwrap();
        let inputs: Vec<_> = inputs()
            .into_iter()
            .map(|mut i| {
                i.logit_codes = Some(vec![37, 36, 37]);
                i
            })
            .collect();
        let references: Vec<_> = inputs
            .iter()
            .map(|i| backend.forward(i.clone()).unwrap())
            .collect();
        let padded = backend.forward_padded_batch(inputs).unwrap();
        for (a, b) in padded.iter().zip(&references) {
            assert_eq!(a.positions(), b.positions());
            let a = calibrate(a.values().data(), 2.40605).unwrap();
            let b = calibrate(b.values().data(), 2.40605).unwrap();
            assert_eq!(argmax(&a), argmax(&b));
            assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 1e-4));
        }
    }
}

#[cfg(feature = "clef")]
#[test]
fn engine_padded_qualification_preserves_frozen_probabilities_and_counts_real_rectangles() {
    use huncho_core::{
        conformance::{GoldenCase, GoldenSuite},
        engine::{Engine, EvalOptions, EvalStats},
        manifest::{BackendId, ModelManifest},
        tokenizer::HfTokenizer,
    };
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
    let cases = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let request: huncho_core::contract::SystemOneRequest =
                serde_json::from_value(case["request"].clone()).unwrap();
            let expected = request
                .questions
                .iter()
                .zip(case["rows"].as_array().unwrap())
                .map(|((id, q), row)| {
                    let prompt = formatter.build(&request.state, q, &tokenizer).unwrap();
                    let p: Vec<f32> = serde_json::from_value(row["probabilities"].clone()).unwrap();
                    (
                        id.clone(),
                        prompt
                            .candidates
                            .into_iter()
                            .zip(p)
                            .map(|(c, p)| (c.label, p))
                            .collect::<BTreeMap<_, _>>(),
                    )
                })
                .collect();
            GoldenCase {
                id: index.to_string(),
                request,
                expected,
                targets: Default::default(),
            }
        })
        .collect();
    let suite = GoldenSuite {
        schema_version: "1.0".into(),
        family: "F2".into(),
        hash: None,
        cases,
    };
    let backend = Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, "fp32").unwrap();
    let engine = Engine::new(
        manifest,
        Box::new(backend),
        Box::new(tokenizer),
        Default::default(),
        BackendId::Candle,
        "fp32",
    )
    .unwrap();
    let opts = EvalOptions {
        max_batch_tokens: Some(1024),
        max_batch_padding_percent: 25,
        extensions: true,
        ..Default::default()
    };
    let report = huncho_core::conformance::run_suite_with_options(
        &engine,
        &suite,
        &Default::default(),
        &opts,
    )
    .unwrap();
    assert!(report.passed, "{report:?}");
    assert_eq!(report.max_batch_padding_percent, 25);
    assert!(report.work.padded_batch_calls > 0);
    assert!(report.work.padded_tokens > 0);
    assert!(report.optimization_parity.unwrap().max_prob_delta <= 1e-4);
    assert!(report.outcome_calibration.is_none());
    let mut work = EvalStats::default();
    let independent = engine
        .eval(
            &suite.cases[0].request,
            &EvalOptions {
                extensions: true,
                ..Default::default()
            },
        )
        .unwrap();
    let padded = engine
        .eval_with_stats(&suite.cases[0].request, &opts, &mut work)
        .unwrap();
    assert_eq!(padded.usage.input_tokens, independent.usage.input_tokens);
    assert_eq!(
        work.processed_tokens,
        padded.usage.input_tokens + work.padded_tokens
    );
    assert_eq!(work.prefill_calls, 0);
    assert!(work.padded_tokens * 100 <= work.processed_tokens * 25);
    let mut singleton = suite.clone();
    singleton.cases[0].request.questions = singleton.cases[0]
        .request
        .questions
        .clone()
        .into_iter()
        .take(1)
        .collect();
    let first = singleton.cases[0]
        .request
        .questions
        .keys()
        .next()
        .unwrap()
        .clone();
    singleton.cases[0].expected.retain(|key, _| key == &first);
    singleton.cases.truncate(1);
    assert!(huncho_core::conformance::run_suite_with_options(
        &engine,
        &singleton,
        &Default::default(),
        &opts
    )
    .is_err());
    for percent in [25, 100] {
        let mut invalid = opts.clone();
        invalid.max_batch_tokens = None;
        assert!(engine.eval(&suite.cases[0].request, &invalid).is_err());
        invalid.max_batch_tokens = Some(1024);
        invalid.max_batch_padding_percent = 101 + percent;
        assert!(engine.eval(&suite.cases[0].request, &invalid).is_err());
    }
    let mut cross_options = opts.clone();
    cross_options.max_batch_padding_percent = 100;
    let report = huncho_core::conformance::run_suite_with_cross_request_batches(
        &engine,
        &suite,
        &Default::default(),
        &cross_options,
        2,
    )
    .unwrap();
    assert!(report.passed);
    assert!(report.work.cross_request_batches > 0 && report.work.padded_batch_calls > 0);
}

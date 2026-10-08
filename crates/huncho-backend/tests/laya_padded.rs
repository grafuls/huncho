//! Actual CPU F1 mask/unpadding checks; fixed synthetic heads are not releases.
#![cfg(feature = "candle")]
#[path = "support/laya.rs"]
#[allow(dead_code)]
mod fixture;
use huncho_backend::CandleBackend;
use huncho_core::{
    backend::{Backend, CacheHandle, ForwardInput, ForwardOutput},
    calibration::{argmax, calibrate},
    conformance::{
        run_suite_with_cross_request_batches, run_suite_with_options, GoldenCase, GoldenSuite,
    },
    contract::{Answer, SystemOneRequest},
    engine::{Engine, EvalOptions, EvalStats},
    manifest::{BackendId, ModelManifest},
    tokenizer::SimpleTokenizer,
};
use std::{collections::BTreeMap, path::Path};

fn package(head: bool) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let root = Path::new("tests/fixtures/tiny_modernbert");
    let mut tensors =
        candle::safetensors::load(root.join("model.safetensors"), &candle::Device::Cpu).unwrap();
    // Include a local layer followed by a global layer, so pad-only local
    // queries cannot introduce nonfinite K/V into later valid outputs.
    let layers: Vec<_> = tensors
        .iter()
        .filter(|(name, _)| name.contains(".layers.0."))
        .map(|(name, tensor)| (name.clone(), tensor.clone()))
        .collect();
    assert!(!layers.is_empty());
    for layer in [1, 2] {
        for (name, tensor) in &layers {
            tensors.insert(
                name.replace(".layers.0.", &format!(".layers.{layer}.")),
                tensor.clone(),
            );
        }
    }
    if head {
        tensors.extend(fixture::head_tensors(8, 2));
    }
    candle::safetensors::save(&tensors, tmp.path().join("model.safetensors")).unwrap();
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
    config["num_hidden_layers"] = serde_json::json!(3);
    config["global_attn_every_n_layers"] = serde_json::json!(2);
    std::fs::write(
        tmp.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    tmp
}

fn load(path: &Path, selected: bool) -> CandleBackend {
    CandleBackend::load(
        path.join("config.json"),
        path.join("model.safetensors"),
        128,
        "fp32",
    )
    .unwrap()
    .with_selected_laya_head(selected)
    .unwrap()
}

fn inputs() -> Vec<ForwardInput> {
    [1, 3, 17, 47, 128]
        .into_iter()
        .enumerate()
        .map(|(row, len)| {
            let mut tokens: Vec<_> = (0..len)
                .map(|i| ((i * 17 + row * 13) % 32768) as u32)
                .collect();
            tokens[0] = 0; // Token zero can also be real input; length owns the mask.
            ForwardInput::new(tokens, vec![len - 1, 0, len / 2, 0]).with_qtype((row % 3) as u32)
        })
        .collect()
}

fn parity(actual: &ForwardOutput, expected: &ForwardOutput) {
    assert_eq!(actual.positions(), expected.positions());
    assert_eq!(actual.values().shape(), expected.values().shape());
    assert!(actual
        .values()
        .data()
        .iter()
        .zip(expected.values().data())
        .all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-5));
    for temp in [0.5, 1.0, 1.9063563, 1.9833995, 1.25143] {
        let a = calibrate(actual.values().data(), temp).unwrap();
        let b = calibrate(expected.values().data(), temp).unwrap();
        assert_eq!(argmax(&a), argmax(&b));
        assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 1e-4));
    }
}

#[test]
fn masked_cpu_batches_preserve_bare_and_typed_heads_lengths_order_and_replicas() {
    for (head, selected) in [(false, false), (true, false), (true, true)] {
        let tmp = package(head);
        let mut backend = load(tmp.path(), selected);
        assert!(backend.supports_padded_batch());
        assert_eq!(
            backend.capabilities().extra["padded_batch_execution"],
            "cpu-right-mask-unpad-head-v1"
        );
        let inputs = inputs();
        let expected: Vec<_> = inputs
            .iter()
            .map(|input| backend.forward(input.clone()).unwrap())
            .collect();
        let actual = backend.forward_padded_batch(inputs.clone()).unwrap();
        actual.iter().zip(&expected).for_each(|(a, b)| parity(a, b));
        let equal = vec![inputs[2].clone(), inputs[2].clone()];
        let a = backend.forward_padded_batch(equal.clone()).unwrap();
        let b = backend.forward_batch(equal).unwrap();
        assert!(a
            .iter()
            .zip(b)
            .all(|(a, b)| a.values().data() == b.values().data()));
        let mut replica = backend.replica().unwrap();
        let repeat = replica.forward_padded_batch(inputs.clone()).unwrap();
        assert!(actual
            .iter()
            .zip(repeat)
            .all(|(a, b)| a.values().data() == b.values().data()));
        let mut wrong = inputs[1].clone();
        wrong.tokens.resize(128, 0);
        assert!(
            backend
                .forward(wrong)
                .unwrap()
                .values()
                .data()
                .iter()
                .zip(actual[1].values().data())
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "fixture must detect unmasked padding"
        );
        for invalid in [
            vec![],
            vec![inputs[0].clone(); 65],
            vec![ForwardInput::new(vec![], vec![])],
            vec![ForwardInput::new(vec![1; 129], vec![0])],
            vec![ForwardInput::new(vec![32768], vec![0])],
            vec![ForwardInput::new(vec![1], vec![1])],
        ] {
            assert!(backend.forward_padded_batch(invalid).is_err());
        }
        for retained in [true, false] {
            let mut invalid = inputs[0].clone();
            invalid.retain_cache = retained;
            invalid.fork_from = (!retained).then_some(CacheHandle { id: 1 });
            assert!(backend.forward_padded_batch(vec![invalid]).is_err());
        }
        let empty = backend
            .forward_padded_batch(vec![
                ForwardInput::new(vec![1; 3], vec![]),
                inputs[2].clone(),
            ])
            .unwrap();
        assert_eq!(empty[0].values().shape(), &[0, 8]);
        assert_eq!(
            backend.forward(inputs[0].clone()).unwrap().values().data(),
            expected[0].values().data()
        );
    }
}

fn probabilities(answer: &Answer) -> BTreeMap<String, f32> {
    match answer {
        Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => {
            probabilities.clone()
        }
        Answer::Noul { noul } => BTreeMap::from([("no".into(), 1.0 - noul), ("yes".into(), *noul)]),
    }
}

#[test]
fn engine_uses_real_masked_rectangles_and_keeps_complete_typed_calibration_gates() {
    let tmp = package(true);
    let mut json: serde_json::Value = serde_json::from_slice(
        &std::fs::read("../../examples/mock-model/huncho-model.json").unwrap(),
    )
    .unwrap();
    json["name"] = serde_json::json!("synthetic-laya-padding");
    json["backbone"]["max_context"] = serde_json::json!(128);
    json["prompt_contract"]["max_len"] = serde_json::json!(128);
    json["prompt_contract"]["head_max_len"] = serde_json::json!(64);
    json["calibration"]["default"]["per_type_temperatures"] =
        serde_json::json!({"choice":1.9063563,"score":1.9833995,"noul":1.25143});
    json["calibration"]["entries"] = serde_json::json!({});
    let manifest: ModelManifest = serde_json::from_value(json).unwrap();
    let make = |selected| {
        Engine::new(
            manifest.clone(),
            Box::new(load(tmp.path(), selected)),
            Box::new(SimpleTokenizer::new(32768)),
            Default::default(),
            BackendId::Candle,
            "fp32",
        )
        .unwrap()
    };
    let baseline = make(false);
    let base: serde_json::Value = serde_json::from_str(r#"{"model":"synthetic-laya-padding","state":"refund","questions":{
        "z_choice":{"type":"choice","instructions":"Team?","criteria":{"shipping":null,"billing":"Charges","returns":"Refunds"}},
        "a_noul":{"type":"noul","instructions":"Urgent?"},
        "m_score":{"type":"score","instructions":"Priority?","criteria":["low","medium","high","very high"]}}}"#).unwrap();
    let mut cases = Vec::new();
    for (index, state) in [
        "refund".to_string(),
        "a customer wants a refund".to_string(),
        "long state ".repeat(160),
    ]
    .into_iter()
    .enumerate()
    {
        let mut value = base.clone();
        value["state"] = serde_json::json!(state);
        let request: SystemOneRequest = serde_json::from_value(value).unwrap();
        let response = baseline.eval(&request, &Default::default()).unwrap();
        cases.push(GoldenCase {
            id: index.to_string(),
            request,
            expected: response
                .answers
                .iter()
                .map(|(id, a)| (id.clone(), probabilities(a)))
                .collect(),
            targets: BTreeMap::from([
                ("z_choice".into(), "billing".into()),
                ("a_noul".into(), "yes".into()),
                ("m_score".into(), "1".into()),
            ]),
        });
    }
    let suite = GoldenSuite {
        schema_version: "1.0".into(),
        family: "F1".into(),
        hash: None,
        cases,
    };
    for selected in [false, true] {
        let engine = make(selected);
        let opts = EvalOptions {
            max_batch_tokens: Some(384),
            max_batch_padding_percent: 25,
            extensions: true,
            ..Default::default()
        };
        let report = run_suite_with_options(&engine, &suite, &Default::default(), &opts).unwrap();
        assert!(report.passed, "{report:?}");
        assert!(report.work.padded_batch_calls > 0 && report.work.padded_tokens > 0);
        assert!(report.optimization_parity.unwrap().max_prob_delta <= 1e-4);
        assert_eq!(report.outcome_calibration.unwrap().questions, 9);
        let mut work = EvalStats::default();
        let actual = engine
            .eval_with_stats(&suite.cases[0].request, &opts, &mut work)
            .unwrap();
        let independent = baseline
            .eval(
                &suite.cases[0].request,
                &EvalOptions {
                    extensions: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(actual.usage.input_tokens, independent.usage.input_tokens);
        assert_eq!(actual.usage.output_tokens, 0);
        assert_eq!(
            work.processed_tokens,
            actual.usage.input_tokens + work.padded_tokens
        );
        assert!(work.padded_tokens * 100 <= work.processed_tokens * 25);
        assert_eq!(actual.answers.len(), 3);
        let mut cross = opts.clone();
        cross.max_batch_padding_percent = 100;
        let report =
            run_suite_with_cross_request_batches(&engine, &suite, &Default::default(), &cross, 2)
                .unwrap();
        assert!(
            report.passed
                && report.work.cross_request_batches > 0
                && report.work.padded_batch_calls > 0
        );
        let mut incomplete = suite.clone();
        incomplete.cases[0].targets.remove("a_noul");
        assert!(run_suite_with_options(&engine, &incomplete, &Default::default(), &opts).is_err());
        let mut drifted = suite.clone();
        drifted.cases[0].expected.insert(
            "z_choice".into(),
            BTreeMap::from([
                ("billing".into(), 0.999),
                ("returns".into(), 0.0005),
                ("shipping".into(), 0.0005),
            ]),
        );
        assert!(
            !run_suite_with_options(&engine, &drifted, &Default::default(), &opts)
                .unwrap()
                .passed
        );
    }
}

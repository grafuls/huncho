//! Offline numerical parity with Cloudflare's reference encoder and joint head.
#![cfg(feature = "clef")]
use huncho_backend::ClefBackend;
use huncho_core::{
    backend::Backend, contract::SystemOneRequest, manifest::ModelManifest, prompt::clef,
    tokenizer::HfTokenizer,
};
use serde_json::Value;
use std::path::Path;
const FIXTURE: &str = "tests/fixtures/tiny_clef";

#[test]
fn native_clef_matches_reference_tokens_spans_and_logits() {
    assert_clef_matches_reference(candle::Device::Cpu, &["fp32", "fp16"]);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU and compatible CUDA kernels"]
fn native_clef_cuda_matches_reference_tokens_spans_and_logits() {
    let device = huncho_backend::clef::device_from_env().unwrap();
    assert!(device.is_cuda(), "CUDA probe unexpectedly fell back to CPU");
    // This also exercises the T4's automatic FP16 selection. Explicit FP32
    // retains the stricter reference tolerance.
    let dtype = huncho_backend::clef::default_dtype().unwrap();
    assert_clef_matches_reference(device, &["fp32", dtype]);
}

fn assert_clef_matches_reference(device: candle::Device, dtypes: &[&str]) {
    let root = Path::new(FIXTURE);
    let manifest = ModelManifest::load(root.join("huncho-model.json")).unwrap();
    let golden: Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let tokenizer = HfTokenizer::from_file_unbounded(root.join("tokenizer.json")).unwrap();
    for &dtype in dtypes {
        let mut model = ClefBackend::load(root, &manifest, dtype, device.clone()).unwrap();
        for (i, case) in golden["cases"].as_array().unwrap().iter().enumerate() {
            let request: SystemOneRequest =
                serde_json::from_value(case["request"].clone()).unwrap();
            let encoded = clef::encode(&request, &tokenizer, 4096).unwrap();
            let expected_tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
            assert_eq!(encoded.input_ids, expected_tokens, "case {i} tokens");
            assert_eq!(
                serde_json::to_value(&encoded.questions).unwrap(),
                case["questions"],
                "case {i} spans"
            );
            let actual = model.forward_request(&request, 4096).unwrap();
            assert_eq!(actual.input_tokens as usize, expected_tokens.len());
            for q in &encoded.questions {
                let mut scores = Vec::new();
                for option in &q.option_ids {
                    let a = actual.logits[&q.question_id][option];
                    let b = case["logits"][&q.question_id][option].as_f64().unwrap() as f32;
                    let tolerance = if dtype == "fp32" { 3e-5 } else { 0.005 };
                    assert!(
                        (a - b).abs() < tolerance,
                        "{dtype} case {i} {} {option}: {a} != {b}",
                        q.question_id
                    );
                    scores.push(a);
                }
                let probabilities = huncho_core::calibration::calibrate(&scores, 1.0).unwrap();
                for (option, actual) in q.option_ids.iter().zip(probabilities) {
                    let expected = case["probabilities"][&q.question_id][option]
                        .as_f64()
                        .unwrap() as f32;
                    assert!((actual - expected).abs() < if dtype == "fp32" { 1e-5 } else { 0.002 });
                }
            }
            // Reject the complete prompt before backbone execution, preserving state.
            let err = model
                .forward_request(&request, expected_tokens.len() - 1)
                .unwrap_err();
            assert!(err.to_string().contains("joint prompt requires"));
        }
    }
}

#[test]
fn native_clef_rejects_unsupported_contracts_and_cpu_bf16() {
    let root = Path::new(FIXTURE);
    let mut manifest = ModelManifest::load(root.join("huncho-model.json")).unwrap();
    let err = ClefBackend::load(root, &manifest, "bf16", candle::Device::Cpu)
        .err()
        .unwrap();
    assert!(err.to_string().contains("bf16 requires CUDA"));
    manifest.prompt_contract.contract_hash = "changed-contract".into();
    let err = ClefBackend::load(root, &manifest, "fp32", candle::Device::Cpu)
        .err()
        .unwrap();
    assert!(err.to_string().contains("prompt contract"));
}

#[test]
fn native_tokenizer_never_truncates_or_pads_prompt_fragments() {
    let root = Path::new(FIXTURE);
    let mut config: Value =
        serde_json::from_slice(&std::fs::read(root.join("tokenizer.json")).unwrap()).unwrap();
    config["truncation"] = serde_json::json!({"direction":"Right","max_length":2,"strategy":"LongestFirst","stride":0});
    config["padding"] = serde_json::json!({"strategy":{"Fixed":10},"direction":"Right","pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"[UNK]"});
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("tokenizer.json");
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    let tokenizer = HfTokenizer::from_file_unbounded(path).unwrap();
    let golden: Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let request = serde_json::from_value(golden["cases"][0]["request"].clone()).unwrap();
    let expected: Vec<u32> = serde_json::from_value(golden["cases"][0]["tokens"].clone()).unwrap();
    assert_eq!(
        clef::encode(&request, &tokenizer, 4096).unwrap().input_ids,
        expected
    );
}

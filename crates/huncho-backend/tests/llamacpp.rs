//! Real CPU GGUF prefill; frozen upstream Kev outcomes remain independent.
#![cfg(feature = "llamacpp")]

use huncho_backend::{llamacpp::LlamaOptions, LlamaCppBackend, Qwen3_5Backend};
use huncho_core::{
    backend::{Backend, ForwardInput},
    calibration::calibrate,
    manifest::{ArtifactRef, BackendId, F3Config, Family, HeadKind, ModelManifest},
};
use std::{collections::BTreeMap, path::Path};

fn manifest() -> ModelManifest {
    let mut m = ModelManifest::load("tests/fixtures/tiny_clef/huncho-model.json").unwrap();
    m.family = Family::F2;
    m.backbone.hidden_size = 16;
    m.backbone.max_context = 512;
    m.backbone.tokenizer = None;
    m.backbone.artifacts = BTreeMap::from([(
        BackendId::LlamaCpp,
        vec![ArtifactRef {
            path: "../llamacpp/kev-f32.gguf".into(),
            dtype: "gguf-f32".into(),
            quantization: None,
        }],
    )]);
    m.head.kind = HeadKind::Pointer;
    m.head.weights = "head.pt".into();
    m.prompt_contract.template = "kev-v1".into();
    m.prompt_contract.max_options = 255;
    m.prompt_contract.state_budget = 256;
    m.prompt_contract.head_budget = 256;
    m.prompt_contract.max_len = 512;
    m.prompt_contract.head_max_len = 192;
    m.validate().unwrap();
    m
}

#[test]
fn native_cpu_pointer_matches_frozen_upstream_and_resets_state() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let m = manifest();
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let mut llama =
        LlamaCppBackend::load(root, &m, "gguf-f32", LlamaOptions { threads: 2 }).unwrap();
    assert!(!llama.capabilities().supports_fork);
    assert!(!llama.supports_batch());
    assert_eq!(llama.capabilities().extra["device"], "CPU");
    let mut worst_raw = 0.0f32;
    let mut worst_probability = 0.0f32;
    for _ in 0..2 {
        for case in golden["cases"].as_array().unwrap() {
            for row in case["rows"].as_array().unwrap() {
                let tokens = serde_json::from_value(row["tokens"].clone()).unwrap();
                let positions: Vec<usize> =
                    serde_json::from_value(row["positions"].clone()).unwrap();
                let output = llama
                    .forward(ForwardInput::new(tokens, positions.clone()))
                    .unwrap();
                assert_eq!(output.values().shape(), &[positions.len(), 1]);
                let expected: Vec<f32> = serde_json::from_value(row["raw_logits"].clone()).unwrap();
                let p: Vec<f32> = serde_json::from_value(row["probabilities"].clone()).unwrap();
                let actual_p = calibrate(output.values().data(), 2.40605).unwrap();
                for (a, b) in output.values().data().iter().zip(expected) {
                    worst_raw = worst_raw.max((a - b).abs());
                }
                for (a, b) in actual_p.iter().zip(p) {
                    worst_probability = worst_probability.max((a - b).abs());
                }
            }
        }
    }
    eprintln!("llama CPU worst raw={worst_raw}, probability={worst_probability}");
    assert!(worst_raw <= 2e-4);
    assert!(worst_probability <= 2e-5);
    let a = llama
        .forward(ForwardInput::new(vec![1, 2], vec![0, 0]))
        .unwrap();
    assert_eq!(a.values().data()[0], a.values().data()[1]);
    assert!(
        (a.values().data()[0] - golden["short_logits"][0].as_f64().unwrap() as f32).abs() <= 2e-4
    );
    assert!(llama
        .forward(ForwardInput::new(vec![384], vec![0]))
        .is_err());
    assert!(llama.forward(ForwardInput::new(vec![1], vec![1])).is_err());
    // A full 512-token prompt must not force all tokens into the readout.
    let long = ForwardInput::new((0..512).map(|i| (i % 380) as u32).collect(), vec![0, 300]);
    let actual = llama.forward(long.clone()).unwrap();
    let mut reference =
        Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, "fp32").unwrap();
    let expected = reference.forward(long).unwrap();
    for (a, b) in actual.values().data().iter().zip(expected.values().data()) {
        assert!((a - b).abs() <= 2e-4);
    }
    let mut cached = ForwardInput::new(vec![1], vec![0]);
    cached.retain_cache = true;
    assert!(llama.forward(cached).is_err());
}

#[test]
fn native_candidate_logits_and_independent_contexts_share_immutable_weights() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let mut m = manifest();
    m.family = Family::F3;
    m.head.kind = HeadKind::CandidateLogit;
    m.f3 = Some(F3Config {
        candidate_codes: vec!["A".into(), "B".into()],
        candidate_token_ids: vec![36, 37],
        system_prompt: String::new(),
        prompt_code_sha256: "fixture".into(),
        max_input_tokens: 512,
    });
    m.validate().unwrap();
    let mut llama = LlamaCppBackend::load(root, &m, "gguf-f32", LlamaOptions::default()).unwrap();
    let mut replica = llama.replica().unwrap();
    let input = ForwardInput::new(vec![1, 31, 14, 63, 2], vec![4, 1, 4]);
    let full = llama.forward(input.clone()).unwrap();
    let mut selected_input = input.clone();
    selected_input.logit_codes = Some(vec![37, 36, 37]);
    let selected = replica.forward(selected_input).unwrap();
    for row in 0..3 {
        for (j, &code) in [37, 36, 37].iter().enumerate() {
            assert_eq!(
                selected.values().data()[row * 3 + j],
                full.values().data()[row * 384 + code]
            );
        }
    }
    let other = replica
        .forward(ForwardInput::new(vec![9, 2], vec![1]))
        .unwrap();
    assert!(other.values().data().iter().all(|v| v.is_finite()));
    let contexts: Vec<_> = (0..3).map(|_| llama.replica().unwrap()).collect();
    let jobs: Vec<_> = contexts
        .into_iter()
        .map(|mut context| {
            let input = input.clone();
            std::thread::spawn(move || context.forward(input).unwrap().values().data().to_vec())
        })
        .collect();
    for job in jobs {
        assert_eq!(job.join().unwrap(), full.values().data());
    }
    let again = llama.forward(input).unwrap();
    assert_eq!(full.values().data(), again.values().data());
    // Compare the GGUF's tied vocabulary projection with the same native HF
    // backbone and FP32 LoRA merge; no synthetic answer probabilities are used.
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
    let mut candle = Qwen3_5Backend::load(temp.path(), Some(root), 512, "fp32").unwrap();
    let expected = candle
        .forward(ForwardInput::new(vec![1, 31, 14, 63, 2], vec![4, 1, 4]))
        .unwrap();
    for (a, b) in full.values().data().iter().zip(expected.values().data()) {
        assert!((a - b).abs() <= 2e-4, "LM logit {a} vs {b}");
    }
}

#[test]
#[ignore = "regenerates a checked-in fixture; needs pinned converter and CPU Python dependencies"]
fn regenerate_gguf_from_native_fp32_merge() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let temp = tempfile::tempdir().unwrap();
    let merged = temp.path().join("merged-hf");
    huncho_backend::qwen3_5::export_merged_hf(root, root, false, &merged).unwrap();
    std::fs::copy(root.join("tokenizer.json"), merged.join("tokenizer.json")).unwrap();
    for dtype in ["f32", "f16"] {
        let status = std::process::Command::new(std::env::var_os("HUNCHO_LLAMA_PYTHON").unwrap())
            .arg("tests/fixtures/llamacpp/generate.py")
            .arg("--tools")
            .arg(std::env::var_os("HUNCHO_LLAMA_TOOLS").unwrap())
            .arg("--merged-hf")
            .arg(&merged)
            .arg("--output")
            .arg(format!("tests/fixtures/llamacpp/kev-{dtype}.gguf"))
            .arg("--outtype")
            .arg(dtype)
            .env("CUDA_VISIBLE_DEVICES", "")
            .env("HF_HUB_OFFLINE", "1")
            .status()
            .unwrap();
        assert!(status.success());
    }
}

#[test]
fn native_f16_pointer_preserves_frozen_probabilities_without_dtype_aliasing() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let mut m = manifest();
    m.backbone
        .artifacts
        .get_mut(&BackendId::LlamaCpp)
        .unwrap()
        .push(ArtifactRef {
            path: "../llamacpp/kev-f16.gguf".into(),
            dtype: "gguf-f16".into(),
            quantization: None,
        });
    let mut backend = LlamaCppBackend::load(root, &m, "gguf-f16", LlamaOptions::default()).unwrap();
    assert_eq!(backend.capabilities().dtype, "gguf-f16");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let mut worst = 0.0f32;
    for case in golden["cases"].as_array().unwrap() {
        for row in case["rows"].as_array().unwrap() {
            let tokens = serde_json::from_value(row["tokens"].clone()).unwrap();
            let positions = serde_json::from_value(row["positions"].clone()).unwrap();
            let out = backend
                .forward(ForwardInput::new(tokens, positions))
                .unwrap();
            let p = calibrate(out.values().data(), 2.40605).unwrap();
            let expected: Vec<f32> = serde_json::from_value(row["probabilities"].clone()).unwrap();
            let argmax = |values: &[f32]| {
                values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0
            };
            assert_eq!(argmax(&p), argmax(&expected));
            for (a, b) in p.iter().zip(expected) {
                worst = worst.max((a - b).abs());
            }
        }
    }
    eprintln!("llama CPU F16 worst probability={worst}");
    assert!(worst <= 1e-4);
    // The exact artifact dtype must agree with the tensor layout.
    m.backbone.artifacts.get_mut(&BackendId::LlamaCpp).unwrap()[0].path =
        "../llamacpp/kev-f16.gguf".into();
    assert!(
        LlamaCppBackend::load(root, &m, "gguf-f32", LlamaOptions::default())
            .err()
            .unwrap()
            .to_string()
            .contains("layout")
    );
}

#[test]
fn hf_export_retains_trained_f3_projection_and_rejects_unsupported_bias_and_overwrite() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    std::fs::create_dir(&base).unwrap();
    std::fs::copy(root.join("config.json"), base.join("config.json")).unwrap();
    let mut tensors =
        candle::safetensors::load(root.join("model.safetensors"), &candle::Device::Cpu).unwrap();
    let embedding = tensors
        .iter()
        .find(|(name, _)| name.ends_with("embed_tokens.weight"))
        .unwrap()
        .1
        .clone();
    let trained = embedding.affine(1.5, 0.01).unwrap();
    tensors.insert("lm_head.weight".into(), trained.clone());
    candle::safetensors::save(&tensors, base.join("model.safetensors")).unwrap();
    let output = temp.path().join("export");
    huncho_backend::qwen3_5::export_merged_hf(&base, root, true, &output).unwrap();
    let actual =
        candle::safetensors::load(output.join("model.safetensors"), &candle::Device::Cpu).unwrap();
    assert_eq!(
        actual["lm_head.weight"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        trained.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    );
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["tie_word_embeddings"], false);
    assert!(huncho_backend::qwen3_5::export_merged_hf(&base, root, true, &output).is_err());
    tensors.insert(
        "lm_head.bias".into(),
        candle::Tensor::zeros(384, candle::DType::F32, &candle::Device::Cpu).unwrap(),
    );
    candle::safetensors::save(&tensors, base.join("model.safetensors")).unwrap();
    let invalid = temp.path().join("invalid");
    assert!(
        huncho_backend::qwen3_5::export_merged_hf(&base, root, true, &invalid)
            .unwrap_err()
            .to_string()
            .contains("bias")
    );
    assert!(!invalid.exists());
}

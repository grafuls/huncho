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
    let mut llama = LlamaCppBackend::load(
        root,
        &m,
        "gguf-f32",
        LlamaOptions {
            threads: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(llama.capabilities().supports_fork);
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
fn native_independent_batches_isolate_sequences_and_preserve_valid_readouts() {
    use huncho_core::calibration::argmax;
    let root = Path::new("tests/fixtures/tiny_kev");
    for dtype in ["gguf-f32", "gguf-f16"] {
        let mut m = manifest();
        let a = &mut m.backbone.artifacts.get_mut(&BackendId::LlamaCpp).unwrap()[0];
        a.dtype = dtype.into();
        a.path = format!("../llamacpp/kev-{}.gguf", dtype.trim_start_matches("gguf-"));
        let mut backend = LlamaCppBackend::load(
            root,
            &m,
            dtype,
            LlamaOptions {
                batch_rows: 4,
                ..Default::default()
            },
        )
        .unwrap();
        let mut replica = backend.replica().unwrap();
        assert!(backend.supports_batch());
        assert_eq!(backend.batch_limits().max_rows, 4);
        assert_eq!(backend.capabilities().extra["llamacpp_batch_rows"], "4");
        for length in [5, 47, 512] {
            let inputs: Vec<_> = (0..4)
                .map(|row| {
                    ForwardInput::new(
                        (0..length)
                            .map(|i| ((i * 13 + row * 31 + 7) % 380) as u32)
                            .collect(),
                        vec![length - 1, 1, length - 1, 0],
                    )
                })
                .collect();
            let independent: Vec<_> = inputs
                .iter()
                .map(|i| backend.forward(i.clone()).unwrap())
                .collect();
            let batch = backend.forward_batch(inputs.clone()).unwrap();
            let other = replica.forward_batch(inputs.clone()).unwrap();
            for ((actual, expected), other) in batch.iter().zip(&independent).zip(other) {
                assert_eq!(actual.positions(), expected.positions());
                assert_eq!(actual.values().data(), other.values().data());
                let p = calibrate(actual.values().data(), 2.40605).unwrap();
                let reference = calibrate(expected.values().data(), 2.40605).unwrap();
                assert_eq!(argmax(&p), argmax(&reference));
                assert!(p.iter().zip(reference).all(|(a, b)| (a - b).abs() <= 1e-4));
            }
            assert!(batch[0]
                .values()
                .data()
                .iter()
                .zip(batch[1].values().data())
                .any(|(a, b)| (a - b).abs() > 1e-4));
            let mut reversed = inputs.clone();
            reversed.reverse();
            for (actual, expected) in backend
                .forward_batch(reversed)
                .unwrap()
                .iter()
                .rev()
                .zip(&batch)
            {
                assert!(actual
                    .values()
                    .data()
                    .iter()
                    .zip(expected.values().data())
                    .all(|(a, b)| (a - b).abs() < 2e-4));
            }
            let again = backend.forward(inputs[0].clone()).unwrap();
            assert_eq!(again.values().data(), independent[0].values().data());
        }
        let parent = backend.prefill(&[1, 2, 3]).unwrap();
        let child = backend.fork(parent).unwrap();
        let input = ForwardInput::new(vec![4, 5, 6], vec![2, 0]);
        let mut branched = input.clone();
        branched.fork_from = Some(child);
        let before = backend.forward(branched.clone()).unwrap();
        backend.forward_batch(vec![input.clone(); 4]).unwrap();
        let replay = backend.fork(parent).unwrap();
        branched.fork_from = Some(replay);
        assert_eq!(
            before.values().data(),
            backend.forward(branched).unwrap().values().data()
        );
        for handle in [parent, child, replay] {
            backend.release_cache(handle).unwrap();
        }
        let mut cached = input.clone();
        cached.fork_from = Some(parent);
        for invalid in [
            vec![],
            vec![input.clone(); 5],
            vec![input.clone(), ForwardInput::new(vec![1], vec![0])],
            vec![input.clone(), ForwardInput::new(vec![384; 3], vec![0])],
            vec![input.clone(), ForwardInput::new(vec![1; 3], vec![3])],
            vec![input.clone(), cached],
            vec![ForwardInput::new(vec![1; 3], vec![0; 128]); 2],
        ] {
            assert!(backend.forward_batch(invalid).is_err());
        }
        assert!(backend
            .forward_batch(vec![input.clone(), input.with_logit_codes(vec![36])])
            .is_err());
    }
    let m = manifest();
    for batch_rows in [0, 9] {
        assert!(LlamaCppBackend::load(
            root,
            &m,
            "gguf-f32",
            LlamaOptions {
                batch_rows,
                ..Default::default()
            }
        )
        .is_err());
    }
}

#[test]
fn native_f3_batches_keep_per_sequence_position_and_code_order() {
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
    for dtype in ["gguf-f32", "gguf-f16"] {
        let a = &mut m.backbone.artifacts.get_mut(&BackendId::LlamaCpp).unwrap()[0];
        a.dtype = dtype.into();
        a.path = format!("../llamacpp/kev-{}.gguf", dtype.trim_start_matches("gguf-"));
        let mut backend = LlamaCppBackend::load(
            root,
            &m,
            dtype,
            LlamaOptions {
                batch_rows: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let mut inputs = vec![
            ForwardInput::new(vec![1, 31, 14, 63, 2], vec![4, 1, 4]),
            ForwardInput::new(vec![63, 2, 31, 14, 1], vec![0, 4, 0]),
        ];
        inputs[1].logit_codes = Some(vec![37, 36, 37]);
        let reference: Vec<_> = inputs
            .iter()
            .map(|i| backend.forward(i.clone()).unwrap())
            .collect();
        for (actual, expected) in backend.forward_batch(inputs).unwrap().iter().zip(reference) {
            assert_eq!(actual.positions(), expected.positions());
            assert_eq!(actual.values().shape(), expected.values().shape());
            assert!(actual
                .values()
                .data()
                .iter()
                .zip(expected.values().data())
                .all(|(a, b)| (a - b).abs() < 2e-4));
        }
    }
}

#[test]
fn full_hybrid_prefix_snapshots_preserve_frozen_pointer_scores_and_isolate_forks() {
    use huncho_core::{
        backend::{CacheHandle, PrefillWork},
        calibration::argmax,
    };
    let root = Path::new("tests/fixtures/tiny_kev");
    let m = manifest();
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for dtype in ["gguf-f32", "gguf-f16"] {
        let mut m = m.clone();
        m.backbone.artifacts.get_mut(&BackendId::LlamaCpp).unwrap()[0].path = format!(
            "../llamacpp/kev-{}.gguf",
            if dtype == "gguf-f32" { "f32" } else { "f16" }
        );
        m.backbone.artifacts.get_mut(&BackendId::LlamaCpp).unwrap()[0].dtype = dtype.into();
        let mut backend = LlamaCppBackend::load(root, &m, dtype, LlamaOptions::default()).unwrap();
        let mut replica = backend.replica().unwrap();
        let mut tested = 0;
        for case in golden["cases"].as_array().unwrap() {
            for row in case["rows"].as_array().unwrap() {
                let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
                let positions: Vec<usize> =
                    serde_json::from_value(row["positions"].clone()).unwrap();
                let independent = backend
                    .forward(ForwardInput::new(tokens.clone(), positions.clone()))
                    .unwrap();
                for split in [1, 3, 7, 19] {
                    if positions.iter().any(|&p| p < split) {
                        continue;
                    }
                    let mut work = PrefillWork::default();
                    let prefill = backend
                        .prefill_cached_with_work(&tokens[..split], 1024 * 1024, &mut work)
                        .unwrap();
                    let parent = prefill.handle;
                    assert!(!prefill.hit);
                    assert_eq!(work.forward_calls, 1);
                    assert_eq!(work.processed_tokens, split as u64);
                    assert!(replica.fork(parent).is_err());
                    let a = backend.fork(parent).unwrap();
                    let b = backend.fork(parent).unwrap();
                    backend.release_cache(parent).unwrap();
                    let mut input = ForwardInput::new(
                        tokens[split..].to_vec(),
                        positions.iter().map(|p| p - split).collect(),
                    );
                    input.fork_from = Some(a);
                    let branched = backend.forward(input.clone()).unwrap();
                    // Independent inference between branches cannot alter saved
                    // recurrent state, convolution history or absolute KV offsets.
                    backend
                        .forward(ForwardInput::new(vec![7, 3, 1], vec![0]))
                        .unwrap();
                    input.fork_from = Some(b);
                    let again = backend.forward(input.clone()).unwrap();
                    assert_eq!(branched.values().data(), again.values().data());
                    let p = calibrate(branched.values().data(), 2.40605).unwrap();
                    let reference = calibrate(independent.values().data(), 2.40605).unwrap();
                    assert_eq!(argmax(&p), argmax(&reference));
                    assert!(p.iter().zip(reference).all(|(a, b)| (a - b).abs() <= 1e-4));
                    let frozen: Vec<f32> =
                        serde_json::from_value(row["probabilities"].clone()).unwrap();
                    assert!(p.iter().zip(frozen).all(|(a, b)| (a - b).abs() <= 1e-3));
                    // Bad validation is transactional: the original branch still
                    // resumes from the full successful prefix+suffix state.
                    let mut bad = ForwardInput::new(vec![384], vec![0]);
                    bad.fork_from = Some(a);
                    assert!(backend.forward(bad).is_err());
                    let mut continuation = ForwardInput::new(vec![1, 2], vec![1, 0, 1]);
                    continuation.fork_from = Some(a);
                    let next = backend.forward(continuation.clone()).unwrap();
                    continuation.fork_from = Some(b);
                    assert_eq!(
                        next.values().data(),
                        backend.forward(continuation).unwrap().values().data()
                    );
                    backend.release_cache(a).unwrap();
                    backend.release_cache(b).unwrap();
                    let mut hit_work = PrefillWork::default();
                    let hit = backend
                        .prefill_cached_with_work(&tokens[..split], 1024 * 1024, &mut hit_work)
                        .unwrap();
                    assert!(hit.hit);
                    assert_eq!(hit_work.forward_calls, 0);
                    assert_eq!(hit_work.processed_tokens, 0);
                    backend.clear_prefix_cache().unwrap();
                    // Clearing retained snapshots does not invalidate live users.
                    let fork = backend.fork(hit.handle).unwrap();
                    backend.release_cache(hit.handle).unwrap();
                    backend.release_cache(fork).unwrap();
                    assert!(backend.release_cache(fork).is_err());
                    tested += 1;
                }
            }
        }
        assert!(tested >= 4, "nonvacuous split-prefix coverage required");
        assert!(backend.fork(CacheHandle { id: u64::MAX }).is_err());
        let mut other = huncho_backend::MockBackend::new();
        let foreign = other.prefill(&[1, 2]).unwrap();
        let own = backend.prefill(&[1, 2]).unwrap();
        assert_ne!(foreign.id, own.id);
        assert!(backend.fork(foreign).is_err());
        assert!(other.fork(own).is_err());
        backend.release_cache(own).unwrap();
        other.release_cache(foreign).unwrap();
        assert!(backend.prefill(&[]).is_err());
        assert!(backend.prefill(&vec![1; 513]).is_err());
        let parent = backend.prefill(&[1, 2, 3]).unwrap();
        let handles: Vec<_> = (0..63).map(|_| backend.fork(parent).unwrap()).collect();
        assert!(backend.fork(parent).is_err());
        assert!(backend.prefill(&[1]).is_err());
        for handle in handles {
            backend.release_cache(handle).unwrap();
        }
        backend.release_cache(parent).unwrap();
        for _ in 0..67 {
            let handle = backend.prefill(&[1]).unwrap();
            backend.release_cache(handle).unwrap();
        }
        let parent = backend.prefill(&vec![1; 512]).unwrap();
        let mut input = ForwardInput::new(vec![1], vec![0]);
        input.fork_from = Some(parent);
        assert!(backend.forward(input).is_err());
        backend.release_cache(parent).unwrap();
    }
}

#[test]
fn pinned_cpu_quantization_preserves_fp32_readouts_and_runs_actual_packed_tensors() {
    use candle::quantized::{gguf_file, GgmlDType};
    use huncho_backend::llamacpp::quantize_gguf_cpu;
    let root = Path::new("tests/fixtures/tiny_kev");
    let source = Path::new("tests/fixtures/llamacpp/kev-f32.gguf");
    let original = std::fs::read(source).unwrap();
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for dtype in ["gguf-q8_0", "gguf-q4_0"] {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("backbone.gguf");
        let stats = quantize_gguf_cpu(source, &output, dtype, 1).unwrap();
        assert!(stats.packed_tensors > 0);
        assert!(!stats.shape_retained_f32.is_empty());
        let content = gguf_file::Content::read(&mut std::fs::File::open(&output).unwrap()).unwrap();
        assert_eq!(
            content.tensor_infos["token_embd.weight"].ggml_dtype,
            GgmlDType::F32
        );
        assert!(content
            .tensor_infos
            .values()
            .all(|info| info.ggml_dtype == GgmlDType::F32
                || info.ggml_dtype
                    == if dtype == "gguf-q8_0" {
                        GgmlDType::Q8_0
                    } else {
                        GgmlDType::Q4_0
                    }));
        let bytes = std::fs::read(&output).unwrap();
        assert!(quantize_gguf_cpu(source, &output, dtype, 1).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), bytes);
        std::fs::copy(root.join("head.pt"), dir.path().join("head.pt")).unwrap();
        let mut m = manifest();
        m.backbone.artifacts = BTreeMap::from([(
            BackendId::LlamaCpp,
            vec![ArtifactRef {
                path: "backbone.gguf".into(),
                dtype: dtype.into(),
                quantization: Some(format!(
                    "llamacpp-qwen35-{}-v1",
                    dtype.trim_start_matches("gguf-")
                )),
            }],
        )]);
        let mut native =
            LlamaCppBackend::load(dir.path(), &m, dtype, LlamaOptions::default()).unwrap();
        let mut batched = LlamaCppBackend::load(
            dir.path(),
            &m,
            dtype,
            LlamaOptions {
                batch_rows: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let mut replica = native.replica().unwrap();
        assert!(native
            .capabilities()
            .extra
            .contains_key("weight_quantization"));
        let mut delta = 0.0f32;
        for case in golden["cases"].as_array().unwrap() {
            for row in case["rows"].as_array().unwrap() {
                let input = ForwardInput::new(
                    serde_json::from_value(row["tokens"].clone()).unwrap(),
                    serde_json::from_value(row["positions"].clone()).unwrap(),
                );
                let out = native.forward(input.clone()).unwrap();
                assert_eq!(
                    out.values().data(),
                    replica.forward(input.clone()).unwrap().values().data()
                );
                assert_eq!(
                    out.values().data(),
                    native.forward(input.clone()).unwrap().values().data()
                );
                let actual = calibrate(out.values().data(), 2.40605).unwrap();
                for output in batched
                    .forward_batch(vec![input.clone(), input.clone()])
                    .unwrap()
                {
                    let p = calibrate(output.values().data(), 2.40605).unwrap();
                    assert_eq!(
                        huncho_core::calibration::argmax(&p),
                        huncho_core::calibration::argmax(&actual)
                    );
                    assert!(p.iter().zip(&actual).all(|(a, b)| (a - b).abs() <= 1e-4));
                }
                if input.positions.iter().all(|&p| p >= 3) {
                    let parent = native.prefill(&input.tokens[..3]).unwrap();
                    let branch = native.fork(parent).unwrap();
                    native.release_cache(parent).unwrap();
                    let mut suffix = ForwardInput::new(
                        input.tokens[3..].to_vec(),
                        input.positions.iter().map(|p| p - 3).collect(),
                    );
                    suffix.fork_from = Some(branch);
                    let p = calibrate(native.forward(suffix).unwrap().values().data(), 2.40605)
                        .unwrap();
                    assert_eq!(
                        huncho_core::calibration::argmax(&p),
                        huncho_core::calibration::argmax(&actual)
                    );
                    assert!(p.iter().zip(&actual).all(|(a, b)| (a - b).abs() <= 1e-4));
                    native.release_cache(branch).unwrap();
                }
                let expected: Vec<f32> =
                    serde_json::from_value(row["probabilities"].clone()).unwrap();
                for (a, b) in actual.iter().zip(expected) {
                    delta = delta.max((a - b).abs());
                }
            }
        }
        eprintln!("{dtype}: unrefitted fixture maximum probability delta={delta}; no observed-label acceptance");
    }
    assert_eq!(std::fs::read(source).unwrap(), original);
    assert!(quantize_gguf_cpu(Path::new("absent.gguf"), Path::new("unused"), "int8", 1).is_err());
    assert!(quantize_gguf_cpu(
        Path::new("absent.gguf"),
        Path::new("unused"),
        "gguf-q8_0",
        0
    )
    .is_err());
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

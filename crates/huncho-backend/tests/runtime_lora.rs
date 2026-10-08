//! Native immutable CPU LoRA dispatch preserves the original typed probability gates.
#![cfg(feature = "candle")]
use huncho_backend::Qwen3_5Backend;
use huncho_core::{
    backend::{Backend, ForwardInput},
    calibration::{argmax, calibrate},
};
use std::path::Path;

fn load(root: &Path, runtime: bool) -> Qwen3_5Backend {
    let backend = if runtime {
        Qwen3_5Backend::load_kev_runtime_lora(root, root, &root.join("head.pt"), 512, "fp32")
    } else {
        Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, "fp32")
    }
    .unwrap();
    if runtime {
        backend.with_cpu_blas_from_env().unwrap()
    } else {
        backend
    }
}
fn parity(a: &[f32], b: &[f32], t: f32, bound: f32) {
    let (a, b) = (calibrate(a, t).unwrap(), calibrate(b, t).unwrap());
    assert_eq!(argmax(&a), argmax(&b), "t={t}: {a:?} vs {b:?}");
    assert!(
        a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= bound),
        "{a:?} vs {b:?}"
    );
}
fn input(row: &serde_json::Value) -> ForwardInput {
    ForwardInput::new(
        serde_json::from_value(row["tokens"].clone()).unwrap(),
        serde_json::from_value(row["positions"].clone()).unwrap(),
    )
}
#[test]
fn runtime_adapters_preserve_frozen_typed_scores_batches_forks_and_owned_outputs() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for optimized in [false, true] {
        let mut merged = load(root, false);
        let mut runtime = load(root, true);
        if optimized {
            runtime = runtime
                .with_kv_page_tokens(16)
                .unwrap()
                .with_prefill_chunk_tokens(3)
                .unwrap()
                .with_grouped_gqa(true)
                .unwrap()
                .with_attention_query_rows(7)
                .unwrap()
                .with_cpu_delta_rule(true)
                .unwrap()
                .with_cpu_causal_conv(true)
                .unwrap()
                .with_cpu_fused_gate(true)
                .unwrap()
                .with_projection_chunk_rows(16)
                .unwrap();
        }
        assert_eq!(
            runtime.capabilities().extra["adapter_execution"],
            "cpu-fp32-runtime-lora-v1"
        );
        assert_eq!(runtime.capabilities().extra["runtime_lora_targets"], "3");
        for case in golden["cases"].as_array().unwrap() {
            let rows = case["rows"].as_array().unwrap();
            let first = input(&rows[0]);
            let shared = rows[0]["prefix_len"].as_u64().unwrap() as usize;
            let cached = runtime
                .prefill_cached(&first.tokens[..shared], 1 << 20)
                .unwrap();
            let held = runtime.forward(first.clone()).unwrap();
            let held_values = held.values().data().to_vec();
            for row in rows {
                let row_input = input(row);
                let expected = merged.forward(row_input.clone()).unwrap();
                let actual = runtime.forward(row_input.clone()).unwrap();
                for t in [0.75, 1., 2.40605] {
                    parity(actual.values().data(), expected.values().data(), t, 1e-4);
                }
                let frozen: Vec<f32> =
                    serde_json::from_value(row["probabilities"].clone()).unwrap();
                let p = calibrate(actual.values().data(), 2.40605).unwrap();
                assert_eq!(argmax(&p), argmax(&frozen));
                assert!(p.iter().zip(&frozen).all(|(a, b)| (a - b).abs() <= 1e-3));
                let group = runtime
                    .forward_batch(vec![row_input.clone(), row_input.clone()])
                    .unwrap();
                for batch in &group {
                    parity(batch.values().data(), actual.values().data(), 2.40605, 1e-4);
                }
                let mut shorter = row_input.clone();
                shorter.tokens.pop();
                if shorter.positions.iter().all(|p| *p < shorter.tokens.len()) {
                    let separate = runtime.forward(shorter.clone()).unwrap();
                    let mixed = runtime
                        .forward_padded_batch(vec![row_input.clone(), shorter])
                        .unwrap();
                    parity(
                        mixed[0].values().data(),
                        actual.values().data(),
                        2.40605,
                        1e-4,
                    );
                    parity(
                        mixed[1].values().data(),
                        separate.values().data(),
                        2.40605,
                        1e-4,
                    );
                }
                let branch = runtime.fork(cached.handle).unwrap();
                let mut suffix = ForwardInput::new(
                    row_input.tokens[shared..].to_vec(),
                    row_input.positions.iter().map(|p| p - shared).collect(),
                );
                suffix.fork_from = Some(branch);
                let forked = runtime.forward(suffix).unwrap();
                parity(
                    forked.values().data(),
                    actual.values().data(),
                    2.40605,
                    1e-4,
                );
                runtime.release_cache(branch).unwrap();
            }
            assert_eq!(held.values().data(), held_values);
            runtime.release_cache(cached.handle).unwrap();
            let hit = runtime
                .prefill_cached(&first.tokens[..shared], 1 << 20)
                .unwrap();
            assert!(hit.hit);
            runtime.release_cache(hit.handle).unwrap();
            runtime.clear_prefix_cache().unwrap();
        }
        let mut replica = runtime.replica().unwrap();
        let first = input(&golden["cases"][0]["rows"][0]);
        let expected = runtime.forward(first.clone()).unwrap();
        drop(runtime);
        let actual = replica.forward(first).unwrap();
        assert_eq!(expected.values().data(), actual.values().data());
    }
}
#[test]
fn separate_live_adapters_keep_weights_and_prefixes_isolated_and_reject_other_precision() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
        }
    }
    let mut a = load(tmp.path(), true);
    let first = input(&golden["cases"][0]["rows"][0]);
    let before = a.forward(first.clone()).unwrap();
    let original = before.values().data().to_vec();
    let handle = a.prefill(&first.tokens[..3]).unwrap();
    let weights = candle::safetensors::load(
        tmp.path().join("adapter_model.safetensors"),
        &candle::Device::Cpu,
    )
    .unwrap();
    let changed: std::collections::HashMap<_, _> = weights
        .into_iter()
        .map(|(n, t)| {
            let t = if n.ends_with(".lora_B.weight") {
                t.affine(-3., 0.).unwrap()
            } else {
                t
            };
            (n, t)
        })
        .collect();
    candle::safetensors::save(&changed, tmp.path().join("adapter_model.safetensors")).unwrap();
    let mut b = load(tmp.path(), true);
    let mut reference = load(tmp.path(), false);
    assert!(b.fork(handle).is_err());
    let different = b.forward(first.clone()).unwrap();
    let merged = reference.forward(first.clone()).unwrap();
    parity(
        different.values().data(),
        merged.values().data(),
        2.40605,
        1e-4,
    );
    assert!(different
        .values()
        .data()
        .iter()
        .zip(&original)
        .any(|(a, b)| (a - b).abs() > 1e-5));
    assert_eq!(a.forward(first).unwrap().values().data(), original);
    assert_eq!(before.values().data(), original);
    a.release_cache(handle).unwrap();
    assert!(
        Qwen3_5Backend::load_kev_runtime_lora(root, root, &root.join("head.pt"), 512, "fp16")
            .is_err()
    );
}

#[test]
fn f3_runtime_adapters_preserve_raw_vocabulary_and_selected_candidate_batch_scores() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    std::fs::copy(root.join("config.json"), tmp.path().join("config.json")).unwrap();
    let mut weights =
        candle::safetensors::load(root.join("model.safetensors"), &candle::Device::Cpu).unwrap();
    let embedding = weights["model.language_model.embed_tokens.weight"].clone();
    let vocab = embedding.dims()[0];
    weights.insert("lm_head.weight".into(), embedding);
    weights.insert(
        "lm_head.bias".into(),
        candle::Tensor::new(
            (0..vocab)
                .map(|n| (n % 23) as f32 * 0.01)
                .collect::<Vec<_>>(),
            &candle::Device::Cpu,
        )
        .unwrap(),
    );
    candle::safetensors::save(&weights, tmp.path().join("model.safetensors")).unwrap();
    let mut merged = Qwen3_5Backend::load(tmp.path(), Some(root), 512, "fp32").unwrap();
    let mut runtime = Qwen3_5Backend::load_runtime_lora(tmp.path(), root, 512, "fp32")
        .unwrap()
        .with_cpu_blas_from_env()
        .unwrap()
        .with_grouped_gqa(true)
        .unwrap()
        .with_attention_query_rows(7)
        .unwrap()
        .with_cpu_delta_rule(true)
        .unwrap()
        .with_cpu_causal_conv(true)
        .unwrap();
    assert!(!runtime.capabilities().supports_fork);
    for case in golden["cases"].as_array().unwrap() {
        for row in case["rows"].as_array().unwrap() {
            let mut full = input(row);
            full.positions = vec![full.tokens.len() - 1];
            let expected = merged.forward(full.clone()).unwrap();
            let actual = runtime.forward(full.clone()).unwrap();
            for codes in [
                vec![31, 7],
                vec![19, 3, 31],
                vec![(vocab - 1) as u32, 1, 19, 7],
            ] {
                let expected_scores: Vec<_> = codes
                    .iter()
                    .map(|c| expected.values().data()[*c as usize])
                    .collect();
                let raw: Vec<_> = codes
                    .iter()
                    .map(|c| actual.values().data()[*c as usize])
                    .collect();
                let selected = full.clone().with_logit_codes(codes.clone());
                let compact = runtime.forward(selected.clone()).unwrap();
                let group = runtime
                    .forward_batch(vec![selected.clone(), selected])
                    .unwrap();
                for t in [0.75, 1., 2.40605] {
                    parity(&raw, &expected_scores, t, 1e-4);
                    for output in std::iter::once(&compact).chain(group.iter()) {
                        let huncho_core::backend::ForwardOutput::SelectedLogits { codes, .. } =
                            output
                        else {
                            panic!("expected selected logits")
                        };
                        let selected_reference: Vec<_> = codes
                            .iter()
                            .map(|code| expected.values().data()[*code as usize])
                            .collect();
                        parity(output.values().data(), &selected_reference, t, 1e-4);
                    }
                }
            }
        }
    }
}

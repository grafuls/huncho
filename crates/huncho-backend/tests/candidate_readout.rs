//! Native F3 compact readout versus the unchanged full-vocabulary reference path.
#![cfg(feature = "candle")]

use huncho_backend::Qwen3_5Backend;
use huncho_core::backend::{Backend, ForwardInput, ForwardOutput};
use huncho_core::calibration::{argmax, calibrate};
use std::path::Path;

#[test]
fn candidate_only_projection_preserves_probabilities_at_existing_temperatures() {
    assert_candidate_projection_parity(candle::Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a compatible CUDA GPU"]
fn cuda_candidate_projection_keeps_the_vocabulary_head_on_the_execution_device() {
    let device = huncho_backend::device::device_from_env().unwrap();
    assert!(device.is_cuda(), "GPU test must run on CUDA");
    assert_candidate_projection_parity(device);
}

fn assert_candidate_projection_parity(device: candle::Device) {
    let root = Path::new("tests/fixtures/tiny_kev");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    // Kev deliberately omits the LM head. Reuse its reference backbone/LoRA
    // and add a deterministic tied projection and bias in a temporary package.
    let package = tempfile::tempdir().unwrap();
    std::fs::copy(root.join("config.json"), package.path().join("config.json")).unwrap();
    let mut tensors =
        candle::safetensors::load(root.join("model.safetensors"), &candle::Device::Cpu).unwrap();
    let embedding = tensors
        .iter()
        .find(|(name, _)| name.ends_with("embed_tokens.weight"))
        .unwrap()
        .1
        .clone();
    tensors.insert("lm_head.weight".into(), embedding);
    tensors.insert(
        "lm_head.bias".into(),
        candle::Tensor::new(
            (0..384)
                .map(|code| (code % 23) as f32 * 0.01)
                .collect::<Vec<_>>(),
            &candle::Device::Cpu,
        )
        .unwrap(),
    );
    candle::safetensors::save(&tensors, package.path().join("model.safetensors")).unwrap();
    for dtype in ["fp32", "fp16"] {
        let mut backend =
            Qwen3_5Backend::load_on_device(package.path(), Some(root), 512, dtype, device.clone())
                .unwrap();
        let mut cpu = Qwen3_5Backend::load(package.path(), Some(root), 512, dtype).unwrap();
        assert_eq!(
            backend
                .capabilities()
                .extra
                .get("device_path")
                .map(String::as_str),
            device.is_cuda().then_some("qwen-f3-cuda")
        );
        for case in golden["cases"].as_array().unwrap() {
            for row in case["rows"].as_array().unwrap() {
                let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
                // The old path projected the same final hidden row per candidate.
                let last = tokens.len() - 1;
                let full = backend
                    .forward(ForwardInput::new(tokens.clone(), vec![last; 3]))
                    .unwrap();
                let cpu_full = cpu
                    .forward(ForwardInput::new(tokens.clone(), vec![last; 3]))
                    .unwrap();
                for codes in [vec![31, 7], vec![19, 3, 31], vec![383, 1, 19, 7]] {
                    let reference: Vec<_> = codes
                        .iter()
                        .map(|&code| cpu_full.values().row(0).unwrap()[code as usize])
                        .collect();
                    let actual: Vec<_> = codes
                        .iter()
                        .map(|&code| full.values().row(0).unwrap()[code as usize])
                        .collect();
                    for temperature in [0.75, 1.0, 2.40605] {
                        let (reference, actual) = (
                            calibrate(&reference, temperature).unwrap(),
                            calibrate(&actual, temperature).unwrap(),
                        );
                        assert_eq!(argmax(&reference), argmax(&actual));
                        assert!(reference
                            .iter()
                            .zip(&actual)
                            .all(|(a, b)| (a - b).abs() <= 1e-3));
                    }
                    let compact = backend
                        .forward(
                            ForwardInput::new(tokens.clone(), vec![last])
                                .with_logit_codes(codes.clone()),
                        )
                        .unwrap();
                    let batch = backend
                        .forward_batch(vec![
                            ForwardInput::new(tokens.clone(), vec![last])
                                .with_logit_codes(codes.clone()),
                            ForwardInput::new(tokens.clone(), vec![last, last])
                                .with_logit_codes(vec![19, 31]),
                        ])
                        .unwrap();
                    let batch_full = backend
                        .forward(ForwardInput::new(tokens.clone(), vec![last, last]))
                        .unwrap();
                    for output in &batch {
                        let ForwardOutput::SelectedLogits { codes, values, .. } = output else {
                            panic!("expected selected batch logits")
                        };
                        for row in 0..values.shape()[0] {
                            let expected: Vec<_> = codes
                                .iter()
                                .map(|&code| batch_full.values().row(row).unwrap()[code as usize])
                                .collect();
                            for temperature in [0.75, 1.0, 2.40605] {
                                let a = calibrate(values.row(row).unwrap(), temperature).unwrap();
                                let b = calibrate(&expected, temperature).unwrap();
                                assert_eq!(argmax(&a), argmax(&b));
                                assert!(a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= 1e-4));
                            }
                        }
                    }
                    let ForwardOutput::SelectedLogits {
                        codes: selected_codes,
                        values,
                        ..
                    } = &compact
                    else {
                        panic!("expected compact logits");
                    };
                    assert_eq!(values.shape(), &[1, codes.len()]);
                    let expected = codes
                        .iter()
                        .map(|&code| full.values().row(0).unwrap()[code as usize])
                        .collect::<Vec<_>>();
                    let actual = codes
                        .iter()
                        .map(|code| {
                            values.row(0).unwrap()[selected_codes
                                .iter()
                                .position(|selected| selected == code)
                                .unwrap()]
                        })
                        .collect::<Vec<_>>();
                    for temperature in [0.75, 1.0, 2.40605] {
                        let reference = calibrate(&expected, temperature).unwrap();
                        let optimized = calibrate(&actual, temperature).unwrap();
                        let max_delta = reference
                            .iter()
                            .zip(&optimized)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f32, f32::max);
                        assert!(max_delta <= 1e-4, "{dtype}: delta={max_delta}, reference={reference:?}, optimized={optimized:?}");
                        assert_eq!(argmax(&reference), argmax(&optimized));
                    }
                }
            }
        }
        assert!(backend
            .forward(ForwardInput::new(vec![1, 2], vec![1]).with_logit_codes(vec![384]))
            .is_err());
        assert!(backend
            .forward(ForwardInput::new(vec![1, 2], vec![1]).with_logit_codes(Vec::new()))
            .is_err());
    }
}

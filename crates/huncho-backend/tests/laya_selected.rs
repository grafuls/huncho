//! Final-layer marker selection uses all keys/values and the trained scorer.
#![cfg(feature = "candle")]
#[path = "support/laya.rs"]
mod fixture;

use huncho_backend::CandleBackend;
use huncho_core::{
    backend::{Backend, ForwardInput, ForwardOutput},
    calibration::{argmax, calibrate},
};
use std::path::Path;

fn parity(actual: &ForwardOutput, reference: &ForwardOutput) {
    assert!(matches!(actual, ForwardOutput::Logits { .. }));
    assert_eq!(actual.positions(), reference.positions());
    assert_eq!(actual.values().shape(), &[actual.positions().len(), 1]);
    for (&a, &b) in actual.values().data().iter().zip(reference.values().data()) {
        assert!((a - b).abs() <= 1e-5, "{a} != {b}");
    }
    for temperature in [0.5, 1., 1.9063563, 1.9833995, 1.25143] {
        let a = calibrate(actual.values().data(), temperature).unwrap();
        let b = calibrate(reference.values().data(), temperature).unwrap();
        assert_eq!(argmax(&a), argmax(&b));
        assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 1e-4));
    }
}

#[test]
fn selected_head_preserves_types_order_repetition_native_batches_and_replicas() {
    let package = tempfile::tempdir().unwrap();
    fixture::write_package(Path::new("tests/fixtures/tiny_modernbert"), package.path());
    let load = || {
        CandleBackend::load(
            package.path().join("config.json"),
            package.path().join("model.safetensors"),
            128,
            "fp32",
        )
        .unwrap()
    };
    let mut reference = load();
    let mut selected = load().with_selected_laya_head(true).unwrap();
    assert!(!reference
        .capabilities()
        .extra
        .contains_key("laya_head_execution"));
    assert_eq!(
        selected.capabilities().extra["laya_head_execution"],
        "marker-queries-last-layer-v1"
    );
    let mut replica = selected.replica().unwrap();
    assert_eq!(replica.capabilities().extra, selected.capabilities().extra);
    for len in [1, 5, 17, 65] {
        let mut inputs = Vec::new();
        for qtype in 0..3 {
            for positions in [
                vec![len - 1],
                vec![len - 1, 0, len / 2, 0],
                (0..len).rev().collect(),
            ] {
                let mut input = ForwardInput::new(
                    (0..len).map(|i| ((i * 19 + 3) % 32768) as u32).collect(),
                    positions,
                );
                input.qtype = qtype;
                parity(
                    &selected.forward(input.clone()).unwrap(),
                    &reference.forward(input.clone()).unwrap(),
                );
                let repeat = replica.forward(input.clone()).unwrap();
                assert_eq!(
                    repeat.values().data(),
                    selected.forward(input.clone()).unwrap().values().data()
                );
                inputs.push(input);
            }
        }
        let a = selected.forward_batch(inputs.clone()).unwrap();
        let b = reference.forward_batch(inputs).unwrap();
        for (a, b) in a.iter().zip(&b) {
            parity(a, b);
        }
    }
    assert!(selected.with_selected_laya_head(false).is_err());
    let mut invalid = ForwardInput::new(vec![1, 2], vec![2]);
    assert!(replica.forward(invalid.clone()).is_err());
    invalid.positions = vec![0];
    invalid.qtype = 3;
    assert!(replica.forward(invalid).is_err());
}

#[test]
fn bare_encoders_refuse_selected_trained_head() {
    let root = Path::new("tests/fixtures/tiny_modernbert");
    let load = || {
        CandleBackend::load(
            root.join("config.json"),
            root.join("model.safetensors"),
            128,
            "fp32",
        )
        .unwrap()
    };
    assert!(load().with_selected_laya_head(true).is_err());
    assert!(load().with_selected_laya_head(false).is_ok());
}

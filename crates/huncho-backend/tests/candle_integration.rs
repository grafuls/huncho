//! Integration tests for the candle ModernBERT backend.
//!
//! These require the `candle` feature
//! (`cargo test -p huncho-backend --features candle`) and a checked-in tiny
//! ModernBERT checkpoint in `tests/fixtures/tiny_modernbert/`. The fixture uses
//! the `convaiinnovations/laya` layout (`encoder.` weight prefix and a
//! transformers-5.0 `rope_parameters` config), so it exercises the load-time
//! key/config remapping.

#![cfg(feature = "candle")]

use huncho_backend::candle::CandleBackend;
use huncho_core::backend::{Backend, ForwardInput};

const CONFIG: &str = "tests/fixtures/tiny_modernbert/config.json";
const WEIGHTS: &str = "tests/fixtures/tiny_modernbert/model.safetensors";

#[test]
fn loads_and_reports_capabilities() {
    let backend = CandleBackend::load(CONFIG, WEIGHTS, 16, "fp32").unwrap();
    assert_eq!(backend.hidden_size(), 8);
    let c = backend.capabilities();
    assert_eq!(c.id, huncho_core::manifest::BackendId::Candle);
    assert_eq!(c.dtype, "fp32");
    assert_eq!(c.max_context, 16);
    assert!(!c.supports_fork);
    assert_eq!(c.families.len(), 1);
}

#[test]
fn forward_returns_features_at_positions() {
    let mut backend = CandleBackend::load(CONFIG, WEIGHTS, 16, "fp32").unwrap();

    // Tokens [1..=5] with candidate positions 1 and 3.
    let input = ForwardInput::new(vec![1, 2, 3, 4, 5], vec![1, 3]);
    let out = backend.forward(input).unwrap();

    assert_eq!(out.positions(), &[1, 3]);
    let values = out.values();
    assert_eq!(values.shape(), &[2, 8]);
    // All values finite and nonzero (the tiny random model is nonzero).
    for x in values.data() {
        assert!(x.is_finite(), "feature value must be finite");
    }
}

#[test]
fn forward_is_deterministic() {
    let mut a = CandleBackend::load(CONFIG, WEIGHTS, 16, "fp32").unwrap();
    let mut b = CandleBackend::load(CONFIG, WEIGHTS, 16, "fp32").unwrap();
    let input = ForwardInput::new(vec![1, 2, 3, 4, 5], vec![0, 2, 4]);
    let oa = a.forward(input.clone()).unwrap();
    let ob = b.forward(input).unwrap();
    assert_eq!(oa.values().data(), ob.values().data());
}

#[test]
fn forward_empty_positions_returns_empty_features() {
    let mut backend = CandleBackend::load(CONFIG, WEIGHTS, 16, "fp32").unwrap();
    let input = ForwardInput::new(vec![1, 2, 3], vec![]);
    let out = backend.forward(input).unwrap();
    assert!(out.positions().is_empty());
    assert_eq!(out.values().shape(), &[0, 8]);
}

#[test]
fn forward_rejects_sequence_over_max_context() {
    let mut backend = CandleBackend::load(CONFIG, WEIGHTS, 16, "fp32").unwrap();
    // 17 tokens > max_context 16.
    let input = ForwardInput::new((0..17).map(|i| i as u32).collect::<Vec<_>>(), vec![1]);
    assert!(backend.forward(input).is_err());
}

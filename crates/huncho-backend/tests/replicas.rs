//! Shared CPU model weights, independently owned mutable execution contexts.
#![cfg(feature = "candle")]
use huncho_backend::{CandleBackend, Qwen3_5Backend};
use huncho_core::backend::{Backend, ForwardInput};
use std::path::Path;

fn concurrent_unchanged_outputs(mut primary: Box<dyn Backend>, input: ForwardInput) {
    let expected = primary.forward(input.clone()).unwrap();
    let expected: Vec<u32> = expected
        .values()
        .data()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let replicas: Vec<_> = (0..4).map(|_| primary.replica().unwrap()).collect();
    std::thread::scope(|scope| {
        let jobs: Vec<_> = replicas
            .into_iter()
            .map(|mut replica| {
                let input = input.clone();
                let expected = &expected;
                scope.spawn(move || {
                    for _ in 0..8 {
                        let result = replica.forward(input.clone()).unwrap();
                        assert_eq!(
                            result
                                .values()
                                .data()
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            *expected
                        );
                    }
                })
            })
            .collect();
        for job in jobs {
            job.join().unwrap();
        }
    });
}

#[test]
fn cpu_modernbert_and_native_kev_replicas_preserve_independent_float_bits_concurrently() {
    let modernbert = Path::new("tests/fixtures/tiny_modernbert");
    concurrent_unchanged_outputs(
        Box::new(
            CandleBackend::load(
                modernbert.join("config.json"),
                modernbert.join("model.safetensors"),
                512,
                "fp32",
            )
            .unwrap(),
        ),
        ForwardInput::new(vec![1, 2, 3, 4], vec![1, 3]),
    );
    let kev = Path::new("tests/fixtures/tiny_kev");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(kev.join("golden.json")).unwrap()).unwrap();
    let row = &golden["cases"][0]["rows"][0];
    let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
    let positions: Vec<usize> = serde_json::from_value(row["positions"].clone()).unwrap();
    for dtype in ["fp32", "fp16"] {
        let backend = Qwen3_5Backend::load_kev(kev, kev, &kev.join("head.pt"), 512, dtype)
            .unwrap()
            .with_cpu_delta_rule(true)
            .unwrap()
            .with_cpu_causal_conv(true)
            .unwrap();
        concurrent_unchanged_outputs(
            Box::new(backend),
            ForwardInput::new(tokens.clone(), positions.clone()),
        );
    }
}

#[cfg(feature = "quantization")]
#[test]
fn packed_cpu_kev_replicas_keep_the_same_packed_arithmetic_and_independent_state() {
    use huncho_backend::qwen3_5::quantized::{convert_kev, Scheme};
    let root = Path::new("tests/fixtures/tiny_kev");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let row = &reference["cases"][0]["rows"][0];
    for scheme in [Scheme::Q8_0, Scheme::Q4_0] {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("backbone.gguf");
        convert_kev(
            root,
            root,
            scheme,
            &mut std::fs::File::create(&path).unwrap(),
        )
        .unwrap();
        let mut backend =
            Qwen3_5Backend::load_quantized_kev(&path, &root.join("head.pt"), 512, scheme.dtype())
                .unwrap();
        let prefix = backend.prefill_cached(&[1, 2, 3], 1 << 20).unwrap();
        let mut replica = backend.replica().unwrap();
        assert!(replica.fork(prefix.handle).is_err());
        let fresh = replica.prefill_cached(&[1, 2, 3], 1 << 20).unwrap();
        assert!(!fresh.hit);
        replica.release_cache(fresh.handle).unwrap();
        backend.release_cache(prefix.handle).unwrap();
        concurrent_unchanged_outputs(
            Box::new(backend),
            ForwardInput::new(
                serde_json::from_value(row["tokens"].clone()).unwrap(),
                serde_json::from_value(row["positions"].clone()).unwrap(),
            ),
        );
    }
}

//! Actual independent LoRA merges over one immutable CPU base residency.
#![cfg(feature = "shared-base")]
use huncho_backend::{shared_base::BaseWeightCache, Qwen3_5Backend};
use huncho_core::backend::{Backend, ForwardInput};
use std::path::Path;

#[test]
fn distinct_adapters_share_only_unmerged_storage_and_preserve_independent_logits() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let other = tempfile::tempdir().unwrap();
    std::fs::copy(
        root.join("adapter_model.safetensors"),
        other.path().join("adapter_model.safetensors"),
    )
    .unwrap();
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("adapter_config.json")).unwrap()).unwrap();
    config["lora_alpha"] = serde_json::json!(7);
    std::fs::write(
        other.path().join("adapter_config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for dtype in ["fp32", "fp16"] {
        let cache = BaseWeightCache::new(8 << 20);
        let mut independent_a =
            Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype).unwrap();
        let mut independent_b =
            Qwen3_5Backend::load_kev(root, other.path(), &root.join("head.pt"), 512, dtype)
                .unwrap();
        let mut shared_a = Qwen3_5Backend::load_kev_with_base_cache(
            root,
            root,
            &root.join("head.pt"),
            512,
            dtype,
            &cache,
        )
        .unwrap();
        let mut shared_b = Qwen3_5Backend::load_kev_with_base_cache(
            root,
            other.path(),
            &root.join("head.pt"),
            512,
            dtype,
            &cache,
        )
        .unwrap();
        assert_eq!(cache.stats().unwrap().hits, 1);
        assert_eq!(cache.stats().unwrap().misses, 1);
        assert_eq!(cache.stats().unwrap().entries, 1);
        assert!(cache.stats().unwrap().charged_bytes <= 8 << 20);
        // Eviction cannot invalidate live models or make an adapter inherit a
        // different adapter's merge. Shared mutable prefix state is absent.
        cache.clear().unwrap();
        let handle = shared_a.prefill_cached(&[1, 2, 3], 1 << 20).unwrap();
        assert!(shared_b.fork(handle.handle).is_err());
        shared_a.release_cache(handle.handle).unwrap();
        let mut different = false;
        for case in golden["cases"].as_array().unwrap() {
            for row in case["rows"].as_array().unwrap() {
                let input = ForwardInput::new(
                    serde_json::from_value(row["tokens"].clone()).unwrap(),
                    serde_json::from_value(row["positions"].clone()).unwrap(),
                );
                let a = independent_a.forward(input.clone()).unwrap();
                let b = independent_b.forward(input.clone()).unwrap();
                let shared_a = shared_a.forward(input.clone()).unwrap();
                let shared_b = shared_b.forward(input).unwrap();
                let bits = |output: &huncho_core::backend::ForwardOutput| {
                    output
                        .values()
                        .data()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&a), bits(&shared_a));
                assert_eq!(bits(&b), bits(&shared_b));
                different |= bits(&a) != bits(&b);
                for temperature in [0.75, 1.0, 2.40605] {
                    for (expected, actual) in [(&a, &shared_a), (&b, &shared_b)] {
                        assert_eq!(
                            huncho_core::calibration::calibrate(
                                expected.values().data(),
                                temperature
                            )
                            .unwrap(),
                            huncho_core::calibration::calibrate(
                                actual.values().data(),
                                temperature
                            )
                            .unwrap()
                        );
                    }
                }
            }
        }
        assert!(
            different,
            "distinct adapter fixture must actually change inference"
        );
    }
}

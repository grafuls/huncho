// Deterministic nonzero test head, never a released trained/calibrated artifact.
use candle::{Device, Tensor};
use std::{collections::HashMap, path::Path};

pub fn head_tensors(d: usize, layer_count: usize) -> HashMap<String, Tensor> {
    let mut tensors = HashMap::new();
    let mut seed = 7u32;
    let mut insert = |key: String, shape: Vec<usize>, norm: bool| {
        let data: Vec<f32> = (0..shape.iter().product())
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let value = (seed >> 8) as f32 / 16777216. - 0.5;
                if norm {
                    1. + value / 8.
                } else {
                    value / (d as f32).sqrt()
                }
            })
            .collect();
        tensors.insert(key, Tensor::from_vec(data, shape, &Device::Cpu).unwrap());
    };
    insert("type_emb.weight".into(), vec![3, d], false);
    for i in 0..layer_count {
        let p = format!("head.layers.{i}");
        for (key, shape, norm) in [
            ("self_attn.in_proj_weight", vec![3 * d, d], false),
            ("self_attn.in_proj_bias", vec![3 * d], false),
            ("self_attn.out_proj.weight", vec![d, d], false),
            ("self_attn.out_proj.bias", vec![d], false),
            ("norm1.weight", vec![d], true),
            ("norm1.bias", vec![d], false),
            ("norm2.weight", vec![d], true),
            ("norm2.bias", vec![d], false),
            ("linear1.weight", vec![2 * d + 1, d], false),
            ("linear1.bias", vec![2 * d + 1], false),
            ("linear2.weight", vec![d, 2 * d + 1], false),
            ("linear2.bias", vec![d], false),
        ] {
            insert(format!("{p}.{key}"), shape, norm);
        }
    }
    for (key, shape, norm) in [
        ("scorer.0.weight", vec![d], true),
        ("scorer.0.bias", vec![d], false),
        ("scorer.1.weight", vec![d / 2 + 1, d], false),
        ("scorer.1.bias", vec![d / 2 + 1], false),
        ("scorer.3.weight", vec![1, d / 2 + 1], false),
        ("scorer.3.bias", vec![1], false),
    ] {
        insert(key.into(), shape, norm);
    }
    tensors
}

pub fn write_package(root: &Path, output: &Path) {
    let mut tensors =
        candle::safetensors::load(root.join("model.safetensors"), &Device::Cpu).unwrap();
    tensors.extend(head_tensors(8, 2));
    candle::safetensors::save(&tensors, output.join("model.safetensors")).unwrap();
    std::fs::copy(root.join("config.json"), output.join("config.json")).unwrap();
}

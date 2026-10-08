//! Standard inference-only CPU FP32 LoRA; immutable bases are never merged.
use super::canonical_weight_name;
use candle::{DType, Tensor};
use candle_nn::{Linear, Module};
use huncho_core::{Error, Result};
use serde_json::Value;
use std::{collections::HashMap, path::Path};

pub(super) struct Adapter {
    a: Linear,
    b: Linear,
    scale: f64,
}
impl Adapter {
    pub(super) fn forward(&self, input: &Tensor) -> candle::Result<Tensor> {
        if !input.device().is_cpu() || input.dtype() != DType::F32 {
            candle::bail!("runtime LoRA requires CPU FP32 activations")
        }
        self.b
            .forward(&self.a.forward(input)?)?
            .affine(self.scale, 0.)
    }
}
fn invalid(reason: &str) -> Error {
    Error::Package(format!("runtime LoRA: {reason}"))
}
fn inactive(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) | Some(Value::Bool(false)) => true,
        Some(Value::Array(a)) => a.is_empty(),
        Some(Value::Object(o)) => o.is_empty(),
        _ => false,
    }
}
fn parameters(config: &Value) -> Result<(usize, f64)> {
    if config.get("peft_type").and_then(Value::as_str) != Some("LORA")
        || config
            .get("bias")
            .is_some_and(|value| value.as_str() != Some("none"))
        || [
            "use_dora",
            "use_rslora",
            "use_qalora",
            "lora_bias",
            "fan_in_fan_out",
            "rank_pattern",
            "alpha_pattern",
            "modules_to_save",
            "alora_invocation_tokens",
            "arrow_config",
            "layer_replication",
            "trainable_token_indices",
            "megatron_config",
            "exclude_modules",
            "layers_to_transform",
            "layers_pattern",
            "loftq_config",
        ]
        .iter()
        .any(|name| !inactive(config.get(*name)))
    {
        return Err(invalid(
            "requires standard constant-rank, alpha/r, bias-free LoRA",
        ));
    }
    if let Some(init) = config.get("init_lora_weights") {
        if !matches!(init, Value::Bool(_)) && init.as_str() != Some("gaussian") {
            return Err(invalid(
                "base-changing or custom LoRA initialization is unsupported",
            ));
        }
    }
    let rank = config
        .get("r")
        .and_then(Value::as_u64)
        .and_then(|r| usize::try_from(r).ok())
        .filter(|r| (1..=256).contains(r))
        .ok_or_else(|| invalid("rank must be 1..256"))?;
    let alpha = config
        .get("lora_alpha")
        .and_then(Value::as_f64)
        .filter(|a| a.is_finite() && *a > 0. && *a <= f32::MAX as f64)
        .ok_or_else(|| invalid("alpha must be finite and positive"))?;
    Ok((rank, alpha / rank as f64))
}

pub(super) fn load(
    directory: &Path,
    base: &HashMap<String, Tensor>,
    head_rank: Option<usize>,
) -> Result<HashMap<String, Adapter>> {
    let config: Value =
        serde_json::from_slice(&std::fs::read(directory.join("adapter_config.json"))?)?;
    let (rank, scale) = parameters(&config)?;
    if head_rank.is_some_and(|head_rank| rank != head_rank) {
        return Err(invalid("adapter rank differs from trained Kev head"));
    }
    let tensors = candle::safetensors::load(
        directory.join("adapter_model.safetensors"),
        &candle::Device::Cpu,
    )
    .map_err(|e| invalid(&format!("loading adapter: {e}")))?;
    from_tensors(&config, tensors, base, rank, scale)
}

fn from_tensors(
    config: &Value,
    mut tensors: HashMap<String, Tensor>,
    base: &HashMap<String, Tensor>,
    rank: usize,
    scale: f64,
) -> Result<HashMap<String, Adapter>> {
    let mut targets = Vec::new();
    for key in tensors.keys() {
        let rest = key
            .strip_prefix("base_model.model.")
            .ok_or_else(|| invalid("unsupported tensor prefix"))?;
        let target = rest
            .strip_suffix(".lora_A.weight")
            .or_else(|| rest.strip_suffix(".lora_B.weight"))
            .ok_or_else(|| invalid("unsupported adapter tensor; complete A/B weights only"))?;
        targets.push(target.to_string());
    }
    targets.sort();
    targets.dedup();
    if targets.is_empty() {
        return Err(invalid("adapter has no projections"));
    }
    let mut adapters = HashMap::new();
    for target in targets {
        let name = canonical_weight_name(&format!("{target}.weight"))
            .filter(|n| n.starts_with("model.language_model.layers."))
            .ok_or_else(|| invalid("only backbone linear projections are supported"))?;
        let weight = base
            .get(&name)
            .ok_or_else(|| invalid("missing targeted base projection"))?;
        let (out, input) = weight
            .dims2()
            .map_err(|_| invalid("base projection must have rank two"))?;
        if weight.dtype() != DType::F32 || !weight.device().is_cpu() {
            return Err(invalid("base must be CPU FP32"));
        }
        let a = tensors
            .remove(&format!("base_model.model.{target}.lora_A.weight"))
            .ok_or_else(|| invalid("missing paired A weight"))?;
        let b = tensors
            .remove(&format!("base_model.model.{target}.lora_B.weight"))
            .ok_or_else(|| invalid("missing paired B weight"))?;
        if a.dims() != [rank, input] || b.dims() != [out, rank] {
            return Err(invalid("A/B projection dimensions do not match rank/base"));
        }
        let checked = |t: Tensor| -> Result<Tensor> {
            if !matches!(t.dtype(), DType::F16 | DType::BF16 | DType::F32) {
                return Err(invalid("adapter requires floating FP16/BF16/FP32 weights"));
            }
            let t = t
                .to_dtype(DType::F32)
                .map_err(|e| invalid(&e.to_string()))?;
            let values = t
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| invalid(&e.to_string()))?;
            if values.iter().any(|v| !v.is_finite()) {
                return Err(invalid("adapter weights must be finite"));
            }
            Ok(t)
        };
        if adapters
            .insert(
                name,
                Adapter {
                    a: Linear::new(checked(a)?, None),
                    b: Linear::new(checked(b)?, None),
                    scale,
                },
            )
            .is_some()
        {
            return Err(invalid("duplicate canonical target"));
        }
    }
    if !tensors.is_empty() {
        return Err(invalid("unconsumed adapter weights"));
    }
    if let Some(modules) = config.get("target_modules").filter(|v| !v.is_null()) {
        let names = modules
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid("target_modules must be a nonempty literal array"))?;
        for target in names {
            let target = target
                .as_str()
                .filter(|s| {
                    !s.is_empty()
                        && s.chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
                })
                .ok_or_else(|| invalid("regex/all-linear target declarations are unsupported"))?;
            let suffix = format!(".{target}.weight");
            let mut found = false;
            for name in base.keys().filter(|name| name.ends_with(&suffix)) {
                found = true;
                if !adapters.contains_key(name) {
                    return Err(invalid("declared target has missing A/B weights"));
                }
            }
            if !found {
                return Err(invalid("declared target is absent from the base"));
            }
        }
        for name in adapters.keys() {
            if !names
                .iter()
                .any(|target| name.ends_with(&format!(".{}.weight", target.as_str().unwrap())))
            {
                return Err(invalid("adapter tensor is outside declared target_modules"));
            }
        }
    }
    Ok(adapters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;
    use serde_json::json;
    fn tensors() -> (HashMap<String, Tensor>, HashMap<String, Tensor>) {
        let tensor =
            |values: Vec<f32>, shape| Tensor::from_vec(values, shape, &Device::Cpu).unwrap();
        (
            HashMap::from([(
                "model.language_model.layers.0.mlp.up_proj.weight".into(),
                tensor(vec![1., 2., 3., 4., 5., 6.], (2, 3)),
            )]),
            HashMap::from([
                (
                    "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.weight"
                        .into(),
                    tensor(vec![1., 2., -1.], (1, 3)),
                ),
                (
                    "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_B.weight"
                        .into(),
                    tensor(vec![3., -2.], (2, 1)),
                ),
            ]),
        )
    }
    fn config() -> Value {
        json!({"peft_type":"LORA", "r":1, "lora_alpha":2, "bias":"none"})
    }
    #[test]
    fn runtime_lora_rejects_nonstandard_incomplete_and_malformed_semantics() {
        for field in [
            "use_dora",
            "use_rslora",
            "use_qalora",
            "lora_bias",
            "fan_in_fan_out",
            "rank_pattern",
            "alpha_pattern",
            "modules_to_save",
            "alora_invocation_tokens",
            "arrow_config",
            "layer_replication",
            "trainable_token_indices",
            "megatron_config",
            "exclude_modules",
            "layers_to_transform",
            "layers_pattern",
            "loftq_config",
        ] {
            let mut c = config();
            c[field] = json!(true);
            assert!(parameters(&c).is_err(), "{field}");
        }
        for r in [0, 257] {
            let mut c = config();
            c["r"] = json!(r);
            assert!(parameters(&c).is_err());
        }
        for alpha in [0., -1.] {
            let mut c = config();
            c["lora_alpha"] = json!(alpha);
            assert!(parameters(&c).is_err());
        }
        for field in ["bias", "init_lora_weights"] {
            let mut c = config();
            c[field] = json!("custom");
            assert!(parameters(&c).is_err());
        }
        let (base, mut weights) = tensors();
        weights.insert(
            "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_B.weight".into(),
            Tensor::from_vec(vec![f32::NAN, 1.], (2, 1), &Device::Cpu).unwrap(),
        );
        assert!(from_tensors(&config(), weights, &base, 1, 2.).is_err());
        let (base, mut weights) = tensors();
        for suffix in ["A", "B"] {
            let tensor = weights[&format!(
                "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_{suffix}.weight"
            )]
                .clone();
            weights.insert(
                format!("base_model.model.layers.0.mlp.up_proj.lora_{suffix}.weight"),
                tensor,
            );
        }
        assert!(from_tensors(&config(), weights, &base, 1, 2.).is_err());
        let (base, mut weights) = tensors();
        weights.remove("base_model.model.model.language_model.layers.0.mlp.up_proj.lora_B.weight");
        assert!(from_tensors(&config(), weights, &base, 1, 2.).is_err());
        let (base, mut weights) = tensors();
        weights.insert(
            "extra.bias".into(),
            Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
        );
        assert!(from_tensors(&config(), weights, &base, 1, 2.).is_err());
        let (base, weights) = tensors();
        assert!(from_tensors(&config(), weights, &base, 2, 2.).is_err());
        let (base, weights) = tensors();
        let mut c = config();
        c["target_modules"] = json!(["down_proj"]);
        assert!(from_tensors(&c, weights, &base, 1, 2.).is_err());
        let (base, weights) = tensors();
        let mut c = config();
        c["target_modules"] = json!("all-linear");
        assert!(from_tensors(&c, weights, &base, 1, 2.).is_err());
    }
    #[test]
    fn runtime_lora_keeps_base_and_applies_two_biased_base_plus_low_rank_rows() {
        let (base, weights) = tensors();
        let mut c = config();
        c["target_modules"] = json!(["up_proj"]);
        let mut adapters = from_tensors(&c, weights, &base, 1, 2.).unwrap();
        let weight = &base["model.language_model.layers.0.mlp.up_proj.weight"];
        let bias = Tensor::new(&[0.5f32, -0.5], &Device::Cpu).unwrap();
        let linear = Linear::new(weight.clone(), Some(bias));
        let x = Tensor::from_vec(vec![1f32, 0., 2., -1., 1., 1.], (1, 2, 3), &Device::Cpu).unwrap();
        let adapter = adapters
            .remove("model.language_model.layers.0.mlp.up_proj.weight")
            .unwrap();
        let result = linear
            .forward(&x)
            .unwrap()
            .broadcast_add(&adapter.forward(&x).unwrap())
            .unwrap();
        assert_eq!(
            result.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.5, 19.5, 4.5, 6.5]
        );
        assert_eq!(
            weight.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1., 2., 3., 4., 5., 6.]
        );
    }
}

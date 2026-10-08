//! CPU-only durable Kev projection quantization. GGUF stores merged backbone
//! projections as Q8_0/Q4_0; embeddings, norms, convolution and pointer head
//! remain FP32. This is an experimental execution, never a calibration approval.

use super::*;
#[cfg(test)]
use candle::quantized::QMatMul;
use candle::quantized::{gguf_file, GgmlDType, QTensor};
use std::io::{Read, Seek, Write};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    Q8_0,
    Q4_0,
}

impl Scheme {
    pub fn from_dtype(dtype: &str) -> Option<Self> {
        match dtype {
            "q8_0-fp32" => Some(Self::Q8_0),
            "q4_0-fp32" => Some(Self::Q4_0),
            _ => None,
        }
    }
    pub fn dtype(self) -> &'static str {
        match self {
            Self::Q8_0 => "q8_0-fp32",
            Self::Q4_0 => "q4_0-fp32",
        }
    }
    pub fn profile(self) -> &'static str {
        match self {
            Self::Q8_0 => "kev-projections-q8_0-fp32-v1",
            Self::Q4_0 => "kev-projections-q4_0-fp32-v1",
        }
    }
    fn ggml(self) -> GgmlDType {
        match self {
            Self::Q8_0 => GgmlDType::Q8_0,
            Self::Q4_0 => GgmlDType::Q4_0,
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct ConversionStats {
    pub scheme: &'static str,
    pub projection_count: usize,
    pub source_projection_bytes: usize,
    pub packed_projection_bytes: usize,
    pub dense_backbone_bytes: usize,
}

fn projection_names(cfg: &Config) -> Vec<String> {
    let mut names = Vec::new();
    for index in 0..cfg.num_hidden_layers {
        let root = format!("model.language_model.layers.{index}");
        for name in ["gate_proj", "up_proj", "down_proj"] {
            names.push(format!("{root}.mlp.{name}.weight"));
        }
        let (kind, projections): (&str, &[&str]) = match cfg
            .layer_types
            .get(index)
            .copied()
            .unwrap_or(LayerType::Linear)
        {
            LayerType::Linear => (
                "linear_attn",
                &[
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "out_proj",
                ],
            ),
            LayerType::Full => ("self_attn", &["q_proj", "k_proj", "v_proj", "o_proj"]),
        };
        for name in projections {
            names.push(format!("{root}.{kind}.{name}.weight"));
        }
    }
    names
}

/// Write one self-contained merged text-backbone GGUF to a caller-owned new
/// file. CPU staging is FP32, including LoRA merging; no temperature is fitted.
/// Conversion currently materializes the full dense backbone and packed output.
pub fn convert_kev<W: Write + Seek>(
    base_dir: &Path,
    adapter_dir: &Path,
    scheme: Scheme,
    writer: &mut W,
) -> CoreResult<ConversionStats> {
    let config_json = std::fs::read_to_string(base_dir.join("config.json"))?;
    let cfg = parse_config(&base_dir.join("config.json"))?;
    let mut tensors = load_base_tensors_filtered(base_dir, &Device::Cpu, DType::F32, false)?;
    let lora =
        candle::safetensors::load(adapter_dir.join("adapter_model.safetensors"), &Device::Cpu)
            .map_err(invalid)?;
    let (rank, alpha) = read_lora_hyperparams(&adapter_dir.join("adapter_config.json"))?;
    if rank == 0 {
        return Err(Error::Package(
            "quantization requires a positive LoRA rank".into(),
        ));
    }
    merge_lora_into_map(&mut tensors, &lora, alpha / rank as f32, DType::F32).map_err(invalid)?;
    for (name, tensor) in &tensors {
        require_finite(tensor).map_err(|e| invalid(format!("{name}: {e}")))?;
    }
    // Validate the actual merged model before producing a durable artifact.
    Model::new(
        &cfg,
        VarBuilder::from_tensors(tensors.clone(), DType::F32, &Device::Cpu),
        &Device::Cpu,
        DType::F32,
    )
    .map_err(invalid)?;
    let names = projection_names(&cfg);
    let mut widths = BTreeMap::new();
    let mut packed = BTreeMap::new();
    let mut stats = ConversionStats {
        scheme: scheme.profile(),
        projection_count: names.len(),
        source_projection_bytes: 0,
        packed_projection_bytes: 0,
        dense_backbone_bytes: 0,
    };
    for name in names {
        let tensor = tensors
            .remove(&name)
            .ok_or_else(|| invalid(format!("missing projection {name}")))?;
        let (rows, width) = tensor.dims2().map_err(invalid)?;
        let padded = width
            .checked_add(31)
            .ok_or_else(|| invalid("projection width overflow"))?
            / 32
            * 32;
        if rows == 0 || width == 0 {
            return Err(invalid("empty projection"));
        }
        stats.source_projection_bytes += tensor.elem_count() * 4;
        let tensor = if width == padded {
            tensor
        } else {
            Tensor::cat(
                &[
                    tensor,
                    Tensor::zeros((rows, padded - width), DType::F32, &Device::Cpu)
                        .map_err(invalid)?,
                ],
                1,
            )
            .map_err(invalid)?
        };
        let quantized = QTensor::quantize(&tensor, scheme.ggml()).map_err(invalid)?;
        stats.packed_projection_bytes += quantized.storage_size_in_bytes();
        widths.insert(name.clone(), width);
        packed.insert(name, quantized);
    }
    for (name, tensor) in tensors {
        stats.dense_backbone_bytes += tensor.elem_count() * 4;
        packed.insert(
            name,
            QTensor::quantize(&tensor, GgmlDType::F32).map_err(invalid)?,
        );
    }
    let metadata = [
        (
            "general.architecture",
            gguf_file::Value::String("huncho-kev-qwen3.5".into()),
        ),
        (
            "huncho.quantization",
            gguf_file::Value::String(scheme.profile().into()),
        ),
        ("huncho.config_json", gguf_file::Value::String(config_json)),
        (
            "huncho.input_widths",
            gguf_file::Value::String(serde_json::to_string(&widths)?),
        ),
    ];
    gguf_file::write(
        writer,
        &metadata.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>(),
        &packed
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
    )
    .map_err(invalid)?;
    Ok(stats)
}

fn invalid(error: impl std::fmt::Display) -> Error {
    Error::Package(format!("Kev quantization: {error}"))
}

fn require_finite(tensor: &Tensor) -> CoreResult<()> {
    let tensor = tensor.force_contiguous().map_err(invalid)?;
    let (storage, layout) = tensor.storage_and_layout();
    let candle::Storage::Cpu(candle::CpuStorage::F32(values)) = &*storage else {
        return Err(invalid("source weights must be CPU FP32"));
    };
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| invalid("noncontiguous source weights"))?;
    if !values[start..end].iter().all(|x| x.is_finite()) {
        return Err(invalid("source weights must be finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conversion_checks_only_visible_finite_cpu_fp32_values() {
        let tensor = Tensor::new(&[f32::NAN, 1.0, 2.0, f32::INFINITY], &Device::Cpu).unwrap();
        assert!(require_finite(&tensor).is_err());
        assert!(require_finite(&tensor.narrow(0, 1, 2).unwrap()).is_ok());
        assert!(require_finite(&tensor.to_dtype(DType::F16).unwrap()).is_err());
    }

    #[test]
    fn direct_packed_construction_keeps_projection_rows_out_of_the_dense_map_and_matches_the_previous_loader(
    ) {
        let root = Path::new("tests/fixtures/tiny_kev");
        let golden: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
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
            let (cfg, mut dense, packed) =
                read_backbone(&mut std::fs::File::open(&path).unwrap(), scheme).unwrap();
            assert_eq!(packed.len(), projection_names(&cfg).len());
            assert!(packed.keys().all(|name| !dense.contains_key(name)));
            // Reference-only reconstruction of the previous dequantize/build/
            // replace loader. Production never materializes these matrices.
            for (name, (weight, width)) in &packed {
                dense.insert(
                    name.clone(),
                    weight
                        .dequantize(&Device::Cpu)
                        .unwrap()
                        .narrow(1, 0, *width)
                        .unwrap()
                        .force_contiguous()
                        .unwrap(),
                );
            }
            let mut model = Model::new(
                &cfg,
                VarBuilder::from_tensors(dense, DType::F32, &Device::Cpu),
                &Device::Cpu,
                DType::F32,
            )
            .unwrap();
            install_packed(&mut model, packed).unwrap();
            let mut previous = Qwen3_5Backend {
                model: Arc::new(model),
                head: Arc::new(Readout::Pointer(
                    PointerHead::load(&root.join("head.pt"), cfg.hidden_size, &Device::Cpu)
                        .unwrap(),
                )),
                vocab_size: 1,
                input_vocab_size: cfg.vocab_size,
                max_context: 512,
                dtype: scheme.dtype().into(),
                device: Device::Cpu,
                caches: BTreeMap::new(),
                prefixes: PrefixSnapshots::default(),
                pending_prefills: BTreeMap::new(),
                prefill_chunk_tokens: 0,
                base_weight_cache: false,
            };
            let mut direct = Qwen3_5Backend::load_quantized_kev(
                &path,
                &root.join("head.pt"),
                512,
                scheme.dtype(),
            )
            .unwrap();
            for case in golden["cases"].as_array().unwrap() {
                for row in case["rows"].as_array().unwrap() {
                    let input = ForwardInput::new(
                        serde_json::from_value(row["tokens"].clone()).unwrap(),
                        serde_json::from_value(row["positions"].clone()).unwrap(),
                    );
                    let actual = direct.forward(input.clone()).unwrap();
                    let expected = previous.forward(input).unwrap();
                    assert_eq!(
                        actual
                            .values()
                            .data()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        expected
                            .values()
                            .data()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                    for temperature in [0.75, 1.0, 2.40605] {
                        assert_eq!(
                            huncho_core::calibration::calibrate(
                                actual.values().data(),
                                temperature
                            )
                            .unwrap(),
                            huncho_core::calibration::calibrate(
                                expected.values().data(),
                                temperature
                            )
                            .unwrap()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn loaded_projection_modules_release_dense_weights_and_retain_only_packed_blocks() {
        let fixture = Path::new("tests/fixtures/tiny_kev");
        for scheme in [Scheme::Q8_0, Scheme::Q4_0] {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("backbone.gguf");
            let stats = convert_kev(
                fixture,
                fixture,
                scheme,
                &mut std::fs::File::create(&path).unwrap(),
            )
            .unwrap();
            let backend = Qwen3_5Backend::load_quantized_kev(
                &path,
                &fixture.join("head.pt"),
                512,
                scheme.dtype(),
            )
            .unwrap();
            let mut projections = Vec::new();
            for layer in &backend.model.layers {
                projections.extend([
                    &layer.mlp.gate_proj,
                    &layer.mlp.up_proj,
                    &layer.mlp.down_proj,
                ]);
                if let Some(a) = &layer.linear_attn {
                    projections.extend([
                        &a.in_proj_qkv,
                        &a.in_proj_z,
                        &a.in_proj_b,
                        &a.in_proj_a,
                        &a.out_proj,
                    ]);
                }
                if let Some(a) = &layer.self_attn {
                    projections.extend([&a.q_proj, &a.k_proj, &a.v_proj, &a.o_proj]);
                }
            }
            assert_eq!(projections.len(), stats.projection_count);
            let mut bytes = 0;
            for projection in projections {
                let Projection::Packed {
                    matmul: QMatMul::QTensor(weight),
                    ..
                } = &projection.linear
                else {
                    panic!("dense weight or fallback retained");
                };
                assert_eq!(Arc::strong_count(weight), 1);
                assert_eq!(weight.dtype(), scheme.ggml());
                bytes += weight.storage_size_in_bytes();
            }
            assert_eq!(bytes, stats.packed_projection_bytes);
            assert!(bytes < stats.source_projection_bytes);
        }
    }
}

impl Qwen3_5Backend {
    /// Load the exact CPU packed artifact. CUDA/Metal and silent dequantized
    /// kernel substitution are excluded from this profile. Projection modules
    /// are constructed directly from packed blocks without temporary dense
    /// projection weights. Embeddings, norms, convolution and head remain FP32.
    pub fn load_quantized_kev(
        artifact: &Path,
        head: &Path,
        max_context: usize,
        dtype: &str,
    ) -> CoreResult<Self> {
        let scheme = Scheme::from_dtype(dtype).ok_or_else(|| invalid("unknown packed dtype"))?;
        let mut file = std::fs::File::open(artifact)?;
        let (cfg, tensors, quantized) = read_backbone(&mut file, scheme)?;
        if max_context == 0 || max_context > cfg.max_position_embeddings {
            return Err(invalid(
                "context budget exceeds embedded model configuration",
            ));
        }
        let model = Model::new_with_projections(
            &cfg,
            VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
            &Device::Cpu,
            DType::F32,
            &mut ProjectionSource::Packed(quantized),
        )
        .map_err(invalid)?;
        let head = Readout::Pointer(PointerHead::load(head, cfg.hidden_size, &Device::Cpu)?);
        Ok(Self {
            model: Arc::new(model),
            head: Arc::new(head),
            vocab_size: 1,
            input_vocab_size: cfg.vocab_size,
            max_context,
            dtype: dtype.into(),
            device: Device::Cpu,
            caches: BTreeMap::new(),
            prefixes: PrefixSnapshots::default(),
            pending_prefills: BTreeMap::new(),
            prefill_chunk_tokens: 0,
            base_weight_cache: false,
        })
    }
}

pub(super) type PackedWeights = HashMap<String, (Arc<QTensor>, usize)>;
fn read_backbone<R: Read + Seek>(
    reader: &mut R,
    scheme: Scheme,
) -> CoreResult<(Config, HashMap<String, Tensor>, PackedWeights)> {
    let content = gguf_file::Content::read(reader).map_err(invalid)?;
    let string = |key: &str| -> CoreResult<&str> {
        content
            .metadata
            .get(key)
            .ok_or_else(|| invalid(format!("missing {key}")))?
            .to_string()
            .map(String::as_str)
            .map_err(invalid)
    };
    if string("general.architecture")? != "huncho-kev-qwen3.5"
        || string("huncho.quantization")? != scheme.profile()
    {
        return Err(invalid(
            "artifact execution profile differs from the requested dtype",
        ));
    }
    let cfg = Config::from_value(&serde_json::from_str(string("huncho.config_json")?)?)
        .map_err(invalid)?;
    let widths: BTreeMap<String, usize> = serde_json::from_str(string("huncho.input_widths")?)?;
    let expected = projection_names(&cfg);
    if widths.len() != expected.len() || expected.iter().any(|name| !widths.contains_key(name)) {
        return Err(invalid("incomplete or unexpected projection schema"));
    }
    let mut dense = HashMap::new();
    let mut packed = HashMap::new();
    for (name, info) in &content.tensor_infos {
        let tensor = content
            .tensor(reader, name, &Device::Cpu)
            .map_err(invalid)?;
        if let Some(&width) = widths.get(name) {
            let (_, padded) = tensor.shape().dims2().map_err(invalid)?;
            let expected_pad = width
                .checked_add(31)
                .ok_or_else(|| invalid("projection width overflow"))?
                / 32
                * 32;
            if width == 0 || padded != expected_pad || info.ggml_dtype != scheme.ggml() {
                return Err(invalid(format!(
                    "invalid quantized projection layout: {name}"
                )));
            }
            packed.insert(name.clone(), (Arc::new(tensor), width));
        } else {
            if info.ggml_dtype != GgmlDType::F32 {
                return Err(invalid(format!("non-projection {name} must retain FP32")));
            }
            dense.insert(
                name.clone(),
                tensor.dequantize(&Device::Cpu).map_err(invalid)?,
            );
        }
    }
    if packed.len() != widths.len() {
        return Err(invalid("missing packed projection data"));
    }
    Ok((cfg, dense, packed))
}

#[cfg(test)]
fn install_packed(model: &mut Model, mut weights: PackedWeights) -> CoreResult<()> {
    let mut replace = |projection: &mut BackboneLinear, name: String| -> CoreResult<()> {
        let (weight, input_width) = weights
            .remove(&name)
            .ok_or_else(|| invalid(format!("missing {name}")))?;
        let Projection::Dense(dense) = &projection.linear else {
            return Err(invalid("projection is already packed"));
        };
        let (rows, width) = dense.weight().dims2().map_err(invalid)?;
        let (packed_rows, packed_width) = weight.shape().dims2().map_err(invalid)?;
        if rows != packed_rows || width != input_width {
            return Err(invalid(format!("projection schema mismatch: {name}")));
        }
        let bias = dense.bias().cloned();
        // Use the packed variant directly: CANDLE_DEQUANTIZE_ALL/_F16 cannot
        // silently change memory, math or recorded execution identity.
        projection.linear = Projection::Packed {
            matmul: QMatMul::QTensor(weight),
            bias,
            input_width,
            packed_width,
        };
        Ok(())
    };
    for (index, layer) in model.layers.iter_mut().enumerate() {
        let root = format!("model.language_model.layers.{index}");
        for (name, projection) in [
            ("gate_proj", &mut layer.mlp.gate_proj),
            ("up_proj", &mut layer.mlp.up_proj),
            ("down_proj", &mut layer.mlp.down_proj),
        ] {
            replace(projection, format!("{root}.mlp.{name}.weight"))?;
        }
        if let Some(attn) = &mut layer.linear_attn {
            for (name, projection) in [
                ("in_proj_qkv", &mut attn.in_proj_qkv),
                ("in_proj_z", &mut attn.in_proj_z),
                ("in_proj_b", &mut attn.in_proj_b),
                ("in_proj_a", &mut attn.in_proj_a),
                ("out_proj", &mut attn.out_proj),
            ] {
                replace(projection, format!("{root}.linear_attn.{name}.weight"))?;
            }
        }
        if let Some(attn) = &mut layer.self_attn {
            for (name, projection) in [
                ("q_proj", &mut attn.q_proj),
                ("k_proj", &mut attn.k_proj),
                ("v_proj", &mut attn.v_proj),
                ("o_proj", &mut attn.o_proj),
            ] {
                replace(projection, format!("{root}.self_attn.{name}.weight"))?;
            }
        }
    }
    if !weights.is_empty() {
        return Err(invalid("unconsumed packed projections"));
    }
    Ok(())
}

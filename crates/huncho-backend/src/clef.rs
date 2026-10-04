//! Native Clef text/JSON inference using Candle, with no executable model code.
use crate::qwen3_5::{load_base_tensors, Config, Model};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use huncho_core::{
    backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput, RequestOutput},
    contract::SystemOneRequest,
    error::{Error, Result},
    manifest::{BackendId, Family, ModelManifest},
    prompt::clef,
    tokenizer::HfTokenizer,
};
use std::{collections::BTreeMap, path::Path};
#[path = "clef_head.rs"]
mod head;

fn backend_error(e: candle::Error) -> Error {
    Error::Backend(format!("Clef: {e}"))
}

/// CUDA builds use the first GPU by default; CPU builds need no GPU libraries.
pub fn device_from_env() -> Result<Device> {
    let name = std::env::var("HUNCHO_CLEF_DEVICE").unwrap_or_else(|_| "auto".into());
    match name.as_str() {
        "cpu" => Ok(Device::Cpu),
        "auto" => {
            #[cfg(feature = "cuda")]
            if candle::utils::cuda_is_available() {
                return Device::new_cuda(0).map_err(backend_error);
            }
            Ok(Device::Cpu)
        }
        _ if name == "cuda" || name.starts_with("cuda:") => {
            let ordinal = name
                .strip_prefix("cuda:")
                .unwrap_or("0")
                .parse::<usize>()
                .map_err(|_| Error::Package(format!("invalid Clef device `{name}`")))?;
            Device::new_cuda(ordinal).map_err(|e| Error::Unsupported(format!("Clef CUDA device unavailable ({e}); use a build with --features cuda or HUNCHO_CLEF_DEVICE=cpu")))
        }
        _ => Err(Error::Package(format!(
            "invalid HUNCHO_CLEF_DEVICE `{name}`; use auto, cpu, cuda, or cuda:N"
        ))),
    }
}

pub fn default_dtype() -> Result<&'static str> {
    Ok(if device_from_env()?.is_cuda() {
        "bf16"
    } else {
        "fp16"
    })
}

pub struct ClefBackend {
    model: Model,
    head: head::JointHead,
    lexical_weight: Tensor,
    tokenizer: HfTokenizer,
    capabilities: Capabilities,
    device: Device,
    head_dtype: DType,
}
impl ClefBackend {
    pub fn load(dir: &Path, manifest: &ModelManifest, dtype: &str, device: Device) -> Result<Self> {
        manifest.validate()?;
        if manifest.family != Family::F5
            || manifest.prompt_contract.template != clef::TEMPLATE
            || manifest.prompt_contract.contract_hash != clef::CONTRACT
        {
            return Err(Error::Package(
                "the native Clef backend requires a clef-native-v1 prompt contract".into(),
            ));
        }
        if manifest.find_artifact(BackendId::Clef, dtype).is_none() {
            return Err(Error::Package(format!(
                "no clef artifact for dtype `{dtype}`"
            )));
        }
        let tensor_dtype = match dtype {
            "fp32" => DType::F32,
            "fp16" => DType::F16,
            "bf16" if device.is_cuda() => DType::BF16,
            "bf16" => {
                return Err(Error::Unsupported(
                    "Clef bf16 requires CUDA; use fp16 or fp32 on CPU".into(),
                ))
            }
            _ => {
                return Err(Error::Unsupported(format!(
                    "unsupported Clef dtype `{dtype}`; use fp32, fp16, or bf16"
                )))
            }
        };
        let config_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let text = config_json.get("text_config").unwrap_or(&config_json);
        if text["model_type"] != "qwen3_5_text" && text["model_type"] != "qwen3_5" {
            return Err(Error::Unsupported(
                "Clef currently supports the Qwen3.5 text backbone".into(),
            ));
        }
        let config = Config::from_value(&config_json).map_err(backend_error)?;
        let head_config: head::HeadConfig =
            serde_json::from_slice(&std::fs::read(dir.join("joint_head_config.json"))?)?;
        head_config.validate(config.hidden_size)?;
        if manifest.backbone.hidden_size != config.hidden_size {
            return Err(Error::Package(
                "Clef manifest/backbone hidden-size mismatch".into(),
            ));
        }
        let tokenizer_path = manifest
            .backbone
            .tokenizer
            .as_deref()
            .ok_or_else(|| Error::Package("Clef requires tokenizer.json".into()))?;
        let tokenizer = HfTokenizer::from_file_unbounded(dir.join(tokenizer_path))?;
        let mut weights = load_base_tensors(dir, &device, tensor_dtype)?;
        let lexical_weight = weights
            .remove("lm_head.weight")
            .or_else(|| {
                if config_json["tie_word_embeddings"] == true || text["tie_word_embeddings"] == true
                {
                    weights
                        .get("model.language_model.embed_tokens.weight")
                        .cloned()
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                Error::Package("Clef requires lm_head.weight for lexical option embeddings".into())
            })?;
        if lexical_weight.dims() != [config.vocab_size, config.hidden_size] {
            return Err(Error::Package(
                "Clef lexical embedding dimensions do not match config".into(),
            ));
        }
        let model = Model::new(
            &config,
            VarBuilder::from_tensors(weights, tensor_dtype, &device),
            &device,
            tensor_dtype,
        )
        .map_err(backend_error)?;
        // The head is small relative to the backbone. Use f32 on CPU for
        // normalization/GELU; keep the native dtype on CUDA.
        let head_dtype = if device.is_cpu() {
            DType::F32
        } else {
            tensor_dtype
        };
        let head_weights = candle::safetensors::load(dir.join(&manifest.head.weights), &device)
            .map_err(backend_error)?;
        let head = head::JointHead::load(
            &head_config,
            VarBuilder::from_tensors(head_weights, head_dtype, &device),
        )
        .map_err(backend_error)?;
        let max_context = manifest
            .backbone
            .max_context
            .min(config.max_position_embeddings)
            .min(manifest.prompt_contract.max_len);
        Ok(Self {
            model,
            head,
            lexical_weight,
            tokenizer,
            device: device.clone(),
            head_dtype,
            capabilities: Capabilities {
                id: BackendId::Clef,
                dtype: dtype.into(),
                max_context,
                families: vec![Family::F5],
                extra: BTreeMap::from([
                    ("runtime".into(), "candle".into()),
                    ("device".into(), format!("{:?}", device.location())),
                    ("media".into(), "text-json-only".into()),
                ]),
                ..Default::default()
            },
        })
    }
}

impl Backend for ClefBackend {
    fn id(&self) -> BackendId {
        BackendId::Clef
    }
    fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }
    fn forward(&mut self, _input: ForwardInput) -> Result<ForwardOutput> {
        Err(Error::Unsupported(
            "Clef requires the complete state and question schema".into(),
        ))
    }
    fn forward_request(
        &mut self,
        request: &SystemOneRequest,
        max_context: usize,
    ) -> Result<RequestOutput> {
        let record = clef::encode(
            request,
            &self.tokenizer,
            max_context.min(self.capabilities.max_context),
        )?;
        let ids = Tensor::new(record.input_ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(backend_error)?;
        let hidden = self
            .model
            .forward(&ids)
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_dtype(self.head_dtype))
            .map_err(backend_error)?;
        let scores = self
            .head
            .forward(&hidden, &self.lexical_weight, &record)
            .map_err(backend_error)?;
        let mut logits = BTreeMap::new();
        for (q, scores) in record.questions.iter().zip(scores) {
            if scores.iter().any(|x| !x.is_finite()) {
                return Err(Error::Backend("Clef returned non-finite logits".into()));
            }
            logits.insert(
                q.question_id.clone(),
                q.option_ids.iter().cloned().zip(scores).collect(),
            );
        }
        Ok(RequestOutput {
            logits,
            input_tokens: record.input_ids.len() as u64,
        })
    }
    fn fork(&mut self, _handle: CacheHandle) -> Result<CacheHandle> {
        Err(Error::Unsupported("Clef does not expose a KV cache".into()))
    }
}

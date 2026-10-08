//! CPU Qwen3.5 GGUF prefill with unpooled trained pointer or raw LM readouts.
//! No sampling, generated tokens, embedding normalization or decode loop.
use candle::{
    quantized::{gguf_file, GgmlDType},
    Device, Tensor,
};
use huncho_core::{
    backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput},
    manifest::{BackendId, Family, ModelManifest},
    tensor::Tensor as CoreTensor,
    Error, Result,
};
use llama_cpp_sys_2 as ffi;
use std::{
    collections::BTreeMap,
    ffi::{CStr, CString},
    path::Path,
    ptr::{self, NonNull},
    sync::{Arc, Once},
};

pub const LLAMA_CPP_REVISION: &str = "26394b4e6749a41c3633db040e0987500a5f7013";
const PIN: &str = "llama-cpp-sys-2=0.1.159;llama.cpp=26394b4e6749a41c3633db040e0987500a5f7013";
const MAX_OUTPUTS: usize = 256;

#[derive(Clone, Copy, Debug)]
pub struct LlamaOptions {
    /// Fixed CPU worker count per context, including prefill (1..=256).
    pub threads: usize,
}
impl Default for LlamaOptions {
    fn default() -> Self {
        Self { threads: 1 }
    }
}

struct Model {
    pointer: NonNull<ffi::llama_model>,
    // Retain the null-terminated CPU device list for the model's full lifetime.
    _devices: Box<[*mut ffi::ggml_backend_device; 2]>,
    hidden: usize,
    vocab: usize,
}
// CPU model tensors are immutable after construction. This wrapper exposes no
// model mutation, LoRA attachment or GPU placement API; contexts own all state.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}
impl Drop for Model {
    fn drop(&mut self) {
        unsafe { ffi::llama_model_free(self.pointer.as_ptr()) }
    }
}
struct Context {
    pointer: NonNull<ffi::llama_context>,
    model: Arc<Model>,
}
// Native operations require &mut Backend. Shared references read only cached
// Rust capability fields, and a context never transfers state into a replica.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ffi::llama_free(self.pointer.as_ptr()) }
    }
}

enum Readout {
    Pointer(crate::kev::PointerHead),
    LanguageModel,
}
pub struct LlamaCppBackend {
    context: Context,
    readout: Arc<Readout>,
    capabilities: Capabilities,
    options: LlamaOptions,
    tokens: Vec<i32>,
    outputs: Vec<i8>,
}

impl LlamaCppBackend {
    pub fn load(
        dir: &Path,
        manifest: &ModelManifest,
        dtype: &str,
        options: LlamaOptions,
    ) -> Result<Self> {
        manifest.validate()?;
        if !(manifest.family == Family::F3
            || (manifest.family == Family::F2 && manifest.prompt_contract.template == "kev-v1"))
        {
            return Err(Error::Unsupported(
                "llama.cpp currently supports Kev F2 and Qwen3.5 F3 only".into(),
            ));
        }
        if !(1..=256).contains(&options.threads)
            || manifest.backbone.max_context == 0
            || manifest.backbone.max_context > 65536
            || manifest.prompt_contract.max_options >= MAX_OUTPUTS
        {
            return Err(Error::Request(
                "llama.cpp requires 1..256 CPU threads, 1..65536 context and at most 255 options"
                    .into(),
            ));
        }
        let artifact = manifest
            .find_artifact(BackendId::LlamaCpp, dtype)
            .ok_or_else(|| Error::Package("missing exact llama.cpp artifact".into()))?;
        let quantization = if matches!(dtype, "gguf-q8_0" | "gguf-q4_0") {
            let profile = format!("llamacpp-qwen35-{}-v1", dtype.trim_start_matches("gguf-"));
            if artifact.quantization.as_deref() != Some(profile.as_str()) {
                return Err(Error::Package(
                    "llama.cpp quantization profile is missing or inconsistent".into(),
                ));
            }
            Some(profile)
        } else {
            if artifact.quantization.is_some() {
                return Err(Error::Package(
                    "dense llama.cpp artifact cannot declare quantization".into(),
                ));
            }
            None
        };
        let path = dir.join(&artifact.path);
        validate_gguf(&path, dtype)?;
        let kernel_build = cpu_kernels()?;
        // Only the CPU device is given to the loader. No device enumeration or
        // GPU capability/driver probing is needed by this adapter.
        let cpu = initialize_cpu()?;
        let mut devices = Box::new([cpu, ptr::null_mut()]);
        let path = CString::new(
            path.to_str()
                .ok_or_else(|| Error::Package("GGUF path must be UTF-8".into()))?,
        )
        .map_err(|_| Error::Package("invalid GGUF path".into()))?;
        let mut params = unsafe { ffi::llama_model_default_params() };
        params.devices = devices.as_mut_ptr();
        params.n_gpu_layers = 0;
        params.load_mode = ffi::LLAMA_LOAD_MODE_NONE;
        params.load_mtp = false;
        params.check_tensors = true;
        params.no_alloc = false;
        params.vocab_only = false;
        let pointer =
            NonNull::new(unsafe { ffi::llama_model_load_from_file(path.as_ptr(), params) })
                .ok_or_else(|| Error::Backend("llama.cpp failed to load validated GGUF".into()))?;
        let hidden = unsafe { ffi::llama_model_n_embd_out(pointer.as_ptr()) };
        let vocab =
            unsafe { ffi::llama_vocab_n_tokens(ffi::llama_model_get_vocab(pointer.as_ptr())) };
        let model = Arc::new(Model {
            pointer,
            _devices: devices,
            hidden: hidden.max(0) as usize,
            vocab: vocab.max(0) as usize,
        });
        if hidden <= 0 || vocab <= 0 || hidden as usize != manifest.backbone.hidden_size {
            return Err(Error::Package(
                "llama.cpp model/head dimensions do not match manifest".into(),
            ));
        }
        let context_limit = unsafe { ffi::llama_model_n_ctx_train(pointer.as_ptr()) };
        if context_limit <= 0 || manifest.backbone.max_context > context_limit as usize {
            return Err(Error::Package(
                "llama.cpp context exceeds GGUF trained context".into(),
            ));
        }
        let readout = if manifest.family == Family::F2 {
            Readout::Pointer(crate::kev::PointerHead::load(
                &dir.join(&manifest.head.weights),
                model.hidden,
                &Device::Cpu,
            )?)
        } else {
            Readout::LanguageModel
        };
        let context = Context::new(model, manifest.backbone.max_context, &readout, options)?;
        let version = unsafe { CStr::from_ptr(ffi::llama_version()) }
            .to_string_lossy()
            .into_owned();
        let mut extra = BTreeMap::from([
            ("device".into(), "CPU".into()),
            ("native_execution".into(), "llamacpp-qwen35-v1".into()),
            ("runtime".into(), PIN.into()),
            ("runtime_version".into(), version),
            (
                "llamacpp_execution".into(),
                "cpu-qwen35-masked-prefill-v1".into(),
            ),
            ("llamacpp_threads".into(), options.threads.to_string()),
            ("llamacpp_cpu_kernels".into(), kernel_build),
            ("kv_dtype".into(), "fp32".into()),
            ("recurrent_state_dtype".into(), "fp32".into()),
            ("weight_load_mode".into(), "copy".into()),
        ]);
        if manifest.family == Family::F2 {
            extra.insert("pointer_head_dtype".into(), "fp32".into());
        }
        if let Some(profile) = quantization {
            extra.insert("weight_quantization".into(), profile);
        }
        Ok(Self {
            context,
            readout: Arc::new(readout),
            options,
            capabilities: Capabilities {
                id: BackendId::LlamaCpp,
                dtype: dtype.into(),
                max_context: manifest.backbone.max_context,
                families: vec![manifest.family],
                extra,
                ..Default::default()
            },
            tokens: Vec::new(),
            outputs: Vec::new(),
        })
    }
}

impl Context {
    fn new(
        model: Arc<Model>,
        context: usize,
        readout: &Readout,
        options: LlamaOptions,
    ) -> Result<Self> {
        let mut params = unsafe { ffi::llama_context_default_params() };
        params.n_ctx = context as u32;
        params.n_batch = context as u32;
        params.n_ubatch = context as u32;
        params.n_seq_max = 1;
        params.n_outputs_max = MAX_OUTPUTS.min(context) as u32;
        params.n_outputs_max_per_seq = params.n_outputs_max;
        params.n_threads = options.threads as i32;
        params.n_threads_batch = options.threads as i32;
        params.pooling_type = ffi::LLAMA_POOLING_TYPE_NONE;
        params.attention_type = ffi::LLAMA_ATTENTION_TYPE_CAUSAL;
        params.flash_attn_type = ffi::LLAMA_FLASH_ATTN_TYPE_DISABLED;
        // Ordinary embeddings mode forces every token to be an output. Use the
        // pinned extension for masked, unnormalized Qwen3.5 post-norm states.
        params.embeddings = false;
        params.type_k = ffi::GGML_TYPE_F32;
        params.type_v = ffi::GGML_TYPE_F32;
        params.offload_kqv = false;
        params.op_offload = false;
        params.no_perf = true;
        params.samplers = ptr::null_mut();
        params.n_samplers = 0;
        let pointer =
            NonNull::new(unsafe { ffi::llama_init_from_model(model.pointer.as_ptr(), params) })
                .ok_or_else(|| {
                    Error::Backend("llama.cpp CPU context initialization failed".into())
                })?;
        if matches!(readout, Readout::Pointer(_)) {
            unsafe { huncho_llama_set_masked_hidden(pointer.as_ptr()) };
        }
        Ok(Self { pointer, model })
    }
}

impl Backend for LlamaCppBackend {
    fn id(&self) -> BackendId {
        BackendId::LlamaCpp
    }
    fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }
    fn replica(&self) -> Result<Box<dyn Backend>> {
        let context = Context::new(
            self.context.model.clone(),
            self.capabilities.max_context,
            &self.readout,
            self.options,
        )?;
        Ok(Box::new(Self {
            context,
            readout: self.readout.clone(),
            capabilities: self.capabilities.clone(),
            options: self.options,
            tokens: Vec::new(),
            outputs: Vec::new(),
        }))
    }
    fn fork(&mut self, _handle: CacheHandle) -> Result<CacheHandle> {
        Err(Error::Unsupported(
            "llama.cpp hybrid prefix forks are not exposed by this profile".into(),
        ))
    }
    fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
        let n = input.tokens.len();
        if n == 0
            || n > self.capabilities.max_context
            || input.positions.is_empty()
            || input.positions.iter().any(|&p| p >= n)
            || input
                .tokens
                .iter()
                .any(|&token| token as usize >= self.context.model.vocab)
            || input.positions.len() >= MAX_OUTPUTS
            || input.fork_from.is_some()
            || input.retain_cache
        {
            return Err(Error::Request(
                "invalid llama.cpp independent tokens/readouts or unsupported cache continuation"
                    .into(),
            ));
        }
        if input.logit_codes.as_ref().is_some_and(|codes| {
            codes.is_empty()
                || codes
                    .iter()
                    .any(|&code| code as usize >= self.context.model.vocab)
        }) {
            return Err(Error::Request(
                "invalid llama.cpp candidate vocabulary codes".into(),
            ));
        }
        if matches!(self.readout.as_ref(), Readout::Pointer(_)) && input.logit_codes.is_some() {
            return Err(Error::Request(
                "vocabulary codes do not apply to a pointer head".into(),
            ));
        }
        self.tokens.clear();
        self.tokens
            .extend(input.tokens.iter().map(|&token| token as i32));
        self.outputs.clear();
        self.outputs.resize(n, 0);
        for &position in &input.positions {
            self.outputs[position] = 1;
        }
        if matches!(self.readout.as_ref(), Readout::Pointer(_)) {
            self.outputs[n - 1] = 1;
        }
        let memory = unsafe { ffi::llama_get_memory(self.context.pointer.as_ptr()) };
        if memory.is_null() {
            return Err(Error::Backend(
                "llama.cpp omitted hybrid model memory".into(),
            ));
        }
        unsafe { ffi::llama_memory_clear(memory, true) };
        let batch = ffi::llama_batch {
            n_tokens: n as i32,
            token: self.tokens.as_mut_ptr(),
            embd: ptr::null_mut(),
            pos: ptr::null_mut(),
            n_seq_id: ptr::null_mut(),
            seq_id: ptr::null_mut(),
            logits: self.outputs.as_mut_ptr(),
        };
        let code = unsafe { ffi::llama_decode(self.context.pointer.as_ptr(), batch) };
        if code != 0 {
            return Err(Error::Backend(format!(
                "llama.cpp prefill failed with status {code}"
            )));
        }
        let values = match self.readout.as_ref() {
            Readout::Pointer(head) => {
                let hidden = self.context.model.hidden;
                let decide = embedding(&self.context, n - 1)?;
                let mut options = Vec::with_capacity(input.positions.len() * hidden);
                for &position in &input.positions {
                    options.extend(embedding(&self.context, position)?);
                }
                let decide =
                    Tensor::from_vec(decide, (1, hidden), &Device::Cpu).map_err(candle_error)?;
                let options =
                    Tensor::from_vec(options, (input.positions.len(), hidden), &Device::Cpu)
                        .map_err(candle_error)?;
                let values = head
                    .forward(&decide, &options)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(candle_error)?;
                return Ok(ForwardOutput::Logits {
                    positions: input.positions.clone(),
                    values: CoreTensor::new(vec![input.positions.len(), 1], values)?,
                });
            }
            Readout::LanguageModel => {
                let mut values = Vec::new();
                for &position in &input.positions {
                    let pointer = unsafe {
                        ffi::llama_get_logits_ith(self.context.pointer.as_ptr(), position as i32)
                    };
                    if pointer.is_null() {
                        return Err(Error::Backend(
                            "llama.cpp omitted requested raw logits".into(),
                        ));
                    }
                    // Width comes from the loaded model; the context and output
                    // allocation remain alive and exclusively borrowed here.
                    let row =
                        unsafe { std::slice::from_raw_parts(pointer, self.context.model.vocab) };
                    if let Some(codes) = &input.logit_codes {
                        values.extend(codes.iter().map(|&code| row[code as usize]));
                    } else {
                        values.extend_from_slice(row);
                    }
                }
                values
            }
        };
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Backend(
                "llama.cpp returned non-finite raw logits".into(),
            ));
        }
        if let Some(codes) = input.logit_codes {
            let width = codes.len();
            Ok(ForwardOutput::SelectedLogits {
                positions: input.positions.clone(),
                codes,
                values: CoreTensor::new(vec![input.positions.len(), width], values)?,
            })
        } else {
            Ok(ForwardOutput::Logits {
                positions: input.positions.clone(),
                values: CoreTensor::new(
                    vec![input.positions.len(), self.context.model.vocab],
                    values,
                )?,
            })
        }
    }
}

fn candle_error(error: candle::Error) -> Error {
    Error::Backend(format!("llama.cpp pointer readout: {error}"))
}

fn initialize_cpu() -> Result<*mut ffi::ggml_backend_device> {
    // Establish the statically linked CPU registry before initialization so
    // llama_backend_init does not search for external device plugins.
    let cpu = unsafe { ffi::ggml_backend_dev_by_type(ffi::GGML_BACKEND_DEVICE_TYPE_CPU) };
    if cpu.is_null() {
        return Err(Error::Backend("llama.cpp CPU backend unavailable".into()));
    }
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe { ffi::llama_backend_init() });
    Ok(cpu)
}

#[derive(Debug, serde::Serialize)]
pub struct QuantizationStats {
    pub dtype: String,
    pub threads: usize,
    pub packed_tensors: usize,
    pub shape_retained_f32: Vec<String>,
    pub cpu_kernels: String,
    pub runtime: String,
}

/// Produce a new CPU Q8_0/Q4_0 GGUF using the exact pinned native quantizer.
/// Source must be unquantized FP32. Embeddings, vocabulary output, norms,
/// convolution and block-incompatible projections stay FP32. No calibration
/// data/temperature is read here; the caller must publish a pending variant.
pub fn quantize_gguf_cpu(
    input: &Path,
    output: &Path,
    dtype: &str,
    threads: usize,
) -> Result<QuantizationStats> {
    let ftype = match dtype {
        "gguf-q8_0" => ffi::LLAMA_FTYPE_MOSTLY_Q8_0,
        "gguf-q4_0" => ffi::LLAMA_FTYPE_MOSTLY_Q4_0,
        _ => {
            return Err(Error::Unsupported(
                "CPU GGUF quantization requires gguf-q8_0 or gguf-q4_0".into(),
            ))
        }
    };
    if !(1..=256).contains(&threads) {
        return Err(Error::Request(
            "GGUF quantization threads must be 1..256".into(),
        ));
    }
    validate_gguf(input, "gguf-f32")?;
    let kernels = cpu_kernels()?;
    initialize_cpu()?;
    let content =
        gguf_file::Content::read(&mut std::fs::File::open(input)?).map_err(candle_error)?;
    let mut retained: Vec<_> = content
        .tensor_infos
        .iter()
        .filter(|(_, info)| {
            info.shape.rank() >= 2
                && info
                    .shape
                    .dims()
                    .last()
                    .is_some_and(|width| width % 32 != 0)
        })
        .map(|(name, _)| name.clone())
        .collect();
    retained.sort();
    let patterns: Vec<_> = retained
        .iter()
        .map(|name| {
            let mut pattern = String::from("^");
            for ch in name.chars() {
                if "\\.^$|?*+()[]{}".contains(ch) {
                    pattern.push('\\');
                }
                pattern.push(ch);
            }
            pattern.push('$');
            CString::new(pattern).map_err(|_| Error::Package("invalid GGUF tensor name".into()))
        })
        .collect::<Result<_>>()?;
    let mut overrides: Vec<_> = patterns
        .iter()
        .map(|pattern| ffi::llama_model_tensor_override {
            pattern: pattern.as_ptr(),
            type_: ffi::GGML_TYPE_F32,
        })
        .collect();
    overrides.push(ffi::llama_model_tensor_override {
        pattern: ptr::null(),
        type_: ffi::GGML_TYPE_F32,
    });
    let c_path = |path: &Path| {
        CString::new(
            path.to_str()
                .ok_or_else(|| Error::Package("GGUF paths must be UTF-8".into()))?,
        )
        .map_err(|_| Error::Package("invalid GGUF path".into()))
    };
    let source = c_path(input)?;
    let destination = c_path(output)?;
    // Reserve a new destination. The native writer truncates this owned file,
    // never an existing user's artifact. Failed output remains unpublished.
    std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?
        .sync_all()?;
    let mut params = unsafe { ffi::llama_model_quantize_default_params() };
    params.nthread = threads as i32;
    params.ftype = ftype;
    params.pure_ = true;
    params.token_embedding_type = ffi::GGML_TYPE_F32;
    params.output_tensor_type = ffi::GGML_TYPE_F32;
    params.quantize_output_tensor = false;
    params.allow_requantize = false;
    params.keep_split = false;
    params.max_buf_size = 64 * 1024 * 1024;
    params.tt_overrides = overrides.as_ptr();
    // Strings and sentinel-terminated overrides remain owned until the
    // synchronous call returns. This routine uses only CPU quantization code.
    let status =
        unsafe { ffi::llama_model_quantize(source.as_ptr(), destination.as_ptr(), &params) };
    if status != 0 {
        return Err(Error::Backend(format!(
            "CPU GGUF quantizer failed with status {status}"
        )));
    }
    std::fs::File::open(output)?.sync_all()?;
    validate_gguf(output, dtype)?;
    let converted =
        gguf_file::Content::read(&mut std::fs::File::open(output)?).map_err(candle_error)?;
    let packed_type = if dtype == "gguf-q8_0" {
        GgmlDType::Q8_0
    } else {
        GgmlDType::Q4_0
    };
    let packed_tensors = converted
        .tensor_infos
        .values()
        .filter(|info| info.ggml_dtype == packed_type)
        .count();
    Ok(QuantizationStats {
        dtype: dtype.into(),
        threads,
        packed_tensors,
        shape_retained_f32: retained,
        cpu_kernels: kernels,
        runtime: PIN.into(),
    })
}
fn embedding(context: &Context, position: usize) -> Result<Vec<f32>> {
    let pointer =
        unsafe { huncho_llama_get_masked_hidden(context.pointer.as_ptr(), position as i32) };
    if pointer.is_null() {
        return Err(Error::Backend(
            "llama.cpp omitted requested unpooled hidden states".into(),
        ));
    }
    let values = unsafe { std::slice::from_raw_parts(pointer, context.model.hidden) }.to_vec();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Backend(
            "llama.cpp returned non-finite hidden states".into(),
        ));
    }
    Ok(values)
}

fn validate_gguf(path: &Path, dtype: &str) -> Result<()> {
    let content =
        gguf_file::Content::read(&mut std::fs::File::open(path)?).map_err(candle_error)?;
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok());
    if arch.map(String::as_str) != Some("qwen35") {
        return Err(Error::Unsupported("llama.cpp requires standard dense Qwen3.5 GGUF (not Huncho's packed projection format)".into()));
    }
    let expected = match dtype {
        "gguf-f32" => (0, GgmlDType::F32),
        "gguf-f16" => (1, GgmlDType::F16),
        "gguf-q8_0" => (7, GgmlDType::Q8_0),
        "gguf-q4_0" => (2, GgmlDType::Q4_0),
        _ => {
            return Err(Error::Unsupported(
                "llama.cpp dtype must be gguf-f32, gguf-f16, gguf-q8_0 or gguf-q4_0".into(),
            ))
        }
    };
    if content
        .metadata
        .get("general.file_type")
        .and_then(|v| v.to_u32().ok())
        != Some(expected.0)
        || content.tensor_infos.is_empty()
        || !content
            .tensor_infos
            .values()
            .any(|t| t.ggml_dtype == expected.1)
        || content
            .tensor_infos
            .values()
            .any(|t| t.ggml_dtype != GgmlDType::F32 && t.ggml_dtype != expected.1)
    {
        return Err(Error::Package(
            "GGUF tensor layout does not match requested dtype".into(),
        ));
    }
    Ok(())
}

// Two pinned staging functions have C++ linkage. The narrow local shim exposes
// a C ABI without relying on private C++ object layouts.
extern "C" {
    fn huncho_llama_set_masked_hidden(context: *mut ffi::llama_context);
    fn huncho_llama_get_masked_hidden(context: *mut ffi::llama_context, position: i32) -> *mut f32;
}
fn cpu_kernels() -> Result<String> {
    let flags = unsafe {
        [
            ("avx", ffi::ggml_cpu_has_avx() != 0),
            ("avx2", ffi::ggml_cpu_has_avx2() != 0),
            ("fma", ffi::ggml_cpu_has_fma() != 0),
            ("f16c", ffi::ggml_cpu_has_f16c() != 0),
            ("avx512f", ffi::ggml_cpu_has_avx512() != 0),
        ]
    };
    #[cfg(target_arch = "x86_64")]
    for (name, enabled) in flags {
        let supported = match name {
            "avx" => std::is_x86_feature_detected!("avx"),
            "avx2" => std::is_x86_feature_detected!("avx2"),
            "fma" => std::is_x86_feature_detected!("fma"),
            "f16c" => std::is_x86_feature_detected!("f16c"),
            "avx512f" => std::is_x86_feature_detected!("avx512f"),
            _ => false,
        };
        if enabled && !supported {
            return Err(Error::Unsupported(format!(
                "llama.cpp requires compiled CPU feature {name}"
            )));
        }
    }
    Ok(flags
        .iter()
        .filter_map(|(name, enabled)| enabled.then_some(*name))
        .collect::<Vec<_>>()
        .join(","))
}

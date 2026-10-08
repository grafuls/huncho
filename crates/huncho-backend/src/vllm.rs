//! Optional pinned CPU vLLM pooling process: raw Kev pointer scores, no decode.
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{mpsc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use huncho_core::backend::{
    Backend, BatchLimits, CacheHandle, Capabilities, ForwardInput, ForwardOutput,
};
use huncho_core::error::{Error, Result};
use huncho_core::manifest::{BackendId, Family, HeadKind, ModelManifest};
use huncho_core::tensor::Tensor;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const DRIVER: &str = include_str!("../runtime/vllm_cpu.py");
const MAX_FRAME: u64 = 32 * 1024 * 1024;
const PROFILE: &str = "kev-pointer-vllm-cpu-bf16-v1";

#[derive(Clone, Debug)]
pub struct VllmOptions {
    /// Explicit absolute executable; retain a venv path rather than resolving it.
    pub python: PathBuf,
    pub threads: usize,
    pub batch_rows: usize,
    pub kv_cache_bytes: usize,
    pub timeout: Duration,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    schema_version: u32,
    profile: String,
    model_dir: String,
    files: BTreeMap<String, String>,
}

fn package_error(message: impl std::fmt::Display) -> Error {
    Error::Package(message.to_string())
}
fn hash_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 128 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn relative_file(root: &Path, name: &str) -> Result<PathBuf> {
    let path = Path::new(name);
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(package_error(
            "vLLM artifact paths must stay inside their package",
        ));
    }
    let resolved = root.join(path).canonicalize()?;
    if !resolved.starts_with(root.canonicalize()?) || !resolved.is_file() {
        return Err(package_error("invalid vLLM artifact file"));
    }
    Ok(resolved)
}

pub struct VllmBackend {
    child: Child,
    input: ChildStdin,
    frames: Mutex<mpsc::Receiver<Result<Value>>>,
    reader: Option<JoinHandle<()>>,
    _driver: tempfile::TempDir,
    capabilities: Capabilities,
    vocab: usize,
    rows: usize,
    timeout: Duration,
    seq: u64,
    failed: bool,
}
impl VllmBackend {
    pub fn load(
        dir: &Path,
        manifest: &ModelManifest,
        dtype: &str,
        options: VllmOptions,
    ) -> Result<Self> {
        manifest.validate()?;
        if !cfg!(all(target_os = "linux", target_arch = "x86_64"))
            || manifest.family != Family::F2
            || manifest.prompt_contract.template != "kev-v1"
            || manifest.head.kind != HeadKind::Pointer
            || dtype != "bf16"
        {
            return Err(Error::Unsupported("vLLM currently requires Linux x86_64 CPU, Kev F2 and bf16 execution with an FP32 pointer head".into()));
        }
        if std::env::var("HUNCHO_DEVICE").is_ok_and(|s| s != "cpu") {
            return Err(Error::Unsupported(
                "vLLM currently requires HUNCHO_DEVICE=cpu or unset".into(),
            ));
        }
        if !options.python.is_absolute()
            || !options.python.is_file()
            || !(1..=64).contains(&options.threads)
            || !(1..=8).contains(&options.batch_rows)
            || !(64u64 * 1024 * 1024..=16u64 * 1024 * 1024 * 1024)
                .contains(&(options.kv_cache_bytes as u64))
            || options.timeout < Duration::from_secs(1)
            || options.timeout > Duration::from_secs(600)
            || !(1..=65536).contains(&manifest.backbone.max_context)
        {
            return Err(Error::Request("vLLM requires an absolute CPU Python executable, 1..64 threads, 1..8 rows, 64 MiB..16 GiB KV bytes, 1..600 seconds timeout and 1..65536 context".into()));
        }
        let reference = manifest
            .find_artifact(BackendId::Vllm, dtype)
            .ok_or_else(|| package_error("missing exact vLLM bf16 artifact"))?;
        if reference.quantization.is_some() {
            return Err(package_error("vLLM quantization is unsupported"));
        }
        let descriptor = relative_file(dir, &reference.path)?;
        let root = descriptor.parent().unwrap();
        let artifact: Artifact = serde_json::from_slice(&std::fs::read(&descriptor)?)?;
        if artifact.schema_version != 1
            || artifact.profile != PROFILE
            || artifact.model_dir != "model"
            || artifact.files.len() != 2
            || !artifact.files.contains_key("model/config.json")
            || !artifact.files.contains_key("model/model.safetensors")
        {
            return Err(package_error("invalid pinned vLLM artifact contract"));
        }
        for (name, digest) in &artifact.files {
            let file = relative_file(root, name)?;
            if digest.len() != 64 || hash_file(&file)? != *digest {
                return Err(package_error("vLLM artifact digest mismatch"));
            }
        }
        let model_dir = root.join("model").canonicalize()?;
        if std::fs::read_dir(&model_dir)?.count() != 2 {
            return Err(package_error(
                "vLLM model directory must contain only pinned config and safetensors",
            ));
        }
        let config: Value = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
        let vocab = config["vocab_size"]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(0);
        let layers = config["layer_types"]
            .as_array()
            .ok_or_else(|| package_error("missing vLLM layer types"))?;
        if config["architectures"] != json!(["HunchoKevForPooling"])
            || config["model_type"] != "qwen3_5_text"
            || !config["quantization_config"].is_null()
            || config["hidden_size"].as_u64() != Some(manifest.backbone.hidden_size as u64)
            || config["max_position_embeddings"].as_u64().unwrap_or(0)
                < manifest.backbone.max_context as u64
            || config["huncho_pointer_dim"].as_u64().unwrap_or(0) == 0
            || vocab == 0
            || layers.is_empty()
            || layers
                .iter()
                .any(|s| s != "linear_attention" && s != "full_attention")
            || (layers.iter().any(|s| s == "linear_attention")
                && (config["linear_key_head_dim"] != 128 || config["linear_value_head_dim"] != 128))
        {
            return Err(package_error(
                "unsupported CPU Qwen3.5/pointer configuration",
            ));
        }
        let driver = tempfile::tempdir()?;
        let script = driver.path().join("huncho_vllm_cpu.py");
        std::fs::write(&script, DRIVER)?;
        let mut command = Command::new(&options.python);
        command.env_clear();
        for name in ["PATH", "HOME", "TMPDIR", "LD_PRELOAD", "LD_LIBRARY_PATH"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        let mut child = command
            .arg("-s")
            .arg("-u")
            .arg(&script)
            .arg(&model_dir)
            .arg(manifest.backbone.max_context.to_string())
            .arg(options.batch_rows.to_string())
            .arg(options.threads.to_string())
            .arg(options.kv_cache_bytes.to_string())
            .env("PYTHONPATH", driver.path())
            .env("VLLM_TARGET_DEVICE", "cpu")
            .env("VLLM_PLUGINS", "")
            .env("VLLM_LOGGING_LEVEL", "ERROR")
            .env("VLLM_NO_USAGE_STATS", "1")
            .env("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .env("OMP_NUM_THREADS", options.threads.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let input = child.stdin.take().unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap());
        let (send, frames) = mpsc::sync_channel(1);
        let reader = std::thread::spawn(move || loop {
            let mut line = Vec::new();
            let result = output.by_ref().take(MAX_FRAME).read_until(b'\n', &mut line);
            let frame = match result {
                Ok(0) => Err(Error::Backend("vLLM worker exited".into())),
                Ok(_) if line.len() as u64 >= MAX_FRAME || line.last() != Some(&b'\n') => Err(
                    Error::Backend("invalid or oversized vLLM protocol frame".into()),
                ),
                Ok(_) => serde_json::from_slice(&line).map_err(Error::from),
                Err(error) => Err(Error::from(error)),
            };
            let failed = frame.is_err();
            if send.try_send(frame).is_err() || failed {
                break;
            }
        });
        let mut backend = Self {
            child,
            input,
            frames: Mutex::new(frames),
            reader: Some(reader),
            _driver: driver,
            capabilities: Capabilities {
                id: BackendId::Vllm,
                dtype: dtype.into(),
                max_context: manifest.backbone.max_context,
                families: vec![Family::F2],
                ..Default::default()
            },
            vocab,
            rows: options.batch_rows,
            timeout: options.timeout,
            seq: 0,
            failed: false,
        };
        let ready = backend.receive()?;
        if ready["protocol"] != 1
            || ready["kind"] != "ready"
            || ready["device"] != "CPU"
            || ready["dtype"] != "bf16"
            || ready["head_dtype"] != "fp32"
            || ready["context"] != manifest.backbone.max_context
            || ready["batch_rows"] != options.batch_rows
            || ready["vllm"] != "0.31.0+cpu"
            || ready["torch"] != "2.13.0+cpu"
            || !ready["runtime_sha256"]
                .as_str()
                .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Err(Error::Backend(
                "vLLM worker returned inconsistent runtime identity".into(),
            ));
        }
        backend.capabilities.extra = BTreeMap::from([
            ("device".into(), "CPU".into()),
            ("vllm_readout".into(), PROFILE.into()),
            (
                "vllm_runtime_sha256".into(),
                ready["runtime_sha256"].as_str().unwrap().into(),
            ),
            (
                "vllm_driver_sha256".into(),
                format!("{:x}", Sha256::digest(DRIVER.as_bytes())),
            ),
            (
                "vllm_python".into(),
                options.python.to_string_lossy().into_owned(),
            ),
            ("vllm_threads".into(), options.threads.to_string()),
            ("vllm_batch_rows".into(), options.batch_rows.to_string()),
            (
                "vllm_kv_cache_bytes".into(),
                options.kv_cache_bytes.to_string(),
            ),
            ("vllm_head_dtype".into(), "fp32".into()),
            ("vllm_prefix_cache".into(), "disabled".into()),
            ("vllm_decode".into(), "disabled".into()),
        ]);
        Ok(backend)
    }
    fn receive(&mut self) -> Result<Value> {
        let result = self
            .frames
            .lock()
            .map_err(|_| Error::Backend("vLLM protocol lock poisoned".into()))?
            .recv_timeout(self.timeout)
            .map_err(|error| {
                Error::Backend(format!("vLLM protocol timeout/disconnection: {error}"))
            })
            .and_then(|frame| frame);
        if result.is_err() {
            self.failed = true;
            let _ = self.child.kill();
        }
        result
    }
    fn execute(&mut self, inputs: Vec<ForwardInput>) -> Result<Vec<ForwardOutput>> {
        if self.failed {
            return Err(Error::Backend(
                "vLLM worker is unavailable after failure".into(),
            ));
        }
        if inputs.is_empty() || inputs.len() > self.rows {
            return Err(Error::Request("vLLM batch exceeds configured rows".into()));
        }
        let length = inputs[0].tokens.len();
        for input in &inputs {
            if input.tokens.is_empty()
                || input.tokens.len() > self.capabilities.max_context
                || input.tokens.len() != length
                || input.positions.len() > 255
                || input.positions.iter().any(|&p| p >= input.tokens.len())
                || input.tokens.iter().any(|&t| t as usize >= self.vocab)
                || input.qtype > 2
                || input.retain_cache
                || input.fork_from.is_some()
                || input.logit_codes.is_some()
            {
                return Err(Error::Request("vLLM needs independent equal nonempty lengths, valid readouts and tokens, without caches or vocabulary-code hints".into()));
            }
        }
        self.seq = self
            .seq
            .checked_add(1)
            .ok_or_else(|| Error::Backend("vLLM sequence exhausted".into()))?;
        let request = json!({"seq":self.seq,"inputs":inputs.iter().map(|i| json!({"tokens":i.tokens,"positions":i.positions})).collect::<Vec<_>>()});
        let mut bytes = serde_json::to_vec(&request)?;
        if bytes.len() as u64 + 1 >= MAX_FRAME {
            return Err(Error::Request("vLLM input frame exceeds 32 MiB".into()));
        }
        bytes.push(b'\n');
        if let Err(error) = self
            .input
            .write_all(&bytes)
            .and_then(|_| self.input.flush())
        {
            self.failed = true;
            let _ = self.child.kill();
            return Err(Error::from(error));
        }
        let frame = self.receive()?;
        let parse = || -> Result<Vec<ForwardOutput>> {
            if frame["protocol"] != 1
                || frame["kind"] != "scores"
                || frame["seq"] != self.seq
                || frame["forward_calls"] != 1
                || frame["processed_tokens"] != inputs.iter().map(|i| i.tokens.len()).sum::<usize>()
            {
                return Err(Error::Backend(
                    "vLLM raw score/work protocol mismatch".into(),
                ));
            }
            let rows: Vec<Vec<f32>> = serde_json::from_value(frame["logits"].clone())?;
            if rows.len() != inputs.len() {
                return Err(Error::Backend("vLLM omitted batch rows".into()));
            }
            rows.into_iter()
                .zip(inputs.iter())
                .map(|(values, input)| {
                    if values.len() != input.positions.len()
                        || values.iter().any(|v| !v.is_finite())
                    {
                        return Err(Error::Backend(
                            "vLLM omitted or returned nonfinite raw pointer scores".into(),
                        ));
                    }
                    Ok(ForwardOutput::Logits {
                        positions: input.positions.clone(),
                        values: Tensor::new(vec![values.len(), 1], values)?,
                    })
                })
                .collect()
        };
        let result = parse();
        if result.is_err() {
            self.failed = true;
            let _ = self.child.kill();
        }
        result
    }
}
impl Backend for VllmBackend {
    fn id(&self) -> BackendId {
        BackendId::Vllm
    }
    fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }
    fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
        Ok(self.execute(vec![input])?.remove(0))
    }
    fn supports_batch(&self) -> bool {
        self.rows > 1
    }
    fn batch_limits(&self) -> BatchLimits {
        BatchLimits {
            max_rows: self.rows,
            max_readouts: Some(256),
        }
    }
    fn forward_batch(&mut self, inputs: Vec<ForwardInput>) -> Result<Vec<ForwardOutput>> {
        self.execute(inputs)
    }
    fn fork(&mut self, _: CacheHandle) -> Result<CacheHandle> {
        Err(Error::Unsupported(
            "vLLM CPU pooling does not expose prefix forks".into(),
        ))
    }
}
impl Drop for VllmBackend {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

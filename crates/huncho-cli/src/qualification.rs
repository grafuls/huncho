//! Content-addressed conformance receipts. They bind retained evidence to a
//! loaded execution, but never replace fresh serving conformance.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use huncho_core::conformance::{ConformanceReport, GoldenSuite};
use huncho_core::engine::{Engine, EvalOptions};
use huncho_core::error::{Error, Result};
use huncho_core::manifest::{BackendId, Family, ModelManifest};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileDigest {
    bytes: u64,
    sha256: String,
}

pub(crate) fn hash_file(path: &Path) -> Result<FileDigest> {
    let mut file = std::fs::File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(Error::Package(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    let mut digest = Sha256::new();
    let mut bytes = 0;
    let mut buffer = vec![0u8; 128 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes += read as u64;
        digest.update(&buffer[..read]);
    }
    Ok(FileDigest {
        bytes,
        sha256: format!("{:x}", digest.finalize()),
    })
}

fn serialized_hash(value: &impl Serialize) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

pub(crate) struct InputSnapshot {
    manifest_path: PathBuf,
    manifest: ModelManifest,
    backend: BackendId,
    dtype: String,
    files: BTreeMap<String, FileDigest>,
}

impl InputSnapshot {
    #[cfg(any(feature = "quantization", feature = "llamacpp"))]
    pub(crate) fn file_digests(&self) -> &BTreeMap<String, FileDigest> {
        &self.files
    }
    /// Run before tokenizer/weights are loaded. Native Qwen records every
    /// loaded base shard, including when the adapter lives in a separate cache.
    pub(crate) fn capture(
        path: &Path,
        manifest: &ModelManifest,
        backend: BackendId,
        dtype: &str,
        dir: &Path,
    ) -> Result<Self> {
        if serialized_hash(manifest)? != serialized_hash(&ModelManifest::load(path)?)? {
            return Err(Error::Package(
                "manifest changed before input capture".into(),
            ));
        }
        let paths = input_paths(path, manifest, backend, dtype, dir)?;
        let files = paths
            .into_iter()
            .map(|(role, path)| Ok((role, hash_file(&path)?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            manifest_path: path.to_path_buf(),
            manifest: manifest.clone(),
            backend,
            dtype: dtype.into(),
            files,
        })
    }

    pub(crate) fn recheck(&self) -> Result<()> {
        let dir = self
            .manifest_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let fresh = Self::capture(
            &self.manifest_path,
            &self.manifest,
            self.backend,
            &self.dtype,
            dir,
        )?;
        if self.files != fresh.files {
            return Err(Error::Conformance(
                "model inputs changed during loading or qualification".into(),
            ));
        }
        Ok(())
    }
}

fn input_paths(
    path: &Path,
    manifest: &ModelManifest,
    backend: BackendId,
    dtype: &str,
    dir: &Path,
) -> Result<BTreeMap<String, PathBuf>> {
    let mut paths = BTreeMap::from([("manifest".into(), path.to_path_buf())]);
    if let Some(tokenizer) = &manifest.backbone.tokenizer {
        paths.insert("tokenizer".into(), dir.join(tokenizer));
    }
    match backend {
        BackendId::Candle if matches!(dtype, "q8_0-fp32" | "q4_0-fp32") => {
            let artifact = manifest.find_artifact(backend, dtype).ok_or_else(|| Error::Package("missing exact packed artifact".into()))?;
            paths.insert("backbone/packed".into(), dir.join(&artifact.path));
            paths.insert("head".into(), dir.join(&manifest.head.weights));
        }
        BackendId::Candle if manifest.family == Family::F3 ||
            (manifest.family == Family::F2 && manifest.prompt_contract.template == "kev-v1") => {
            let base = crate::load::adapter_base_dir(manifest, dir);
            paths.insert("base/config.json".into(), base.join("config.json"));
            add_base_shards(&mut paths, &base)?;
            paths.insert("adapter/config".into(), dir.join("adapter_config.json"));
            paths.insert("adapter/weights".into(), dir.join("adapter_model.safetensors"));
            if manifest.family == Family::F2 { paths.insert("head".into(), dir.join(&manifest.head.weights)); }
        }
        BackendId::Candle => {
            paths.insert("backbone/config".into(), dir.join("config.json"));
            let artifact = manifest.find_artifact(backend, dtype).ok_or_else(|| Error::Package("missing selected artifact".into()))?;
            paths.insert("backbone/weights".into(), dir.join(&artifact.path));
        }
        BackendId::Clef => {
            paths.insert("base/config.json".into(), dir.join("config.json"));
            add_base_shards(&mut paths, dir)?;
            paths.insert("head/config".into(), dir.join("joint_head_config.json"));
            paths.insert("head/weights".into(), dir.join(&manifest.head.weights));
        }
        BackendId::LlamaCpp => {
            let artifact = manifest.find_artifact(backend, dtype).ok_or_else(|| Error::Package("missing exact llama.cpp artifact".into()))?;
            paths.insert("backbone/gguf".into(), dir.join(&artifact.path));
            if manifest.family == Family::F2 { paths.insert("head".into(), dir.join(&manifest.head.weights)); }
        }
        BackendId::Onnx => {
            let artifact = manifest.find_artifact(backend, dtype).ok_or_else(|| Error::Package("missing selected ONNX artifact".into()))?;
            let graph = dir.join(&artifact.path);
            paths.insert("onnx/graph".into(), graph.clone());
            // External ONNX tensor data may use arbitrary filenames. Conservatively
            // capture every regular file in the graph directory tree; directory
            // symlinks fail closed, and individual HF file symlinks are hashed.
            add_tree(&mut paths, graph.parent().unwrap_or(dir), "onnx/files")?;
        }
        _ => return Err(Error::Unsupported("qualification records are implemented for native Candle/Clef, llama.cpp and ONNX artifact loaders".into())),
    }
    Ok(paths)
}

fn add_base_shards(paths: &mut BTreeMap<String, PathBuf>, dir: &Path) -> Result<()> {
    let mut count = 0;
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| Error::Package("non-UTF8 base shard name".into()))?;
        if path.extension().is_some_and(|e| e == "safetensors")
            && name != "adapter_model.safetensors"
            && name != "joint_head.safetensors"
        {
            paths.insert(format!("base/{name}"), path);
            count += 1;
        }
    }
    if count == 0 {
        return Err(Error::Package(
            "no base shards for qualification identity".into(),
        ));
    }
    Ok(())
}

fn add_tree(paths: &mut BTreeMap<String, PathBuf>, dir: &Path, role: &str) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| Error::Package("non-UTF8 artifact filename".into()))?;
        let role = format!("{role}/{name}");
        if path.is_dir() {
            if entry.file_type()?.is_symlink() {
                return Err(Error::Package(
                    "artifact directory symlinks are not supported by qualification capture".into(),
                ));
            }
            add_tree(paths, &path, &role)?;
        } else if path.is_file() {
            paths.insert(role, path);
        } else {
            return Err(Error::Package(
                "artifact tree contains a non-file entry".into(),
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExecutionIdentity {
    program: FileDigest,
    artifacts: BTreeMap<String, FileDigest>,
    manifest_sha256: String,
    calibration_sha256: String,
    backend: String,
    dtype: String,
    device: String,
    metadata: BTreeMap<String, String>,
    options_sha256: String,
    cross_request_max_requests: Option<usize>,
    runtime: BTreeMap<String, String>,
    libraries: BTreeMap<String, FileDigest>,
}

impl ExecutionIdentity {
    pub(crate) fn capture(
        engine: &Engine,
        inputs: &InputSnapshot,
        options: &EvalOptions,
        cross: Option<usize>,
    ) -> Result<Self> {
        inputs.recheck()?;
        if serialized_hash(engine.manifest())? != serialized_hash(&inputs.manifest)?
            || engine.backend_id() != inputs.backend
            || engine.dtype() != inputs.dtype
        {
            return Err(Error::Conformance(
                "loaded engine does not match captured inputs".into(),
            ));
        }
        let mut runtime = BTreeMap::from([
            ("os".into(), std::env::consts::OS.into()),
            ("arch".into(), std::env::consts::ARCH.into()),
            (
                "parallelism".into(),
                std::thread::available_parallelism()
                    .map_or_else(|_| "unknown".into(), |n| n.to_string()),
            ),
        ]);
        for name in [
            "RAYON_NUM_THREADS",
            "CANDLE_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "HUNCHO_DEVICE",
            "HUNCHO_PROJECTION_CHUNK_ROWS",
            "HUNCHO_ATTENTION_FP32",
            "HUNCHO_ATTENTION_QUERY_ROWS",
            "HUNCHO_GROUPED_GQA",
            "HUNCHO_CLEF_VECTOR_HEAD",
            "HUNCHO_CLEF_GROUPED_POOL",
            "HUNCHO_BASE_CACHE_BYTES",
            "HUNCHO_ONNX_COMPACT_READOUT",
            "HUNCHO_ONNX_INTEGRATED_HEAD",
            "HUNCHO_ONNX_OUTPUT_BUFFER_BYTES",
            "HUNCHO_ONNX_EP",
            "HUNCHO_ONNX_THREADS",
            "HUNCHO_LLAMA_THREADS",
            "HUNCHO_LLAMA_BATCH_ROWS",
            "HUNCHO_ONNX_NATIVE_BATCH",
            "HUNCHO_ONNX_SHARED_INITIALIZERS",
            "HUNCHO_CPU_DELTA_RULE",
            "HUNCHO_CPU_CAUSAL_CONV",
            "HUNCHO_CPU_FUSED_GATE",
            "HUNCHO_CPU_BLAS_LIBRARY",
            "HUNCHO_CPU_BLAS_THREADS",
            "HUNCHO_LAYA_SELECTED_HEAD",
            "HUNCHO_COOPERATIVE_PREFILL",
            "HUNCHO_PREFILL_CHUNK_TOKENS",
            "ORT_DYLIB_PATH",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
        ] {
            runtime.insert(name.into(), std::env::var(name).unwrap_or_default());
        }
        for (name, path) in [
            ("kernel", "/proc/sys/kernel/osrelease"),
            ("process_status", "/proc/self/status"),
            ("cpu", "/proc/cpuinfo"),
        ] {
            if let Ok(contents) = std::fs::read_to_string(path) {
                let selected: std::collections::BTreeSet<_> = contents
                    .lines()
                    .filter(|line| match name {
                        "process_status" => line.starts_with("Cpus_allowed_list:"),
                        "cpu" => [
                            "model name",
                            "vendor_id",
                            "flags",
                            "Features",
                            "CPU implementer",
                            "CPU part",
                        ]
                        .iter()
                        .any(|prefix| line.starts_with(prefix)),
                        _ => true,
                    })
                    .map(str::to_owned)
                    .collect();
                runtime.insert(
                    name.into(),
                    selected.into_iter().collect::<Vec<_>>().join("\n"),
                );
            }
        }
        let mut libraries = BTreeMap::new();
        if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
            for line in maps.lines() {
                let columns: Vec<_> = line.split_whitespace().collect();
                if columns.len() == 6 && columns[1].contains('x') && columns[5].starts_with('/') {
                    let path = Path::new(columns[5]);
                    #[cfg(target_os = "linux")]
                    {
                        use std::os::unix::fs::MetadataExt;
                        if std::fs::metadata(path)?.ino().to_string() != columns[4] {
                            return Err(Error::Conformance(
                                "loaded executable mapping no longer matches its file inode".into(),
                            ));
                        }
                    }
                    if path != std::env::current_exe()? {
                        libraries
                            .entry(columns[5].into())
                            .or_insert(hash_file(path)?);
                    }
                }
            }
        }
        Ok(Self {
            program: hash_file(&std::env::current_exe()?)?,
            artifacts: inputs.files.clone(),
            manifest_sha256: serialized_hash(engine.manifest())?,
            calibration_sha256: serialized_hash(engine.calibration())?,
            backend: engine.backend_id().to_string(),
            dtype: engine.dtype().into(),
            device: engine.device().into(),
            metadata: engine.execution_metadata().clone(),
            options_sha256: options_hash(engine, options)?,
            cross_request_max_requests: cross,
            runtime,
            libraries,
        })
    }
}

fn options_hash(engine: &Engine, options: &EvalOptions) -> Result<String> {
    let mut options = options.clone();
    options.extensions = false;
    if engine.family() != Family::F3 {
        options.reference_readout = false;
    }
    serialized_hash(&options)
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QualificationRecord {
    schema_version: u32,
    scope: String,
    unix_seconds: u64,
    identity: ExecutionIdentity,
    golden: FileDigest,
    question_count: usize,
    outcome_gates_passed: bool,
    report: ConformanceReport,
    limits: Vec<String>,
}

fn numerical_pass(report: &ConformanceReport) -> bool {
    let finite = [report.max_prob_delta, report.argmax_agreement, report.ece]
        .iter()
        .all(|n| n.is_finite());
    finite
        && report.passed
        && report.max_prob_delta <= 1e-3
        && report.argmax_agreement == 1.0
        && report.ece <= 0.02
        && report.work.result_cache_hits == 0
        && report.work.prompt_cache_hits == 0
        && !report.cases.is_empty()
        && report.max_prob_delta >= 0.0
        && report.ece >= 0.0
        && report.cases.iter().all(|case| {
            case.max_prob_delta.is_finite()
                && case.max_prob_delta >= 0.0
                && case.max_prob_delta <= 1e-3
                && case.ece.is_finite()
                && case.argmax_match
        })
        && match report.optimization_parity.as_ref() {
            Some(p) => {
                p.max_prob_delta.is_finite()
                    && p.max_prob_delta <= 1e-4
                    && p.argmax_agreement == 1.0
            }
            None => true,
        }
}

fn strict_pass(report: &ConformanceReport, questions: usize) -> bool {
    numerical_pass(report)
        && report.outcome_calibration.as_ref().is_some_and(|o| {
            o.questions == questions
                && questions > 0
                && [o.backend_ece, o.reference_ece]
                    .iter()
                    .all(|n| n.is_finite())
                && o.backend_brier.is_finite()
                && o.reference_brier.is_finite()
        })
}

impl QualificationRecord {
    pub(crate) fn create(
        engine: &Engine,
        inputs: &InputSnapshot,
        options: &EvalOptions,
        suite: &GoldenSuite,
        golden_path: &Path,
        report: ConformanceReport,
    ) -> Result<Self> {
        let question_count = suite.cases.iter().map(|c| c.request.questions.len()).sum();
        let cross = report.cross_request_max_requests;
        let identity = ExecutionIdentity::capture(engine, inputs, options, cross)?;
        Ok(Self {
            schema_version: 1, scope: "fresh-conformance audit; never cached serving authorization".into(),
            unix_seconds: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e| Error::Conformance(e.to_string()))?.as_secs(),
            identity, golden: hash_file(golden_path)?, question_count,
            outcome_gates_passed: engine.calibration().status != huncho_core::manifest::CalibrationStatus::Pending && strict_pass(&report, question_count), report,
            limits: vec![
                "Acceptance applies only to supplied independent vectors and observed outcomes; no universal calibration or speed claim.".into(),
                "This receipt does not prove fit/evaluation separation or absence of model-training overlap.".into(),
                "Fresh startup conformance is mandatory; records do not authorize another model, backend, precision, option set or hardware.".into(),
                "Linux captures loaded executable mappings, kernel, CPU features and affinity; other platforms may have incomplete runtime-library identity. GPU hardware is not independently inventoried.".into(),
            ],
        })
    }

    pub(crate) fn write_new(&self, path: &Path) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(self)?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        Ok(())
    }

    pub(crate) fn load(path: &Path) -> Result<Self> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }

    pub(crate) fn verify(
        &self,
        engine: &Engine,
        inputs: &InputSnapshot,
        options: &EvalOptions,
        golden: &Path,
        cross: Option<usize>,
        require_outcomes: bool,
    ) -> Result<()> {
        let fresh = ExecutionIdentity::capture(engine, inputs, options, cross)?;
        if self.schema_version != 1 || self.identity != fresh || self.golden != hash_file(golden)? {
            return Err(Error::Conformance("qualification record does not match current artifacts, runtime, options or golden bytes".into()));
        }
        let expected_outcomes = engine.calibration().status
            != huncho_core::manifest::CalibrationStatus::Pending
            && strict_pass(&self.report, self.question_count);
        if self.outcome_gates_passed != expected_outcomes
            || !numerical_pass(&self.report)
            || self.report.model != engine.manifest().name
            || self.report.backend != fresh.backend
            || self.report.dtype != fresh.dtype
            || self.report.device != fresh.device
            || self.report.execution_metadata != fresh.metadata
            || self.report.cross_request_max_requests != cross
            || self.report.prefix_cache != options.prefix_cache
            || self.report.persistent_prefix_bytes != options.persistent_prefix_bytes
            || self.report.max_batch_tokens != options.max_batch_tokens
            || self.report.max_batch_padding_percent != options.max_batch_padding_percent
            || (options.max_batch_padding_percent > 0
                && (self.report.work.padded_batch_calls == 0
                    || self.report.work.padded_tokens == 0))
            || self.report.prepare_all != options.prepare_all
            || self.report.cooperative_prefill != options.cooperative_prefill
            || (engine.family() == Family::F3
                && self.report.reference_readout != options.reference_readout)
            || (options.prefix_cache && self.report.work.cache_forks == 0)
            || (fresh.metadata.contains_key("prefill_chunk_tokens")
                && options.prefix_cache
                && self.report.work.chunked_prefills == 0)
            || (options.persistent_prefix_bytes > 0 && self.report.work.persistent_prefix_hits == 0)
            || (options.max_batch_tokens.is_some() && self.report.work.batch_calls == 0)
            || (options.prepare_all && self.report.work.prepared_questions == 0)
            || (options.cooperative_prefill
                && (self.report.work.prefill_yields == 0
                    || self.report.work.prefill_interleaves == 0))
            || (cross.is_some() && self.report.work.cross_request_batches == 0)
            || (require_outcomes
                && (!self.outcome_gates_passed || !strict_pass(&self.report, self.question_count)))
        {
            return Err(Error::Conformance(
                "qualification record does not pass required outcome gates".into(),
            ));
        }
        Ok(())
    }
}

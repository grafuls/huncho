//! Export a new, pending CPU GGUF package through the pinned upstream converter.
use crate::qualification::{hash_file, FileDigest, InputSnapshot};
use clap::Args;
use huncho_backend::llamacpp::{LlamaOptions, LLAMA_CPP_REVISION};
use huncho_core::{
    backend::Backend,
    manifest::{
        ArtifactRef, BackendId, CalibrationConfig, CalibrationStatus, Family, ModelManifest,
    },
};
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Component, Path, PathBuf},
    process::Command,
};

#[derive(Args)]
pub struct ExportArgs {
    /// Original local unquantized Kev F2 or Qwen3.5 F3 package. Never modified.
    #[arg(long)]
    model: PathBuf,
    /// New package directory; must not exist.
    #[arg(long)]
    output: PathBuf,
    /// Clean llama.cpp checkout at the exact runtime revision (no download).
    #[arg(long)]
    tool_dir: PathBuf,
    /// Python environment with the upstream converter's CPU dependencies.
    #[arg(long)]
    python: PathBuf,
    #[arg(long, default_value = "gguf-f32", value_parser = ["gguf-f32", "gguf-f16", "gguf-q8_0", "gguf-q4_0"])]
    dtype: String,
    /// CPU quantizer workers (1–256), applied only to Q8_0/Q4_0 export.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..=256))]
    quantization_threads: u16,
}
fn new_file(path: &Path) -> anyhow::Result<std::fs::File> {
    Ok(std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?)
}
fn copy(source: &Path, target: &Path) -> anyhow::Result<()> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut input = std::fs::File::open(source)?;
    let mut output = new_file(target)?;
    std::io::copy(&mut input, &mut output)?;
    output.sync_all()?;
    Ok(())
}
fn asset(source: &Path, output: &Path, relative: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !relative.is_empty()
            && Path::new(relative)
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "package asset must be a local relative path"
    );
    copy(&source.join(relative), &output.join(relative))
}
fn tools(dir: &Path) -> anyhow::Result<BTreeMap<String, FileDigest>> {
    let revision = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "HEAD"])
        .output()?;
    anyhow::ensure!(
        revision.status.success()
            && std::str::from_utf8(&revision.stdout)?.trim() == LLAMA_CPP_REVISION,
        "converter checkout must be pinned to {LLAMA_CPP_REVISION}"
    );
    anyhow::ensure!(
        Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["diff", "--quiet", "HEAD", "--"])
            .status()?
            .success(),
        "converter checkout has tracked modifications"
    );
    let files = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args([
            "ls-files",
            "-z",
            "--",
            "convert_hf_to_gguf.py",
            "conversion",
            "gguf-py",
        ])
        .output()?;
    anyhow::ensure!(
        files.status.success(),
        "could not enumerate converter inputs"
    );
    let mut hashes = BTreeMap::new();
    for path in files.stdout.split(|&b| b == 0).filter(|v| !v.is_empty()) {
        let name = std::str::from_utf8(path)?;
        let path = dir.join(name);
        anyhow::ensure!(
            !path.is_symlink(),
            "converter source symlinks are unsupported"
        );
        hashes.insert(name.to_string(), hash_file(&path)?);
    }
    anyhow::ensure!(
        hashes.contains_key("convert_hf_to_gguf.py"),
        "pinned converter script missing"
    );
    Ok(hashes)
}

pub fn run(args: ExportArgs) -> anyhow::Result<()> {
    let path = if args.model.is_dir() {
        args.model.join("huncho-model.json")
    } else {
        args.model.clone()
    };
    let mut manifest = ModelManifest::load(&path)?;
    anyhow::ensure!(
        manifest.family == Family::F3
            || (manifest.family == Family::F2 && manifest.prompt_contract.template == "kev-v1"),
        "GGUF export requires Kev F2 or native Qwen3.5 F3"
    );
    anyhow::ensure!(
        !manifest
            .backbone
            .artifacts
            .values()
            .flatten()
            .any(|a| a.quantization.is_some()),
        "export from the original unquantized package"
    );
    let source = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let tokenizer = manifest
        .backbone
        .tokenizer
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("GGUF export requires the original tokenizer.json"))?;
    let base = crate::load::adapter_base_dir(&manifest, source);
    let inputs = InputSnapshot::capture(&path, &manifest, BackendId::Candle, "fp32", source)?;
    let tool_dir = args.tool_dir.canonicalize()?;
    let tool_files = tools(&tool_dir)?;
    // Keep the venv path: resolving its Python symlink changes sys.prefix
    // and can silently invoke the base interpreter without converter packages.
    let python = std::env::current_dir()?.join(&args.python);
    let python_digest = hash_file(&python)?;
    let python_version = Command::new(&python).arg("-c").arg("import sys,torch,numpy,transformers,safetensors; print(sys.version); print('torch='+torch.__version__+';numpy='+numpy.__version__+';transformers='+transformers.__version__+';safetensors='+safetensors.__version__)")
        .env("CUDA_VISIBLE_DEVICES", "").env("HF_HUB_OFFLINE", "1").output()?;
    anyhow::ensure!(
        python_version.status.success(),
        "converter Python dependencies are missing"
    );
    // Additional tokenizer assets affect converter metadata, so capture them
    // too. The manifest's exact JSON always supplies inference token IDs.
    let mut auxiliaries = BTreeMap::new();
    for name in [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    ] {
        let candidate = [source.join(name), base.join(name)]
            .into_iter()
            .find(|p| p.is_file());
        if let Some(path) = candidate {
            auxiliaries.insert(name, (path.clone(), hash_file(&path)?));
        }
    }
    std::fs::create_dir(&args.output)?;
    let staging = args.output.join("merged-hf");
    let dense_bytes = huncho_backend::qwen3_5::export_merged_hf(
        &base,
        source,
        manifest.family == Family::F3,
        &staging,
    )?;
    copy(&source.join(tokenizer), &staging.join("tokenizer.json"))?;
    for (name, (path, _)) in &auxiliaries {
        copy(path, &staging.join(name))?;
    }
    let merged_files = ["config.json", "model.safetensors", "tokenizer.json"]
        .into_iter()
        .map(|name| Ok((name, hash_file(&staging.join(name))?)))
        .collect::<anyhow::Result<BTreeMap<_, _>>>()?;
    let artifact_path = "backbone.gguf";
    let quantized = matches!(args.dtype.as_str(), "gguf-q8_0" | "gguf-q4_0");
    let conversion_path = if quantized {
        "merged-f32.gguf"
    } else {
        artifact_path
    };
    let converted = Command::new(&python)
        .arg(tool_dir.join("convert_hf_to_gguf.py"))
        .arg(staging.canonicalize()?)
        .arg("--outtype")
        .arg(if quantized {
            "f32"
        } else {
            args.dtype.trim_start_matches("gguf-")
        })
        .arg("--outfile")
        .arg(args.output.canonicalize()?.join(conversion_path))
        .arg("--no-mtp")
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PYTHONNOUSERSITE", "1")
        .output()?;
    let mut log = new_file(&args.output.join("conversion.log"))?;
    log.write_all(&converted.stdout)?;
    log.write_all(&converted.stderr)?;
    log.sync_all()?;
    anyhow::ensure!(
        converted.status.success(),
        "pinned converter failed; inspect conversion.log in the unpublished output package"
    );
    let quantization_source = quantized
        .then(|| hash_file(&args.output.join(conversion_path)))
        .transpose()?;
    let quantizer = if quantized {
        Some(huncho_backend::llamacpp::quantize_gguf_cpu(
            &args.output.join(conversion_path),
            &args.output.join(artifact_path),
            &args.dtype,
            usize::from(args.quantization_threads),
        )?)
    } else {
        None
    };
    if let Some(expected) = &quantization_source {
        anyhow::ensure!(*expected == hash_file(&args.output.join(conversion_path))?, "FP32 quantization source changed during export");
    }
    std::fs::File::open(args.output.join(artifact_path))?.sync_all()?;
    if manifest.family == Family::F2 {
        asset(source, &args.output, &manifest.head.weights)?;
    }
    asset(source, &args.output, tokenizer)?;
    inputs.recheck()?;
    anyhow::ensure!(
        tool_files == tools(&tool_dir)? && python_digest == hash_file(&python)?,
        "converter inputs changed during export"
    );
    for (path, digest) in auxiliaries.values() {
        anyhow::ensure!(
            *digest == hash_file(path)?,
            "tokenizer metadata changed during export"
        );
    }
    let source_manifest = manifest.clone();
    manifest.backbone.artifacts = BTreeMap::from([(
        BackendId::LlamaCpp,
        vec![ArtifactRef {
            path: artifact_path.into(),
            dtype: args.dtype.clone(),
            quantization: quantized.then(|| {
                format!(
                    "llamacpp-qwen35-{}-v1",
                    args.dtype.trim_start_matches("gguf-")
                )
            }),
        }],
    )]);
    if manifest.family == Family::F3 {
        // The trained vocabulary projection is inside the self-contained GGUF.
        manifest.head.weights = artifact_path.into();
    }
    let mut pending = manifest.calibration.default.clone();
    pending.temperature = 1.0;
    pending.per_type_temperatures = None;
    pending.temperature_by_options = None;
    pending.status = CalibrationStatus::Pending;
    manifest.calibration = CalibrationConfig {
        default: pending.clone(),
        entries: BTreeMap::from([(format!("llamacpp:{}", args.dtype), pending)]),
        eval_set_hash: None,
    };
    manifest.reference = None;
    manifest.validate()?;
    let backend = huncho_backend::LlamaCppBackend::load(
        &args.output,
        &manifest,
        &args.dtype,
        LlamaOptions::default(),
    )?;
    let capabilities = backend.capabilities();
    drop(backend);
    let provenance = serde_json::json!({
        "schema_version": 1, "source_manifest": source_manifest, "source_files": inputs.file_digests(),
        "converter_revision": LLAMA_CPP_REVISION, "converter_files": tool_files,
        "converter_python": python_digest, "converter_python_versions": String::from_utf8(python_version.stdout)?,
        "quantizer": quantizer, "quantization_source": quantization_source,
        "exporter_binary": hash_file(&std::env::current_exe()?)?, "merged_hf_files": merged_files, "merged_hf_payload_bytes": dense_bytes,
        "tokenizer_auxiliaries": auxiliaries.iter().map(|(name,(_,digest))| (*name,digest)).collect::<BTreeMap<_,_>>(),
        "artifact": hash_file(&args.output.join(artifact_path))?, "runtime_capabilities": capabilities.extra,
        "qualified": false,
        "limits": ["CPU FP32 LoRA merge and pinned upstream Qwen3.5 tensor/normalization/value-head conversion.", "F2 uses the original external pointer head; its unused auxiliary vocabulary projection is tied to embeddings.", "Q8/Q4 keep embeddings, vocabulary output, norms, convolution and block-incompatible projections FP32; no re-quantization or importance matrix.", "No fitting or held-out acceptance. Temperatures start pending; quantized serving requires an exact backend:dtype refit and fresh complete observed-outcome conformance.", "Conversion temporarily retains dense HF staging and GGUF; payload bytes do not measure peak RSS."]
    });
    let mut provenance_file = new_file(&args.output.join("llamacpp-provenance.json"))?;
    provenance_file.write_all(&serde_json::to_vec_pretty(&provenance)?)?;
    provenance_file.sync_all()?;
    std::fs::remove_dir_all(&staging)?;
    if quantized {
        std::fs::remove_file(args.output.join(conversion_path))?;
    }
    let mut published = new_file(&args.output.join("huncho-model.json"))?;
    published.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
    published.write_all(b"\n")?;
    published.sync_all()?;
    println!("{}", serde_json::to_string_pretty(&provenance)?);
    Ok(())
}

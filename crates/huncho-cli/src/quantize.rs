//! Durable conversion is distinct from fitting and held-out acceptance.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use clap::Args;
use huncho_backend::qwen3_5::quantized::{convert_kev, Scheme};
use huncho_core::manifest::{
    ArtifactRef, BackendId, CalibrationConfig, CalibrationStatus, Family, ModelManifest,
};

#[derive(Args)]
pub struct QuantizeArgs {
    /// Local unquantized Kev package directory or manifest. Never modified.
    #[arg(long)]
    model: PathBuf,
    /// New output package directory; existing directories are rejected.
    #[arg(long)]
    output: PathBuf,
    /// Packed projection weights with FP32 activations/state/head.
    #[arg(long, value_parser = ["q8_0-fp32", "q4_0-fp32"])]
    dtype: String,
}

fn new_file(path: &Path) -> anyhow::Result<std::fs::File> {
    Ok(std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?)
}

fn copy_declared(source: &Path, destination: &Path, relative: &str) -> anyhow::Result<()> {
    let path = Path::new(relative);
    anyhow::ensure!(
        !path.as_os_str().is_empty()
            && path.components().all(|c| matches!(c, Component::Normal(_))),
        "package assets must use local relative paths"
    );
    let target = destination.join(path);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut input = std::fs::File::open(source.join(path))?;
    let mut output = new_file(&target)?;
    std::io::copy(&mut input, &mut output)?;
    output.sync_all()?;
    Ok(())
}

pub fn run(args: QuantizeArgs) -> anyhow::Result<()> {
    let path = if args.model.is_dir() {
        args.model.join("huncho-model.json")
    } else {
        args.model.clone()
    };
    let mut manifest = ModelManifest::load(&path)?;
    anyhow::ensure!(
        manifest.family == Family::F2 && manifest.prompt_contract.template == "kev-v1",
        "conversion currently supports unquantized Kev F2 packages only"
    );
    anyhow::ensure!(
        !manifest.backbone.artifacts.values().flatten().any(|a| a
            .quantization
            .as_deref()
            .is_some_and(|q| q.starts_with("kev-projections-"))),
        "convert from the original unquantized package"
    );
    let source = path.parent().unwrap_or_else(|| Path::new("."));
    let scheme = Scheme::from_dtype(&args.dtype).unwrap();
    let inputs = crate::qualification::InputSnapshot::capture(
        &path,
        &manifest,
        BackendId::Candle,
        "fp32",
        source,
    )?;
    let base = crate::load::adapter_base_dir(&manifest, source);
    // Ownership is established by create_dir; no existing evidence/package can
    // be overwritten. On failure leave partial output for inspection without a
    // published manifest. Conversion uses CPU directly and never probes GPUs.
    std::fs::create_dir(&args.output)?;
    let artifact_path = "backbone.gguf";
    let mut artifact = new_file(&args.output.join(artifact_path))?;
    let stats = convert_kev(&base, source, scheme, &mut artifact)?;
    artifact.sync_all()?;
    copy_declared(source, &args.output, &manifest.head.weights)?;
    if let Some(tokenizer) = &manifest.backbone.tokenizer {
        copy_declared(source, &args.output, tokenizer)?;
    }
    inputs.recheck()?;
    let provenance = serde_json::json!({
        "schema_version": 1,
        "source_manifest": manifest,
        "source_files": inputs.file_digests(),
        "converter_binary": crate::qualification::hash_file(&std::env::current_exe()?)?,
        "artifact": crate::qualification::hash_file(&args.output.join(artifact_path))?,
        "statistics": stats,
        "qualified": false,
        "limits": ["CPU only; FP32 LoRA merge followed by packed Q8_0/Q4_0 projections.", "QMatMul internally quantizes activation blocks; FP32 activations and state do not imply unchanged arithmetic.", "Conversion and loading temporarily materialize dense projection weights; statistics cover payload, not peak RSS.", "No fitting data or held-out outcomes were evaluated; do not serve before refit and fresh labeled conformance."]
    });
    new_file(&args.output.join("quantization-provenance.json"))?
        .write_all(&serde_json::to_vec_pretty(&provenance)?)?;
    manifest.backbone.artifacts = BTreeMap::from([(
        BackendId::Candle,
        vec![ArtifactRef {
            path: artifact_path.into(),
            dtype: args.dtype.clone(),
            quantization: Some(scheme.profile().into()),
        }],
    )]);
    // Preserve source pins as provenance, but the merged artifact loads no
    // external adapter/base weights. Tokenizer, trained pointer and contract
    // stay unchanged. Source temperatures are not copied into the new variant.
    let mut pending = manifest.calibration.default.clone();
    pending.temperature = 1.0;
    pending.per_type_temperatures = None;
    pending.temperature_by_options = None;
    pending.status = CalibrationStatus::Pending;
    manifest.calibration = CalibrationConfig {
        default: pending.clone(),
        entries: BTreeMap::from([(format!("candle:{}", args.dtype), pending)]),
        eval_set_hash: None,
    };
    manifest.reference = None;
    manifest.validate()?;
    let mut published = new_file(&args.output.join("huncho-model.json"))?;
    published.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
    published.write_all(b"\n")?;
    published.sync_all()?;
    println!("{}", serde_json::to_string_pretty(&provenance)?);
    Ok(())
}

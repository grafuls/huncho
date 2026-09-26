//! `huncho calibrate` — fit temperatures for a backend × dtype (CONV-02).

use std::path::{Path, PathBuf};

use clap::Args;
use serde::Deserialize;
use huncho_core::calibration::fit_temperature;
use huncho_core::manifest::{BackendId, CalibrationEntry, CalibrationStatus, ModelManifest};

use crate::load::resolve_model;

#[derive(Args)]
pub struct CalibrateArgs {
    /// Path to the model manifest to update. Mutually exclusive with `--model`.
    #[arg(long, conflicts_with = "model")]
    pub manifest: Option<String>,

    /// A model reference to resolve: a local package dir/path or an HF repo id.
    /// Mutually exclusive with `--manifest`. When used, the resolved (cached)
    /// manifest is updated in place.
    #[arg(long, conflicts_with = "manifest")]
    pub model: Option<String>,

    /// Git revision to resolve an HF `--model` at.
    #[arg(long)]
    pub revision: Option<String>,

    /// Hugging Face access token (defaults to HF_TOKEN / login cache).
    #[arg(long)]
    pub token: Option<String>,

    /// Model cache directory (also used by HF resolution).
    #[arg(long)]
    pub cache_dir: Option<String>,

    /// Backend id (onnx|llamacpp|mlx|vllm).
    #[arg(long)]
    pub backend: String,

    /// Dtype the temperature is being fitted for.
    #[arg(long)]
    pub dtype: String,

    /// Path to a JSON file with `{"rows": [[...logits]], "targets": [idx]}`.
    #[arg(long)]
    pub data: String,

    /// Write the fitted temperature back into the manifest (default true).
    #[arg(long, default_value_t = true)]
    pub save: bool,
}

#[derive(Deserialize)]
struct FitData {
    rows: Vec<Vec<f32>>,
    targets: Vec<usize>,
}

pub fn run(args: CalibrateArgs) -> anyhow::Result<()> {
    let manifest_path: PathBuf = match (&args.model, &args.manifest) {
        (Some(model), None) => resolve_model(
            model,
            None,
            Some(&args.dtype),
            args.revision.clone(),
            args.token.clone(),
            args.cache_dir.clone(),
            false,
        )?,
        (None, Some(path)) => PathBuf::from(path),
        (Some(_), Some(_)) => anyhow::bail!("`--model` and `--manifest` are mutually exclusive"),
        (None, None) => anyhow::bail!("one of `--model` or `--manifest` is required"),
    };

    let backend = BackendId::parse(&args.backend)?;
    let data: FitData = serde_json::from_slice(&std::fs::read(&args.data)?)?;
    tracing::info!("fitting temperature on {} rows", data.rows.len());
    let (temperature, nll) = fit_temperature(&data.rows, &data.targets)?;
    tracing::info!(
        "fitted temperature={temperature:.4} (nll={nll:.4}) for {backend}:{}",
        args.dtype
    );

    let manifest_path = Path::new(&manifest_path);
    let mut manifest = ModelManifest::load(manifest_path)?;
    let key = huncho_core::manifest::CalibrationConfig::key_for(&backend.to_string(), &args.dtype);
    let conf = manifest.calibration.default.confidence.clone();
    manifest.calibration.entries.insert(
        key.clone(),
        CalibrationEntry {
            temperature,
            per_type_temperatures: manifest
                .calibration
                .default
                .per_type_temperatures
                .clone(),
            confidence: conf,
            status: CalibrationStatus::Refit,
        },
    );
    if args.save {
        let bytes = serde_json::to_vec_pretty(&manifest)?;
        std::fs::write(manifest_path, bytes)?;
        tracing::info!(
            "wrote calibrated temperature for {key} into {}",
            manifest_path.display()
        );
    } else {
        tracing::info!("dry-run: not writing manifest");
    }

    Ok(())
}

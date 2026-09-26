//! `huncho calibrate` — fit temperatures for a backend × dtype (CONV-02).

use std::path::Path;

use clap::Args;
use serde::Deserialize;
use huncho_core::calibration::fit_temperature;
use huncho_core::manifest::{
    BackendId, CalibrationEntry, CalibrationStatus, ModelManifest,
};

#[derive(Args)]
pub struct CalibrateArgs {
    /// Path to the model manifest to update.
    #[arg(long)]
    pub manifest: String,

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
    let backend = BackendId::parse(&args.backend)?;
    let data: FitData = serde_json::from_slice(&std::fs::read(&args.data)?)?;
    tracing::info!(
        "fitting temperature on {} rows",
        data.rows.len()
    );
    let (temperature, nll) = fit_temperature(&data.rows, &data.targets)?;
    tracing::info!("fitted temperature={temperature:.4} (nll={nll:.4}) for {backend}:{}", args.dtype);

    let manifest_path = Path::new(&args.manifest);
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

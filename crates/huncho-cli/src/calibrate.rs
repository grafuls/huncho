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

    /// Backend id (onnx|llamacpp|mlx|vllm|candle).
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
            temperature_by_options: manifest
                .calibration
                .default
                .temperature_by_options
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

#[cfg(test)]
mod tests {
    use super::*;
    use huncho_core::manifest::{
        self, Backbone, BackboneSource, CalibrationConfig, CalibrationEntry, CalibrationStatus,
        ConfidenceDef, Family, HeadConfig, HeadKind, ModelManifest, PromptContract,
    };
    use std::collections::BTreeMap;
    use std::fs;

    fn test_manifest(path: &Path) {
        let m = ModelManifest {
            schema_version: manifest::MANIFEST_SCHEMA_VERSION.into(),
            name: "laya".into(),
            family: Family::F1,
            backbone: Backbone {
                source: BackboneSource::Hf {
                    repo: "convaiinnovations/laya".into(),
                    revision: "main".into(),
                },
                artifacts: Default::default(),
                hidden_size: 1024,
                max_context: 4096,
                tokenizer: None,
            },
            adapter: None,
            head: HeadConfig {
                kind: HeadKind::OptionMarker,
                weights: "head.safetensors".into(),
                width: 1,
                pointer_offset: None,
            },
            prompt_contract: PromptContract {
                template: "f1-v1".into(),
                option_marker_tokens: vec!["<option:0>".into()],
                state_budget: 1024,
                head_budget: 1024,
                max_options: 255,
                contract_hash: "0123456789abcdef".into(),
                max_len: 512,
                head_max_len: 192,
            },
            calibration: CalibrationConfig {
                default: CalibrationEntry {
                    temperature: 1.0,
                    per_type_temperatures: None,
                    temperature_by_options: None,
                    confidence: ConfidenceDef::Peak,
                    status: CalibrationStatus::Pending,
                },
                entries: BTreeMap::new(),
                eval_set_hash: None,
            },
            reference: None,
            capabilities: Default::default(),
        };
        fs::write(path, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    }

    #[test]
    fn fit_temperature_lowers_nll_and_writes_entry() {
        // Logits where a temperature > 1 sharpens the distribution toward the
        // correct class, so `fit_temperature` must return a finite, positive
        // temperature.
        let rows = vec![
            vec![3.0f32, 1.0, 1.0],
            vec![0.5, 3.0, 0.5],
            vec![0.2, 0.3, 4.0],
            vec![2.5, 2.0, 1.0],
        ];
        let targets = vec![0, 1, 2, 0];
        let (temp, nll) = fit_temperature(&rows, &targets).unwrap();
        assert!(temp.is_finite() && temp > 0.0);
        assert!(nll.is_finite());

        // Now drive `run` against a temp package and assert the manifest got the
        // fitted entry with status Refit.
        let tmp = tempfile::tempdir().unwrap();
        let mf = tmp.path().join("huncho-model.json");
        test_manifest(&mf);
        let data = tmp.path().join("data.json");
        let data_json = serde_json::json!({ "rows": rows, "targets": targets });
        fs::write(&data, data_json.to_string()).unwrap();

        let args = CalibrateArgs {
            manifest: Some(mf.to_string_lossy().to_string()),
            model: None,
            revision: None,
            token: None,
            cache_dir: None,
            backend: "onnx".into(),
            dtype: "fp32".into(),
            data: data.to_string_lossy().to_string(),
            save: true,
        };
        run(args).unwrap();

        let m = ModelManifest::load(&mf).unwrap();
        let key = CalibrationConfig::key_for("onnx", "fp32");
        let entry = m.calibration.entries.get(&key).expect("entry written");
        assert_eq!(entry.status, CalibrationStatus::Refit);
        assert!(entry.temperature > 0.0);
    }

    #[test]
    fn fit_temperature_dry_run_does_not_write() {
        let tmp = tempfile::tempdir().unwrap();
        let mf = tmp.path().join("huncho-model.json");
        test_manifest(&mf);
        let data = tmp.path().join("data.json");
        fs::write(&data, serde_json::json!({ "rows": [[1.0, 2.0]], "targets": [1] }).to_string()).unwrap();

        let args = CalibrateArgs {
            manifest: Some(mf.to_string_lossy().to_string()),
            model: None,
            revision: None,
            token: None,
            cache_dir: None,
            backend: "onnx".into(),
            dtype: "fp32".into(),
            data: data.to_string_lossy().to_string(),
            save: false,
        };
        run(args).unwrap();

        let m = ModelManifest::load(&mf).unwrap();
        assert!(m.calibration.entries.is_empty());
    }
}

//! `huncho calibrate` — fit temperatures for a backend × dtype (CONV-02).

use std::path::{Path, PathBuf};

use clap::Args;
use huncho_core::calibration::{bucket_size, fit_temperature};
use huncho_core::manifest::{BackendId, CalibrationEntry, CalibrationStatus, ModelManifest};
use serde::Deserialize;

use crate::load::{resolve_model, BackendChoice};

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
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub save: bool,

    /// Emit fitted entry and fitting statistics as standalone JSON.
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

#[derive(Deserialize)]
struct FitData {
    rows: Vec<Vec<f32>>,
    targets: Vec<usize>,
    /// Optional row-aligned question types for per-type/cardinality fitting.
    #[serde(default)]
    qtypes: Option<Vec<String>>,
}

fn fit_entry(
    data: &FitData,
    confidence: huncho_core::manifest::ConfidenceDef,
) -> anyhow::Result<(CalibrationEntry, f64)> {
    let (temperature, nll) = fit_temperature(&data.rows, &data.targets)?;
    let mut per_type = std::collections::BTreeMap::new();
    let mut by_options = std::collections::BTreeMap::new();
    if let Some(qtypes) = &data.qtypes {
        anyhow::ensure!(
            qtypes.len() == data.rows.len(),
            "qtypes must match the number of logit rows"
        );
        let mut groups = std::collections::BTreeMap::<String, Vec<usize>>::new();
        for (index, qtype) in qtypes.iter().enumerate() {
            anyhow::ensure!(
                matches!(qtype.as_str(), "choice" | "score" | "noul"),
                "unknown question type `{qtype}`"
            );
            anyhow::ensure!(
                qtype != "noul" || data.rows[index].len() == 2,
                "noul fit rows require two logits"
            );
            groups.entry(qtype.clone()).or_default().push(index);
            groups
                .entry(format!("{qtype}:{}", bucket_size(data.rows[index].len())))
                .or_default()
                .push(index);
        }
        for (key, indices) in groups {
            let rows = indices
                .iter()
                .map(|&index| data.rows[index].clone())
                .collect::<Vec<_>>();
            let targets = indices
                .iter()
                .map(|&index| data.targets[index])
                .collect::<Vec<_>>();
            let fitted = fit_temperature(&rows, &targets)?.0;
            if key.contains(':') {
                by_options.insert(key, fitted);
            } else {
                per_type.insert(key, fitted);
            }
        }
    }
    Ok((
        CalibrationEntry {
            temperature,
            // Never inherit DEFAULT overrides that could shadow this variant's fit.
            per_type_temperatures: (!per_type.is_empty()).then_some(per_type),
            temperature_by_options: (!by_options.is_empty()).then_some(by_options),
            confidence,
            status: CalibrationStatus::Refit,
        },
        nll,
    ))
}

pub fn run(args: CalibrateArgs) -> anyhow::Result<()> {
    let manifest_path: PathBuf = match (&args.model, &args.manifest) {
        (Some(model), None) => resolve_model(
            model,
            BackendChoice::Mock,
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
    let manifest_path = Path::new(&manifest_path);
    let mut manifest = ModelManifest::load(manifest_path)?;
    let key = huncho_core::manifest::CalibrationConfig::key_for(&backend.to_string(), &args.dtype);
    let confidence = manifest
        .calibration
        .resolve(&backend.to_string(), &args.dtype)
        .confidence;
    let (entry, nll) = fit_entry(&data, confidence)?;
    let temperature = entry.temperature;
    tracing::info!(
        "fitted temperature={temperature:.4} (nll={nll:.4}) for {backend}:{}",
        args.dtype
    );

    manifest.calibration.entries.insert(key.clone(), entry);
    // The old shared evaluation hash cannot identify newly supplied fit data.
    manifest.calibration.eval_set_hash = None;
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

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "manifest": manifest_path,
                "backend": backend.to_string(), "dtype": args.dtype,
                "fitting_rows": data.rows.len(), "fit_nll": nll,
                "entry": manifest.calibration.entries[&key], "saved": args.save,
                "qualification": "refit only; held-out runtime conformance remains required"
            }))?
        );
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
            f3: None,
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
            json: false,
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
        fs::write(
            &data,
            serde_json::json!({ "rows": [[1.0, 2.0]], "targets": [1] }).to_string(),
        )
        .unwrap();

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
            json: false,
        };
        run(args).unwrap();

        let m = ModelManifest::load(&mf).unwrap();
        assert!(m.calibration.entries.is_empty());
    }

    #[test]
    fn scalar_refit_does_not_inherit_overrides_or_replace_variant_confidence() {
        let tmp = tempfile::tempdir().unwrap();
        let mf = tmp.path().join("huncho-model.json");
        test_manifest(&mf);
        let mut manifest = ModelManifest::load(&mf).unwrap();
        manifest.calibration.default.per_type_temperatures =
            Some(BTreeMap::from([("choice".into(), 99.0)]));
        manifest.calibration.default.temperature_by_options =
            Some(BTreeMap::from([("choice:2".into(), 88.0)]));
        manifest.calibration.eval_set_hash = Some("old evaluation set".into());
        let mut old = manifest.calibration.default.clone();
        old.confidence = ConfidenceDef::Entropy;
        manifest.calibration.entries.insert("onnx:fp32".into(), old);
        fs::write(&mf, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let data = tmp.path().join("fit.json");
        fs::write(
            &data,
            serde_json::json!({"rows": [[1.0, 2.0]], "targets": [1]}).to_string(),
        )
        .unwrap();
        run(CalibrateArgs {
            manifest: Some(mf.to_string_lossy().into()),
            model: None,
            revision: None,
            token: None,
            cache_dir: None,
            backend: "onnx".into(),
            dtype: "fp32".into(),
            data: data.to_string_lossy().into(),
            save: true,
            json: false,
        })
        .unwrap();
        let updated = ModelManifest::load(&mf).unwrap();
        let entry = updated.calibration.resolve("onnx", "fp32");
        assert!(entry.per_type_temperatures.is_none());
        assert!(entry.temperature_by_options.is_none());
        assert!(matches!(entry.confidence, ConfidenceDef::Entropy));
        assert!(updated.calibration.eval_set_hash.is_none());
        assert_eq!(
            updated.calibration.default.temperature_by_options.unwrap()["choice:2"],
            88.0
        );
    }

    #[test]
    fn typed_refit_fits_only_supplied_types_and_buckets() {
        let data = FitData {
            rows: vec![
                vec![2.0, 0.0],
                vec![0.0, 2.0],
                vec![1.0, 2.0, 0.0],
                vec![1.0, 0.0],
            ],
            targets: vec![0, 0, 1, 1],
            qtypes: Some(vec![
                "choice".into(),
                "choice".into(),
                "score".into(),
                "noul".into(),
            ]),
        };
        let (entry, _) = fit_entry(&data, ConfidenceDef::Peak).unwrap();
        let buckets = entry.temperature_by_options.unwrap();
        assert_eq!(buckets.len(), 3);
        assert_eq!(
            buckets["choice:2"],
            fit_temperature(&data.rows[..2], &data.targets[..2])
                .unwrap()
                .0
        );
        assert_eq!(
            buckets["score:3-5"],
            fit_temperature(&data.rows[2..3], &data.targets[2..3])
                .unwrap()
                .0
        );
        assert!(!buckets.contains_key("choice:11+"));
        let mut bad = data;
        bad.qtypes = Some(vec!["choice".into()]);
        assert!(fit_entry(&bad, ConfidenceDef::Peak).is_err());
        bad.qtypes = Some(vec!["unknown".into(); 4]);
        assert!(fit_entry(&bad, ConfidenceDef::Peak).is_err());
    }
}

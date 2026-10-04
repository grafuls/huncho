//! `huncho conform` — run golden vectors against a backend (CONF-02/03).

use std::path::Path;

use clap::Args;
use huncho_core::conformance::{self, ConformanceReport, ConformanceThresholds};
use huncho_core::engine::Engine;
use huncho_core::manifest::{BackendId, Family, ModelManifest};

use crate::load::{engine_from_resolved_manifest, mock_engine, resolve_model, BackendChoice};

#[derive(Args)]
pub struct ConformArgs {
    /// A local model manifest path (huncho-model.json). Mutually exclusive with `--model`.
    #[arg(long, conflicts_with = "model")]
    pub manifest: Option<String>,

    /// A model reference to resolve: a local package dir/path or an HF repo id.
    /// Mutually exclusive with `--manifest`.
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

    /// Path to the golden-vector JSON suite. When omitted with `--model`, it is
    /// derived from the resolved manifest's `reference.golden`.
    #[arg(long)]
    pub golden: Option<String>,

    /// Backend override (auto|onnx|candle|clef|mock). Auto selects from model metadata.
    #[arg(long, default_value = "auto")]
    pub backend: String,

    /// Override dtype for manifest/model-loaded runs.
    #[arg(long)]
    pub dtype: Option<String>,

    /// Override max tolerated probability delta.
    #[arg(long)]
    pub max_delta: Option<f32>,

    /// Emit the report as JSON.
    #[arg(long, default_value_t = false)]
    pub json: bool,

    /// Name for the built-in mock model when no manifest/model is given.
    #[arg(long, default_value = "mock")]
    pub mock_model: String,
}

pub fn run(args: ConformArgs) -> anyhow::Result<()> {
    let backend = BackendChoice::parse(&args.backend)?;
    let dtype = args.dtype.as_deref();

    let engine: Engine;
    let golden_path: String;

    if let Some(model) = &args.model {
        let manifest_path = resolve_model(
            model,
            backend,
            dtype,
            args.revision.clone(),
            args.token.clone(),
            args.cache_dir.clone(),
            true,
        )?;
        golden_path = match &args.golden {
            Some(g) => g.clone(),
            None => {
                let m = ModelManifest::load(&manifest_path)?;
                m.reference
                    .map(|r| {
                        let parent = manifest_path.parent().unwrap_or_else(|| Path::new("."));
                        parent.join(&r.golden).to_string_lossy().to_string()
                    })
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "`--model` resolved a manifest with no `reference.golden`; pass `--golden`"
                        )
                    })?
            }
        };
        engine = engine_from_resolved_manifest(&manifest_path, backend, dtype)?;
    } else if let Some(path) = &args.manifest {
        golden_path = args
            .golden
            .clone()
            .ok_or_else(|| anyhow::anyhow!("`--golden` is required when using `--manifest`"))?;
        engine = engine_from_resolved_manifest(std::path::Path::new(path), backend, dtype)?;
    } else {
        golden_path = args
            .golden
            .clone()
            .ok_or_else(|| anyhow::anyhow!("`--golden` is required when no manifest/model is given"))?;
        engine = mock_engine(&args.mock_model, Family::F1, BackendId::Onnx, "fp32", 1.0)?;
    }

    let suite = conformance::load_suite(&golden_path)?;
    let thresholds = ConformanceThresholds {
        max_prob_delta: args.max_delta.unwrap_or(1e-3),
        ..Default::default()
    };

    let report = conformance::run_suite(&engine, &suite, &thresholds)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&report);
    }

    if !report.passed {
        std::process::exit(1);
    }
    Ok(())
}

fn print_report(report: &ConformanceReport) {
    println!(
        "Conformance: model={} backend={} dtype={}",
        report.model, report.backend, report.dtype
    );
    println!("  cases: {}", report.cases.len());
    println!("  max probability delta: {:.6}", report.max_prob_delta);
    println!("  argmax agreement:      {:.3}", report.argmax_agreement);
    println!("  ECE drift:             {:.6}", report.ece);
    println!(
        "  status: {}",
        if report.passed { "PASS" } else { "FAIL" }
    );
    if !report.passed {
        for c in &report.cases {
            if !c.argmax_match {
                println!(
                    "    - case `{}` argmax mismatch (max delta {:.6})",
                    c.id, c.max_prob_delta
                );
            }
        }
    }
}

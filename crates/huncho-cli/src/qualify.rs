//! `huncho qualify` — guided serving qualification for one model.
//!
//! Loads a model, discovers its golden suite (the package's upstream golden
//! or an explicit `--golden` path), and runs the qualification tier the suite
//! supports:
//!
//! - complete observed labels -> full qualification (probability delta,
//!   argmax and ECE/Brier outcome gates) — the tier `--require-qualified`
//!   serving demands;
//! - no labels -> numerical qualification (probability delta and argmax
//!   against the reference vectors) — served in the default trial mode, and
//!   upgradeable to full by filling in observed outcomes
//!   (`--template` writes the fill-in file).
//!
//! This is the single command behind the flag scatter
//! (`--qualification-golden MODEL=PATH`), so first-run users get one verb
//! instead of an error message.

use std::path::{Path, PathBuf};

use clap::Args;
use huncho_core::{
    conformance::{load_suite, GoldenSuite},
    engine::{Engine, EvalOptions},
};

use crate::load::{engine_from_ref, BackendChoice};

#[derive(Args)]
pub struct QualifyArgs {
    /// Model reference: a local package dir/manifest path or an HF repo id
    /// (`owner/repo`). Hub references require `hf` (included by `clef`).
    #[arg(long)]
    pub model: String,

    /// Backend override (auto|onnx|candle|clef|llamacpp|vllm|mock).
    #[arg(long, default_value = "auto", env = "HUNCHO_BACKEND")]
    pub backend: String,

    /// Override the dtype for manifest/manually-loaded models.
    #[arg(long, env = "HUNCHO_DTYPE")]
    pub dtype: Option<String>,

    /// Git revision to resolve HF model references at (default: repo default branch).
    #[arg(long)]
    pub revision: Option<String>,

    /// Hugging Face access token (defaults to HF_TOKEN / login cache).
    #[arg(long)]
    pub token: Option<String>,

    /// Model cache directory (also used by HF resolution).
    #[arg(long, env = "HUNCHO_CACHE_DIR")]
    pub cache_dir: Option<String>,

    /// Pinned golden suite path. Default: the package's own reference golden
    /// (`reference.golden` in the manifest).
    #[arg(long)]
    pub golden: Option<PathBuf>,

    /// Write a fill-in labeled template from the golden suite when it carries
    /// no observed labels, then exit without qualifying.
    #[arg(long)]
    pub template: Option<PathBuf>,

    /// Emit the conformance report as JSON.
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

pub fn run(args: QualifyArgs) -> anyhow::Result<()> {
    let backend = BackendChoice::parse(&args.backend)
        .map_err(|error| anyhow::anyhow!("invalid --backend: {error}"))?;
    let dtype = args.dtype.as_deref();
    let (engine, manifest_path) = engine_from_ref(
        &args.model,
        backend,
        dtype,
        args.revision.clone(),
        args.token.clone(),
        args.cache_dir.clone(),
    )?;

    let golden_path = resolve_golden(&engine, &manifest_path, args.golden.as_deref())?;
    let suite = load_suite(&golden_path)
        .map_err(|error| anyhow::anyhow!("loading golden `{}`: {error}", golden_path.display()))?;
    let labeled = suite.cases.iter().any(|case| !case.targets.is_empty());

    println!(
        "model={} backend={} dtype={} device={}",
        engine.manifest().name,
        engine.backend_id(),
        engine.dtype(),
        engine.device()
    );
    println!("golden: {}", golden_path.display());

    if !labeled {
        if let Some(template) = &args.template {
            write_template(&suite, template)?;
            println!(
                "wrote labeled template: {}\nfill each case's `targets` with the observed answer \
                 ids (one per question), then re-run `huncho qualify` with `--golden {}`",
                template.display(),
                template.display()
            );
            return Ok(());
        }
        println!(
            "suite carries no observed target labels: running numerical gates only \
             (probability delta + argmax)"
        );
    }

    let options = EvalOptions {
        reference_readout: true,
        ..Default::default()
    };
    let report = if labeled {
        engine.qualify_for_serving(&suite, &options, None)?
    } else {
        engine.qualify_for_serving_numerical(&suite, &options, None)?
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_summary(&report);
    }

    let status = engine.serving_qualification_status(&options, None);
    if report.passed {
        println!("qualification status: {}", status.as_str());
        if !labeled {
            println!(
                "next: collect observed outcomes and re-run with a labeled suite for full \
                 qualification (`huncho qualify --model {} --template labeled-golden.json`)",
                args.model
            );
        }
        Ok(())
    } else {
        anyhow::bail!("qualification failed: golden gates rejected this model/profile");
    }
}

/// Explicit `--golden` wins; otherwise resolve the manifest's
/// `reference.golden` relative to the manifest directory.
fn resolve_golden(
    engine: &Engine,
    manifest_path: &Path,
    explicit: Option<&Path>,
) -> anyhow::Result<PathBuf> {
    if let Some(path) = explicit {
        anyhow::ensure!(path.is_file(), "golden file not found: {}", path.display());
        return Ok(path.to_path_buf());
    }
    if let Some(reference) = &engine.manifest().reference {
        let path = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&reference.golden);
        if path.is_file() {
            return Ok(path);
        }
        anyhow::bail!(
            "the package's reference golden `{}` was not found next to {} (downloaded package \
             incomplete?); pass --golden /path/to/pinned-golden.json, or build one with \
             `huncho conform`",
            reference.golden,
            manifest_path.display()
        );
    }
    anyhow::bail!(
        "the package declares no reference golden and none was supplied; pass --golden \
         /path/to/pinned-golden.json, or build one with `huncho conform`"
    )
}

/// Write the golden suite as a fill-in labeled template: same cases and
/// expected vectors, `targets` left empty for observed outcomes.
fn write_template(suite: &GoldenSuite, path: &Path) -> anyhow::Result<()> {
    let mut template = suite.clone();
    for case in &mut template.cases {
        case.targets.clear();
    }
    huncho_core::conformance::save_suite(&template, path)
        .map_err(|error| anyhow::anyhow!("writing template `{}`: {error}", path.display()))
}

fn print_summary(report: &huncho_core::conformance::ConformanceReport) {
    println!(
        "  cases: {}; max probability delta: {:.6}; argmax agreement: {:.3}; ECE drift: {:.6}",
        report.cases.len(),
        report.max_prob_delta,
        report.argmax_agreement,
        report.ece
    );
    if let Some(outcomes) = &report.outcome_calibration {
        println!(
            "  outcome gates: ECE {:.6} / Brier {:.6} ({} labeled questions)",
            outcomes.backend_ece, outcomes.backend_brier, outcomes.questions
        );
    } else {
        println!("  outcome gates: not run (no observed labels in the suite)");
    }
    println!("  status: {}", if report.passed { "PASS" } else { "FAIL" });
}

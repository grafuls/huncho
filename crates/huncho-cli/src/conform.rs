//! `huncho conform` — run golden vectors against a backend (CONF-02/03).

use std::path::Path;

use clap::Args;
use huncho_core::conformance::{self, ConformanceReport, ConformanceThresholds};
use huncho_core::engine::{Engine, EvalOptions};
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

    /// Backend override (auto|onnx|candle|clef|llamacpp|vllm|mock). Auto selects from model metadata.
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

    /// Persist an execution-bound audit receipt (requires `qualification`).
    /// Refuses to overwrite an existing file; never replaces fresh serving gates.
    #[cfg(feature = "qualification")]
    #[arg(long)]
    pub write_qualification: Option<String>,

    /// Check the legacy F3 readout against the same golden suite.
    #[arg(long, default_value_t = false)]
    pub reference_readout: bool,

    /// Qualify request-local Kev prefix reuse against the unchanged golden suite.
    #[arg(long, default_value_t = false)]
    pub prefix_cache: bool,

    /// Qualify actual interleaved CPU Kev prefix chunks across golden cases.
    #[arg(long, default_value_t = false, requires = "prefix_cache")]
    pub cooperative_prefill: bool,

    /// Qualify exact retained native prefixes; requires prefix reuse and real hits.
    #[arg(long, default_value = "0")]
    pub persistent_prefix_bytes: usize,

    /// Qualify native equal-length question batching against the same suite.
    #[arg(long)]
    pub max_batch_tokens: Option<usize>,

    /// Allow supported CPU mixed lengths with at most this percent padding (0..100).
    #[arg(long, default_value_t = 0, requires = "max_batch_tokens")]
    pub max_batch_padding_percent: usize,

    /// Qualify batches across this many cases (2–64), requiring actual mixing.
    #[arg(long, requires = "max_batch_tokens", conflicts_with = "prefix_cache")]
    pub batch_max_requests: Option<usize>,

    /// Qualify upfront prompt preparation against unchanged independent forwards.
    #[arg(long, default_value_t = false)]
    pub prepare_all: bool,

    /// Name for the built-in mock model when no manifest/model is given.
    #[arg(long, default_value = "mock")]
    pub mock_model: String,
}

pub fn run(args: ConformArgs) -> anyhow::Result<()> {
    #[cfg(feature = "qualification")]
    if let Some(path) = &args.write_qualification {
        anyhow::ensure!(
            !Path::new(path).exists(),
            "qualification output already exists"
        );
        anyhow::ensure!(
            args.model.is_some() || args.manifest.is_some(),
            "qualification records require a real manifest/model"
        );
    }
    let backend = BackendChoice::parse(&args.backend)?;
    let dtype = args.dtype.as_deref();

    let engine: Engine;
    let golden_path: String;
    #[cfg(feature = "qualification")]
    let mut inputs = None;

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
        engine = load_engine(
            &manifest_path,
            backend,
            dtype,
            #[cfg(feature = "qualification")]
            (&args.write_qualification, &mut inputs),
        )?;
    } else if let Some(path) = &args.manifest {
        golden_path = args
            .golden
            .clone()
            .ok_or_else(|| anyhow::anyhow!("`--golden` is required when using `--manifest`"))?;
        engine = load_engine(
            Path::new(path),
            backend,
            dtype,
            #[cfg(feature = "qualification")]
            (&args.write_qualification, &mut inputs),
        )?;
    } else {
        golden_path = args.golden.clone().ok_or_else(|| {
            anyhow::anyhow!("`--golden` is required when no manifest/model is given")
        })?;
        engine = mock_engine(&args.mock_model, Family::F1, BackendId::Onnx, "fp32", 1.0)?;
    }

    #[cfg(feature = "qualification")]
    let golden_identity = args
        .write_qualification
        .as_ref()
        .map(|_| crate::qualification::hash_file(Path::new(&golden_path)))
        .transpose()?;
    let suite = conformance::load_suite(&golden_path)?;
    let thresholds = ConformanceThresholds {
        max_prob_delta: args.max_delta.unwrap_or(1e-3),
        ..Default::default()
    };

    let options = EvalOptions {
        reference_readout: args.reference_readout,
        prefix_cache: args.prefix_cache,
        cooperative_prefill: args.cooperative_prefill,
        persistent_prefix_bytes: args.persistent_prefix_bytes,
        max_batch_tokens: args.max_batch_tokens,
        max_batch_padding_percent: args.max_batch_padding_percent,
        prepare_all: args.prepare_all
            || args.cooperative_prefill
            || args.batch_max_requests.is_some(),
        ..Default::default()
    };
    let report = if let Some(rows) = args.batch_max_requests {
        conformance::run_suite_with_cross_request_batches(
            &engine,
            &suite,
            &thresholds,
            &options,
            rows,
        )?
    } else {
        conformance::run_suite_with_options(&engine, &suite, &thresholds, &options)?
    };

    #[cfg(feature = "qualification")]
    if let Some(path) = &args.write_qualification {
        anyhow::ensure!(
            golden_identity.as_ref()
                == Some(&crate::qualification::hash_file(Path::new(&golden_path))?),
            "golden bytes changed during qualification"
        );
        let record = crate::qualification::QualificationRecord::create(
            &engine,
            inputs
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing artifact capture"))?,
            &options,
            &suite,
            Path::new(&golden_path),
            report.clone(),
        )?;
        record.write_new(Path::new(path))?;
    }

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

fn load_engine(
    path: &Path,
    backend: BackendChoice,
    dtype: Option<&str>,
    #[cfg(feature = "qualification")] evidence: (
        &Option<String>,
        &mut Option<crate::qualification::InputSnapshot>,
    ),
) -> huncho_core::error::Result<Engine> {
    #[cfg(feature = "qualification")]
    if evidence.0.is_some() {
        let engine = crate::load::engine_from_resolved_manifest_observed(
            path,
            backend,
            dtype,
            |manifest, backend, dtype, dir| {
                *evidence.1 = Some(crate::qualification::InputSnapshot::capture(
                    path, manifest, backend, dtype, dir,
                )?);
                Ok(())
            },
        )?;
        evidence.1.as_ref().unwrap().recheck()?;
        return Ok(engine);
    }
    engine_from_resolved_manifest(path, backend, dtype)
}

fn print_report(report: &ConformanceReport) {
    println!(
        "Conformance: model={} backend={} dtype={} device={} reference_readout={} prefix_cache={} batch_tokens={:?} prepare_all={}",
        report.model,
        report.backend,
        report.dtype,
        report.device,
        report.reference_readout,
        report.prefix_cache,
        report.max_batch_tokens,
        report.prepare_all
    );
    println!("  cases: {}", report.cases.len());
    println!(
        "  native batches: {}; cross-request batches: {}; forks: {}; physical token positions: {}",
        report.work.batch_calls,
        report.work.cross_request_batches,
        report.work.cache_forks,
        report.work.processed_tokens
    );
    println!("  max probability delta: {:.6}", report.max_prob_delta);
    println!("  argmax agreement:      {:.3}", report.argmax_agreement);
    println!("  ECE drift:             {:.6}", report.ece);
    if let Some(parity) = &report.optimization_parity {
        println!(
            "  independent parity:    delta={:.6}; argmax={:.3} (requires <=0.0001 and 1.0)",
            parity.max_prob_delta, parity.argmax_agreement
        );
    }
    if let Some(outcomes) = &report.outcome_calibration {
        println!(
            "  ECE basis:             observed outcomes ({} questions)",
            outcomes.questions
        );
        println!(
            "  backend/reference ECE: {:.6} / {:.6}",
            outcomes.backend_ece, outcomes.reference_ece
        );
        println!(
            "  backend/reference Brier: {:.6} / {:.6}",
            outcomes.backend_brier, outcomes.reference_brier
        );
    } else {
        println!("  ECE basis:             reference agreement (unlabeled suite)");
    }
    println!("  status: {}", if report.passed { "PASS" } else { "FAIL" });
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

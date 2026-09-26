//! `huncho conform` — run golden vectors against a backend (CONF-02/03).

use clap::Args;
use huncho_core::conformance::{
    self, ConformanceReport, ConformanceThresholds,
};
use huncho_core::engine::Engine;
use huncho_core::manifest::{BackendId, Family};

use crate::load::{engine_from_manifest, mock_engine, mock_engine_from_manifest};

#[derive(Args)]
pub struct ConformArgs {
    /// Model manifest path. If omitted, a built-in mock model is used.
    #[arg(long)]
    pub manifest: Option<String>,

    /// Path to the golden-vector JSON suite.
    #[arg(long)]
    pub golden: String,

    /// Backend to use (onnx|mock).
    #[arg(long, default_value = "mock")]
    pub backend: String,

    /// Override dtype for manifest-loaded models.
    #[arg(long)]
    pub dtype: Option<String>,

    /// Override max tolerated probability delta.
    #[arg(long)]
    pub max_delta: Option<f32>,

    /// Emit the report as JSON.
    #[arg(long, default_value_t = false)]
    pub json: bool,

    /// Name for the mock model when no manifest is given.
    #[arg(long, default_value = "mock")]
    pub model: String,
}

pub fn run(args: ConformArgs) -> anyhow::Result<()> {
    let engine: Engine = match &args.manifest {
        Some(path) => {
            if args.backend.eq_ignore_ascii_case("mock") {
                mock_engine_from_manifest(path)?
            } else {
                let backend = BackendId::parse(&args.backend)?;
                engine_from_manifest(path, backend, args.dtype.as_deref())?
            }
        }
        None => mock_engine(&args.model, Family::F1, BackendId::Onnx, "fp32", 1.0)?,
    };

    let suite = conformance::load_suite(&args.golden)?;
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
    println!("Conformance: model={} backend={} dtype={}", report.model, report.backend, report.dtype);
    println!("  cases: {}", report.cases.len());
    println!("  max probability delta: {:.6}", report.max_prob_delta);
    println!("  argmax agreement:      {:.3}", report.argmax_agreement);
    println!("  ECE drift:             {:.6}", report.ece);
    println!("  status: {}", if report.passed { "PASS" } else { "FAIL" });
    if !report.passed {
        for c in &report.cases {
            if !c.argmax_match {
                println!("    - case `{}` argmax mismatch (max delta {:.6})", c.id, c.max_prob_delta);
            }
        }
    }
}

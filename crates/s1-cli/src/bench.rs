//! `s1 bench` — latency and throughput micro-benchmark (CONF-04).

use std::collections::BTreeMap;
use std::time::Instant;

use clap::Args;
use s1_core::contract::{Instructions, Question, StateValue, SystemOneRequest};
use s1_core::engine::{Engine, EvalOptions};
use s1_core::manifest::{BackendId, Family};

use crate::load::{engine_from_manifest, mock_engine, mock_engine_from_manifest};

#[derive(Args)]
pub struct BenchArgs {
    /// Model manifest path; mock is used when omitted.
    #[arg(long)]
    pub manifest: Option<String>,

    /// Backend to use (onnx|mock).
    #[arg(long, default_value = "mock")]
    pub backend: String,

    /// Number of questions per request (1, 5, or 20).
    #[arg(long, default_value = "1")]
    pub questions: usize,

    /// Number of timed iterations.
    #[arg(long, default_value = "50")]
    pub iterations: usize,

    /// Use a long state (~1500 chars) instead of a short one.
    #[arg(long, default_value_t = false)]
    pub long_state: bool,

    /// Name for the mock model when no manifest is given.
    #[arg(long, default_value = "mock")]
    pub model: String,
}

pub fn run(args: BenchArgs) -> anyhow::Result<()> {
    let engine: Engine = match &args.manifest {
        Some(path) => {
            if args.backend.eq_ignore_ascii_case("mock") {
                mock_engine_from_manifest(path)?
            } else {
                let backend = BackendId::parse(&args.backend)?;
                engine_from_manifest(path, backend, None)?
            }
        }
        None => mock_engine(&args.model, Family::F1, BackendId::Onnx, "fp32", 1.0)?,
    };

    let request = make_request(args.questions, args.long_state);

    // Warm-up.
    let _ = engine.eval(&request, &EvalOptions::default());

    let mut latencies = Vec::with_capacity(args.iterations);
    let n = args.iterations.max(1);
    let start = Instant::now();
    for _ in 0..n {
        let t0 = Instant::now();
        let resp = engine.eval(&request, &EvalOptions::default())?;
        let dt = t0.elapsed();
        latencies.push(dt.as_secs_f64() * 1000.0);
        let _ = resp;
    }
    let wall = start.elapsed().as_secs_f64();

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    let p99 = percentile(&latencies, 0.99);
    let mean = latencies.iter().sum::<f64>() / n as f64;
    let qps = n as f64 / wall;

    println!("Bench: model={} questions={} backend={}", engine.manifest().name, args.questions, engine.backend_id());
    println!("  mean   : {mean:.3} ms");
    println!("  p50    : {p50:.3} ms");
    println!("  p95    : {p95:.3} ms");
    println!("  p99    : {p99:.3} ms");
    println!("  req/s  : {qps:.2}");
    Ok(())
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn make_request(questions: usize, long_state: bool) -> SystemOneRequest {
    let state = if long_state {
        let mut s = String::from("Support conversation:\n");
        for i in 0..40 {
            s.push_str(&format!("customer: message number {i} about the product issue.\n"));
        }
        StateValue::from(s)
    } else {
        StateValue::from("A support ticket where the customer asks for a refund.")
    };

    let mut qs = BTreeMap::new();
    for i in 0..questions {
        let q = Question::Choice {
            instructions: Instructions::from(serde_json::Value::String("Which team handles this?".into())),
            criteria: [
                ("returns".to_string(), Some(serde_json::Value::String("money back".into()))),
                ("billing".to_string(), Some(serde_json::Value::String("charge issue".into()))),
                ("shipping".to_string(), Some(serde_json::Value::String("delivery problem".into()))),
            ]
            .into_iter()
            .collect(),
        };
        qs.insert(format!("q{i}"), q);
    }

    SystemOneRequest {
        state,
        model: "mock".into(),
        questions: qs,
    }
}

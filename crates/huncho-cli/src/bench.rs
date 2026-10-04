//! `huncho bench` — latency and throughput micro-benchmark (CONF-04).

use std::collections::BTreeMap;
use std::time::Instant;

use clap::Args;
use huncho_core::contract::{Instructions, Question, StateValue, SystemOneRequest};
use huncho_core::engine::{Engine, EvalOptions};
use huncho_core::manifest::{BackendId, Family};

use crate::load::{
    engine_from_manifest, engine_from_resolved_manifest, mock_engine, mock_engine_from_manifest,
    resolve_model,
};

#[derive(Args)]
pub struct BenchArgs {
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

    /// Backend to use (onnx|mock|candle|clef).
    #[arg(long, default_value = "mock")]
    pub backend: String,

    /// Override dtype for manifest/model-loaded runs.
    #[arg(long)]
    pub dtype: Option<String>,

    /// Number of questions per request (1, 5, or 20).
    #[arg(long, default_value = "1")]
    pub questions: usize,

    /// Number of timed iterations.
    #[arg(long, default_value = "50")]
    pub iterations: usize,

    /// Use a long state (~1500 chars) instead of a short one.
    #[arg(long, default_value_t = false)]
    pub long_state: bool,

    /// Name for the built-in mock model when no manifest/model is given.
    #[arg(long, default_value = "mock")]
    pub mock_model: String,
}

pub fn run(args: BenchArgs) -> anyhow::Result<()> {
    let is_mock = args.backend.eq_ignore_ascii_case("mock");
    let backend_id = if is_mock {
        None
    } else {
        Some(BackendId::parse(&args.backend)?)
    };
    let dtype = args.dtype.as_deref();
    let engine: Engine = if let Some(model) = &args.model {
        let manifest_path = resolve_model(
            model,
            backend_id,
            dtype,
            args.revision.clone(),
            args.token.clone(),
            args.cache_dir.clone(),
            false,
        )?;
        engine_from_resolved_manifest(&manifest_path, backend_id, dtype)?
    } else if let Some(path) = &args.manifest {
        if is_mock {
            mock_engine_from_manifest(path)?
        } else {
            engine_from_manifest(path, backend_id.expect("non-mock backend"), dtype)?
        }
    } else {
        mock_engine(&args.mock_model, Family::F1, BackendId::Onnx, "fp32", 1.0)?
    };

    let request = make_request(args.questions, args.long_state);

    // Warm-up.
    let _ = engine.eval(&request, &EvalOptions::default())?;

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

    println!(
        "Bench: model={} questions={} backend={}",
        engine.manifest().name,
        args.questions,
        engine.backend_id()
    );
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
            instructions: Instructions::from(serde_json::Value::String(
                "Which team handles this?".into(),
            )),
            criteria: [
                (
                    "returns".to_string(),
                    Some(serde_json::Value::String("money back".into())),
                ),
                (
                    "billing".to_string(),
                    Some(serde_json::Value::String("charge issue".into())),
                ),
                (
                    "shipping".to_string(),
                    Some(serde_json::Value::String("delivery problem".into())),
                ),
            ]
            .into_iter()
            .collect(),
        };
        qs.insert(format!("q{i}"), q);
    }

    SystemOneRequest {
        state,
        model: "mock".into(),
        questions: qs.into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_bounds() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&v, 0.50) - 3.0).abs() < 1e-6);
        assert!((percentile(&v, 0.99) - 5.0).abs() < 1e-6);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn make_request_builds_expected_questions() {
        let r = make_request(5, false);
        assert_eq!(r.model, "mock");
        assert_eq!(r.questions.len(), 5);
        assert!(r.questions.keys().all(|k| k.starts_with('q')));
        for q in r.questions.values() {
            if let Question::Choice { criteria, .. } = q {
                assert!(criteria.contains_key("returns"));
                assert!(criteria.contains_key("billing"));
            } else {
                panic!("expected a choice question");
            }
        }
        let long = serde_json::to_string(&make_request(1, true).state).unwrap();
        assert!(long.len() > 500);
    }

    #[test]
    fn run_mock_bench_completes() {
        // Smoke-test the whole bench path on the in-process mock backend.
        let args = BenchArgs {
            manifest: None,
            model: None,
            revision: None,
            token: None,
            cache_dir: None,
            backend: "mock".into(),
            dtype: None,
            questions: 5,
            iterations: 5,
            long_state: false,
            mock_model: "mock".into(),
        };
        // Capturing stdout isn't necessary; we just assert it runs cleanly.
        run(args).unwrap();
    }
}

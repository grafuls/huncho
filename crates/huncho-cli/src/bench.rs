//! `huncho bench` — latency and throughput micro-benchmark (CONF-04).

use std::collections::BTreeMap;
use std::time::Instant;

use clap::{Args, ValueEnum};
use huncho_core::contract::{Instructions, Question, StateValue, SystemOneRequest};
use huncho_core::engine::{Engine, EvalOptions, EvalStats};
use huncho_core::manifest::{BackendId, Family};

use crate::load::{engine_from_resolved_manifest, mock_engine, resolve_model, BackendChoice};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Workload {
    Choice,
    Mixed,
}

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

    /// Backend override (auto|onnx|candle|clef|llamacpp|vllm|mock). Auto selects from model metadata.
    #[arg(long, default_value = "auto")]
    pub backend: String,

    /// Override dtype for manifest/model-loaded runs.
    #[arg(long)]
    pub dtype: Option<String>,

    /// Number of questions per request (typically 1, 5, or 20).
    #[arg(long, default_value = "1")]
    pub questions: usize,

    /// Number of timed iterations.
    #[arg(long, default_value = "50")]
    pub iterations: usize,

    /// Concurrent in-process clients; reports closed-loop throughput.
    #[arg(long, default_value = "1")]
    pub concurrency: usize,

    /// Shared-weight CPU execution contexts (1–8). Clients bind round-robin
    /// to contexts; each context is warmed before timing.
    #[arg(long, env = "HUNCHO_REPLICAS", default_value_t = 1,
        value_parser = clap::value_parser!(u8).range(1..=8))]
    pub replicas: u8,

    /// Choice questions only, or a mix of choice/score/noul.
    #[arg(long, value_enum, default_value = "choice")]
    pub workload: Workload,

    /// Repeat identical requests instead of varying state per iteration.
    #[arg(long, default_value_t = false)]
    pub repeat_inputs: bool,

    /// Per-model charged-byte budget for exact response reuse (0 disables).
    #[arg(long, default_value = "0")]
    pub result_cache_bytes: usize,

    /// Emit benchmark results and execution/workload metadata as JSON.
    #[arg(long, default_value_t = false)]
    pub json: bool,

    /// Benchmark the legacy F3 readout instead of candidate-only projection.
    #[arg(long, default_value_t = false)]
    pub reference_readout: bool,

    /// Benchmark opt-in native Kev request-local prefix reuse.
    #[arg(long, default_value_t = false)]
    pub prefix_cache: bool,

    /// Charged native prefix-snapshot budget; requires prefix reuse (0 disables).
    #[arg(long, default_value = "0")]
    pub persistent_prefix_bytes: usize,

    /// Opt-in native equal-length question batches bounded by total tokens.
    #[arg(long, conflicts_with = "prefix_cache")]
    pub max_batch_tokens: Option<usize>,

    /// Allow supported CPU mixed lengths with at most this percent padding (0..100).
    #[arg(long, default_value_t = 0, requires = "max_batch_tokens")]
    pub max_batch_padding_percent: usize,

    /// Collate up to this many complete requests per timed client group.
    /// F5 retains whole schemas; other families retain their question prompts.
    #[arg(long, requires = "max_batch_tokens", value_parser = clap::value_parser!(u16).range(2..=64))]
    pub batch_max_requests: Option<u16>,

    /// Use a long state (~1500 chars) instead of a short one.
    #[arg(long, default_value_t = false)]
    pub long_state: bool,

    /// Name for the built-in mock model when no manifest/model is given.
    #[arg(long, default_value = "mock")]
    pub mock_model: String,
}

pub fn run(args: BenchArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.questions > 0 && args.iterations > 0 && args.concurrency > 0,
        "questions, iterations and concurrency must be positive"
    );
    let replica_count = usize::from(args.replicas);
    anyhow::ensure!((1..=8).contains(&replica_count), "replicas must be 1..8");
    anyhow::ensure!(
        replica_count <= args.concurrency.min(args.iterations),
        "replicas must not exceed timed clients or iterations"
    );
    anyhow::ensure!(
        args.persistent_prefix_bytes == 0 || args.prefix_cache,
        "persistent prefix bytes requires prefix reuse"
    );
    anyhow::ensure!(
        args.persistent_prefix_bytes == 0 || args.persistent_prefix_bytes >= replica_count,
        "persistent prefix budget must provide at least one byte per replica"
    );
    let backend = BackendChoice::parse(&args.backend)?;
    let dtype = args.dtype.as_deref();
    let engine: Engine = if let Some(model) = &args.model {
        let manifest_path = resolve_model(
            model,
            backend,
            dtype,
            args.revision.clone(),
            args.token.clone(),
            args.cache_dir.clone(),
            false,
        )?;
        engine_from_resolved_manifest(&manifest_path, backend, dtype)?
    } else if let Some(path) = &args.manifest {
        engine_from_resolved_manifest(std::path::Path::new(path), backend, dtype)?
    } else {
        mock_engine(&args.mock_model, Family::F1, BackendId::Onnx, "fp32", 1.0)?
    };

    let engine = engine.with_result_cache(args.result_cache_bytes);
    anyhow::ensure!(
        replica_count == 1 || engine.device() == "CPU",
        "replica benchmarks currently support CPU only"
    );
    let mut engines = vec![engine];
    for _ in 1..replica_count {
        let replica = engines[0].replica()?;
        engines.push(replica);
    }
    let engine = &engines[0];
    anyhow::ensure!(
        args.batch_max_requests.is_none() || engine.supports_batch(),
        "cross-request benchmarks require a native batch backend"
    );
    anyhow::ensure!(
        engine.family() != Family::F5
            || args.max_batch_tokens.is_none()
            || args.batch_max_requests.is_some(),
        "F5 batching benchmarks require --batch-max-requests; questions within one schema cannot be split"
    );
    let warmup_case = if args.repeat_inputs { 0 } else { usize::MAX };
    let options = EvalOptions {
        reference_readout: args.reference_readout,
        prefix_cache: args.prefix_cache,
        persistent_prefix_bytes: args.persistent_prefix_bytes / replica_count,
        max_batch_tokens: args.max_batch_tokens,
        max_batch_padding_percent: args.max_batch_padding_percent,
        prepare_all: args.batch_max_requests.is_some(),
        ..Default::default()
    };
    for context in &engines {
        let requests = (0..usize::from(args.batch_max_requests.unwrap_or(1)).min(args.iterations))
            .map(|_| {
                make_request(
                    args.questions,
                    args.long_state,
                    args.workload,
                    warmup_case,
                    &context.manifest().name,
                )
            })
            .collect();
        evaluate_group(
            context,
            requests,
            &options,
            args.batch_max_requests.is_some(),
            &mut Default::default(),
        )?;
    }
    let n = args.iterations;
    let workers = args.concurrency.min(n);
    let barrier = std::sync::Barrier::new(workers + 1);
    let (mut latencies, work, replica_work, wall) =
        std::thread::scope(|scope| -> anyhow::Result<_> {
            let mut handles = Vec::with_capacity(workers);
            for worker in 0..workers {
                let replica = worker % replica_count;
                let engine = &engines[replica];
                let args = &args;
                let barrier = &barrier;
                let options = &options;
                handles.push(scope.spawn(
                    move || -> anyhow::Result<(usize, Vec<f64>, EvalStats)> {
                        let mut latencies = Vec::new();
                        let mut work = EvalStats::default();
                        barrier.wait();
                        let iterations: Vec<_> = (worker..n).step_by(workers).collect();
                        for group in
                            iterations.chunks(usize::from(args.batch_max_requests.unwrap_or(1)))
                        {
                            let requests = group
                                .iter()
                                .map(|&iteration| {
                                    let case = if args.repeat_inputs { 0 } else { iteration };
                                    make_request(
                                        args.questions,
                                        args.long_state,
                                        args.workload,
                                        case,
                                        &engine.manifest().name,
                                    )
                                })
                                .collect();
                            let start = Instant::now();
                            let mut stats = EvalStats::default();
                            evaluate_group(
                                engine,
                                requests,
                                options,
                                args.batch_max_requests.is_some(),
                                &mut stats,
                            )?;
                            work.accumulate(&stats);
                            // Every request completes with its group. Do not
                            // divide latency by batch size or hide preparation.
                            latencies.extend(
                                std::iter::repeat(start.elapsed().as_secs_f64() * 1000.0)
                                    .take(group.len()),
                            );
                        }
                        Ok((replica, latencies, work))
                    },
                ));
            }
            let start = Instant::now();
            barrier.wait();
            let mut latencies = Vec::with_capacity(n);
            let mut work = EvalStats::default();
            let mut replica_work: Vec<_> = (0..replica_count)
                .map(|index| ReplicaWork {
                    index,
                    requests: 0,
                    work: EvalStats::default(),
                })
                .collect();
            for handle in handles {
                let (replica, worker_latencies, worker_work) = handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("benchmark worker panicked"))??;
                replica_work[replica].requests += worker_latencies.len();
                replica_work[replica].work.accumulate(&worker_work);
                latencies.extend(worker_latencies);
                work.accumulate(&worker_work);
            }
            Ok((latencies, work, replica_work, start.elapsed().as_secs_f64()))
        })?;

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    anyhow::ensure!(
        args.batch_max_requests.is_none()
            || work.cross_request_batches > 0
            || work.result_cache_hits > 0,
        "cross-request benchmark submitted no actual batch; use more iterations per client or a larger token budget"
    );
    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    let p99 = percentile(&latencies, 0.99);
    let mean = latencies.iter().sum::<f64>() / n as f64;
    let qps = n as f64 / wall;

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": engine.manifest().name, "backend": engine.backend_id().to_string(),
                "dtype": engine.dtype(), "device": engine.device(), "execution_metadata": engine.execution_metadata(), "questions": args.questions,
                "iterations": n, "concurrency": workers, "workload": args.workload,
                "replicas": replica_count, "replica_work": replica_work,
                "client_assignment": "worker modulo replica count",
                "persistent_prefix_bytes_per_replica": options.persistent_prefix_bytes,
            "repeat_inputs": args.repeat_inputs, "long_state": args.long_state,
            "result_cache_bytes": args.result_cache_bytes,
            "reference_readout": args.reference_readout,
            "prefix_cache": args.prefix_cache, "persistent_prefix_bytes": args.persistent_prefix_bytes, "max_batch_tokens": args.max_batch_tokens, "max_batch_padding_percent": args.max_batch_padding_percent, "batch_max_requests": args.batch_max_requests, "work": work,
                "mean_ms": mean, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
                "requests_per_second": qps, "questions_per_second": qps * args.questions as f64,
                "measurement": if args.batch_max_requests.is_some() { "warm closed-loop client groups; each request's latency is its entire group's completion time" } else { "warm closed-loop in-process engine evaluation" }
            }))?
        );
        return Ok(());
    }

    println!(
        "Bench: model={} questions={} backend={} dtype={} device={}",
        engine.manifest().name,
        args.questions,
        engine.backend_id(),
        engine.dtype(),
        engine.device()
    );
    println!(
        "  workload: {:?}; concurrency: {workers}; replicas: {replica_count}; repeated inputs: {}",
        args.workload, args.repeat_inputs
    );
    println!("  mean   : {mean:.3} ms");
    println!("  p50    : {p50:.3} ms");
    println!("  p95    : {p95:.3} ms");
    println!("  p99    : {p99:.3} ms");
    println!("  req/s  : {qps:.2}");
    println!("  questions/s: {:.2}", qps * args.questions as f64);
    println!(
        "  physical tokens: {}; prefills: {}; forwards: {}; forks: {}; reused prefix positions: {}",
        work.processed_tokens,
        work.prefill_calls,
        work.forward_calls,
        work.cache_forks,
        work.reused_prefix_tokens
    );
    println!("  exact result cache hits: {}", work.result_cache_hits);
    println!("  prepared prompt cache hits: {}", work.prompt_cache_hits);
    Ok(())
}

#[derive(serde::Serialize)]
struct ReplicaWork {
    index: usize,
    requests: usize,
    work: EvalStats,
}

fn evaluate_group(
    engine: &Engine,
    requests: Vec<SystemOneRequest>,
    options: &EvalOptions,
    grouped: bool,
    work: &mut EvalStats,
) -> huncho_core::Result<()> {
    if grouped {
        let packets = requests
            .into_iter()
            .map(|request| {
                engine.prepare_eval_with_stats(request, options.clone(), &mut Default::default())
            })
            .collect::<huncho_core::Result<Vec<_>>>()?;
        let budget = options.max_batch_tokens.ok_or_else(|| {
            huncho_core::Error::Request("cross-request benchmark needs a token budget".into())
        })?;
        engine.eval_prepared_batch_with_stats(packets, budget, work)?;
    } else {
        engine.eval_with_stats(&requests[0], options, work)?;
    }
    Ok(())
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn make_request(
    questions: usize,
    long_state: bool,
    workload: Workload,
    case: usize,
    model: &str,
) -> SystemOneRequest {
    let state = if long_state {
        let mut s = format!("Support conversation for ticket {case}:\n");
        for i in 0..40 {
            s.push_str(&format!(
                "customer: message number {i} about the product issue.\n"
            ));
        }
        StateValue::from(s)
    } else {
        StateValue::from(format!("Ticket {case}: the customer asks for a refund."))
    };

    let mut qs = BTreeMap::new();
    for i in 0..questions {
        let q = Question::Choice {
            instructions: Instructions::from(serde_json::Value::String(format!(
                "Which team handles routing question {i}?"
            ))),
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
        let q = match (workload, i % 3) {
            (Workload::Mixed, 1) => Question::Score {
                instructions: Instructions::from(serde_json::Value::String(format!(
                    "Severity for question {i}?"
                ))),
                criteria: vec![
                    serde_json::json!("low"),
                    serde_json::json!("medium"),
                    serde_json::json!("high"),
                ],
            },
            (Workload::Mixed, 2) => Question::Noul {
                instructions: Instructions::from(serde_json::Value::String(format!(
                    "Question {i}: does the customer request a refund?"
                ))),
                criteria: None,
            },
            _ => q,
        };
        qs.insert(format!("q{i:04}"), q);
    }

    SystemOneRequest {
        state,
        model: model.into(),
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
        let r = make_request(5, false, Workload::Choice, 0, "mock");
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
        let long = serde_json::to_string(&make_request(1, true, Workload::Choice, 0, "mock").state)
            .unwrap();
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
            concurrency: 3,
            replicas: 2,
            workload: Workload::Mixed,
            repeat_inputs: false,
            result_cache_bytes: 0,
            json: true,
            reference_readout: false,
            prefix_cache: false,
            persistent_prefix_bytes: 0,
            max_batch_tokens: None,
            max_batch_padding_percent: 0,
            batch_max_requests: None,
            long_state: false,
            mock_model: "mock".into(),
        };
        // Capturing stdout isn't necessary; we just assert it runs cleanly.
        run(args).unwrap();
    }

    #[test]
    fn benchmark_mix_is_valid_distinct_and_uses_loaded_model_name() {
        let request = make_request(5, false, Workload::Mixed, 42, "kev");
        request.validate().unwrap();
        assert_eq!(request.model, "kev");
        let types = request
            .questions
            .values()
            .map(Question::type_name)
            .collect::<Vec<_>>();
        assert_eq!(types, vec!["choice", "score", "noul", "choice", "score"]);
        let other = make_request(5, false, Workload::Mixed, 43, "kev");
        assert_ne!(
            serde_json::to_value(request.state).unwrap(),
            serde_json::to_value(other.state).unwrap()
        );
    }
}

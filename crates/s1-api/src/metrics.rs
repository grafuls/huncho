//! Prometheus metrics (API-04).

use prometheus::{
    Encoder, HistogramVec, IntCounter, IntCounterVec, IntGauge, Opts, Registry, TextEncoder,
};

/// A thin wrapper around the Prometheus registry and the metrics `s1` exposes.
pub struct Metrics {
    registry: Registry,
    /// Request latency histogram (seconds).
    pub request_latency: HistogramVec,
    /// Number of requests currently queued/waiting.
    pub queue_depth: IntGauge,
    /// Tokens prefilled across all requests.
    pub tokens_prefilled: IntCounter,
    /// Number of KV forks performed.
    pub fork_count: IntCounter,
    /// Requests handled, by model.
    pub requests_total: IntCounterVec,
    /// Tokens prefilled, by model.
    pub model_tokens: IntCounterVec,
}

impl Metrics {
    /// Build the metric set.
    pub fn new() -> Metrics {
        let registry = Registry::new();
        let request_latency = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "s1_request_latency_seconds",
                "Request latency in seconds.",
            )
            .buckets(vec![
                0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]),
            &["model"],
        )
        .unwrap();
        let queue_depth = IntGauge::with_opts(Opts::new(
            "s1_queue_depth",
            "Number of requests currently queued or in flight.",
        ))
        .unwrap();
        let tokens_prefilled =
            IntCounter::with_opts(Opts::new("s1_tokens_prefilled", "Tokens prefilled.")).unwrap();
        let fork_count =
            IntCounter::with_opts(Opts::new("s1_fork_count", "Number of KV forks.")).unwrap();
        let requests_total = IntCounterVec::new(
            Opts::new("s1_requests_total", "Total requests by model."),
            &["model"],
        )
        .unwrap();
        let model_tokens = IntCounterVec::new(
            Opts::new("s1_model_tokens", "Tokens prefilled by model."),
            &["model"],
        )
        .unwrap();

        registry
            .register(Box::new(request_latency.clone()))
            .unwrap();
        registry.register(Box::new(queue_depth.clone())).unwrap();
        registry
            .register(Box::new(tokens_prefilled.clone()))
            .unwrap();
        registry.register(Box::new(fork_count.clone())).unwrap();
        registry
            .register(Box::new(requests_total.clone()))
            .unwrap();
        registry.register(Box::new(model_tokens.clone())).unwrap();

        Metrics {
            registry,
            request_latency,
            queue_depth,
            tokens_prefilled,
            fork_count,
            requests_total,
            model_tokens,
        }
    }

    /// Render the metrics as OpenMetrics/Prometheus text.
    pub fn render(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        let _ = encoder.encode(&metric_families, &mut buffer);
        String::from_utf8_lossy(&buffer).to_string()
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics::new()
    }
}

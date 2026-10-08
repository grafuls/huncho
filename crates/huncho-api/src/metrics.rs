//! Prometheus metrics (API-04).

use prometheus::{
    Encoder, HistogramVec, IntCounter, IntCounterVec, IntGauge, Opts, Registry, TextEncoder,
};

/// Cancellation-safe gauge tracking. Moving this into a blocking job keeps the
/// admitted count accurate even when its HTTP caller disconnects.
pub(crate) struct GaugeGuard(IntGauge);

impl GaugeGuard {
    pub(crate) fn new(gauge: &IntGauge) -> Self {
        gauge.inc();
        Self(gauge.clone())
    }
}

impl Drop for GaugeGuard {
    fn drop(&mut self) {
        self.0.dec();
    }
}

/// A thin wrapper around the Prometheus registry and the metrics `huncho` exposes.
pub struct Metrics {
    registry: Registry,
    /// Request latency histogram (seconds).
    pub request_latency: HistogramVec,
    /// Admitted requests currently waiting or executing (legacy metric).
    pub queue_depth: IntGauge,
    /// Admitted requests waiting for their model's execution slot.
    pub requests_waiting: IntGauge,
    pub queue_wait: HistogramVec,
    /// Engine evaluation time, excluding queue wait.
    pub evaluation_latency: HistogramVec,
    pub preparation_wait: HistogramVec,
    pub preparation_latency: HistogramVec,
    pub requests_preparing: IntGauge,
    pub prepared_waiting: IntGauge,
    pub questions_prepared: IntCounter,
    /// Physical token positions submitted across all forward/prefill calls,
    /// including failed evaluations and disconnected HTTP callers.
    pub tokens_prefilled: IntCounter,
    pub prefill_calls: IntCounter,
    pub chunked_prefills: IntCounter,
    /// Number of KV forks performed.
    pub fork_count: IntCounter,
    pub reused_prefix_tokens: IntCounter,
    pub batch_count: IntCounter,
    pub cross_request_batch_count: IntCounter,
    pub persistent_prefix_hits: IntCounter,
    pub result_cache_hits: IntCounter,
    pub prompt_cache_hits: IntCounter,
    pub requests_coalesced: IntCounter,
    pub coalesced_waiting: IntGauge,
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
                "huncho_request_latency_seconds",
                "Request latency in seconds.",
            )
            .buckets(vec![
                0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]),
            &["model"],
        )
        .unwrap();
        let queue_depth = IntGauge::with_opts(Opts::new(
            "huncho_queue_depth",
            "Number of requests currently queued or in flight.",
        ))
        .unwrap();
        let tokens_prefilled = IntCounter::with_opts(Opts::new(
            "huncho_tokens_prefilled",
            "Physical token positions submitted to model inference (including failed attempts).",
        ))
        .unwrap();
        let fork_count =
            IntCounter::with_opts(Opts::new("huncho_fork_count", "Number of KV forks.")).unwrap();
        let prefill_calls = IntCounter::new(
            "huncho_prefill_calls",
            "Native prefix forward attempts, including every chunk.",
        )
        .unwrap();
        let chunked_prefills = IntCounter::new(
            "huncho_chunked_prefills",
            "Prefixes actually submitted across more than one native call.",
        )
        .unwrap();
        let batch_count = IntCounter::new(
            "huncho_batch_count",
            "Native backbone calls containing multiple independent sequences.",
        )
        .unwrap();
        let cross_request_batch_count = IntCounter::new(
            "huncho_cross_request_batch_count",
            "Native backbone batches containing sequences from distinct requests.",
        )
        .unwrap();
        let persistent_prefix_hits = IntCounter::new(
            "huncho_persistent_prefix_hits",
            "Caller-owned native prefix handles cloned from retained immutable snapshots.",
        )
        .unwrap();
        let reused_prefix_tokens = IntCounter::new(
            "huncho_reused_prefix_tokens",
            "Prefix token positions reused across question forks.",
        )
        .unwrap();
        let result_cache_hits = IntCounter::new(
            "huncho_result_cache_hits",
            "Exact successful responses reused without physical inference work.",
        )
        .unwrap();
        let prompt_cache_hits = IntCounter::new(
            "huncho_prompt_cache_hits",
            "Exact prepared prompts reused while model inference still executes.",
        )
        .unwrap();
        let requests_coalesced = IntCounter::new(
            "huncho_requests_coalesced",
            "Admitted callers sharing an identical in-flight evaluation.",
        )
        .unwrap();
        let coalesced_waiting = IntGauge::new(
            "huncho_coalesced_waiting",
            "Admitted callers waiting for a shared in-flight result.",
        )
        .unwrap();
        let requests_total = IntCounterVec::new(
            Opts::new("huncho_requests_total", "Total requests by model."),
            &["model"],
        )
        .unwrap();
        let model_tokens = IntCounterVec::new(
            Opts::new(
                "huncho_model_tokens",
                "Physical token positions submitted by model.",
            ),
            &["model"],
        )
        .unwrap();

        let requests_waiting = IntGauge::new(
            "huncho_requests_waiting",
            "Admitted requests waiting for a model execution slot.",
        )
        .unwrap();
        let queue_wait = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "huncho_queue_wait_seconds",
                "Time waiting for a model execution slot.",
            ),
            &["model"],
        )
        .unwrap();
        let evaluation_latency = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "huncho_evaluation_seconds",
                "Engine evaluation time excluding queue wait.",
            ),
            &["model"],
        )
        .unwrap();
        let preparation_wait = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "huncho_preparation_wait_seconds",
                "Time waiting for bounded prompt preparation capacity.",
            ),
            &["model"],
        )
        .unwrap();
        let preparation_latency = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "huncho_preparation_seconds",
                "CPU prompt preparation time, excluding capacity wait.",
            ),
            &["model"],
        )
        .unwrap();
        let requests_preparing = IntGauge::new(
            "huncho_requests_preparing",
            "Requests actively preparing prompts.",
        )
        .unwrap();
        let prepared_waiting = IntGauge::new(
            "huncho_prepared_waiting",
            "Prepared requests awaiting a model execution slot.",
        )
        .unwrap();
        let questions_prepared = IntCounter::new(
            "huncho_questions_prepared",
            "Questions prepared ahead of model work, including later canceled jobs.",
        )
        .unwrap();

        registry
            .register(Box::new(request_latency.clone()))
            .unwrap();
        registry.register(Box::new(queue_depth.clone())).unwrap();
        registry
            .register(Box::new(requests_waiting.clone()))
            .unwrap();
        registry.register(Box::new(queue_wait.clone())).unwrap();
        registry
            .register(Box::new(evaluation_latency.clone()))
            .unwrap();
        registry
            .register(Box::new(preparation_wait.clone()))
            .unwrap();
        registry
            .register(Box::new(preparation_latency.clone()))
            .unwrap();
        registry
            .register(Box::new(requests_preparing.clone()))
            .unwrap();
        registry
            .register(Box::new(prepared_waiting.clone()))
            .unwrap();
        registry
            .register(Box::new(questions_prepared.clone()))
            .unwrap();
        registry
            .register(Box::new(tokens_prefilled.clone()))
            .unwrap();
        registry.register(Box::new(fork_count.clone())).unwrap();
        registry.register(Box::new(prefill_calls.clone())).unwrap();
        registry
            .register(Box::new(chunked_prefills.clone()))
            .unwrap();
        registry.register(Box::new(batch_count.clone())).unwrap();
        registry
            .register(Box::new(cross_request_batch_count.clone()))
            .unwrap();
        registry
            .register(Box::new(persistent_prefix_hits.clone()))
            .unwrap();
        registry
            .register(Box::new(result_cache_hits.clone()))
            .unwrap();
        registry
            .register(Box::new(prompt_cache_hits.clone()))
            .unwrap();
        registry
            .register(Box::new(requests_coalesced.clone()))
            .unwrap();
        registry
            .register(Box::new(coalesced_waiting.clone()))
            .unwrap();
        registry
            .register(Box::new(reused_prefix_tokens.clone()))
            .unwrap();
        registry.register(Box::new(requests_total.clone())).unwrap();
        registry.register(Box::new(model_tokens.clone())).unwrap();

        Metrics {
            registry,
            request_latency,
            queue_depth,
            requests_waiting,
            queue_wait,
            evaluation_latency,
            preparation_wait,
            preparation_latency,
            requests_preparing,
            prepared_waiting,
            questions_prepared,
            tokens_prefilled,
            prefill_calls,
            chunked_prefills,
            fork_count,
            reused_prefix_tokens,
            batch_count,
            cross_request_batch_count,
            persistent_prefix_hits,
            result_cache_hits,
            prompt_cache_hits,
            requests_coalesced,
            coalesced_waiting,
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

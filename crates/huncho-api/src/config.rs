//! Server configuration.

/// Server configuration for the `huncho` HTTP API.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// The address to bind (host:port).
    pub bind: String,
    /// Optional bearer token required on every request (API-03). When set,
    /// requests must present `Authorization: Bearer <token>`.
    pub auth_token: Option<String>,
    /// Whether to mount the `/metrics` endpoint (API-04).
    pub metrics: bool,
    /// Whether engine extensions are enabled by default. They are surfaced in
    /// the response only when a request specifically asks for them.
    pub default_extensions: bool,
    /// Maximum waiting requests per model, excluding the running evaluation.
    /// Zero allows one running request and rejects all concurrent admissions.
    pub max_queued_per_model: u16,
    /// Maximum requests preparing or holding prepared prompts per model.
    /// Zero disables preprocessing overlap; running work releases its slot.
    pub max_prepared_per_model: u16,
    /// Per-model charged metadata budget for exact in-flight request sharing.
    /// Zero disables sharing; every caller still consumes admission capacity.
    pub coalesce_bytes: usize,
    /// Enable native Kev request-local prefix reuse for qualified models/devices.
    pub prefix_cache: bool,
    /// Opt-in CPU Kev scheduling at prefix-chunk and question boundaries.
    /// Requires one context, prefix reuse and configured native chunking.
    pub cooperative_prefill: bool,
    /// Optional exact native prefix snapshots under a charged-byte budget.
    /// Zero disables retention; requires prefix_cache and separate qualification.
    pub persistent_prefix_bytes: usize,
    /// Opt-in native equal-length question batching token budget.
    pub max_batch_tokens: Option<usize>,
    /// Zero keeps exact lengths; otherwise supported CPU padding percent (1..100).
    pub max_batch_padding_percent: usize,
    /// Optional cross-request collation (2–64 requests). Requires native batch
    /// support and max_batch_tokens; F5 remains whole-request inference.
    pub batch_max_requests: Option<u16>,
    /// Maximum collation wait after the first prepared request, in milliseconds.
    pub batch_wait_ms: u16,
    /// Enable qualified F3 candidate-only projection. Otherwise serving uses
    /// the legacy duplicated-position/full-vocabulary reference readout.
    pub candidate_readout: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            bind: "127.0.0.1:8080".into(),
            auth_token: None,
            metrics: true,
            default_extensions: false,
            max_queued_per_model: 32,
            max_prepared_per_model: 0,
            coalesce_bytes: 0,
            prefix_cache: false,
            cooperative_prefill: false,
            persistent_prefix_bytes: 0,
            max_batch_tokens: None,
            max_batch_padding_percent: 0,
            batch_max_requests: None,
            batch_wait_ms: 2,
            candidate_readout: false,
        }
    }
}

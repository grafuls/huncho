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
    /// Opt-in native per-request question batching. No cross-request batching yet.
    pub max_batch_tokens: Option<usize>,
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
            max_batch_tokens: None,
            candidate_readout: false,
        }
    }
}

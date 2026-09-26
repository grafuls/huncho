//! Server configuration.

/// Server configuration for the `s1` HTTP API.
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
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            bind: "127.0.0.1:8080".into(),
            auth_token: None,
            metrics: true,
            default_extensions: false,
        }
    }
}

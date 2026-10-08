//! `huncho-api` — the HTTP/gRPC-facing API layer for the `huncho` engine.
//!
//! Implements the Jev `/v1/systemone` contract, `/health`, `/v1/models`,
//! optional bearer auth, Prometheus `/metrics`, and engine extensions.

pub mod auth;
mod coalesce;
pub mod config;
pub mod metrics;
pub mod routes;
pub mod server;
pub mod state;

pub use config::ServerConfig;
pub use metrics::Metrics;
pub use routes::router;
pub use server::serve;
pub use state::{AppState, ModelRegistry};

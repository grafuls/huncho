//! Shared application state for the HTTP server.

use std::collections::BTreeMap;
use std::sync::Arc;

use huncho_core::engine::Engine;
use tokio::sync::RwLock;

use crate::config::ServerConfig;
use crate::metrics::Metrics;

/// A registry of loaded models keyed by the `model` string clients send.
pub struct ModelRegistry {
    models: BTreeMap<String, Engine>,
}

impl ModelRegistry {
    pub fn new() -> ModelRegistry {
        ModelRegistry {
            models: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, engine: Engine) {
        self.models.insert(name.into(), engine);
    }

    pub fn get(&self, name: &str) -> Option<&Engine> {
        self.models.get(name)
    }

    pub fn names(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    pub fn models(&self) -> &BTreeMap<String, Engine> {
        &self.models
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        ModelRegistry::new()
    }
}

/// The state shared across request handlers.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub registry: Arc<RwLock<ModelRegistry>>,
    pub metrics: Arc<Metrics>,
    /// Time the server started.
    pub started: std::time::Instant,
}

impl AppState {
    pub fn new(config: ServerConfig, registry: ModelRegistry, metrics: Metrics) -> AppState {
        AppState {
            config: Arc::new(config),
            registry: Arc::new(RwLock::new(registry)),
            metrics: Arc::new(metrics),
            started: std::time::Instant::now(),
        }
    }
}

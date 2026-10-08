//! Shared application state for the HTTP server.

use std::collections::BTreeMap;
use std::sync::Arc;

use huncho_core::engine::Engine;
use tokio::sync::{RwLock, Semaphore};

use crate::coalesce::RequestFlights;
use crate::config::ServerConfig;
use crate::metrics::Metrics;

/// A registry of loaded models keyed by the `model` string clients send.
pub struct ModelRegistry {
    models: BTreeMap<String, ModelHandle>,
    max_queued: u16,
    coalesce_bytes: usize,
    max_prepared: u16,
}

/// Shareable model and bounded serving admission. Immutable model metadata can
/// be read after releasing the registry lock; blocking evaluation owns permits.
#[derive(Clone)]
pub struct ModelHandle {
    pub(crate) engine: Arc<Engine>,
    pub(crate) admission: Arc<Semaphore>,
    pub(crate) execution: Arc<Semaphore>,
    pub(crate) flights: Arc<RequestFlights>,
    pub(crate) preparation: Option<Arc<Semaphore>>,
}

impl ModelHandle {
    fn new(engine: Arc<Engine>, max_queued: u16, coalesce_bytes: usize, max_prepared: u16) -> Self {
        let preparation = (max_prepared > 0 && engine.family() != huncho_core::Family::F5)
            .then(|| Arc::new(Semaphore::new(usize::from(max_prepared))));
        Self {
            engine,
            admission: Arc::new(Semaphore::new(usize::from(max_queued) + 1)),
            execution: Arc::new(Semaphore::new(1)),
            flights: RequestFlights::new(coalesce_bytes),
            preparation,
        }
    }
}

impl std::ops::Deref for ModelHandle {
    type Target = Engine;
    fn deref(&self) -> &Engine {
        &self.engine
    }
}

impl ModelRegistry {
    pub fn new() -> ModelRegistry {
        ModelRegistry {
            models: BTreeMap::new(),
            max_queued: 32,
            coalesce_bytes: 0,
            max_prepared: 0,
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, engine: Engine) {
        self.models.insert(
            name.into(),
            ModelHandle::new(
                Arc::new(engine),
                self.max_queued,
                self.coalesce_bytes,
                self.max_prepared,
            ),
        );
    }

    pub fn get(&self, name: &str) -> Option<ModelHandle> {
        self.models.get(name).cloned()
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

    pub fn models(&self) -> &BTreeMap<String, ModelHandle> {
        &self.models
    }

    /// Called before serving starts. New registrations inherit the same limit.
    fn configure_queue(&mut self, max_queued: u16, coalesce_bytes: usize, max_prepared: u16) {
        self.max_queued = max_queued;
        self.coalesce_bytes = coalesce_bytes;
        self.max_prepared = max_prepared;
        for model in self.models.values_mut() {
            *model = ModelHandle::new(
                model.engine.clone(),
                max_queued,
                coalesce_bytes,
                max_prepared,
            );
        }
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
    pub fn new(config: ServerConfig, mut registry: ModelRegistry, metrics: Metrics) -> AppState {
        registry.configure_queue(
            config.max_queued_per_model,
            config.coalesce_bytes,
            config.max_prepared_per_model,
        );
        AppState {
            config: Arc::new(config),
            registry: Arc::new(RwLock::new(registry)),
            metrics: Arc::new(metrics),
            started: std::time::Instant::now(),
        }
    }
}

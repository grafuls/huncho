//! Shared application state for the HTTP server.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, Weak};

use huncho_core::engine::Engine;
use tokio::sync::{RwLock, Semaphore};

use crate::batch::BatchQueue;
use crate::coalesce::RequestFlights;
use crate::config::ServerConfig;
use crate::metrics::Metrics;
use crate::replicas::ReplicaPool;
use crate::residency::{ModelDescription, Residency, ServingPolicy};

/// A registry of loaded models keyed by the `model` string clients send.
pub struct ModelRegistry {
    models: BTreeMap<String, ModelHandle>,
    max_queued: u16,
    coalesce_bytes: usize,
    max_prepared: u16,
    batch: Option<(u16, u16, usize)>,
    pub(crate) residency: Option<Arc<Residency>>,
}

/// Shareable model and bounded serving admission. Immutable model metadata can
/// be read after releasing the registry lock; blocking evaluation owns permits.
#[derive(Clone)]
pub struct ModelHandle {
    capacity: usize,
    pub(crate) engine: Arc<Engine>,
    pub(crate) admission: Arc<Semaphore>,
    pub(crate) pool: Arc<ReplicaPool>,
    pub(crate) flights: Arc<RequestFlights>,
    pub(crate) preparation: Option<Arc<Semaphore>>,
    pub(crate) batch: Option<Arc<BatchQueue>>,
    pub(crate) last_prefill: Arc<Mutex<Option<(Weak<()>, bool)>>>,
}

impl ModelHandle {
    pub(crate) fn new(
        engines: Vec<Arc<Engine>>,
        max_queued: u16,
        coalesce_bytes: usize,
        max_prepared: u16,
        batch: Option<(u16, u16, usize)>,
    ) -> Self {
        let engine = engines[0].clone();
        let capacity = usize::from(max_queued) + engines.len();
        let batch = batch.filter(|(rows, _, tokens)| {
            (2..=64).contains(rows) && *tokens > 0 && engine.supports_batch()
        });
        let max_prepared = batch.map_or(max_prepared, |(rows, _, _)| max_prepared.max(rows));
        let preparation = (max_prepared > 0 && engine.family() != huncho_core::Family::F5)
            .then(|| Arc::new(Semaphore::new(usize::from(max_prepared))));
        Self {
            capacity,
            engine,
            admission: Arc::new(Semaphore::new(capacity)),
            pool: ReplicaPool::new(engines),
            flights: RequestFlights::new(coalesce_bytes),
            preparation,
            batch: batch.map(|(rows, wait, tokens)| BatchQueue::new(capacity, rows, wait, tokens)),
            last_prefill: Arc::new(Mutex::new(None)),
        }
    }

    /// Independent execution contexts for startup qualification. Their model
    /// identity and prepared-packet ownership belong to one immutable group.
    pub fn replica_engines(&self) -> &[Arc<Engine>] {
        &self.pool.engines
    }

    /// Registry ownership alone is insufficient: jobs can own permits or
    /// cloned engine Arcs after their HTTP future/model handle disappears.
    pub(crate) fn can_unload(&self) -> bool {
        Arc::strong_count(&self.pool) == 1
            && self.admission.available_permits() == self.capacity
            && self
                .pool
                .engines
                .iter()
                .enumerate()
                .all(|(index, engine)| Arc::strong_count(engine) == if index == 0 { 2 } else { 1 })
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
            batch: None,
            residency: None,
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, engine: Engine) {
        let name = name.into();
        assert!(
            !self
                .lazy_descriptions()
                .iter()
                .any(|entry| entry.name == name),
            "duplicate lazy/eager model name"
        );
        self.models.insert(
            name,
            ModelHandle::new(
                vec![Arc::new(engine)],
                self.max_queued,
                self.coalesce_bytes,
                self.max_prepared,
                self.batch,
            ),
        );
    }

    /// Configure a bounded CPU pool before serving. Construction is atomic:
    /// unsupported backends leave all existing model handles unchanged.
    pub fn set_replicas(&mut self, count: usize) -> huncho_core::Result<()> {
        use huncho_core::error::Error;
        if !(1..=8).contains(&count) {
            return Err(Error::Request("replicas must be between 1 and 8".into()));
        }
        if !self.lazy_descriptions().is_empty() {
            return Err(Error::Unsupported(
                "lazy replica counts are fixed by their registration and factory".into(),
            ));
        }
        if count > 1 && self.batch.is_some() {
            return Err(Error::Unsupported(
                "replicas and cross-request collation cannot be combined yet".into(),
            ));
        }
        let mut replacement = BTreeMap::new();
        for (name, model) in &self.models {
            if count > 1 && model.device() != "CPU" {
                return Err(Error::Unsupported(
                    "replica serving currently supports CPU only".into(),
                ));
            }
            let mut engines = vec![model.engine.clone()];
            for _ in 1..count {
                engines.push(Arc::new(model.engine.replica()?));
            }
            replacement.insert(
                name.clone(),
                ModelHandle::new(
                    engines,
                    self.max_queued,
                    self.coalesce_bytes,
                    self.max_prepared,
                    self.batch,
                ),
            );
        }
        self.models = replacement;
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<ModelHandle> {
        self.models
            .get(name)
            .cloned()
            .or_else(|| self.residency.as_ref()?.resident(name))
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.models.keys().cloned().collect();
        names.extend(self.lazy_descriptions().into_iter().map(|entry| entry.name));
        names.sort();
        names
    }

    pub fn len(&self) -> usize {
        self.models.len() + self.lazy_descriptions().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn models(&self) -> &BTreeMap<String, ModelHandle> {
        &self.models
    }

    /// Optional CPU model slots, separate from eagerly registered models.
    /// This is a count bound, not an estimated resident-byte budget.
    pub fn enable_lazy(
        &mut self,
        max_models: usize,
        idle: std::time::Duration,
    ) -> huncho_core::Result<()> {
        if self.residency.is_some() {
            return Err(huncho_core::error::Error::Request(
                "lazy residency already configured".into(),
            ));
        }
        self.residency = Some(Residency::new(max_models, idle, self.policy())?);
        Ok(())
    }

    /// The factory owns artifact pinning and fresh complete labeled gates.
    /// It is run on a blocking worker once per actual cold load, never at
    /// registration. Returned contexts must match the immutable description.
    pub fn register_lazy(
        &mut self,
        description: ModelDescription,
        factory: impl Fn() -> huncho_core::Result<Vec<Arc<Engine>>> + Send + Sync + 'static,
    ) -> huncho_core::Result<()> {
        if self.models.contains_key(&description.name) {
            return Err(huncho_core::error::Error::Request(
                "duplicate eager/lazy model name".into(),
            ));
        }
        self.residency
            .as_ref()
            .ok_or_else(|| {
                huncho_core::error::Error::Request(
                    "enable lazy residency before registration".into(),
                )
            })?
            .register(description, Arc::new(factory))
    }

    pub fn lazy_descriptions(&self) -> Vec<ModelDescription> {
        self.residency
            .as_ref()
            .map_or_else(Vec::new, |residency| residency.descriptions())
    }

    fn policy(&self) -> ServingPolicy {
        ServingPolicy {
            max_queued: self.max_queued,
            coalesce_bytes: self.coalesce_bytes,
            max_prepared: self.max_prepared,
            batch: self.batch,
        }
    }

    /// Called before serving starts. New registrations inherit the same limit.
    fn configure_queue(
        &mut self,
        max_queued: u16,
        coalesce_bytes: usize,
        max_prepared: u16,
        batch: Option<(u16, u16, usize)>,
    ) {
        self.max_queued = max_queued;
        self.coalesce_bytes = coalesce_bytes;
        self.max_prepared = max_prepared;
        self.batch = batch;
        for model in self.models.values_mut() {
            *model = ModelHandle::new(
                model.pool.engines.clone(),
                max_queued,
                coalesce_bytes,
                max_prepared,
                batch,
            );
        }
        if let Some(residency) = &self.residency {
            residency.configure(self.policy());
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
    /// Release the registry lock before any load, qualification or waiting.
    pub async fn resolve_model(&self, name: &str) -> huncho_core::Result<Option<ModelHandle>> {
        let residency = {
            let registry = self.registry.read().await;
            if let Some(model) = registry.get(name) {
                return Ok(Some(model));
            }
            registry.residency.clone()
        };
        match residency {
            Some(residency) => residency.resolve(name).await,
            None => Ok(None),
        }
    }

    pub async fn evict_idle_models(&self) -> usize {
        let residency = self.registry.read().await.residency.clone();
        match residency {
            Some(residency) => tokio::task::spawn_blocking(move || residency.evict_idle())
                .await
                .unwrap_or(0),
            None => 0,
        }
    }

    pub fn new(config: ServerConfig, mut registry: ModelRegistry, metrics: Metrics) -> AppState {
        registry.configure_queue(
            config.max_queued_per_model,
            config.coalesce_bytes,
            if config.cooperative_prefill {
                config.max_prepared_per_model.max(1)
            } else {
                config.max_prepared_per_model
            },
            config
                .batch_max_requests
                .zip(config.max_batch_tokens)
                .filter(|_| !config.prefix_cache)
                .map(|(rows, tokens)| (rows, config.batch_wait_ms, tokens)),
        );
        AppState {
            config: Arc::new(config),
            registry: Arc::new(RwLock::new(registry)),
            metrics: Arc::new(metrics),
            started: std::time::Instant::now(),
        }
    }
}

//! Bounded CPU residency. Factories must qualify the exact engines they return.
//! A canceled cold request does not abandon its load or release its reservation.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use huncho_core::engine::Engine;
use huncho_core::error::{Error, Result};
use huncho_core::manifest::{BackendId, Family};
use serde::Serialize;
use tokio::sync::{watch, Semaphore};

use crate::state::ModelHandle;

#[derive(Clone, Debug, Serialize)]
pub struct ModelDescription {
    pub name: String,
    pub family: Family,
    pub backend: BackendId,
    pub dtype: String,
    pub max_context: usize,
    pub replicas: usize,
    pub residency: String,
}

type Factory = Arc<dyn Fn() -> Result<Vec<Arc<Engine>>> + Send + Sync>;
type LoadResult = std::result::Result<ModelHandle, Arc<str>>;

#[derive(Clone, Copy)]
pub(crate) struct ServingPolicy {
    pub max_queued: u16,
    pub coalesce_bytes: usize,
    pub max_prepared: u16,
    pub batch: Option<(u16, u16, usize)>,
}

struct Entry {
    description: ModelDescription,
    factory: Factory,
    resident: Option<ModelHandle>,
    loading: Option<watch::Receiver<Option<LoadResult>>>,
    failure: Option<Arc<str>>,
    last_use: Instant,
    waiting: Arc<Semaphore>,
}

struct State {
    entries: BTreeMap<String, Entry>,
    policy: ServingPolicy,
}

pub(crate) struct Residency {
    state: Mutex<State>,
    max_models: usize,
    pub(crate) idle: Duration,
}

impl Residency {
    pub(crate) fn new(
        max_models: usize,
        idle: Duration,
        policy: ServingPolicy,
    ) -> Result<Arc<Self>> {
        if !(1..=64).contains(&max_models) || idle.is_zero() {
            return Err(Error::Request(
                "lazy residency requires 1..64 slots and a positive idle duration".into(),
            ));
        }
        Ok(Arc::new(Self {
            state: Mutex::new(State {
                entries: BTreeMap::new(),
                policy,
            }),
            max_models,
            idle,
        }))
    }

    pub(crate) fn register(&self, description: ModelDescription, factory: Factory) -> Result<()> {
        if description.name.is_empty()
            || !(1..=8).contains(&description.replicas)
            || description.max_context == 0
        {
            return Err(Error::Request("invalid lazy model description".into()));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::Backend("resident registry poisoned".into()))?;
        if state.entries.contains_key(&description.name) {
            return Err(Error::Request("duplicate lazy model name".into()));
        }
        let capacity = usize::from(state.policy.max_queued) + description.replicas;
        state.entries.insert(
            description.name.clone(),
            Entry {
                description,
                factory,
                resident: None,
                loading: None,
                failure: None,
                last_use: Instant::now(),
                waiting: Arc::new(Semaphore::new(capacity)),
            },
        );
        Ok(())
    }

    // Startup only: AppState applies one immutable admission/preparation policy
    // before any cold loads can begin.
    pub(crate) fn configure(&self, policy: ServingPolicy) {
        let mut state = self
            .state
            .lock()
            .expect("resident registry poisoned at startup");
        assert!(state
            .entries
            .values()
            .all(|e| e.resident.is_none() && e.loading.is_none()));
        state.policy = policy;
        for entry in state.entries.values_mut() {
            entry.waiting = Arc::new(Semaphore::new(
                usize::from(policy.max_queued) + entry.description.replicas,
            ));
        }
    }

    pub(crate) fn descriptions(&self) -> Vec<ModelDescription> {
        let Ok(state) = self.state.lock() else {
            return Vec::new();
        };
        state
            .entries
            .values()
            .map(|entry| {
                let mut description = entry.description.clone();
                description.residency = if entry.resident.is_some() {
                    "resident"
                } else if entry.loading.is_some() {
                    "loading"
                } else if entry.failure.is_some() {
                    "failed"
                } else {
                    "cold"
                }
                .into();
                description
            })
            .collect()
    }

    pub(crate) fn resident(&self, name: &str) -> Option<ModelHandle> {
        let mut state = self.state.lock().ok()?;
        let entry = state.entries.get_mut(name)?;
        let model = entry.resident.clone()?;
        entry.last_use = Instant::now();
        Some(model)
    }

    pub(crate) fn evict_idle(&self) -> usize {
        let removed = {
            let Ok(mut state) = self.state.lock() else {
                return 0;
            };
            state
                .entries
                .values_mut()
                .filter_map(|entry| {
                    if entry.last_use.elapsed() >= self.idle
                        && entry.resident.as_ref().is_some_and(ModelHandle::can_unload)
                    {
                        entry.resident.take()
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };
        let count = removed.len();
        // Native destructors do not run with the registry mutex held.
        drop(removed);
        count
    }

    pub(crate) async fn resolve(self: &Arc<Self>, name: &str) -> Result<Option<ModelHandle>> {
        let (mut receiver, waiting, evicted, start) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| Error::Backend("resident registry poisoned".into()))?;
            let Some(entry) = state.entries.get_mut(name) else {
                return Ok(None);
            };
            if let Some(model) = &entry.resident {
                entry.last_use = Instant::now();
                return Ok(Some(model.clone()));
            }
            if let Some(failure) = &entry.failure {
                return Err(Error::Backend(format!("cold model qualification/load failed: {failure}; restart after correcting the package")));
            }
            let waiting = entry
                .waiting
                .clone()
                .try_acquire_owned()
                .map_err(|_| Error::Backend("cold model waiting queue is full".into()))?;
            if let Some(receiver) = &entry.loading {
                (receiver.clone(), waiting, None, None)
            } else {
                let used = state
                    .entries
                    .values()
                    .filter(|e| e.resident.is_some() || e.loading.is_some())
                    .count();
                let evicted = if used >= self.max_models {
                    let victim = state
                        .entries
                        .iter()
                        .filter(|(_, e)| e.resident.as_ref().is_some_and(ModelHandle::can_unload))
                        .min_by_key(|(_, e)| e.last_use)
                        .map(|(name, _)| name.clone())
                        .ok_or_else(|| {
                            Error::Backend("all resident model slots are busy or loading".into())
                        })?;
                    state.entries.get_mut(&victim).unwrap().resident.take()
                } else {
                    None
                };
                let policy = state.policy;
                let entry = state.entries.get_mut(name).unwrap();
                let (sender, receiver) = watch::channel(None);
                entry.loading = Some(receiver.clone());
                (
                    receiver,
                    waiting,
                    evicted,
                    Some((
                        sender,
                        entry.factory.clone(),
                        entry.description.clone(),
                        policy,
                    )),
                )
            }
        };
        drop(evicted);
        if let Some((sender, factory, description, policy)) = start {
            let owner = self.clone();
            // This task owns the reservation until load/qualification finishes,
            // even if every HTTP waiter disappears or its factory panics.
            tokio::spawn(async move {
                let name = description.name.clone();
                tracing::info!(model = %name, "loading CPU model for fresh qualification");
                let result = tokio::task::spawn_blocking(move || {
                    let engines = factory()?;
                    if engines.len() != description.replicas || engines.iter().any(|engine| {
                        engine.manifest().name != description.name || engine.family() != description.family
                            || engine.backend_id() != description.backend || engine.dtype() != description.dtype
                            || engine.manifest().backbone.max_context != description.max_context || engine.device() != "CPU"
                    }) {
                        return Err(Error::Conformance("lazy factory returned a different model, runtime, replica count or device".into()));
                    }
                    Ok(ModelHandle::new(engines, policy.max_queued, policy.coalesce_bytes, policy.max_prepared, policy.batch))
                }).await;
                let result: LoadResult = match result {
                    Ok(Ok(model)) => Ok(model),
                    Ok(Err(error)) => Err(error.to_string().into()),
                    Err(_) => Err("cold model worker panicked".into()),
                };
                match &result {
                    Ok(_) => tracing::info!(model = %name, "qualified CPU model is resident"),
                    Err(error) => {
                        tracing::warn!(model = %name, error = %error, "cold model failed; serving remains disabled")
                    }
                }
                if let Ok(mut state) = owner.state.lock() {
                    let entry = state.entries.get_mut(&name).unwrap();
                    match &result {
                        Ok(model) => {
                            entry.resident = Some(model.clone());
                            entry.last_use = Instant::now();
                        }
                        Err(error) => entry.failure = Some(error.clone()),
                    }
                    entry.loading = None;
                }
                let _ = sender.send(Some(result));
            });
        }
        let _waiting = waiting;
        loop {
            if let Some(result) = receiver.borrow().clone() {
                return result
                    .map(Some)
                    .map_err(|error| Error::Backend(error.to_string()));
            }
            receiver
                .changed()
                .await
                .map_err(|_| Error::Backend("cold model worker unavailable".into()))?;
        }
    }
}

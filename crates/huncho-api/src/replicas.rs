//! Idle execution contexts leased through the entire blocking model job.

use huncho_core::engine::Engine;
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub(crate) struct ReplicaPool {
    pub(crate) engines: Vec<Arc<Engine>>,
    idle: Mutex<Vec<usize>>,
    available: Arc<Semaphore>,
}

impl ReplicaPool {
    pub(crate) fn new(engines: Vec<Arc<Engine>>) -> Arc<Self> {
        Arc::new(Self {
            idle: Mutex::new((0..engines.len()).rev().collect()),
            available: Arc::new(Semaphore::new(engines.len())),
            engines,
        })
    }

    pub(crate) async fn acquire(self: &Arc<Self>) -> Result<ReplicaLease, ()> {
        let permit = self
            .available
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ())?;
        let index = self.idle.lock().map_err(|_| ())?.pop().ok_or(())?;
        Ok(ReplicaLease {
            pool: self.clone(),
            index,
            _permit: permit,
        })
    }
}

pub(crate) struct ReplicaLease {
    pool: Arc<ReplicaPool>,
    index: usize,
    _permit: OwnedSemaphorePermit,
}

impl std::ops::Deref for ReplicaLease {
    type Target = Engine;
    fn deref(&self) -> &Engine {
        &self.pool.engines[self.index]
    }
}

impl Drop for ReplicaLease {
    fn drop(&mut self) {
        // Publish the idle context before the owned permit becomes available.
        // Unwind/canceled HTTP futures cannot release a running blocking job.
        if let Ok(mut idle) = self.pool.idle.lock() {
            idle.push(self.index);
        } else {
            self.pool.available.close();
        }
    }
}

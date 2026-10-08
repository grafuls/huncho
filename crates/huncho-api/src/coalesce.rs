//! Bounded, exact in-flight request sharing. No completed-response retention.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use huncho_core::contract::SystemOneResponse;
use huncho_core::error::Result;
use tokio::sync::watch;

// Results are already shared through Arc; boxing the response would add an
// allocation/indirection on every inference without reducing retained payload.
#[allow(clippy::large_enum_variant)]
pub(crate) enum JobResult {
    Finished(Result<SystemOneResponse>),
    Unavailable,
    WorkerFailed,
}

pub(crate) type SharedResult = Option<Arc<JobResult>>;
pub(crate) type Receiver = watch::Receiver<SharedResult>;

struct Table {
    entries: HashMap<Arc<[u8]>, watch::Sender<SharedResult>>,
    charged_bytes: usize,
}

pub(crate) struct RequestFlights {
    max_bytes: usize,
    table: Mutex<Table>,
}

pub(crate) enum Join {
    Leader(FlightOwner, Receiver),
    Follower(Receiver),
}

/// Drop removes the key and closes unfinished receivers, including unwinds.
pub(crate) struct FlightOwner {
    flights: Arc<RequestFlights>,
    key: Arc<[u8]>,
    pub(crate) sender: watch::Sender<SharedResult>,
}

impl RequestFlights {
    pub(crate) fn new(max_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            max_bytes,
            table: Mutex::new(Table {
                entries: HashMap::new(),
                charged_bytes: 0,
            }),
        })
    }

    pub(crate) fn enabled(&self) -> bool {
        self.max_bytes > 0
    }

    pub(crate) fn join(self: &Arc<Self>, key: Vec<u8>) -> Option<Join> {
        // Charge retained key bytes and conservative channel/hash-node overhead.
        // This bounds retained metadata, not allocator RSS or response bodies.
        let cost = key.len().checked_add(512)?;
        let mut table = self.table.lock().ok()?;
        if let Some(sender) = table.entries.get(key.as_slice()) {
            return Some(Join::Follower(sender.subscribe()));
        }
        if cost > self.max_bytes
            || table.charged_bytes > self.max_bytes - cost
            || table.entries.len() >= 1024
        {
            return None;
        }
        let key: Arc<[u8]> = key.into();
        let (sender, receiver) = watch::channel(None);
        table.entries.insert(key.clone(), sender.clone());
        table.charged_bytes += cost;
        Some(Join::Leader(
            FlightOwner {
                flights: self.clone(),
                key,
                sender,
            },
            receiver,
        ))
    }
}

impl FlightOwner {
    pub(crate) fn complete(self, result: Arc<JobResult>) {
        self.sender.send_replace(Some(result));
        // Drop removes this registration. Existing callers own their result;
        // later requests must evaluate again unless result caching is enabled.
    }
}

impl Drop for FlightOwner {
    fn drop(&mut self) {
        if let Ok(mut table) = self.flights.table.lock() {
            if table.entries.remove(self.key.as_ref()).is_some() {
                table.charged_bytes -= self.key.len() + 512;
            }
        }
    }
}

pub(crate) async fn wait(mut receiver: Receiver) -> Arc<JobResult> {
    loop {
        let result = receiver.borrow_and_update().clone();
        if let Some(result) = result {
            return result;
        }
        if receiver.changed().await.is_err() {
            return Arc::new(JobResult::WorkerFailed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bounds_cleanup_and_unfinished_owner_release() {
        let flights = RequestFlights::new(516);
        let Some(Join::Leader(owner, receiver)) = flights.join(b"same".to_vec()) else {
            panic!()
        };
        let Some(Join::Follower(follower)) = flights.join(b"same".to_vec()) else {
            panic!()
        };
        assert!(flights.join(b"other".to_vec()).is_none());
        drop(receiver);
        assert_eq!(owner.sender.receiver_count(), 1);
        drop(owner);
        assert!(matches!(&*wait(follower).await, JobResult::WorkerFailed));
        assert_eq!(flights.table.lock().unwrap().charged_bytes, 0);
        assert!(matches!(
            flights.join(b"next".to_vec()),
            Some(Join::Leader(_, _))
        ));
        assert!(RequestFlights::new(0).join(Vec::new()).is_none());
    }
}

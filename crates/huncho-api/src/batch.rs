//! Bounded cross-request collation. The worker owns physical-work permits;
//! queued cancellation never abandons another caller's batch.

use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use huncho_core::engine::{EvalStats, PreparedEvaluation};
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit};

use crate::coalesce::JobResult;
use crate::metrics::{GaugeGuard, Metrics};
use crate::replicas::ReplicaPool;

pub(crate) struct BatchJob {
    pub prepared: PreparedEvaluation,
    pub model: String,
    pub admission: OwnedSemaphorePermit,
    pub admitted: GaugeGuard,
    pub waiting: GaugeGuard,
    pub prepared_waiting: Option<GaugeGuard>,
    pub preparation_slot: Option<OwnedSemaphorePermit>,
    pub queued: Instant,
}

struct Pending {
    job: BatchJob,
    reply: oneshot::Sender<Arc<JobResult>>,
}

pub(crate) struct BatchQueue {
    sender: mpsc::Sender<Pending>,
    receiver: Mutex<Option<mpsc::Receiver<Pending>>>,
    max_requests: usize,
    wait: Duration,
    tokens: usize,
}

impl BatchQueue {
    pub(crate) fn new(capacity: usize, rows: u16, wait_ms: u16, tokens: usize) -> Arc<Self> {
        let (sender, receiver) = mpsc::channel(capacity);
        Arc::new(Self {
            sender,
            receiver: Mutex::new(Some(receiver)),
            max_requests: usize::from(rows),
            wait: Duration::from_millis(u64::from(wait_ms)),
            tokens,
        })
    }

    pub(crate) async fn submit(
        &self,
        job: BatchJob,
        pool: Arc<ReplicaPool>,
        metrics: Arc<Metrics>,
    ) -> Arc<JobResult> {
        let receiver = match self.receiver.lock() {
            Ok(mut receiver) => receiver.take(),
            Err(_) => return Arc::new(JobResult::Unavailable),
        };
        if let Some(receiver) = receiver {
            // The worker does not retain the queue's sender. Removing the last
            // model handle closes the queue and releases the engine after drain.
            tokio::spawn(run(
                receiver,
                Arc::downgrade(&pool),
                metrics,
                self.max_requests,
                self.wait,
                self.tokens,
            ));
        }
        let (reply, receive) = oneshot::channel();
        if self.sender.try_send(Pending { job, reply }).is_err() {
            return Arc::new(JobResult::Unavailable);
        }
        receive
            .await
            .unwrap_or_else(|_| Arc::new(JobResult::WorkerFailed))
    }
}

async fn run(
    mut receiver: mpsc::Receiver<Pending>,
    pool: Weak<ReplicaPool>,
    metrics: Arc<Metrics>,
    max_requests: usize,
    wait: Duration,
    tokens: usize,
) {
    while let Some(first) = receiver.recv().await {
        if first.reply.is_closed() {
            continue;
        }
        // Use the first job's enqueue time: delayed workers never add another
        // full waiting window after a previous batch finishes.
        let deadline = tokio::time::Instant::from_std(first.job.queued + wait);
        let mut group = vec![first];
        while group.len() < max_requests {
            let next = match receiver.try_recv() {
                Ok(next) => Some(next),
                Err(mpsc::error::TryRecvError::Disconnected) => None,
                Err(mpsc::error::TryRecvError::Empty) => {
                    tokio::time::timeout_at(deadline, receiver.recv())
                        .await
                        .unwrap_or_default()
                }
            };
            let Some(next) = next else {
                break;
            };
            if !next.reply.is_closed() {
                group.push(next);
            }
        }
        group.retain(|pending| !pending.reply.is_closed());
        if group.is_empty() {
            continue;
        }
        let Some(pool) = pool.upgrade() else {
            break;
        };
        let Ok(backend) = pool.acquire().await else {
            break;
        };
        group.retain(|pending| !pending.reply.is_closed());
        if group.is_empty() {
            continue;
        }
        let job_metrics = metrics.clone();
        // Replies are held outside the blocking task so a panic closes neither
        // the worker nor future admission; every affected caller receives 500.
        let (jobs, replies): (Vec<_>, Vec<_>) = group
            .into_iter()
            .map(|pending| (pending.job, pending.reply))
            .unzip();
        let count = replies.len();
        let results = tokio::task::spawn_blocking(move || {
            let mut packets = Vec::with_capacity(jobs.len());
            let mut owners = Vec::with_capacity(jobs.len());
            for job in jobs {
                job_metrics
                    .queue_wait
                    .with_label_values(&[&job.model])
                    .observe(job.queued.elapsed().as_secs_f64());
                drop(job.waiting);
                drop(job.prepared_waiting);
                drop(job.preparation_slot);
                packets.push(job.prepared);
                owners.push((job.model, job.admission, job.admitted));
            }
            let start = Instant::now();
            let mut stats = EvalStats::default();
            let result = backend.eval_prepared_batch_with_stats(packets, tokens, &mut stats);
            // All names resolve to this same immutable engine. Physical work is
            // counted once, under its registry name; per-caller usage is logical.
            let name = backend.manifest().name.as_str();
            crate::routes::record_execution(&job_metrics, name, &stats);
            job_metrics
                .evaluation_latency
                .with_label_values(&[name])
                .observe(start.elapsed().as_secs_f64());
            drop(owners);
            result
        })
        .await;
        let results: Vec<Arc<JobResult>> = match results {
            Ok(Ok(responses)) if responses.len() == count => responses
                .into_iter()
                .map(|response| Arc::new(JobResult::Finished(Ok(response))))
                .collect(),
            Ok(Err(error)) => {
                let shared = Arc::new(JobResult::Finished(Err(error)));
                vec![shared; count]
            }
            _ => {
                let shared = Arc::new(JobResult::WorkerFailed);
                vec![shared; count]
            }
        };
        for (reply, result) in replies.into_iter().zip(results) {
            let _ = reply.send(result);
        }
    }
}

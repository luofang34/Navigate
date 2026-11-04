//! Bounded single-slot supersede and bounded FIFO background workers with watchdog counters.

use crate::error::FlightVisualError;
use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

struct SupersedeSlot<J> {
    job: Option<J>,
    closed: bool,
}

struct SharedState<J> {
    slot: Mutex<SupersedeSlot<J>>,
    cv: Condvar,
    alive: AtomicBool,
    faulted: AtomicBool,
    busy: AtomicBool,
    superseded: AtomicU32,
    dropped: AtomicU32,
}

/// Single-slot supersede background worker (`capacity = 1` pending job, bounded result channel).
///
/// Submitting a new job while an earlier job is still queued in the slot overwrites the stale job
/// and increments [`Self::superseded_count`].
pub struct SupersedeWorker<J, R> {
    name: String,
    shared: Arc<SharedState<J>>,
    results: Receiver<R>,
    handle: Option<JoinHandle<()>>,
    shutdown_timeout: Duration,
}

impl<J: Send + 'static, R: Send + 'static> SupersedeWorker<J, R> {
    /// Spawn a named worker thread running `handler` on the most recent submitted job.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError::WorkerFault`] when the OS cannot spawn the thread.
    pub fn spawn<F>(
        name: &str,
        shutdown_timeout: Duration,
        mut handler: F,
    ) -> Result<Self, FlightVisualError>
    where
        F: FnMut(J) -> R + Send + 'static,
    {
        let shared = Arc::new(SharedState {
            slot: Mutex::new(SupersedeSlot {
                job: None,
                closed: false,
            }),
            cv: Condvar::new(),
            alive: AtomicBool::new(true),
            faulted: AtomicBool::new(false),
            busy: AtomicBool::new(false),
            superseded: AtomicU32::new(0),
            dropped: AtomicU32::new(0),
        });
        let (res_tx, res_rx) = mpsc::sync_channel::<R>(2);
        let worker_shared = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                loop {
                    let next_job = {
                        let Ok(mut guard) = worker_shared.slot.lock() else {
                            worker_shared.faulted.store(true, Ordering::Release);
                            break;
                        };
                        while guard.job.is_none() && !guard.closed {
                            let Ok(waited) = worker_shared.cv.wait(guard) else {
                                worker_shared.faulted.store(true, Ordering::Release);
                                worker_shared.alive.store(false, Ordering::Release);
                                return;
                            };
                            guard = waited;
                        }
                        if let Some(j) = guard.job.take() {
                            worker_shared.busy.store(true, Ordering::Release);
                            j
                        } else {
                            break;
                        }
                    };
                    let out = handler(next_job);
                    worker_shared.busy.store(false, Ordering::Release);
                    if res_tx.send(out).is_err() {
                        break;
                    }
                }
                worker_shared.alive.store(false, Ordering::Release);
            })
            .map_err(|e| FlightVisualError::WorkerFault {
                worker: name.to_owned(),
                message: format!("cannot spawn worker thread: {e}"),
            })?;
        Ok(Self {
            name: name.to_owned(),
            shared,
            results: res_rx,
            handle: Some(handle),
            shutdown_timeout,
        })
    }

    /// Submit `job`, superseding any queued job that has not started execution yet.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError::WorkerFault`] if the worker thread stopped or poisoned its lock.
    pub fn submit_supersede(&self, job: J) -> Result<(), FlightVisualError> {
        if !self.is_alive() {
            return Err(self.fault("worker thread stopped"));
        }
        let mut guard = self
            .shared
            .slot
            .lock()
            .map_err(|_| self.fault("worker mutex poisoned"))?;
        if guard.job.replace(job).is_some() {
            self.shared.superseded.fetch_add(1, Ordering::Relaxed);
        }
        self.shared.cv.notify_one();
        Ok(())
    }

    /// Submit `job` only when the worker is neither executing nor holding a queued job.
    ///
    /// Increments [`Self::dropped_count`] and returns `Ok(false)` when busy.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError::WorkerFault`] if the worker thread stopped or faulted.
    pub fn submit_if_idle(&self, build_job: impl FnOnce() -> J) -> Result<bool, FlightVisualError> {
        if !self.is_alive() {
            return Err(self.fault("worker thread stopped"));
        }
        if self.shared.busy.load(Ordering::Acquire) {
            self.shared.dropped.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let mut guard = self
            .shared
            .slot
            .lock()
            .map_err(|_| self.fault("worker mutex poisoned"))?;
        if guard.job.is_some() || self.shared.busy.load(Ordering::Acquire) {
            self.shared.dropped.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        guard.job = Some(build_job());
        self.shared.cv.notify_one();
        Ok(true)
    }

    /// Poll for a completed result without blocking.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError::WorkerFault`] if the worker result channel disconnected.
    pub fn try_recv(&self) -> Result<Option<R>, FlightVisualError> {
        match self.results.try_recv() {
            Ok(r) => Ok(Some(r)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                self.shared.faulted.store(true, Ordering::Release);
                Err(self.fault("worker result channel disconnected"))
            }
        }
    }

    /// Block until the next completed result arrives.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError::WorkerFault`] if the worker result channel disconnected.
    pub fn recv_blocking(&self) -> Result<R, FlightVisualError> {
        self.results.recv().map_err(|_| {
            self.shared.faulted.store(true, Ordering::Release);
            self.fault("worker result channel disconnected")
        })
    }

    /// Record an application-level worker fault on the watchdog flag.
    pub fn mark_faulted(&self) {
        self.shared.faulted.store(true, Ordering::Release);
    }

    /// Return true when the worker thread is alive.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.shared.alive.load(Ordering::Acquire)
    }

    /// Return true when the worker recorded a fault.
    #[must_use]
    pub fn is_faulted(&self) -> bool {
        self.shared.faulted.load(Ordering::Acquire)
    }

    /// Return cumulative count of superseded jobs.
    #[must_use]
    pub fn superseded_count(&self) -> u32 {
        self.shared.superseded.load(Ordering::Relaxed)
    }

    /// Return cumulative count of busy-dropped jobs.
    #[must_use]
    pub fn dropped_count(&self) -> u32 {
        self.shared.dropped.load(Ordering::Relaxed)
    }

    fn fault(&self, msg: &str) -> FlightVisualError {
        FlightVisualError::WorkerFault {
            worker: self.name.clone(),
            message: msg.to_owned(),
        }
    }
}

impl<J, R> Drop for SupersedeWorker<J, R> {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.shared.slot.lock() {
            guard.job = None;
            guard.closed = true;
            self.shared.cv.notify_all();
        }
        loop {
            match self.results.recv_timeout(self.shutdown_timeout) {
                Ok(_) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    if let Some(handle) = self.handle.take()
                        && handle.join().is_err()
                    {
                        self.shared.faulted.store(true, Ordering::Release);
                        tracing::error!(worker = %self.name, "background worker panicked");
                    }
                    return;
                }
                Err(RecvTimeoutError::Timeout) => {
                    self.shared.faulted.store(true, Ordering::Release);
                    tracing::error!(worker = %self.name, "background worker timed out on shutdown");
                    return;
                }
            }
        }
    }
}

/// Bounded FIFO background worker (`sync_channel(capacity)`) with atomic drop and fault counters.
pub struct BoundedFifoWorker<J, R> {
    name: String,
    jobs: Option<SyncSender<J>>,
    results: Receiver<R>,
    alive: Arc<AtomicBool>,
    faulted: Arc<AtomicBool>,
    dropped: AtomicU32,
    handle: Option<JoinHandle<()>>,
    shutdown_timeout: Duration,
}

impl<J: Send + 'static, R: Send + 'static> BoundedFifoWorker<J, R> {
    /// Spawn a named FIFO worker with bounded queue `capacity`.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError::WorkerFault`] when the OS cannot spawn the thread.
    pub fn spawn<F>(
        name: &str,
        capacity: usize,
        shutdown_timeout: Duration,
        mut handler: F,
    ) -> Result<Self, FlightVisualError>
    where
        F: FnMut(J) -> R + Send + 'static,
    {
        let cap = capacity.max(1);
        let (job_tx, job_rx) = mpsc::sync_channel::<J>(cap);
        let (res_tx, res_rx) = mpsc::sync_channel::<R>(cap.max(4));
        let alive = Arc::new(AtomicBool::new(true));
        let faulted = Arc::new(AtomicBool::new(false));
        let worker_alive = Arc::clone(&alive);
        let handle = thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                for job in job_rx {
                    if res_tx.send(handler(job)).is_err() {
                        break;
                    }
                }
                worker_alive.store(false, Ordering::Release);
            })
            .map_err(|e| FlightVisualError::WorkerFault {
                worker: name.to_owned(),
                message: format!("cannot spawn worker thread: {e}"),
            })?;
        Ok(Self {
            name: name.to_owned(),
            jobs: Some(job_tx),
            results: res_rx,
            alive,
            faulted,
            dropped: AtomicU32::new(0),
            handle: Some(handle),
            shutdown_timeout,
        })
    }

    /// Try to enqueue `job` without blocking; increments [`Self::dropped_count`] when full.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError::WorkerFault`] if the worker channel disconnected.
    pub fn try_submit(&self, job: J) -> Result<bool, FlightVisualError> {
        let Some(tx) = self.jobs.as_ref() else {
            return Err(self.fault("worker closed"));
        };
        match tx.try_send(job) {
            Ok(()) => Ok(true),
            Err(TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                Ok(false)
            }
            Err(TrySendError::Disconnected(_)) => {
                self.faulted.store(true, Ordering::Release);
                Err(self.fault("worker job channel disconnected"))
            }
        }
    }

    /// Poll for a finished result without blocking.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError::WorkerFault`] if the worker result channel disconnected.
    pub fn try_recv(&self) -> Result<Option<R>, FlightVisualError> {
        match self.results.try_recv() {
            Ok(r) => Ok(Some(r)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                self.faulted.store(true, Ordering::Release);
                Err(self.fault("worker result channel disconnected"))
            }
        }
    }

    /// Block until the next finished result arrives.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError::WorkerFault`] if the worker result channel disconnected.
    pub fn recv_blocking(&self) -> Result<R, FlightVisualError> {
        self.results.recv().map_err(|_| {
            self.faulted.store(true, Ordering::Release);
            self.fault("worker result channel disconnected")
        })
    }

    /// Mark an application-level worker fault.
    pub fn mark_faulted(&self) {
        self.faulted.store(true, Ordering::Release);
    }

    /// Return true when the worker thread is alive.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }

    /// Return true when the worker recorded a fault.
    #[must_use]
    pub fn is_faulted(&self) -> bool {
        self.faulted.load(Ordering::Acquire)
    }

    /// Return cumulative count of dropped jobs due to full queue.
    #[must_use]
    pub fn dropped_count(&self) -> u32 {
        self.dropped.load(Ordering::Relaxed)
    }

    fn fault(&self, msg: &str) -> FlightVisualError {
        FlightVisualError::WorkerFault {
            worker: self.name.clone(),
            message: msg.to_owned(),
        }
    }
}

impl<J, R> Drop for BoundedFifoWorker<J, R> {
    fn drop(&mut self) {
        drop(self.jobs.take());
        loop {
            match self.results.recv_timeout(self.shutdown_timeout) {
                Ok(_) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    if let Some(handle) = self.handle.take()
                        && handle.join().is_err()
                    {
                        self.faulted.store(true, Ordering::Release);
                        tracing::error!(worker = %self.name, "background worker panicked");
                    }
                    return;
                }
                Err(RecvTimeoutError::Timeout) => {
                    self.faulted.store(true, Ordering::Release);
                    tracing::error!(worker = %self.name, "background worker timed out on shutdown");
                    return;
                }
            }
        }
    }
}

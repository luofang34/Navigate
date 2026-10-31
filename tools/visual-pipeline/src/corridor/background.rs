//! Named background worker thread with job and result channels.
//!
//! The corridor tracker sends slow work (`Segformer` refresh, map-anchor
//! acquisition) to a [`BackgroundWorker`] so that the per-frame loop does not
//! wait for it. The loop polls for results with [`BackgroundWorker::try_recv`].

use std::{
    sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError},
    thread::{self, JoinHandle},
    time::Duration,
};
use tracing::{debug, error};

use crate::PipelineError;

/// One worker thread that runs `handler` for each job in submission order.
pub(super) struct BackgroundWorker<J, R> {
    name: String,
    jobs: Option<Sender<J>>,
    results: Receiver<R>,
    handle: Option<JoinHandle<()>>,
    shutdown_timeout: Duration,
}

impl<J: Send + 'static, R: Send + 'static> BackgroundWorker<J, R> {
    /// Spawn the thread `name`. It runs `handler` until the job sender closes.
    pub(super) fn spawn<F>(
        name: &str,
        shutdown_timeout: Duration,
        mut handler: F,
    ) -> Result<Self, PipelineError>
    where
        F: FnMut(J) -> R + Send + 'static,
    {
        let (job_tx, job_rx) = mpsc::channel::<J>();
        let (result_tx, result_rx) = mpsc::channel::<R>();
        let handle = thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                for job in job_rx {
                    if result_tx.send(handler(job)).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| PipelineError::Tool {
                tool: name.to_owned(),
                message: format!("cannot spawn worker thread: {e}"),
            })?;
        Ok(Self {
            name: name.to_owned(),
            jobs: Some(job_tx),
            results: result_rx,
            handle: Some(handle),
            shutdown_timeout,
        })
    }

    /// Queue `job` for the worker thread.
    pub(super) fn submit(&self, job: J) -> Result<(), PipelineError> {
        self.jobs
            .as_ref()
            .and_then(|tx| tx.send(job).ok())
            .ok_or_else(|| self.stopped())
    }

    /// Return a finished result without waiting, or `None` when no result is ready.
    pub(super) fn try_recv(&self) -> Result<Option<R>, PipelineError> {
        match self.results.try_recv() {
            Ok(result) => Ok(Some(result)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(self.stopped()),
        }
    }

    /// Wait for the next result.
    pub(super) fn recv_blocking(&self) -> Result<R, PipelineError> {
        self.results.recv().map_err(|_| self.stopped())
    }

    fn stopped(&self) -> PipelineError {
        PipelineError::Tool {
            tool: self.name.clone(),
            message: "background worker thread stopped".into(),
        }
    }
}

impl<J, R> Drop for BackgroundWorker<J, R> {
    fn drop(&mut self) {
        // Closing the job channel ends the worker loop after its current job.
        drop(self.jobs.take());
        loop {
            match self.results.recv_timeout(self.shutdown_timeout) {
                Ok(_) => debug!(worker = %self.name, "discarded a late background result"),
                Err(RecvTimeoutError::Disconnected) => {
                    if let Some(handle) = self.handle.take()
                        && handle.join().is_err()
                    {
                        error!(worker = %self.name, "background worker thread panicked");
                    }
                    return;
                }
                Err(RecvTimeoutError::Timeout) => {
                    error!(
                        worker = %self.name,
                        timeout_ms = self.shutdown_timeout.as_millis(),
                        "background worker did not exit before the shutdown timeout; detaching it"
                    );
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;

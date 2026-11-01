//! `Segformer` key-layer refresh, inline or on a background worker thread.

use super::super::background::BackgroundWorker;
use crate::PipelineError;
use image::RgbImage;
use nalgebra::Matrix3;
pub(super) use navigate_segment::amortized::{RefreshLayers, SegEngine};
use std::time::Duration;

/// Time that a dropped masker waits for an in-flight refresh to finish.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

type RefreshWorker = BackgroundWorker<RgbImage, Result<RefreshLayers, PipelineError>>;

/// Which frame the cached key layers or the pending job belong to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum KeyRole {
    #[default]
    Earlier,
    Current,
    Submitted,
}

/// Cached `Segformer` layers that later frames warp with their attitude.
#[derive(Debug, Default)]
pub(super) struct KeyLayers {
    pub(super) dims: [u32; 2],
    pub(super) airframe_keep: Vec<u8>,
    pub(super) sky_keep: Vec<u8>,
    pub(super) rot: Option<Matrix3<f64>>,
    frames_since_refresh: usize,
    role: KeyRole,
}

impl KeyLayers {
    pub(super) fn usable(&self, dims: [u32; 2]) -> bool {
        let total = (dims[0] as usize).saturating_mul(dims[1] as usize);
        self.dims == dims && self.airframe_keep.len() == total && self.sky_keep.len() == total
    }

    fn install(&mut self, layers: RefreshLayers, rot: Option<Matrix3<f64>>, frames: usize) {
        self.dims = layers.dims;
        self.airframe_keep = layers.airframe_keep;
        self.sky_keep = layers.sky_keep;
        self.rot = rot;
        self.frames_since_refresh = frames;
    }
}

struct Pending {
    submit_rot: Option<Matrix3<f64>>,
    solved_rot: Option<Matrix3<f64>>,
    frames_since_submit: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RefreshStep {
    pub(super) refreshed: bool,
    pub(super) submitted: bool,
    pub(super) segformer_ms: Option<f64>,
}

pub(super) enum Refresher {
    Inline(Box<SegEngine>),
    Background(Box<BackgroundRefresh>),
}

pub(super) struct BackgroundRefresh {
    worker: RefreshWorker,
    pending: Option<Pending>,
}

impl Refresher {
    pub(super) fn into_background(self) -> Result<Self, PipelineError> {
        match self {
            Self::Inline(mut engine) => {
                engine.clamp_input_side(256);
                Self::spawn_background(move |rgb| engine.refresh(&rgb).map_err(PipelineError::from))
            }
            background @ Self::Background(_) => Ok(background),
        }
    }

    fn spawn_background<F>(refresh: F) -> Result<Self, PipelineError>
    where
        F: FnMut(RgbImage) -> Result<RefreshLayers, PipelineError> + Send + 'static,
    {
        Ok(Self::Background(Box::new(BackgroundRefresh {
            worker: BackgroundWorker::spawn(
                "corridor-segformer-refresh",
                SHUTDOWN_TIMEOUT,
                refresh,
            )?,
            pending: None,
        })))
    }

    pub(super) fn step(
        &mut self,
        key: &mut KeyLayers,
        rgb: &RgbImage,
        cur_rot: Option<&Matrix3<f64>>,
        interval: usize,
        allow_submit: bool,
    ) -> Result<RefreshStep, PipelineError> {
        key.role = KeyRole::Earlier;
        let dims = [rgb.width(), rgb.height()];
        let mut step = RefreshStep::default();
        match self {
            Self::Inline(engine) => {
                if !key.usable(dims) || (allow_submit && key.frames_since_refresh >= interval) {
                    let layers = engine.refresh(rgb)?;
                    step = installed(key, layers, cur_rot.copied(), 1);
                    key.role = KeyRole::Current;
                } else {
                    key.frames_since_refresh += 1;
                }
            }
            Self::Background(background) => {
                let BackgroundRefresh { worker, pending } = &mut **background;
                if let Some(job) = pending.as_mut() {
                    job.frames_since_submit += 1;
                    if let Some(layers) = worker.try_recv()? {
                        let rot = job.solved_rot.or(job.submit_rot);
                        let frames = job.frames_since_submit;
                        *pending = None;
                        step = installed(key, layers?, rot, frames);
                    }
                }
                if !key.usable(dims) {
                    if pending.is_none() {
                        worker.submit(rgb.clone())?;
                        *pending = Some(Pending {
                            submit_rot: cur_rot.copied(),
                            solved_rot: None,
                            frames_since_submit: 1,
                        });
                        key.role = KeyRole::Submitted;
                        step.submitted = true;
                    }
                } else {
                    if !step.refreshed {
                        key.frames_since_refresh += 1;
                    }
                    if allow_submit
                        && pending.is_none()
                        && key.role != KeyRole::Current
                        && key.frames_since_refresh >= interval
                    {
                        worker.submit(rgb.clone())?;
                        *pending = Some(Pending {
                            submit_rot: cur_rot.copied(),
                            solved_rot: None,
                            frames_since_submit: 1,
                        });
                        key.role = KeyRole::Submitted;
                        step.submitted = true;
                    }
                }
            }
        }
        Ok(step)
    }

    pub(super) fn note_solved_rotation(&mut self, key: &mut KeyLayers, rot: Matrix3<f64>) {
        match (key.role, self) {
            (KeyRole::Current, _) => key.rot = Some(rot),
            (KeyRole::Submitted, Self::Background(background)) => {
                if let Some(job) = background.pending.as_mut() {
                    job.solved_rot = Some(rot);
                }
            }
            _ => {}
        }
    }

    pub(super) fn initial_fast_pass(
        &mut self,
        key: &mut KeyLayers,
        rgb: &RgbImage,
        cur_rot: Option<&Matrix3<f64>>,
    ) -> Result<Option<f64>, PipelineError> {
        let dims = [rgb.width(), rgb.height()];
        if key.usable(dims) {
            return Ok(None);
        }
        match self {
            Self::Inline(engine) => {
                let layers = match engine.refresh(rgb) {
                    Ok(l) => l,
                    Err(e) => {
                        tracing::warn!("inline Segformer refresh failed: {e}");
                        return Ok(None);
                    }
                };
                *key = KeyLayers::default();
                let step = installed(key, layers, cur_rot.copied(), 1);
                key.role = KeyRole::Current;
                Ok(Some(step.segformer_ms.unwrap_or(0.0)))
            }
            Self::Background(background) => {
                let BackgroundRefresh { worker, pending } = &mut **background;
                if pending.is_none() {
                    worker.submit(rgb.clone())?;
                    *pending = Some(Pending {
                        submit_rot: cur_rot.copied(),
                        solved_rot: None,
                        frames_since_submit: 1,
                    });
                }
                let layers = match worker.recv_blocking()? {
                    Ok(l) => l,
                    Err(e) => {
                        tracing::warn!("background Segformer refresh failed: {e}");
                        *pending = None;
                        return Ok(None);
                    }
                };
                let rot = pending.as_ref().and_then(|p| p.solved_rot.or(p.submit_rot));
                *pending = None;
                *key = KeyLayers::default();
                let step = installed(key, layers, rot, 1);
                key.role = KeyRole::Current;
                Ok(Some(step.segformer_ms.unwrap_or(0.0)))
            }
        }
    }
}

fn installed(
    key: &mut KeyLayers,
    layers: RefreshLayers,
    rot: Option<Matrix3<f64>>,
    frames: usize,
) -> RefreshStep {
    let segformer_ms = Some(layers.eval_ms);
    key.install(layers, rot, frames);
    RefreshStep {
        refreshed: true,
        submitted: false,
        segformer_ms,
    }
}

#[cfg(test)]
mod tests;

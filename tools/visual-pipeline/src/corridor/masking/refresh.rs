//! `Segformer` key-layer refresh, inline or on a background worker thread.
//!
//! In background mode the masker keeps warping the cached sky layer with the
//! current attitude while a newer refresh runs. The new layers belong to the
//! frame that was sent, so their key rotation is the solved rotation of that
//! frame, not of the frame on which they arrive.

use super::super::background::BackgroundWorker;
use crate::PipelineError;
use burn::tensor::Device;
use image::RgbImage;
use nalgebra::Matrix3;
use navigate_segment::{
    ColorMode, MaskPolicy, PreprocessorConfig, Segformer, morphology::dilate_rect, surface_mask,
};
use std::time::{Duration, Instant};

/// Time that a dropped masker waits for an in-flight refresh to finish.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

pub(super) struct SegEngine {
    pub(super) model: Segformer,
    pub(super) preproc: PreprocessorConfig,
    pub(super) airplane_indices: Vec<usize>,
    pub(super) sky_label: u8,
    pub(super) is_four_class: bool,
    pub(super) device: Device,
    pub(super) policy: MaskPolicy,
    pub(super) margin_px: usize,
}

/// Key layers from one `Segformer` evaluation.
pub(super) struct RefreshLayers {
    dims: [u32; 2],
    airframe_keep: Vec<u8>,
    sky_keep: Vec<u8>,
    eval_ms: f64,
}

impl SegEngine {
    fn refresh(&self, rgb: &RgbImage) -> Result<RefreshLayers, PipelineError> {
        let t0 = Instant::now();
        let (airframe_keep, sky_keep) = self.eval(rgb)?;
        Ok(RefreshLayers {
            dims: [rgb.width(), rgb.height()],
            airframe_keep,
            sky_keep,
            eval_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn eval(&self, rgb: &RgbImage) -> Result<(Vec<u8>, Vec<u8>), PipelineError> {
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let pred = self.model.predict_image(
            rgb,
            &self.preproc,
            ColorMode::Color,
            &self.airplane_indices,
            &self.device,
        )?;
        if self.is_four_class {
            Ok(four_class_keep_layers(
                &pred.labels,
                self.sky_label,
                w,
                h,
                self.margin_px,
            ))
        } else {
            let air_probs = pred.selected_probs.first().map(Vec::as_slice);
            let (combined_keep, _) = surface_mask(
                rgb,
                &pred.labels,
                &pred.confidence,
                air_probs,
                self.policy,
                self.margin_px,
            )?;
            let raw_sky: Vec<bool> = pred
                .labels
                .iter()
                .zip(&pred.confidence)
                .map(|(&l, &c)| l == self.sky_label && c >= 0.35)
                .collect();
            let sky_occ = dilate_rect(&raw_sky, w, h, self.margin_px);
            let sky_keep: Vec<u8> = sky_occ.iter().map(|&b| if b { 0 } else { 255 }).collect();
            let air_keep: Vec<u8> = combined_keep
                .iter()
                .zip(&sky_occ)
                .map(|(&keep_px, &is_sky)| if keep_px || is_sky { 255 } else { 0 })
                .collect();
            Ok((air_keep, sky_keep))
        }
    }
}

type RefreshWorker = BackgroundWorker<RgbImage, Result<RefreshLayers, PipelineError>>;

/// Which frame the cached key layers or the pending job belong to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum KeyRole {
    /// Neither the key layers nor a pending job come from the current frame.
    #[default]
    Earlier,
    /// The key layers come from the current frame.
    Current,
    /// The current frame went to the worker thread.
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

/// A frame that the worker thread evaluates.
struct Pending {
    submit_rot: Option<Matrix3<f64>>,
    solved_rot: Option<Matrix3<f64>>,
    frames_since_submit: usize,
}

/// Refresh events of one frame.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RefreshStep {
    /// New key layers took effect on this frame.
    pub(super) refreshed: bool,
    /// This frame went to the worker thread.
    pub(super) submitted: bool,
    /// `Segformer` evaluation time of the layers that took effect.
    pub(super) segformer_ms: Option<f64>,
}

pub(super) enum Refresher {
    Inline(Box<SegEngine>),
    Background(Box<BackgroundRefresh>),
}

/// Worker thread and the frame that it evaluates.
pub(super) struct BackgroundRefresh {
    worker: RefreshWorker,
    pending: Option<Pending>,
}

impl Refresher {
    /// Move the engine to a named worker thread and cap its input side to `256` pixels.
    pub(super) fn into_background(self) -> Result<Self, PipelineError> {
        match self {
            Self::Inline(mut engine) => {
                let (w, h) = engine.preproc.target_size();
                engine.preproc.size = serde_json::json!({
                    "width": w.min(256),
                    "height": h.min(256),
                });
                Self::spawn_background(move |rgb| engine.refresh(&rgb))
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

    /// Bring `key` up to date for frame `rgb` with estimated rotation `cur_rot`.
    pub(super) fn step(
        &mut self,
        key: &mut KeyLayers,
        rgb: &RgbImage,
        cur_rot: Option<&Matrix3<f64>>,
        interval: usize,
    ) -> Result<RefreshStep, PipelineError> {
        key.role = KeyRole::Earlier;
        let dims = [rgb.width(), rgb.height()];
        let mut step = RefreshStep::default();
        match self {
            Self::Inline(engine) => {
                if !key.usable(dims) || key.frames_since_refresh >= interval {
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
                    if pending.is_none()
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

    /// Record the solved rotation of the current frame for the layers made from it.
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

    /// Run a fast initial Segformer pass on frame 1 to provide ground and sky masks before anchor jobs.
    pub(super) fn initial_fast_pass(
        &mut self,
        key: &mut KeyLayers,
        rgb: &RgbImage,
        cur_rot: Option<&Matrix3<f64>>,
    ) -> Result<bool, PipelineError> {
        let dims = [rgb.width(), rgb.height()];
        if key.usable(dims) {
            return Ok(false);
        }
        match self {
            Self::Inline(engine) => {
                let layers = match engine.refresh(rgb) {
                    Ok(l) => l,
                    Err(e) => {
                        tracing::warn!("inline Segformer refresh failed: {e}");
                        return Ok(false);
                    }
                };
                *key = KeyLayers::default();
                let _ = installed(key, layers, cur_rot.copied(), 1);
                key.role = KeyRole::Current;
                Ok(true)
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
                        return Ok(false);
                    }
                };
                let rot = pending.as_ref().and_then(|p| p.solved_rot.or(p.submit_rot));
                *pending = None;
                *key = KeyLayers::default();
                let _ = installed(key, layers, rot, 1);
                key.role = KeyRole::Current;
                Ok(true)
            }
        }
    }
}

fn four_class_keep_layers(
    labels: &[u8],
    sky_label: u8,
    w: usize,
    h: usize,
    margin_px: usize,
) -> (Vec<u8>, Vec<u8>) {
    let raw_sky: Vec<bool> = labels.iter().map(|&l| l == sky_label).collect();
    let raw_air: Vec<bool> = labels
        .iter()
        .map(|&l| matches!(l, navigate_segment::AIRFRAME | navigate_segment::PROPELLER))
        .collect();
    let sky_occ = dilate_rect(&raw_sky, w, h, margin_px);
    let air_occ = dilate_rect(&raw_air, w, h, margin_px);
    (
        air_occ
            .into_iter()
            .map(|b| if b { 0 } else { 255 })
            .collect(),
        sky_occ
            .into_iter()
            .map(|b| if b { 0 } else { 255 })
            .collect(),
    )
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

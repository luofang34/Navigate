//! `Segformer` key-layer refresh via [`SupersedeWorker`] with valid-fraction sanity gating.

use crate::{error::FlightVisualError, worker::SupersedeWorker};
use image::RgbImage;
use nalgebra::Matrix3;
pub(super) use navigate_match::segment::amortized::{RefreshLayers, SegEngine};
use std::time::Duration;

const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

type RefreshWorker = SupersedeWorker<RgbImage, Result<RefreshLayers, FlightVisualError>>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum KeyRole {
    #[default]
    Earlier,
    Current,
    Submitted,
}

#[derive(Debug, Default)]
pub(super) struct KeyLayers {
    pub(super) dims: [u32; 2],
    pub(super) airframe_keep: Vec<u8>,
    pub(super) sky_keep: Vec<u8>,
    pub(super) airframe_keep_count: usize,
    pub(super) rot: Option<Matrix3<f64>>,
    frames_since_refresh: usize,
    role: KeyRole,
}

impl KeyLayers {
    pub(super) fn usable(&self, dims: [u32; 2]) -> bool {
        let total = (dims[0] as usize).saturating_mul(dims[1] as usize);
        self.dims == dims && self.airframe_keep.len() == total && self.sky_keep.len() == total
    }

    pub(super) const fn frames_since_refresh(&self) -> usize {
        self.frames_since_refresh
    }

    fn install(&mut self, layers: RefreshLayers, rot: Option<Matrix3<f64>>, frames: usize) {
        self.airframe_keep_count = layers.airframe_keep.iter().filter(|&&b| b >= 128).count();
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
    pub(super) fn into_background(self) -> Result<Self, FlightVisualError> {
        match self {
            Self::Inline(mut engine) => {
                engine.clamp_input_side(256);
                Ok(Self::Background(Box::new(BackgroundRefresh {
                    worker: SupersedeWorker::spawn(
                        "corridor-segformer-refresh",
                        SHUTDOWN_TIMEOUT,
                        move |rgb| engine.refresh(&rgb).map_err(FlightVisualError::from),
                    )?,
                    pending: None,
                })))
            }
            background @ Self::Background(_) => Ok(background),
        }
    }

    pub(super) fn superseded_count(&self) -> u32 {
        match self {
            Self::Inline(_) => 0,
            Self::Background(bg) => bg.worker.superseded_count(),
        }
    }

    pub(super) fn has_pending(&self) -> bool {
        matches!(self, Self::Background(bg) if bg.pending.is_some())
    }

    pub(super) fn clamp_input_side(&mut self, max_side: u32) {
        if let Self::Inline(engine) = self {
            engine.clamp_input_side(max_side);
        }
    }

    pub(super) fn step(
        &mut self,
        key: &mut KeyLayers,
        dims: [u32; 2],
        rgb: Option<&RgbImage>,
        cur_rot: Option<&Matrix3<f64>>,
        interval: usize,
        allow_submit: bool,
    ) -> Result<RefreshStep, FlightVisualError> {
        key.role = KeyRole::Earlier;
        let rgb_valid = rgb.filter(|img| img.dimensions() == (dims[0], dims[1]));
        let mut step = RefreshStep::default();
        match self {
            Self::Inline(engine) => {
                if (!key.usable(dims) || (allow_submit && key.frames_since_refresh >= interval))
                    && let Some(rgb_img) = rgb_valid
                {
                    let layers = engine.refresh(rgb_img)?;
                    step = maybe_install(key, layers, cur_rot.copied(), 1);
                    if step.refreshed {
                        key.role = KeyRole::Current;
                    }
                } else {
                    key.frames_since_refresh = key.frames_since_refresh.saturating_add(1);
                }
            }
            Self::Background(background) => {
                let BackgroundRefresh { worker, pending } = &mut **background;
                if let Some(job) = pending.as_mut() {
                    job.frames_since_submit = job.frames_since_submit.saturating_add(1);
                    if let Some(layers_res) = worker.try_recv()? {
                        let rot = job.solved_rot.or(job.submit_rot);
                        let frames = job.frames_since_submit;
                        *pending = None;
                        match layers_res {
                            Ok(layers) => {
                                step = maybe_install(key, layers, rot, frames);
                            }
                            Err(err) => {
                                worker.mark_faulted();
                                return Err(err);
                            }
                        }
                    }
                }
                if !key.usable(dims) {
                    if pending.is_none()
                        && allow_submit
                        && let Some(rgb_img) = rgb_valid
                    {
                        worker.submit_supersede(rgb_img.clone())?;
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
                        key.frames_since_refresh = key.frames_since_refresh.saturating_add(1);
                    }
                    if allow_submit
                        && pending.is_none()
                        && key.role != KeyRole::Current
                        && key.frames_since_refresh >= interval
                        && let Some(rgb_img) = rgb_valid
                    {
                        worker.submit_supersede(rgb_img.clone())?;
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
        if key.rot.is_none() {
            key.rot = Some(rot);
        }
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
    ) -> Result<Option<f64>, FlightVisualError> {
        let dims = [rgb.width(), rgb.height()];
        if key.usable(dims) {
            return Ok(None);
        }
        match self {
            Self::Inline(engine) => {
                let saved = engine.preproc.size.clone();
                engine.clamp_input_side(256);
                let res = engine.refresh(rgb);
                engine.preproc.size = saved;
                let layers = res?;
                let step = maybe_install(key, layers, cur_rot.copied(), 1);
                if step.refreshed {
                    key.role = KeyRole::Current;
                }
                Ok(step.segformer_ms)
            }
            Self::Background(background) => {
                let BackgroundRefresh { worker, pending } = &mut **background;
                if pending.is_none() {
                    worker.submit_supersede(rgb.clone())?;
                    *pending = Some(Pending {
                        submit_rot: cur_rot.copied(),
                        solved_rot: None,
                        frames_since_submit: 1,
                    });
                    key.role = KeyRole::Submitted;
                }
                Ok(None)
            }
        }
    }
}

fn maybe_install(
    key: &mut KeyLayers,
    layers: RefreshLayers,
    rot: Option<Matrix3<f64>>,
    frames: usize,
) -> RefreshStep {
    let total = layers.airframe_keep.len();
    let valid_airframe = layers.airframe_keep.iter().filter(|&&v| v > 0).count();
    let eval_ms = layers.eval_ms;
    // Reject an empty airframe mask whose valid area is less than 5% of the image.
    if total == 0 || valid_airframe * 100 < total * 5 {
        key.frames_since_refresh = 0;
        return RefreshStep {
            refreshed: false,
            submitted: false,
            segformer_ms: Some(eval_ms),
        };
    }
    key.install(layers, rot, frames);
    RefreshStep {
        refreshed: true,
        submitted: false,
        segformer_ms: Some(eval_ms),
    }
}

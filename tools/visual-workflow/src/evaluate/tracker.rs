//! Persistent tracking execution selected independently of neural matching.
use crate::Error;
use navigate_visual::{ImageMatcher, PointTracker, PyramidalMatcher};
use serde::Deserialize;

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TrackerBackend {
    #[default]
    Cpu,
    Wgpu,
}

pub(crate) struct Tracker {
    pub backend: Box<dyn PointTracker>,
    pub identity: String,
}
impl Tracker {
    pub fn new_blocking(backend: TrackerBackend) -> Result<Self, Error> {
        match backend {
            TrackerBackend::Cpu => Ok(Self {
                identity: ImageMatcher::identity(&PyramidalMatcher).into(),
                backend: Box::new(PyramidalMatcher),
            }),
            TrackerBackend::Wgpu => Self::gpu_blocking(false),
        }
    }
    #[cfg(feature = "gpu")]
    pub fn gpu_blocking(reuse: bool) -> Result<Self, Error> {
        let mut backend = pollster::block_on(navigate_visual::GpuPyramidalMatcher::new())?;
        backend.set_gpu_image_reuse(reuse);
        Ok(Self {
            identity: ImageMatcher::identity(&backend).into(),
            backend: Box::new(backend),
        })
    }
    #[cfg(not(feature = "gpu"))]
    pub fn gpu_blocking(_reuse: bool) -> Result<Self, Error> {
        Err(Error::Input(
            "wgpu tracking requires the gpu build feature".into(),
        ))
    }
}

#[cfg(test)]
mod tests;

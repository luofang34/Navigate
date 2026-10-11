//! Persistent tracking execution selected independently of neural matching.
use crate::Error;
use navigate_visual::{ImageMatcher, PointTracker, PyramidalMatcher};
use serde::Deserialize;

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TrackerBackend {
    #[default]
    Cpu,
    Wgpu,
}

pub(super) struct Tracker {
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
            TrackerBackend::Wgpu => Self::gpu_blocking(),
        }
    }
    #[cfg(feature = "gpu")]
    fn gpu_blocking() -> Result<Self, Error> {
        let backend = pollster::block_on(navigate_visual::GpuPyramidalMatcher::new())?;
        Ok(Self {
            identity: ImageMatcher::identity(&backend).into(),
            backend: Box::new(backend),
        })
    }
    #[cfg(not(feature = "gpu"))]
    fn gpu_blocking() -> Result<Self, Error> {
        Err(Error::Input(
            "wgpu tracking requires the gpu build feature".into(),
        ))
    }
}

#[cfg(test)]
mod tests;

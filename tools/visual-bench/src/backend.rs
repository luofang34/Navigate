//! Explicit matcher selection. Backend failures do not trigger a silent fallback.

use clap::ValueEnum;
use navigate_visual::{
    GpuPyramidalMatcher, ImageMatcher, PixelMatch, PyramidalMatcher, VisualError,
};

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum BackendKind {
    #[default]
    Cpu,
    Gpu,
}

pub(crate) enum Backend {
    Cpu(PyramidalMatcher),
    Gpu(Box<GpuPyramidalMatcher>),
    Matches(crate::matches::VerifiedMatches),
}

impl Backend {
    pub fn validate_reference(
        &self,
        reference: &navigate_visual::ReferenceView,
    ) -> Result<(), VisualError> {
        match self {
            Self::Matches(matches) => matches.validate_depth(reference),
            _ => Ok(()),
        }
    }
    pub async fn new(kind: BackendKind) -> Result<Self, VisualError> {
        match kind {
            BackendKind::Cpu => Ok(Self::Cpu(PyramidalMatcher)),
            BackendKind::Gpu => Ok(Self::Gpu(Box::new(GpuPyramidalMatcher::new().await?))),
        }
    }
}

impl ImageMatcher for Backend {
    fn identity(&self) -> &str {
        match self {
            Self::Cpu(m) => m.identity(),
            Self::Gpu(m) => m.identity(),
            Self::Matches(m) => m.identity(),
        }
    }
    fn match_images_blocking(
        &mut self,
        reference: &image::GrayImage,
        query: &image::GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        match self {
            Self::Cpu(m) => m.match_images_blocking(reference, query),
            Self::Gpu(m) => m.match_images_blocking(reference, query),
            Self::Matches(m) => m.match_images_blocking(reference, query),
        }
    }
}

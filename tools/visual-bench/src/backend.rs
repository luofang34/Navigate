//! Explicit matcher selection. Backend failures do not trigger a silent fallback.

use clap::{ValueEnum, builder::PossibleValue};
use navigate_visual::{
    GpuPyramidalMatcher, ImageMatcher, PixelMatch, PyramidalMatcher, VisualError,
};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) enum BackendKind {
    #[default]
    Cpu,
    Gpu,
}

impl ValueEnum for BackendKind {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Cpu, Self::Gpu]
    }
    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(PossibleValue::new(match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }))
    }
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

impl navigate_visual::PointTracker for Backend {
    fn features_blocking(
        &mut self,
        image: &image::GrayImage,
    ) -> Result<Vec<nalgebra::Vector2<f64>>, VisualError> {
        match self {
            Self::Cpu(m) => navigate_visual::PointTracker::features_blocking(m, image),
            Self::Gpu(m) => navigate_visual::PointTracker::features_blocking(m.as_mut(), image),
            Self::Matches(_) => Err(VisualError::Invalid {
                field: "external correspondences cannot select tracked features",
            }),
        }
    }
    fn track_points_blocking(
        &mut self,
        reference: &image::GrayImage,
        query: &image::GrayImage,
        points: &[nalgebra::Vector2<f64>],
    ) -> Result<Vec<Option<nalgebra::Vector2<f64>>>, VisualError> {
        match self {
            Self::Cpu(m) => {
                navigate_visual::PointTracker::track_points_blocking(m, reference, query, points)
            }
            Self::Gpu(m) => navigate_visual::PointTracker::track_points_blocking(
                m.as_mut(),
                reference,
                query,
                points,
            ),
            Self::Matches(_) => Err(VisualError::Invalid {
                field: "external correspondences cannot track feature identities",
            }),
        }
    }
}

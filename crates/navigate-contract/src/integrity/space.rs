//! Integrity vocabulary of a solution in a Cartesian frame (ADR-0015).
//!
//! A Cartesian frame has no local vertical, so the surface pair of a
//! horizontal and a vertical 1-sigma has no meaning. A space solution
//! states its position uncertainty in orbital axes, or in the principal
//! axes of the covariance when the orbit plane is not defined.

use crate::composition::SourceComposition;

use super::{FaultDetection, Redundancy, SolutionQuality};

/// The one-sigma position uncertainty of a space solution, in meters.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SpaceUncertainty {
    /// The uncertainty in the radial, along-track, and cross-track axes
    /// of the orbit. With the position `r` and the velocity `v` in the
    /// solution frame, the radial axis is `R = r/|r|`, the cross-track
    /// axis is `N = (r × v)/|r × v|`, and the along-track axis is
    /// `T = N × R`.
    OrbitalRtn {
        /// One-sigma along `R`.
        radial_1sigma_m: f64,
        /// One-sigma along `T`.
        along_track_1sigma_m: f64,
        /// One-sigma along `N`.
        cross_track_1sigma_m: f64,
        /// One-sigma along the largest principal axis of the position
        /// covariance.
        max_axis_1sigma_m: f64,
    },
    /// The uncertainty along the principal axes of the position
    /// covariance, largest first. A producer uses this form when
    /// `|r × v|` is too small to define an orbit plane, for example in a
    /// hover or in a radial descent.
    PrincipalAxes {
        /// One-sigma along each principal axis, in decreasing order.
        axes_1sigma_m: [f64; 3],
        /// One-sigma along the largest principal axis.
        max_axis_1sigma_m: f64,
    },
}

impl SpaceUncertainty {
    /// One-sigma along the largest principal axis of the position
    /// covariance. Every form carries it.
    #[must_use]
    pub const fn max_axis_1sigma_m(&self) -> f64 {
        match self {
            Self::OrbitalRtn {
                max_axis_1sigma_m, ..
            }
            | Self::PrincipalAxes {
                max_axis_1sigma_m, ..
            } => *max_axis_1sigma_m,
        }
    }
}

/// The integrity assessment published with every
/// [`crate::CartesianSolution`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SpaceIntegrityAssessment {
    /// Derived quality classification.
    pub quality: SolutionQuality,
    /// Source classes that contributed within the assessment window.
    pub contributing: SourceComposition,
    /// Redundancy actually available.
    pub redundancy: Redundancy,
    /// Position uncertainty, from the covariance.
    pub uncertainty: SpaceUncertainty,
    /// Fault-detection capability statement.
    pub fault_detection: FaultDetection,
    /// The radius in meters of a sphere about the position that bounds
    /// the position error at the integrity risk of an implementation.
    /// It is absent until an implementation earns it (ADR-0004).
    pub protection_radius_m: Option<f64>,
}

impl SpaceIntegrityAssessment {
    /// Builds an assessment from its parts. The protection radius starts
    /// absent; [`Self::with_protection_radius`] adds it.
    #[must_use]
    pub const fn new(
        quality: SolutionQuality,
        contributing: SourceComposition,
        redundancy: Redundancy,
        uncertainty: SpaceUncertainty,
        fault_detection: FaultDetection,
    ) -> Self {
        Self {
            quality,
            contributing,
            redundancy,
            uncertainty,
            fault_detection,
            protection_radius_m: None,
        }
    }

    /// This assessment with a protection radius in meters.
    #[must_use]
    pub const fn with_protection_radius(mut self, radius_m: f64) -> Self {
        self.protection_radius_m = Some(radius_m);
        self
    }
}

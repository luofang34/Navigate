//! Integrity vocabulary: what a solution's confidence actually rests on
//! (ADR-0004). The assessment never overstates redundancy — with a single
//! source, fault detection is `Unavailable`, not implied.

use crate::composition::SourceComposition;

/// Quality classification of a navigation solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SolutionQuality {
    /// Within configured accuracy bounds and admission health.
    Good,
    /// Usable with caution: bounds exceeded or sources degraded.
    Degraded,
    /// Not usable for any navigation decision.
    Unusable,
}

/// How much independent information supports the solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Redundancy {
    /// One source class contributes; nothing can cross-check it.
    None,
    /// Multiple sources of the same class contribute.
    SameClass,
    /// Multiple independent source classes contribute.
    IndependentClasses,
}

/// Whether the solution can detect a faulty source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FaultDetection {
    /// No redundant information exists; a single source agreeing with
    /// itself proves nothing.
    Unavailable,
    /// Cross-source consistency monitoring is active.
    Monitoring,
}

/// RAIM-class integrity bounds. Reserved: absent until an
/// implementation earns them (ADR-0004).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ProtectionLevels {
    /// Horizontal protection level in meters.
    pub horizontal_m: f64,
    /// Vertical protection level in meters.
    pub vertical_m: f64,
}

impl ProtectionLevels {
    /// Builds protection levels from their parts.
    #[must_use]
    pub const fn new(horizontal_m: f64, vertical_m: f64) -> Self {
        Self {
            horizontal_m,
            vertical_m,
        }
    }
}

/// The integrity assessment published with every solution.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct IntegrityAssessment {
    /// Derived quality classification.
    pub quality: SolutionQuality,
    /// Source classes that contributed within the assessment window.
    pub contributing: SourceComposition,
    /// Redundancy actually available.
    pub redundancy: Redundancy,
    /// One-sigma horizontal uncertainty in meters, from the covariance.
    pub horizontal_1sigma_m: f64,
    /// One-sigma vertical uncertainty in meters, from the covariance.
    pub vertical_1sigma_m: f64,
    /// Fault-detection capability statement.
    pub fault_detection: FaultDetection,
    /// RAIM-class bounds when an implementation provides them.
    pub protection_levels: Option<ProtectionLevels>,
}

impl IntegrityAssessment {
    /// Builds an assessment from its parts. Protection levels start
    /// absent; [`Self::with_protection_levels`] adds them.
    #[must_use]
    pub const fn new(
        quality: SolutionQuality,
        contributing: SourceComposition,
        redundancy: Redundancy,
        horizontal_1sigma_m: f64,
        vertical_1sigma_m: f64,
        fault_detection: FaultDetection,
    ) -> Self {
        Self {
            quality,
            contributing,
            redundancy,
            horizontal_1sigma_m,
            vertical_1sigma_m,
            fault_detection,
            protection_levels: None,
        }
    }

    /// This assessment with protection levels attached.
    #[must_use]
    pub const fn with_protection_levels(mut self, levels: ProtectionLevels) -> Self {
        self.protection_levels = Some(levels);
        self
    }
}

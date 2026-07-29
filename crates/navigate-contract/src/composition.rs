//! Source composition: the declared provenance of a value.
//!
//! Composition is how double-counting is prevented (ADR-0003): every
//! observation and every solution declares which sensor classes fed it,
//! so a consumer can refuse information derived from measurements it
//! already holds. An empty composition is inadmissible, never presumed
//! independent.

/// Classes of navigation information sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SensorClass {
    /// Satellite navigation receivers.
    Gnss,
    /// Celestial fixes (star/sun trackers).
    Celestial,
    /// Visual odometry (incremental ego-motion from cameras).
    VisualOdometry,
    /// Visual absolute fixes (landmark / terrain matching).
    VisualLandmark,
    /// Barometric altitude.
    Barometric,
    /// Raw inertial measurements.
    Inertial,
    /// State exported by the flight controller's own estimator.
    FcState,
    /// A published Navigate solution (feeding one back is circular).
    NavigationSolution,
}

impl SensorClass {
    const fn bit(self) -> u32 {
        match self {
            Self::Gnss => 1 << 0,
            Self::Celestial => 1 << 1,
            Self::VisualOdometry => 1 << 2,
            Self::VisualLandmark => 1 << 3,
            Self::Barometric => 1 << 4,
            Self::Inertial => 1 << 5,
            Self::FcState => 1 << 6,
            Self::NavigationSolution => 1 << 7,
        }
    }
}

/// A set of [`SensorClass`]es describing what fed a value.
///
/// Deliberately not `Default`: the empty composition is inadmissible as
/// a declaration and must be asked for by name via [`Self::empty`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceComposition(u32);

impl SourceComposition {
    /// The empty composition. Inadmissible as a declaration; exists so
    /// sets can be built incrementally.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// A composition of exactly one class.
    #[must_use]
    pub const fn of(class: SensorClass) -> Self {
        Self(class.bit())
    }

    /// This composition with `class` added.
    #[must_use]
    pub const fn with(self, class: SensorClass) -> Self {
        Self(self.0 | class.bit())
    }

    /// The set union of this composition and `other`: every class
    /// declared by either — how a fused value declares the provenance of
    /// all of its inputs.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether `class` is declared.
    #[must_use]
    pub const fn contains(self, class: SensorClass) -> bool {
        self.0 & class.bit() != 0
    }

    /// Whether no class is declared.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether any declared class is shared with `other`.
    #[must_use]
    pub const fn overlaps(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// Number of declared classes.
    #[must_use]
    pub const fn class_count(self) -> u32 {
        self.0.count_ones()
    }

    /// Whether this value is derived from an estimator output rather
    /// than independent measurements — circular as an aid (ADR-0003).
    #[must_use]
    pub const fn is_estimator_derived(self) -> bool {
        self.contains(SensorClass::FcState) || self.contains(SensorClass::NavigationSolution)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::{SensorClass, SourceComposition};

    #[test]
    fn composition_set_operations() {
        let c = SourceComposition::of(SensorClass::Gnss).with(SensorClass::Barometric);
        assert!(c.contains(SensorClass::Gnss));
        assert!(!c.contains(SensorClass::Celestial));
        assert_eq!(c.class_count(), 2);
        assert!(c.overlaps(SourceComposition::of(SensorClass::Gnss)));
        assert!(!c.overlaps(SourceComposition::of(SensorClass::Inertial)));
    }

    #[test]
    fn union_declares_every_class_from_either_side() {
        let left = SourceComposition::of(SensorClass::Gnss).with(SensorClass::Barometric);
        let right = SourceComposition::of(SensorClass::Barometric).with(SensorClass::Inertial);
        let merged = left.union(right);
        assert!(merged.contains(SensorClass::Gnss));
        assert!(merged.contains(SensorClass::Barometric));
        assert!(merged.contains(SensorClass::Inertial));
        assert_eq!(merged.class_count(), 3);
        assert_eq!(left.union(SourceComposition::empty()), left);
    }

    #[test]
    fn estimator_derived_is_flagged() {
        assert!(SourceComposition::of(SensorClass::FcState).is_estimator_derived());
        assert!(SourceComposition::of(SensorClass::NavigationSolution).is_estimator_derived());
        assert!(!SourceComposition::of(SensorClass::Gnss).is_estimator_derived());
    }
}

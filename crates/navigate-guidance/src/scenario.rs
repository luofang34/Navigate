//! Deterministic scenarios the derivation tests share: one clock domain,
//! one solution builder, one equatorial leg.
//!
//! Sharing them is what lets a test assert that both derivations judge
//! *the same* solution identically rather than two look-alike copies.

#![allow(clippy::expect_used, clippy::panic)]

use navigate_contract::{
    ClockDomainId, FaultDetection, GeodeticPosition, IntegrityAssessment, MonotonicNanos,
    NavigationSolution, NedVelocity, Redundancy, SensorClass, SolutionQuality, SolutionStamp,
    SourceComposition, SourceEpoch, SymmetricCov3, Waypoint, WrappingSequence,
};

/// The single clock domain of these tests: solutions are stamped on it
/// and `now` is read from it unless a test probes the mismatch.
pub(crate) const CLOCK: ClockDomainId = ClockDomainId::new(0);

/// A position from degrees, the unit these scenarios are written in.
pub(crate) fn deg(latitude_deg: f64, longitude_deg: f64, altitude_m: f64) -> GeodeticPosition {
    GeodeticPosition::new(
        latitude_deg.to_radians(),
        longitude_deg.to_radians(),
        altitude_m,
    )
}

/// A single-source GNSS solution at `position`, stamped `solved_at` on
/// [`CLOCK`] and carrying `quality`.
pub(crate) fn solution(
    quality: SolutionQuality,
    position: GeodeticPosition,
    solved_at: MonotonicNanos,
) -> NavigationSolution {
    let composition = SourceComposition::of(SensorClass::Gnss);
    NavigationSolution::new(
        SolutionStamp::new(
            SourceEpoch::new(1),
            WrappingSequence::new(42),
            solved_at,
            CLOCK,
        ),
        position,
        NedVelocity::new(0.0, 0.0, 0.0),
        SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
        SymmetricCov3::from_diagonal(0.1, 0.1, 0.1),
        IntegrityAssessment::new(
            quality,
            composition,
            Redundancy::None,
            1.0,
            1.5,
            FaultDetection::Unavailable,
        ),
        composition,
    )
}

/// Guidance time far enough past zero that a test can step back from it.
pub(crate) fn now() -> MonotonicNanos {
    MonotonicNanos::from_nanos(2_000_000_000)
}

/// Eastbound track along the equator: the geometry with the least
/// ambiguous bearing (exactly 90°) and cross-track sign (right is south).
pub(crate) fn equator_leg() -> (GeodeticPosition, Waypoint) {
    (
        deg(0.0, 0.0, 0.0),
        Waypoint::new("END".into(), deg(0.0, 1.0, 0.0)),
    )
}

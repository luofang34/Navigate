//! The orbital uncertainty separates the radial, along-track, and
//! cross-track axes, and falls back to the principal axes without an
//! orbit plane.

use nalgebra::{Matrix3, Vector3};
use navigate_contract::{
    ClockDomainId, MonotonicNanos, ObservationStamp, SensorClass, SolutionQuality,
    SourceComposition, SourceEpoch, SourceId, SpaceUncertainty, WrappingSequence,
};

use crate::config::FusionConfig;
use crate::provenance::{ConstrainedState, ProvenanceLedger};

use super::{derive, derive_space, space_uncertainty};

/// A circular low Earth orbit: 7000 km radius, 7546 m/s.
const LEO_RADIUS_M: f64 = 7_000_000.0;
const LEO_SPEED_MPS: f64 = 7_546.0;

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-9 * expected.abs().max(1.0)
}

fn rtn(uncertainty: SpaceUncertainty) -> (f64, f64, f64, f64) {
    match uncertainty {
        SpaceUncertainty::OrbitalRtn {
            radial_1sigma_m,
            along_track_1sigma_m,
            cross_track_1sigma_m,
            max_axis_1sigma_m,
        } => (
            radial_1sigma_m,
            along_track_1sigma_m,
            cross_track_1sigma_m,
            max_axis_1sigma_m,
        ),
        other => panic!("expected orbital axes, got {other:?}"),
    }
}

#[test]
fn a_circular_orbit_separates_radial_along_track_and_cross_track() {
    // Radial along x, along-track along y, cross-track along z.
    let position = Vector3::new(LEO_RADIUS_M, 0.0, 0.0);
    let velocity = Vector3::new(0.0, LEO_SPEED_MPS, 0.0);
    let cov = Matrix3::from_diagonal(&Vector3::new(4.0, 400.0, 25.0));
    let (radial, along, cross, max_axis) = rtn(space_uncertainty(position, velocity, cov));
    assert!(close(radial, 2.0) && close(along, 20.0) && close(cross, 5.0));
    assert!(close(max_axis, 20.0));
    assert!(radial != along && along != cross && radial != cross);
}

#[test]
fn the_orbital_axes_follow_the_state_and_not_the_frame_axes() {
    // The same orbit turned 90° about z: radial along y, along-track
    // along -x. The frame covariance turns with it.
    let position = Vector3::new(0.0, LEO_RADIUS_M, 0.0);
    let velocity = Vector3::new(-LEO_SPEED_MPS, 0.0, 0.0);
    let cov = Matrix3::from_diagonal(&Vector3::new(400.0, 4.0, 25.0));
    let (radial, along, cross, _) = rtn(space_uncertainty(position, velocity, cov));
    assert!(close(radial, 2.0) && close(along, 20.0) && close(cross, 5.0));
    // A retrograde orbit flips N, and the variances do not change.
    let (radial, along, cross, _) = rtn(space_uncertainty(position, -velocity, cov));
    assert!(close(radial, 2.0) && close(along, 20.0) && close(cross, 5.0));
}

#[test]
fn a_correlated_covariance_turns_into_the_orbital_axes() {
    // An inclined orbit: r along x, v in the y-z plane at 45°.
    let position = Vector3::new(LEO_RADIUS_M, 0.0, 0.0);
    let s = core::f64::consts::FRAC_1_SQRT_2;
    let velocity = Vector3::new(0.0, s * LEO_SPEED_MPS, s * LEO_SPEED_MPS);
    // 100 m² along v, 9 m² across the plane, 1 m² radial.
    let along = Vector3::new(0.0, s, s);
    let normal = Vector3::new(0.0, -s, s);
    let cov = Matrix3::from_diagonal(&Vector3::new(1.0, 0.0, 0.0))
        + along * along.transpose() * 100.0
        + normal * normal.transpose() * 9.0;
    let (radial, along_1sigma, cross, max_axis) = rtn(space_uncertainty(position, velocity, cov));
    assert!(close(radial, 1.0), "{radial}");
    assert!(close(along_1sigma, 10.0), "{along_1sigma}");
    assert!(close(cross, 3.0), "{cross}");
    assert!(close(max_axis, 10.0));
}

#[test]
fn a_hover_and_a_radial_descent_fall_back_to_the_principal_axes() {
    let position = Vector3::new(1_737_400.0 + 100.0, 0.0, 0.0);
    let cov = Matrix3::from_diagonal(&Vector3::new(9.0, 36.0, 1.0));
    for velocity in [Vector3::zeros(), Vector3::new(-1_600.0, 0.0, 0.0)] {
        match space_uncertainty(position, velocity, cov) {
            SpaceUncertainty::PrincipalAxes {
                axes_1sigma_m,
                max_axis_1sigma_m,
            } => {
                assert!(close(axes_1sigma_m[0], 6.0));
                assert!(close(axes_1sigma_m[1], 3.0));
                assert!(close(axes_1sigma_m[2], 1.0));
                assert!(close(max_axis_1sigma_m, 6.0));
            }
            other => panic!("expected principal axes, got {other:?}"),
        }
    }
    // A position at the frame origin has no radial direction.
    let at_origin = space_uncertainty(Vector3::zeros(), Vector3::new(0.0, 1.0, 0.0), cov);
    assert!(matches!(at_origin, SpaceUncertainty::PrincipalAxes { .. }));
}

#[test]
fn a_non_finite_covariance_gives_a_non_finite_largest_axis() {
    let position = Vector3::new(LEO_RADIUS_M, 0.0, 0.0);
    let velocity = Vector3::new(0.0, LEO_SPEED_MPS, 0.0);
    let mut cov = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 1.0));
    cov[(0, 1)] = f64::NAN;
    assert!(
        !space_uncertainty(position, velocity, cov)
            .max_axis_1sigma_m()
            .is_finite()
    );
}

#[test]
fn a_gate_rejection_storm_degrades_good_quality_until_an_observation_admits() {
    let config = FusionConfig::default();
    let clock = ClockDomainId::new(1);
    let now = MonotonicNanos::from_nanos(1_000_000_000);
    let mut ledger = ProvenanceLedger::default();
    let first = ObservationStamp::new(
        SourceId::new(1),
        SourceEpoch::new(1),
        WrappingSequence::new(1),
        now,
        clock,
    );
    ledger.commit(
        &first,
        SourceComposition::of(SensorClass::Gnss),
        ConstrainedState::POSITION,
        now,
        config.assessment_window,
    );

    assert_eq!(
        derive(&ledger, now, 1.0, 1.0, &config).quality,
        SolutionQuality::Good
    );
    let leo_uncertainty = space_uncertainty(
        Vector3::new(LEO_RADIUS_M, 0.0, 0.0),
        Vector3::new(0.0, LEO_SPEED_MPS, 0.0),
        Matrix3::from_diagonal(&Vector3::new(1.0, 4.0, 1.0)),
    );
    assert_eq!(
        derive_space(&ledger, now, leo_uncertainty, &config).quality,
        SolutionQuality::Good
    );

    for _ in 0..config.max_consecutive_gate_rejections {
        ledger.record_gate_rejection(ConstrainedState::POSITION);
    }
    assert_eq!(
        derive(&ledger, now, 1.0, 1.0, &config).quality,
        SolutionQuality::Degraded
    );
    assert_eq!(
        derive_space(&ledger, now, leo_uncertainty, &config).quality,
        SolutionQuality::Degraded
    );

    // An admitted attitude update does NOT clear a position rejection storm.
    let attitude_time = MonotonicNanos::from_nanos(1_250_000_000);
    let attitude_admission = ObservationStamp::new(
        SourceId::new(2),
        SourceEpoch::new(1),
        WrappingSequence::new(1),
        attitude_time,
        clock,
    );
    ledger.commit(
        &attitude_admission,
        SourceComposition::of(SensorClass::Celestial),
        ConstrainedState::ATTITUDE,
        attitude_time,
        config.assessment_window,
    );
    assert_eq!(
        derive(&ledger, attitude_time, 1.0, 1.0, &config).quality,
        SolutionQuality::Degraded
    );

    let next_time = MonotonicNanos::from_nanos(1_500_000_000);
    let second = ObservationStamp::new(
        SourceId::new(1),
        SourceEpoch::new(1),
        WrappingSequence::new(2),
        next_time,
        clock,
    );
    ledger.commit(
        &second,
        SourceComposition::of(SensorClass::Gnss),
        ConstrainedState::POSITION,
        next_time,
        config.assessment_window,
    );
    assert_eq!(
        derive(&ledger, next_time, 1.0, 1.0, &config).quality,
        SolutionQuality::Good
    );
}

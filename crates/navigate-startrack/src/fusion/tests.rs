use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use navigate_contract::{
    ClockDomainId, MonotonicNanos, NedVelocity, SourceEpoch, SourceId, WrappingSequence,
};
use navigate_fusion::{InertialConfig, InertialNavigationFilter, IngestOutcome, PlanetaryBody};

use super::*;
use crate::synthetic::{pipeline, position};

const CLOCK: ClockDomainId = ClockDomainId::new(3);
const AT: MonotonicNanos = MonotonicNanos::from_nanos(1_000_000_000);
const ARCSEC: f64 = std::f64::consts::PI / (180.0 * 3600.0);

fn stamp(sequence: u32) -> ObservationStamp {
    ObservationStamp::new(
        SourceId::new(40),
        SourceEpoch::new(1),
        WrappingSequence::new(sequence),
        AT,
        CLOCK,
    )
}

fn earth_frame() -> FrameId {
    PlanetaryBody::EARTH_WGS84.ellipsoid.frame()
}

/// A surface filter at the true position with the body axes along the
/// local NED axes, which is the true attitude.
fn filter() -> InertialNavigationFilter {
    let mut filter = InertialNavigationFilter::new_surface(
        InertialConfig::default(),
        CLOCK,
        PlanetaryBody::EARTH_WGS84,
    );
    filter
        .initialize_surface(
            position(),
            NedVelocity::new(0.0, 0.0, 0.0),
            AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            AT,
            SourceComposition::of(SensorClass::Gnss),
        )
        .expect("initial state");
    filter
}

#[test]
fn line_of_sight_observations_carry_the_star_directions() {
    let run = pipeline(5.0, 9, &[]);
    let mount_variance = 1e-10;
    let observations = line_of_sight_observations(
        &run.solution,
        stamp(u32::MAX),
        earth_frame(),
        mount_variance,
    )
    .expect("observations");
    assert_eq!(observations.len(), run.solution.matches.len());
    for (i, (obs, star)) in observations.iter().zip(&run.solution.matches).enumerate() {
        assert_eq!(obs.stamp.sequence.get(), u32::MAX.wrapping_add(i as u32));
        assert_eq!(obs.stamp.acquired_at, AT);
        assert_eq!(obs.frame, Some(earth_frame()));
        assert_eq!(
            obs.composition,
            SourceComposition::of(SensorClass::Celestial)
        );
        let ObservationValue::LineOfSight {
            measured_body_unit,
            reference:
                LineOfSightReference::InertialDirection {
                    direction_in_frame_unit,
                },
            angular_variance_rad2,
        } = obs.value
        else {
            panic!("a line of sight with an inertial direction");
        };
        assert_eq!(
            Vector3::from(measured_body_unit),
            star.body_unit.normalize()
        );
        assert_eq!(Vector3::from(direction_in_frame_unit), star.ecef_unit);
        assert_eq!(
            angular_variance_rad2,
            star.direction_sigma_rad.powi(2) + mount_variance
        );
    }
    assert!(matches!(
        line_of_sight_observations(&run.solution, stamp(0), earth_frame(), -1e-12),
        Err(StarTrackError::Invalid { .. })
    ));
}

#[test]
fn a_surface_filter_accepts_the_star_lines_of_sight() {
    let run = pipeline(5.0, 9, &[]);
    let observations = line_of_sight_observations(&run.solution, stamp(1), earth_frame(), 0.0)
        .expect("observations");
    let mut filter = filter();
    for obs in &observations {
        assert_eq!(filter.ingest(obs, AT), IngestOutcome::Accepted);
    }
    assert_eq!(filter.rejections().sequence_not_admitted, 0);
}

#[test]
fn the_local_ned_attitude_fix_is_near_the_truth_and_accepted() {
    let run = pipeline(5.0, 9, &[]);
    let obs = attitude_fix_local_ned(&run.solution, position(), stamp(1), earth_frame());
    let ObservationValue::AttitudeFix {
        attitude,
        covariance,
    } = obs.value
    else {
        panic!("an attitude fix");
    };
    // The true body axes are the local NED axes.
    let q = UnitQuaternion::from_quaternion(Quaternion::new(
        attitude.w, attitude.x, attitude.y, attitude.z,
    ));
    assert!(q.angle() < 10.0 * ARCSEC, "{} arcsec", q.angle() / ARCSEC);
    let (xx, _, zz) = covariance.diagonal();
    assert!(zz > xx, "the roll about the boresight is the weakest axis");
    let mut filter = filter();
    assert_eq!(filter.ingest(&obs, AT), IngestOutcome::Accepted);
}

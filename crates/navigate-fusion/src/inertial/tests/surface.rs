//! Admission, publication, attitude updates, and the exact split of an
//! IMU interval on a body surface.

use nalgebra::UnitQuaternion;
use navigate_contract::{CartesianPosition, DurationNanos, SolutionQuality, SymmetricCov3};
use navigate_geodesy::{BodyEllipsoid, NedOffset};

use super::*;
use crate::inertial::local_level::LocalLevel;
use crate::inertial::models::vehicle_level;
use crate::inertial::nominal::{ATT, to_contract};
use crate::observation::{MeasurementKind, Observation, ObservationValue};
use crate::rejection::RejectionReason;
use navigate_astrodynamics::{SpaceEnvironment, SpaceGravityModel};

fn position_fix(sequence: u32, at_ms: u64, position: GeodeticPosition) -> Observation {
    Observation::new(
        stamp(1, sequence, at_ms),
        ObservationValue::PositionFix {
            position,
            covariance: SymmetricCov3::from_diagonal(4.0, 4.0, 9.0),
        },
        gnss(),
    )
}

fn rejection(outcome: IngestOutcome) -> RejectionReason {
    match outcome {
        IngestOutcome::Rejected(reason) => reason,
        IngestOutcome::Accepted => panic!("expected a rejection"),
    }
}

#[test]
fn refusals_before_initialization_and_without_imu_data() {
    let config = InertialConfig::default();
    let mut filter =
        InertialNavigationFilter::new_surface(config, CLOCK, PlanetaryBody::EARTH_WGS84);
    let rates = ([0.0, 0.0, -9.8], [0.0; 3]);
    assert_eq!(
        rejection(filter.ingest_imu(&imu(1, 0, rates, 0.01), t(0))),
        RejectionReason::NotInitialized
    );
    assert_eq!(
        rejection(filter.ingest(&position_fix(1, 0, earth_origin()), t(0))),
        RejectionReason::NotInitialized
    );
    assert!(filter.tick(t(0)).is_none());
    let mut filter = surface_filter(config, PlanetaryBody::EARTH_WGS84, earth_origin());
    // A fix at the state time needs no propagation.
    assert_accepted(filter.ingest(&position_fix(1, 0, earth_origin()), t(0)));
    assert_eq!(
        rejection(filter.ingest(&position_fix(2, 100, earth_origin()), t(100))),
        RejectionReason::NoInertialData
    );
    assert!(filter.tick(t(100)).is_none());
    assert_eq!(filter.rejections().no_inertial_data, 1);
    assert_eq!(filter.rejections().not_initialized, 0);
}

#[test]
fn each_regime_refuses_the_models_that_it_does_not_have() {
    let mut surface = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    );
    // The state is in the body-fixed frame, so a surface filter also
    // takes a Cartesian fix in that frame.
    let p = earth_body_fixed(&earth_origin());
    let cartesian = ObservationValue::CartesianPositionFix {
        position: CartesianPosition::new(p.x + 1.0, p.y - 1.0, p.z),
        covariance: SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
    };
    let obs = Observation::new(stamp(4, 1, 0), cartesian, gnss());
    assert_accepted(surface.ingest(&obs, t(0)));
    let pseudorange = ObservationValue::Pseudorange {
        satellite_ecef_m: [1.0e7, 2.0e7, 1.0e7],
        pseudorange_m: 2.1e7,
        variance_m2: 9.0,
    };
    let obs = Observation::new(stamp(4, 2, 0), pseudorange, gnss());
    assert_eq!(
        rejection(surface.ingest(&obs, t(0))),
        RejectionReason::UnsupportedMeasurement {
            kind: pseudorange.kind()
        }
    );
    let env = SpaceEnvironment::inertial(FrameId::EARTH_J2000, SpaceGravityModel::Zero);
    let mut space = InertialNavigationFilter::new_space(InertialConfig::default(), CLOCK, env);
    let obs = position_fix(1, 0, earth_origin()).with_frame(FrameId::EARTH_J2000);
    assert_eq!(
        rejection(space.ingest(&obs, t(0))),
        RejectionReason::UnsupportedMeasurement {
            kind: MeasurementKind::PositionFix
        }
    );
    assert_eq!(space.rejections().unsupported_measurement, 1);
    assert_eq!(surface.rejections().unsupported_measurement, 1);
}

#[test]
fn a_lunar_filter_requires_the_lunar_frame() {
    let origin = GeodeticPosition::new(-1.5, 0.4, 0.0);
    let mut filter = InertialNavigationFilter::new_surface(
        InertialConfig::default(),
        CLOCK,
        PlanetaryBody::MOON_ME,
    );
    filter
        .initialize_surface(
            origin,
            navigate_contract::NedVelocity::new(0.0, 0.0, 0.0),
            identity_attitude(),
            SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            t(0),
            gnss(),
        )
        .expect("plausible");
    let rates = rest_rates(&filter);
    assert_eq!(
        rejection(filter.ingest_imu(&imu(1, 10, rates, 0.01), t(10))),
        RejectionReason::MissingFrame {
            expected: FrameId::MOON_ME
        }
    );
    assert_eq!(
        rejection(filter.ingest_imu(&imu_in(1, 10, rates, 0.01, FrameId::MOON_PA), t(10))),
        RejectionReason::FrameMismatch {
            expected: FrameId::MOON_ME,
            got: FrameId::MOON_PA
        }
    );
    assert_accepted(filter.ingest_imu(&imu_in(1, 10, rates, 0.01, FrameId::MOON_ME), t(10)));
    let solution = filter.tick(t(10)).expect("initialized");
    assert_eq!(solution.ellipsoid, navigate_contract::Ellipsoid::MOON_ME);
}

#[test]
fn imu_samples_do_not_reset_the_observation_silence() {
    let mut filter = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    );
    let rates = rest_rates(&filter);
    for k in 1..=6_u32 {
        let at_ms = u64::from(k) * 1_000;
        assert_accepted(filter.ingest_imu(&imu(k, at_ms, rates, 1.0), t(at_ms)));
    }
    let solution = filter.tick(t(6_000)).expect("initialized");
    assert!(solution.integrity.horizontal_1sigma_m < 10.0);
    assert!(DurationNanos::from_millis(6_000) >= filter.config().fusion.quality_silence_unusable);
    assert_eq!(solution.integrity.quality, SolutionQuality::Unusable);
    assert!(solution.composition.contains(SensorClass::Inertial));
    assert!(solution.composition.contains(SensorClass::Gnss));
    assert!(solution.integrity.contributing.is_empty());
}

#[test]
fn the_published_attitude_is_in_the_local_axes_at_the_vehicle() {
    let mut filter = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    );
    let plane = navigate_geodesy::LocalTangentPlane::new(earth_origin(), BodyEllipsoid::WGS84)
        .expect("plausible");
    let vehicle = plane.from_ned(&NedOffset::new(50_000.0, 20_000.0, 0.0));
    let local = LocalLevel::at(vehicle);
    let q_local = UnitQuaternion::from_euler_angles(0.05, -0.02, 0.6);
    {
        let state = filter.core.as_mut().expect("initialized");
        state.nominal.p = earth_body_fixed(&vehicle);
        state.nominal.q = local.bf_to_ned_quaternion().inverse() * q_local;
    }
    let solution = filter.tick(t(0)).expect("initialized");
    let published = solution.attitude.expect("attitude");
    let expected = to_contract(&q_local);
    let dot = published.w * expected.w
        + published.x * expected.x
        + published.y * expected.y
        + published.z * expected.z;
    assert!(
        (dot.abs() - 1.0).abs() < 1e-12,
        "{published:?} vs {expected:?}"
    );
    assert!((solution.position.latitude_rad - vehicle.latitude_rad).abs() < 1e-12);
    assert!((solution.position.longitude_rad - vehicle.longitude_rad).abs() < 1e-12);
    assert!((solution.position.altitude_m - vehicle.altitude_m).abs() < 1e-6);
    // The body-fixed attitude is far from the local attitude.
    let raw = to_contract(&core(&filter).nominal.q);
    let raw_dot = raw.w * expected.w + raw.x * expected.x + raw.y * expected.y + raw.z * expected.z;
    assert!((raw_dot.abs() - 1.0).abs() > 1e-3);
    assert!(solution.attitude_cov.is_some());
}

/// The attitude of the nominal state in the local NED axes at the
/// vehicle.
fn local_attitude(filter: &InertialNavigationFilter) -> UnitQuaternion<f64> {
    let state = core(filter);
    vehicle_level(state, BodyEllipsoid::WGS84).bf_to_ned_quaternion() * state.nominal.q
}

fn attitude_fix(sequence: u32, at_ms: u64, q: UnitQuaternion<f64>) -> Observation {
    Observation::new(
        stamp(5, sequence, at_ms),
        ObservationValue::AttitudeFix {
            attitude: to_contract(&q),
            covariance: SymmetricCov3::from_diagonal(1e-8, 1e-8, 1e-8),
        },
        SourceComposition::of(SensorClass::Celestial),
    )
}

fn pose(sequence: u32, at_ms: u64, north_m: f64, attitude: UnitQuaternion<f64>) -> Observation {
    let surface_plane = navigate_geodesy::LocalTangentPlane::new(
        earth_origin(),
        navigate_geodesy::BodyEllipsoid::WGS84,
    )
    .expect("plausible");
    Observation::new(
        stamp(6, sequence, at_ms),
        ObservationValue::VisualPose {
            position: surface_plane.from_ned(&NedOffset::new(north_m, 0.0, 0.0)),
            position_covariance: SymmetricCov3::from_diagonal(4.0, 4.0, 4.0),
            attitude: to_contract(&attitude),
            attitude_covariance: SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
        },
        SourceComposition::of(SensorClass::VisualLandmark),
    )
}

#[test]
fn attitude_fixes_and_visual_poses_update_the_attitude() {
    let mut filter = InertialNavigationFilter::new_surface(
        InertialConfig::default(),
        CLOCK,
        PlanetaryBody::EARTH_WGS84,
    );
    filter
        .initialize_surface(
            earth_origin(),
            navigate_contract::NedVelocity::new(0.0, 0.0, 0.0),
            identity_attitude(),
            SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            SymmetricCov3::from_diagonal(1e-4, 1e-4, 1e-4),
            t(0),
            gnss(),
        )
        .expect("plausible");
    let rotated = UnitQuaternion::from_euler_angles(0.0, 0.0, 0.01);
    assert_accepted(filter.ingest(&attitude_fix(1, 0, rotated), t(0)));
    let error =
        crate::inertial::nominal::rotation_vector(&(local_attitude(&filter).inverse() * rotated));
    assert!(error.norm() < 1e-5, "{error}");
    assert!(core(&filter).p[(ATT + 2, ATT + 2)] < 1e-7);
    assert_accepted(filter.ingest(&pose(1, 0, 0.0, rotated), t(0)));
    let outlier = rejection(filter.ingest(&pose(2, 0, 1_000.0, rotated), t(0)));
    assert!(
        matches!(outlier, RejectionReason::InnovationGate { threshold, .. } if threshold == 22.46)
    );
    let stats = filter
        .innovation_stats(MeasurementKind::VisualPose)
        .expect("recorded");
    assert_eq!(
        (stats.admitted_count, stats.consecutive_gate_rejections),
        (1, 1)
    );
}

/// Runs a script of IMU samples and one fix and returns the core.
fn run_script(events: &[(bool, u64, u64, usize)]) -> crate::inertial::estimate::InertialCore {
    let config = InertialConfig {
        fusion: crate::config::FusionConfig {
            staleness_bound: DurationNanos::from_millis(5_000),
            ..InertialConfig::default().fusion
        },
        ..InertialConfig::default()
    };
    let mut filter = surface_filter(config, PlanetaryBody::EARTH_WGS84, earth_origin());
    let rest = rest_rates(&filter);
    let rates = [
        rest,
        (
            [rest.0[0] + 0.4, rest.0[1] - 0.2, rest.0[2]],
            [0.02, -0.01, 0.05],
        ),
        (
            [rest.0[0] - 0.3, rest.0[1], rest.0[2] + 0.1],
            [-0.03, 0.02, -0.04],
        ),
    ];
    let fix = surface_plane_fix();
    let mut sequence = 0_u32;
    for &(is_imu, at_ms, now_ms, rate_index) in events {
        sequence = sequence.wrapping_add(1);
        if is_imu {
            let previous = events
                .iter()
                .filter(|e| e.0 && e.1 < at_ms)
                .map(|e| e.1)
                .max()
                .unwrap_or(0);
            let dt_s = (at_ms - previous) as f64 * 1e-3;
            assert_accepted(
                filter.ingest_imu(&imu(sequence, at_ms, rates[rate_index], dt_s), t(now_ms)),
            );
        } else {
            assert_accepted(filter.ingest(&position_fix(1, at_ms, fix), t(now_ms)));
        }
    }
    *core(&filter)
}

fn surface_plane_fix() -> GeodeticPosition {
    let plane = navigate_geodesy::LocalTangentPlane::new(
        earth_origin(),
        navigate_geodesy::BodyEllipsoid::WGS84,
    )
    .expect("plausible");
    plane.from_ned(&NedOffset::new(3.0, -2.0, 1.0))
}

#[test]
fn a_delayed_fix_inside_an_imu_interval_splits_the_interval_exactly() {
    // The reference has the interval split at the fix time and the fix
    // in order.
    let reference = run_script(&[
        (true, 1_000, 1_000, 1),
        (true, 1_500, 1_500, 2),
        (false, 1_500, 1_500, 0),
        (true, 2_000, 2_000, 2),
        (true, 3_000, 3_000, 1),
    ]);
    let delayed = run_script(&[
        (true, 1_000, 1_000, 1),
        (true, 2_000, 2_000, 2),
        (true, 3_000, 3_000, 1),
        (false, 1_500, 3_000, 0),
    ]);
    assert!((reference.nominal.p - delayed.nominal.p).norm() < 1e-9);
    assert!((reference.nominal.v - delayed.nominal.v).norm() < 1e-12);
    assert!(reference.nominal.q.angle_to(&delayed.nominal.q) < 1e-12);
    assert!((reference.p - delayed.p).abs().max() < 1e-12);
    assert_eq!(reference.state_at, delayed.state_at);
    // A fix in order before its covering sample holds the older rates,
    // so it ends at a different state.
    let held = run_script(&[
        (true, 1_000, 1_000, 1),
        (false, 1_500, 1_500, 0),
        (true, 2_000, 2_000, 2),
        (true, 3_000, 3_000, 1),
    ]);
    assert!((held.nominal.p - reference.nominal.p).norm() > 1e-6);
}

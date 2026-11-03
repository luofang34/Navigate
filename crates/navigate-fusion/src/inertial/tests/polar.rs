//! The local axes at the poles, and a straight flight over the North
//! Pole with no change of frame and no anchor.

use std::vec::Vec;

use core::f64::consts::{FRAC_PI_2, PI};

use nalgebra::{Matrix3, Vector3};
use navigate_contract::{DurationNanos, NavigationSolution};
use navigate_geodesy::{
    BodyEllipsoid, BodyFixedPosition, LocalTangentPlane, body_fixed_to_geodetic,
};

use super::*;
use crate::inertial::local_level::{LocalLevel, quaternion_of};
use crate::inertial::ned_to_body_fixed_dcm;
use crate::inertial::nominal::{POS, to_contract};
use crate::observation::{Observation, ObservationValue};
use navigate_astrodynamics::SpaceEnvironment;

#[test]
fn the_local_axes_are_orthonormal_at_each_latitude_and_match_the_tangent_plane() {
    for lat_deg in [-90.0_f64, -89.999_999, -45.0, 0.0, 47.0, 89.999_999, 90.0] {
        for lon_deg in [-180.0_f64, -80.6, 0.0, 8.0, 135.0] {
            let r = ned_to_body_fixed_dcm(lat_deg.to_radians(), lon_deg.to_radians());
            let error = (r.transpose() * r - Matrix3::identity()).abs().max();
            assert!(error < 1e-15, "{lat_deg}, {lon_deg}: {error}");
            assert!((r.determinant() - 1.0).abs() < 1e-15);
        }
    }
    let north_pole = ned_to_body_fixed_dcm(FRAC_PI_2, 0.3);
    assert!((north_pole.column(2) - Vector3::new(0.0, 0.0, -1.0)).norm() < 1e-15);
    let south_pole = ned_to_body_fixed_dcm(-FRAC_PI_2, 0.3);
    assert!((south_pole.column(2) - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-15);
    let origin = earth_origin();
    let plane = LocalTangentPlane::new(origin, BodyEllipsoid::WGS84).expect("plausible");
    let rows = plane.body_fixed_to_ned_rows();
    let r = ned_to_body_fixed_dcm(origin.latitude_rad, origin.longitude_rad);
    let from_plane = Matrix3::from_fn(|i, j| rows[i][j]);
    assert!((r.transpose() - from_plane).abs().max() < 1e-15);
}

const ALTITUDE_M: f64 = 10_000.0;
/// The flight crosses the pole at this time.
const POLE_S: f64 = 40.0;
const END_S: f64 = 80.0;
const IMU_DT_S: f64 = 0.1;
/// Rate of the meridian angle in rad/s: 0.1° in 40 s, about 280 m/s.
const MERIDIAN_RATE: f64 = 0.1 / 40.0 * PI / 180.0;

/// The meridian angle at `time_s`. Above `π/2` the flight is on the far
/// side of the pole, at the longitude 180°.
fn meridian_angle(time_s: f64) -> f64 {
    FRAC_PI_2 + MERIDIAN_RATE * (time_s - POLE_S)
}

/// The true body-fixed position on the meridian plane of the longitude
/// 0°.
fn truth_position(time_s: f64) -> Vector3<f64> {
    let position = GeodeticPosition::new(meridian_angle(time_s), 0.0, ALTITUDE_M);
    earth_body_fixed(&position)
}

fn truth_velocity(time_s: f64) -> Vector3<f64> {
    let h = 0.01;
    (truth_position(time_s + h) - truth_position(time_s - h)) / (2.0 * h)
}

fn truth_acceleration(time_s: f64) -> Vector3<f64> {
    let h = 0.1;
    (truth_position(time_s + h) - 2.0 * truth_position(time_s) + truth_position(time_s - h))
        / (h * h)
}

/// The body axes along the track: `x` forward, `y` along the constant
/// east axis of the meridian, and `z` down along the normal.
fn truth_attitude(time_s: f64) -> Matrix3<f64> {
    ned_to_body_fixed_dcm(meridian_angle(time_s), 0.0)
}

/// The IMU sample of the interval that ends at `end_s`, from the truth
/// at the midpoint of the interval.
fn truth_imu(sequence: u32, end_s: f64) -> ImuSample {
    let mid_s = end_s - 0.5 * IMU_DT_S;
    let env = SpaceEnvironment::from_body(PlanetaryBody::EARTH_WGS84);
    let p = truth_position(mid_s);
    let v = truth_velocity(mid_s);
    let c = truth_attitude(mid_s);
    let omega = env.frame_rotation();
    let (g, _) = env.gravity(&p);
    let f = c.transpose() * (truth_acceleration(mid_s) - g + 2.0 * omega.cross(&v));
    let w = c.transpose() * omega + Vector3::new(0.0, -MERIDIAN_RATE, 0.0);
    let at_ms = (end_s * 1e3).round() as u64;
    imu(sequence, at_ms, (f.into(), w.into()), IMU_DT_S)
}

fn geodetic(p: &Vector3<f64>) -> GeodeticPosition {
    body_fixed_to_geodetic(&BodyFixedPosition::new(p.x, p.y, p.z), BodyEllipsoid::WGS84)
}

fn heading_rad(solution: &NavigationSolution) -> f64 {
    let attitude = solution.attitude.expect("attitude");
    let q = crate::inertial::nominal::to_unit_quaternion(&attitude);
    let forward = q * Vector3::x();
    forward.y.atan2(forward.x)
}

/// Starts at 89.9°N, flies over the pole, and ends at 89.9°N on the far
/// side, with a GNSS fix each second.
fn fly_over_the_pole() -> (InertialNavigationFilter, Vec<(f64, NavigationSolution)>) {
    let config = InertialConfig {
        fusion: crate::config::FusionConfig {
            staleness_bound: DurationNanos::from_millis(5_000),
            ..InertialConfig::default().fusion
        },
        ..InertialConfig::default()
    };
    let start = geodetic(&truth_position(0.0));
    let local = LocalLevel::at(start);
    let v_ned = local.bf_to_ned() * truth_velocity(0.0);
    let attitude = quaternion_of(&(local.bf_to_ned() * truth_attitude(0.0)));
    let mut filter =
        InertialNavigationFilter::new_surface(config, CLOCK, PlanetaryBody::EARTH_WGS84);
    filter
        .initialize_surface(
            start,
            NedVelocity::new(v_ned.x, v_ned.y, v_ned.z),
            to_contract(&attitude),
            SymmetricCov3::from_diagonal(4.0, 4.0, 9.0),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            t(0),
            gnss(),
        )
        .expect("plausible");
    let mut solutions = Vec::new();
    let steps = (END_S / IMU_DT_S).round() as u32;
    for k in 1..=steps {
        let end_s = f64::from(k) * IMU_DT_S;
        let at_ms = (end_s * 1e3).round() as u64;
        assert_accepted(filter.ingest_imu(&truth_imu(k, end_s), t(at_ms)));
        if k % 10 != 0 {
            continue;
        }
        let fix = Observation::new(
            stamp(1, k, at_ms),
            ObservationValue::PositionFix {
                position: geodetic(&truth_position(end_s)),
                covariance: SymmetricCov3::from_diagonal(4.0, 4.0, 9.0),
            },
            gnss(),
        );
        assert_accepted(filter.ingest(&fix, t(at_ms)));
        let solution = filter.tick(t(at_ms)).expect("a surface regime publishes");
        solutions.push((end_s, solution));
    }
    (filter, solutions)
}

#[test]
fn a_straight_flight_over_the_north_pole_has_no_singularity() {
    let (filter, solutions) = fly_over_the_pole();
    assert_eq!(filter.frame(), FrameId::EARTH_ITRF93);
    assert_eq!(
        *filter.rejections(),
        crate::rejection::RejectionCounters::default()
    );
    let state = core(&filter);
    let error = (state.nominal.p - truth_position(END_S)).norm();
    assert!(error < 0.1, "{error}");
    let attitude_error = state
        .nominal
        .q
        .angle_to(&quaternion_of(&truth_attitude(END_S)));
    assert!(attitude_error < 1e-6, "{attitude_error}");
    let p_pos = state.p.fixed_view::<3, 3>(POS, POS).into_owned();
    assert!(p_pos.iter().all(|v| v.is_finite()));
    let mut previous_sigma: Option<f64> = None;
    for (time_s, solution) in &solutions {
        let truth = geodetic(&truth_position(*time_s));
        let position = solution.position;
        assert!(position.is_plausible(), "{time_s}: {position:?}");
        assert!(
            (position.altitude_m - truth.altitude_m).abs() < 5.0,
            "{time_s}"
        );
        let sigma = solution.integrity.horizontal_1sigma_m;
        assert!(
            sigma.is_finite() && sigma > 0.1 && sigma < 5.0,
            "{time_s}: {sigma}"
        );
        if let Some(previous) = previous_sigma {
            assert!(
                (sigma / previous - 1.0).abs() < 0.3,
                "{time_s}: {sigma} vs {previous}"
            );
        }
        previous_sigma = Some(sigma);
        // The heading in the local axes is north before the pole and
        // south after it.
        let heading = heading_rad(solution);
        if *time_s < POLE_S - 5.0 {
            assert!(heading.abs() < 0.05, "{time_s}: {heading}");
        }
        if *time_s > POLE_S + 5.0 {
            assert!(heading.abs() > PI - 0.05, "{time_s}: {heading}");
        }
    }
    let (_, at_pole) = solutions
        .iter()
        .find(|(time_s, _)| (*time_s - POLE_S).abs() < 1e-9)
        .expect("a solution at the pole");
    assert!((at_pole.position.latitude_rad - FRAC_PI_2).abs() < 1e-5);
}

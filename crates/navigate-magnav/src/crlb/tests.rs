use std::vec;
use std::vec::Vec;

use nalgebra::{Rotation3, UnitQuaternion, Vector3};
use navigate_astrodynamics::{GravityField, PlanetaryBody, SpaceEnvironment};
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, NedVelocity, SymmetricCov3};
use navigate_fusion::RejectionReason;
use navigate_fusion::inertial::{InertialConfig, ned_to_body_fixed_dcm};
use navigate_geodesy::{BodyEllipsoid, LocalTangentPlane, NedOffset, geodetic_to_body_fixed};

use super::{CrlbTracker, ReferencePoint};

fn origin() -> GeodeticPosition {
    GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 1_000.0)
}

fn tracker_at(start: GeodeticPosition) -> CrlbTracker {
    CrlbTracker::new_surface(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        start,
        SymmetricCov3::from_diagonal(1e4, 1e4, 100.0),
        SymmetricCov3::from_diagonal(0.25, 0.25, 0.25),
        SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
    )
    .expect("plausible start")
}

/// Flight at 50 m/s along the north axis of the tangent plane at `start`,
/// with the body axes along the axes of that plane. Close to a pole the
/// flight crosses the pole, so the local axes turn under the vehicle.
fn reference_from(start: GeodeticPosition, time_s: f64) -> ReferencePoint {
    let plane = LocalTangentPlane::new(start, BodyEllipsoid::WGS84).expect("plausible");
    let position = plane.from_ned(&NedOffset::new(50.0 * time_s, 0.0, 0.0));
    let env = SpaceEnvironment::from_body(PlanetaryBody::EARTH_WGS84);
    let r = geodetic_to_body_fixed(&position, BodyEllipsoid::WGS84);
    let (g, _) = env.gravity(&Vector3::new(r.x_m, r.y_m, r.z_m));
    let body = ned_to_body_fixed_dcm(start.latitude_rad, start.longitude_rad);
    let local = ned_to_body_fixed_dcm(position.latitude_rad, position.longitude_rad);
    let v = body * Vector3::new(50.0, 0.0, 0.0);
    let v_ned = local.transpose() * v;
    let omega = env.frame_rotation();
    let f = body.transpose() * (-g + 2.0 * omega.cross(&v));
    let to_local = Rotation3::from_matrix_unchecked(local.transpose() * body);
    let q = UnitQuaternion::from_rotation_matrix(&to_local).into_inner();
    ReferencePoint {
        position,
        velocity: NedVelocity::new(v_ned.x, v_ned.y, v_ned.z),
        attitude: AttitudeQuaternion::new(q.w, q.i, q.j, q.k),
        specific_force_body_mps2: f.into(),
        angular_rate_body_rad_s: (body.transpose() * omega).into(),
    }
}

/// The horizontal bound after each second of a 120 s leg from `start`
/// with a map of the gradient magnitude `gradient_t_per_m`. The gradient
/// direction turns with the track over the map features.
fn leg_from(start: GeodeticPosition, gradient_t_per_m: f64) -> Vec<f64> {
    let mut bound = tracker_at(start);
    let mut drms = vec![bound.horizontal_drms_m()];
    for step in 0..120 {
        let time_s = f64::from(step);
        bound
            .propagate(&reference_from(start, time_s), 1.0)
            .expect("valid step");
        let angle = 0.3 * time_s;
        let g = [
            gradient_t_per_m * angle.cos(),
            gradient_t_per_m * angle.sin(),
            0.0,
        ];
        bound.update_map_gradient(g, 1e-18).expect("valid update");
        drms.push(bound.horizontal_drms_m());
    }
    drms
}

#[test]
fn a_gradient_rich_leg_reduces_the_bound_and_a_flat_leg_does_not() {
    let rich = leg_from(origin(), 3e-10);
    let flat = leg_from(origin(), 1e-14);
    for pair in rich.windows(2).take(60) {
        assert!(pair[1] <= pair[0] + 1e-9, "{pair:?}");
    }
    let (start, rich_end, flat_end) = (rich[0], rich[120], flat[120]);
    assert!(rich_end < 0.1 * start, "{rich_end} vs {start}");
    assert!(flat_end >= 0.99 * start, "{flat_end} vs {start}");
    assert!(rich_end < 0.1 * flat_end);
}

#[test]
fn the_bound_has_no_singularity_at_a_pole() {
    let pole = GeodeticPosition::new(89.999_f64.to_radians(), 0.0, 1_000.0);
    let rich = leg_from(pole, 3e-10);
    let near_origin = leg_from(origin(), 3e-10);
    assert!(rich.iter().all(|d| d.is_finite()));
    let (start, end) = (rich[0], rich[120]);
    assert!((start - near_origin[0]).abs() < 1e-9, "{start}");
    assert!(end < 0.1 * start, "{end} vs {start}");
    // The same leg near 47°N gives almost the same bound.
    let ratio = end / near_origin[120];
    assert!((ratio - 1.0).abs() < 0.05, "{end} vs {}", near_origin[120]);
}

#[test]
fn the_bound_reports_each_axis_and_refuses_implausible_inputs() {
    let mut bound = tracker_at(origin());
    let sigma = bound.position_1sigma_m();
    assert!((sigma[0] - 100.0).abs() < 1e-9 && (sigma[2] - 10.0).abs() < 1e-9);
    let attitude = bound.attitude_1sigma_rad();
    assert!((attitude[2] - 1e-5_f64.sqrt()).abs() < 1e-12);
    let before = bound;
    assert_eq!(
        bound.update_map_gradient([f64::NAN, 0.0, 0.0], 1e-18),
        Err(RejectionReason::NonFiniteValue)
    );
    assert_eq!(
        bound.update_map_gradient([1e-10, 0.0, 0.0], 0.0),
        Err(RejectionReason::ImplausibleCovariance)
    );
    assert_eq!(
        bound.propagate(&reference_from(origin(), 0.0), 0.0),
        Err(RejectionReason::NonFiniteValue)
    );
    let mut skewed = reference_from(origin(), 0.0);
    skewed.attitude = AttitudeQuaternion::new(2.0, 0.0, 0.0, 0.0);
    assert_eq!(
        bound.propagate(&skewed, 1.0),
        Err(RejectionReason::NonFiniteValue)
    );
    assert_eq!(bound, before);
}

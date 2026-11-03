use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, SymmetricCov3};
use navigate_geodesy::LocalTangentPlane;

use super::{CrlbTracker, ReferencePoint};
use crate::environment::{GravityField, PlanetaryBody, SurfaceFrame};
use crate::inertial::InertialConfig;
use crate::rejection::RejectionReason;

fn origin() -> GeodeticPosition {
    GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 1_000.0)
}

fn tracker() -> CrlbTracker {
    CrlbTracker::new_surface(
        InertialConfig::default(),
        PlanetaryBody::earth_wgs84(),
        origin(),
        SymmetricCov3::from_diagonal(1e4, 1e4, 100.0),
        SymmetricCov3::from_diagonal(0.25, 0.25, 0.25),
        SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
    )
    .expect("plausible start")
}

/// Straight and level flight north at 50 m/s with the body axes along
/// the plane axes.
fn reference(time_s: f64) -> ReferencePoint {
    let plane = LocalTangentPlane::new(origin()).expect("plausible");
    let frame = SurfaceFrame::new(plane, PlanetaryBody::earth_wgs84());
    let p = Vector3::new(50.0 * time_s, 0.0, 0.0);
    let v = Vector3::new(50.0, 0.0, 0.0);
    let (g, _) = frame.gravity(&p);
    let omega = frame.frame_rotation();
    let f = -g + 2.0 * omega.cross(&v);
    let q = UnitQuaternion::<f64>::identity();
    let raw = q.quaternion();
    ReferencePoint {
        position_plane_m: p.into(),
        velocity_plane_mps: v.into(),
        attitude: AttitudeQuaternion::new(raw.w, raw.i, raw.j, raw.k),
        specific_force_body_mps2: f.into(),
        angular_rate_body_rad_s: omega.into(),
    }
}

/// The horizontal bound after each second of a 120 s leg with a map of
/// the gradient magnitude `gradient_t_per_m`. The gradient direction
/// turns with the track over the map features.
fn leg(gradient_t_per_m: f64) -> Vec<f64> {
    let mut bound = tracker();
    let mut drms = vec![bound.horizontal_drms_m()];
    for step in 0..120 {
        let time_s = f64::from(step);
        bound
            .propagate(&reference(time_s), 1.0)
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
    let rich = leg(3e-10);
    let flat = leg(1e-14);
    for pair in rich.windows(2).take(60) {
        assert!(pair[1] <= pair[0] + 1e-9, "{pair:?}");
    }
    let (start, rich_end, flat_end) = (rich[0], rich[120], flat[120]);
    assert!(rich_end < 0.1 * start, "{rich_end} vs {start}");
    assert!(flat_end >= 0.99 * start, "{flat_end} vs {start}");
    assert!(rich_end < 0.1 * flat_end);
}

#[test]
fn the_bound_reports_each_axis_and_refuses_implausible_inputs() {
    let mut bound = tracker();
    let sigma = bound.position_1sigma_m();
    assert!((sigma[0] - 100.0).abs() < 1e-9 && (sigma[2] - 10.0).abs() < 1e-9);
    let attitude = bound.attitude_1sigma_rad();
    assert!((attitude[2] - 1e-5_f64.sqrt()).abs() < 1e-12);
    let before = bound.clone();
    assert_eq!(
        bound.update_map_gradient([f64::NAN, 0.0, 0.0], 1e-18),
        Err(RejectionReason::NonFiniteValue)
    );
    assert_eq!(
        bound.update_map_gradient([1e-10, 0.0, 0.0], 0.0),
        Err(RejectionReason::ImplausibleCovariance)
    );
    assert_eq!(
        bound.propagate(&reference(0.0), 0.0),
        Err(RejectionReason::NonFiniteValue)
    );
    assert_eq!(bound, before);
}

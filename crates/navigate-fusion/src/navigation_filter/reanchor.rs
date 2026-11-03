//! Tangent-plane re-anchoring for long-range flights.
//!
//! When the vehicle moves beyond `config.reanchor_distance_m` from the active
//! tangent-plane origin, this module moves the origin to the current position
//! estimate and rotates the velocity state and state covariance into the new
//! North-East-Down frame. Every plane in this module is on the ellipsoid
//! of the active plane (ADR-0013).

use nalgebra::{Matrix3, Vector3};
use navigate_contract::GeodeticPosition;
use navigate_geodesy::{LocalTangentPlane, NedOffset};

use super::Core;
use crate::config::FusionConfig;
use crate::filter::{Mat6, Vec6, rotation_between};

pub(crate) fn vehicle_to_plane(
    plane: &LocalTangentPlane,
    vehicle: GeodeticPosition,
    v: Vector3<f64>,
    r: Matrix3<f64>,
) -> (Vector3<f64>, Matrix3<f64>) {
    let Ok(local) = LocalTangentPlane::with_ellipsoid(vehicle, plane.ellipsoid()) else {
        return (v, r);
    };
    if local.origin() == plane.origin() {
        return (v, r);
    }
    let rot = rotation_between(&local, plane);
    let r_rot = rot * r * rot.transpose();
    (rot * v, (r_rot + r_rot.transpose()) * 0.5)
}

pub(crate) fn plane_to_vehicle(
    plane: &LocalTangentPlane,
    vehicle: GeodeticPosition,
    v: Vector3<f64>,
    pos_cov: Matrix3<f64>,
    vel_cov: Matrix3<f64>,
) -> (Vector3<f64>, Matrix3<f64>, Matrix3<f64>) {
    let Ok(local) = LocalTangentPlane::with_ellipsoid(vehicle, plane.ellipsoid()) else {
        return (v, pos_cov, vel_cov);
    };
    if local.origin() == plane.origin() {
        return (v, pos_cov, vel_cov);
    }
    let rot = rotation_between(plane, &local);
    let p_rot = rot * pos_cov * rot.transpose();
    let v_rot = rot * vel_cov * rot.transpose();
    (
        rot * v,
        (p_rot + p_rot.transpose()) * 0.5,
        (v_rot + v_rot.transpose()) * 0.5,
    )
}

pub(super) fn reanchor_if_needed(core: &mut Core, config: &FusionConfig) {
    let threshold_m = config.reanchor_distance_m;
    if !threshold_m.is_finite() || threshold_m <= 0.0 {
        return;
    }
    let x = &core.state.x;
    let horizontal_m = x[0].hypot(x[1]);
    if !horizontal_m.is_finite() || horizontal_m <= threshold_m {
        return;
    }
    let current = core.plane.from_ned(&NedOffset::new(x[0], x[1], x[2]));
    let Ok(new_plane) = LocalTangentPlane::with_ellipsoid(current, core.plane.ellipsoid()) else {
        return;
    };
    let r = rotation_between(&core.plane, &new_plane);
    let residual = new_plane.to_ned(&current);
    let v_new = r * nalgebra::Vector3::new(x[3], x[4], x[5]);
    let mut j = Mat6::zeros();
    for row in 0..3 {
        for col in 0..3 {
            j[(row, col)] = r[(row, col)];
            j[(row + 3, col + 3)] = r[(row, col)];
        }
    }
    let rotated_p = j * core.state.p * j.transpose();
    let symmetric_p = (rotated_p + rotated_p.transpose()) * 0.5;
    core.plane = new_plane;
    core.state.x = Vec6::new(
        residual.north_m,
        residual.east_m,
        residual.down_m,
        v_new[0],
        v_new[1],
        v_new[2],
    );
    core.state.p = symmetric_p;
}

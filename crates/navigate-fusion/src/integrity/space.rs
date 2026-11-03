//! The position uncertainty of a space solution in orbital axes
//! (ADR-0015).

use nalgebra::{Matrix3, Vector3};
use navigate_contract::SpaceUncertainty;

/// A position closer to the frame origin than this has no radial
/// direction to judge.
const MIN_RADIUS_M: f64 = 1.0;

/// Below this specific angular momentum `|r × v|` the orbit plane is not
/// defined, for example in a hover or in a radial descent.
const MIN_ANGULAR_MOMENTUM_M2_S: f64 = 1e-3;

/// Below this fraction of `|r|·|v|`, the direction of `r × v` is the
/// rounding error of the cross product, not a plane.
const MIN_RELATIVE_ANGULAR_MOMENTUM: f64 = 1e-12;

/// The 1-sigma position uncertainty of `position_cov` for the state
/// `(position, velocity)`, all in the axes of one frame.
///
/// When the orbit plane is defined, the covariance turns into the
/// radial, along-track, and cross-track axes:
/// `R = r/|r|`, `N = (r × v)/|r × v|`, `T = N × R`, and
/// `P_RTN = M·P·Mᵀ` with the rows of `M` equal to `R`, `T`, and `N`.
/// Otherwise the result is the principal 1-sigma values, largest first.
/// Each form also gives the square root of the largest eigenvalue. A
/// non-finite input gives non-finite values, which no quality bound
/// admits.
pub(crate) fn space_uncertainty(
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    position_cov: Matrix3<f64>,
) -> SpaceUncertainty {
    let cov = (position_cov + position_cov.transpose()) * 0.5;
    // The eigenvalue iteration does not converge on a non-finite matrix.
    if !cov.iter().all(|value| value.is_finite()) {
        return SpaceUncertainty::PrincipalAxes {
            axes_1sigma_m: [f64::NAN; 3],
            max_axis_1sigma_m: f64::NAN,
        };
    }
    let mut eigenvalues: [f64; 3] = cov.symmetric_eigenvalues().into();
    eigenvalues.sort_by(|a, b| b.total_cmp(a));
    let axes_1sigma_m = eigenvalues.map(sigma);
    let max_axis_1sigma_m = axes_1sigma_m[0];
    let Some(rows) = rtn_rows(position, velocity) else {
        return SpaceUncertainty::PrincipalAxes {
            axes_1sigma_m,
            max_axis_1sigma_m,
        };
    };
    let rtn = rows * cov * rows.transpose();
    SpaceUncertainty::OrbitalRtn {
        radial_1sigma_m: sigma(rtn[(0, 0)]),
        along_track_1sigma_m: sigma(rtn[(1, 1)]),
        cross_track_1sigma_m: sigma(rtn[(2, 2)]),
        max_axis_1sigma_m,
    }
}

/// The rows `R`, `T`, `N` of the rotation into orbital axes, or `None`
/// when the state does not define an orbit plane.
fn rtn_rows(position: Vector3<f64>, velocity: Vector3<f64>) -> Option<Matrix3<f64>> {
    let radius_m = position.norm();
    let momentum = position.cross(&velocity);
    let momentum_norm = momentum.norm();
    let defined = radius_m >= MIN_RADIUS_M
        && momentum_norm > MIN_ANGULAR_MOMENTUM_M2_S
        && momentum_norm > MIN_RELATIVE_ANGULAR_MOMENTUM * radius_m * velocity.norm();
    if !defined {
        return None;
    }
    let radial = position / radius_m;
    let cross_track = momentum / momentum_norm;
    let along_track = cross_track.cross(&radial);
    Some(Matrix3::from_rows(&[
        radial.transpose(),
        along_track.transpose(),
        cross_track.transpose(),
    ]))
}

/// The 1-sigma of a variance. Rounding can make a variance a little
/// negative, which gives 0.
fn sigma(variance: f64) -> f64 {
    variance.max(0.0).sqrt()
}

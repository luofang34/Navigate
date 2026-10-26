//! IAU 1976 precession and mean obliquity (Lieske et al. 1977).
//!
//! The angles are accurate to better than 0.1 arcseconds within two
//! centuries of J2000. `t` is TDB Julian centuries from J2000.

use nalgebra::Matrix3;

const ARCSECOND: f64 = std::f64::consts::PI / 648_000.0;

/// Mean obliquity of the ecliptic of date, in radians.
pub(crate) fn mean_obliquity(t: f64) -> f64 {
    (84_381.448 + t * (-46.815 + t * (-0.000_59 + t * 0.001_813))) * ARCSECOND
}

/// Rotate mean ecliptic-of-date axes to mean equator-of-date axes.
pub(crate) fn ecliptic_to_equator(t: f64) -> Matrix3<f64> {
    frame_x(-mean_obliquity(t))
}

/// Rotate J2000 mean equator axes to mean equator-of-date axes.
pub(crate) fn j2000_to_mean_of_date(t: f64) -> Matrix3<f64> {
    let zeta = t * (2_306.218_1 + t * (0.301_88 + t * 0.017_998)) * ARCSECOND;
    let z = t * (2_306.218_1 + t * (1.094_68 + t * 0.018_203)) * ARCSECOND;
    let theta = t * (2_004.310_9 + t * (-0.426_65 - t * 0.041_833)) * ARCSECOND;
    frame_z(-z) * frame_y(theta) * frame_z(-zeta)
}

/// A rotation of the coordinate axes by `angle` about x.
pub(crate) fn frame_x(angle: f64) -> Matrix3<f64> {
    let (sin, cos) = angle.sin_cos();
    Matrix3::new(1.0, 0.0, 0.0, 0.0, cos, sin, 0.0, -sin, cos)
}

/// A rotation of the coordinate axes by `angle` about z.
pub(crate) fn frame_z(angle: f64) -> Matrix3<f64> {
    let (sin, cos) = angle.sin_cos();
    Matrix3::new(cos, sin, 0.0, -sin, cos, 0.0, 0.0, 0.0, 1.0)
}

fn frame_y(angle: f64) -> Matrix3<f64> {
    let (sin, cos) = angle.sin_cos();
    Matrix3::new(cos, 0.0, -sin, 0.0, 1.0, 0.0, sin, 0.0, cos)
}

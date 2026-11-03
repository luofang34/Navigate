use std::vec::Vec;

use nalgebra::{Matrix3, Rotation3, Vector3};

use super::{
    HarmonicFieldError, HarmonicNormalization, MAX_INLINE_HARMONIC_DEGREE, SphericalHarmonicField,
    harmonic_coefficient_count, harmonic_index,
};
use crate::PlanetaryBody;

const EARTH: PlanetaryBody = PlanetaryBody::EARTH_WGS84;

fn radius() -> f64 {
    EARTH.ellipsoid.semi_major_axis_m()
}

/// The point mass and `J2` as unnormalized or normalized coefficients.
fn j2_coefficients(normalization: HarmonicNormalization) -> ([f64; 6], [f64; 6]) {
    let mut c = [0.0; 6];
    c[0] = 1.0;
    c[harmonic_index(2, 0)] = match normalization {
        HarmonicNormalization::Unnormalized => -EARTH.j2,
        HarmonicNormalization::FullyNormalized => -EARTH.j2 / 5.0_f64.sqrt(),
    };
    (c, [0.0; 6])
}

/// A degree-8 field with normalized coefficients of the size of a real
/// planet. The values are fixed so that each run is the same.
fn degree_eight() -> (Vec<f64>, Vec<f64>) {
    let count = harmonic_coefficient_count(8);
    let mut c = std::vec![0.0; count];
    let mut s = std::vec![0.0; count];
    for n in 2..=8 {
        for m in 0..=n {
            let phase = 1.3 * n as f64 + 0.7 * m as f64 + 0.1;
            c[harmonic_index(n, m)] = 2e-6 * phase.sin();
            if m > 0 {
                s[harmonic_index(n, m)] = 1.5e-6 * (0.9 * phase).cos();
            }
        }
    }
    c[0] = 1.0;
    c[harmonic_index(2, 0)] = -4.841_65e-4;
    (c, s)
}

fn test_points() -> [Vector3<f64>; 6] {
    let r = radius();
    [
        Vector3::new(0.0, 0.0, r),
        Vector3::new(0.0, 0.0, -r),
        Vector3::new(r, 0.0, 0.0),
        Vector3::new(0.0, -r - 400e3, 0.0),
        Vector3::new(3.1e6, -4.2e6, 3.9e6),
        Vector3::new(-1.0e3, 2.0e3, 7.0e6),
    ]
}

fn relative(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn degree_two_matches_the_central_body_at_the_poles_and_the_equator() {
    let central = EARTH.central_gravity();
    for normalization in [
        HarmonicNormalization::Unnormalized,
        HarmonicNormalization::FullyNormalized,
    ] {
        let (c, s) = j2_coefficients(normalization);
        let field =
            SphericalHarmonicField::new(EARTH.gm_m3_per_s2, radius(), 2, normalization, &c, &s)
                .expect("plausible");
        for r in test_points() {
            let (g, gradient) = field.evaluate(&r);
            let expected_g = central.acceleration(&r);
            let expected_gradient = central.gradient(&r);
            let g_error = (g - expected_g).norm() / expected_g.norm();
            let gradient_error =
                (gradient - expected_gradient).abs().max() / expected_gradient.abs().max();
            assert!(g_error < 1e-13, "{normalization:?} {r}: {g_error}");
            assert!(
                gradient_error < 1e-13,
                "{normalization:?} {r}: {gradient_error}"
            );
        }
    }
}

#[test]
fn degree_eight_at_the_exact_poles_matches_central_differences() {
    let (c, s) = degree_eight();
    let field = SphericalHarmonicField::new(
        EARTH.gm_m3_per_s2,
        radius(),
        8,
        HarmonicNormalization::FullyNormalized,
        &c,
        &s,
    )
    .expect("plausible");
    let r = radius();
    for pole in [Vector3::new(0.0, 0.0, r), Vector3::new(0.0, 0.0, -r)] {
        let (g, gradient) = field.evaluate(&pole);
        assert!(g.iter().chain(gradient.iter()).all(|v| v.is_finite()));
        let step = 50.0;
        let mut numeric = Matrix3::zeros();
        for axis in 0..3 {
            let mut d = Vector3::zeros();
            d[axis] = step;
            let column =
                (field.evaluate(&(pole + d)).0 - field.evaluate(&(pole - d)).0) / (2.0 * step);
            numeric.set_column(axis, &column);
        }
        let error = (gradient - numeric).abs().max() / gradient.abs().max();
        assert!(error < 1e-8, "{pole}: {error}\n{gradient}\n{numeric}");
        // The tesseral terms give a horizontal acceleration at the pole.
        assert!(g.x.abs() > 1e-6 || g.y.abs() > 1e-6, "{g}");
    }
}

#[test]
fn the_gradient_is_symmetric_and_free_of_trace_and_the_acceleration_is_the_potential_slope() {
    let (c, s) = degree_eight();
    let field = SphericalHarmonicField::new(
        EARTH.gm_m3_per_s2,
        radius(),
        8,
        HarmonicNormalization::FullyNormalized,
        &c,
        &s,
    )
    .expect("plausible");
    for r in test_points() {
        let (g, gradient) = field.evaluate(&r);
        let scale = gradient.abs().max();
        assert!((gradient - gradient.transpose()).abs().max() < 1e-12 * scale);
        assert!(gradient.trace().abs() < 1e-12 * scale, "{r}");
        let step = 10.0;
        for axis in 0..3 {
            let mut d = Vector3::zeros();
            d[axis] = step;
            let slope = (field.potential(&(r + d)) - field.potential(&(r - d))) / (2.0 * step);
            assert!((slope - g[axis]).abs() < 1e-7 * g.norm(), "{r} {axis}");
        }
    }
}

#[test]
fn an_orientation_turns_the_field_into_the_frame_axes() {
    let (c, s) = degree_eight();
    let body = SphericalHarmonicField::new(
        EARTH.gm_m3_per_s2,
        radius(),
        8,
        HarmonicNormalization::FullyNormalized,
        &c,
        &s,
    )
    .expect("plausible");
    let rotation = Rotation3::from_euler_angles(0.3, -0.2, 1.1).into_inner();
    let framed = body.with_orientation(rotation).expect("orthonormal");
    let r_frame = Vector3::new(2.0e6, -5.5e6, 3.0e6);
    let (g_body, gradient_body) = body.evaluate(&(rotation * r_frame));
    let (g_frame, gradient_frame) = framed.evaluate(&r_frame);
    assert!((rotation.transpose() * g_body - g_frame).norm() < 1e-14 * g_body.norm());
    let expected = rotation.transpose() * gradient_body * rotation;
    assert!((expected - gradient_frame).abs().max() < 1e-14 * expected.abs().max());
    assert_eq!(
        body.with_orientation(Matrix3::identity() * 2.0),
        Err(HarmonicFieldError::NotPlausible)
    );
}

#[test]
fn implausible_fields_are_refused() {
    let (c, s) = j2_coefficients(HarmonicNormalization::Unnormalized);
    let gm = EARTH.gm_m3_per_s2;
    let unnormalized = HarmonicNormalization::Unnormalized;
    let too_high = MAX_INLINE_HARMONIC_DEGREE + 1;
    assert_eq!(
        SphericalHarmonicField::new(gm, radius(), too_high, unnormalized, &c, &s),
        Err(HarmonicFieldError::DegreeTooHigh {
            degree: too_high,
            max: MAX_INLINE_HARMONIC_DEGREE
        })
    );
    assert_eq!(
        SphericalHarmonicField::new(gm, radius(), 3, unnormalized, &c, &s),
        Err(HarmonicFieldError::TooFewCoefficients {
            needed: 10,
            c_len: 6,
            s_len: 6
        })
    );
    let mut bad = c;
    bad[3] = f64::NAN;
    assert_eq!(
        SphericalHarmonicField::new(gm, radius(), 2, unnormalized, &bad, &s),
        Err(HarmonicFieldError::NotPlausible)
    );
    assert_eq!(
        SphericalHarmonicField::new(-gm, radius(), 2, unnormalized, &c, &s),
        Err(HarmonicFieldError::NotPlausible)
    );
    let field =
        SphericalHarmonicField::new(gm, radius(), 2, unnormalized, &c, &s).expect("plausible");
    assert_eq!(field.evaluate(&Vector3::zeros()).0, Vector3::zeros());
    assert_eq!(field.degree(), 2);
}

#[test]
fn the_full_inline_degree_is_finite_on_the_surface() {
    let count = harmonic_coefficient_count(MAX_INLINE_HARMONIC_DEGREE);
    let mut c = std::vec![1e-7; count];
    let s = std::vec![-1e-7; count];
    c[0] = 1.0;
    let field = SphericalHarmonicField::new(
        EARTH.gm_m3_per_s2,
        radius(),
        MAX_INLINE_HARMONIC_DEGREE,
        HarmonicNormalization::FullyNormalized,
        &c,
        &s,
    )
    .expect("plausible");
    for r in test_points() {
        let (g, gradient) = field.evaluate(&r);
        assert!(g.iter().chain(gradient.iter()).all(|v| v.is_finite()));
        assert!(relative(g.norm(), EARTH.gm_m3_per_s2 / r.norm_squared()) < 1e-2);
        assert!(gradient.trace().abs() < 1e-10 * gradient.abs().max());
    }
}

use super::*;

#[cfg(test)]
const MAP: MapSigmas = MapSigmas {
    horizontal_m: 3.0,
    height_m: 4.0,
};

#[test]
fn a_nadir_view_adds_variances_and_not_sigmas() {
    let view = ViewGeometry {
        off_nadir_rad: 0.0,
        ground_direction: [0.0, 0.0],
        range_m: 1_000.0,
    };
    // Calibration: 1 mrad at 1 km is 1 m.
    let p = external_covariance_ned(&MAP, 1e-3, &view);
    assert!((p[(0, 0)] - 10.0).abs() < 1e-12, "3² + 1², not (3 + 1)²");
    assert!((p[(1, 1)] - 10.0).abs() < 1e-12);
    assert!((p[(2, 2)] - 17.0).abs() < 1e-12, "4² + 1²");
    assert!(p[(0, 1)].abs() < 1e-12 && p[(0, 2)].abs() < 1e-12);
}

#[test]
fn an_oblique_view_takes_the_height_error_along_its_ground_direction() {
    let view = ViewGeometry {
        off_nadir_rad: core::f64::consts::FRAC_PI_4,
        // The camera looks north-east.
        ground_direction: [
            core::f64::consts::FRAC_1_SQRT_2,
            core::f64::consts::FRAC_1_SQRT_2,
        ],
        range_m: 0.0,
    };
    let p = external_covariance_ned(&MAP, 0.0, &view);
    // tan 45° = 1: the 4 m height error is 4 m along the view.
    let along = Vector3::new(1.0, 1.0, 0.0).normalize();
    let across = Vector3::new(1.0, -1.0, 0.0).normalize();
    assert!(((along.transpose() * p * along)[(0, 0)] - 25.0).abs() < 1e-9);
    assert!(((across.transpose() * p * across)[(0, 0)] - 9.0).abs() < 1e-9);
    assert!((p[(2, 2)] - 16.0).abs() < 1e-12);
    assert!(p.symmetric_eigen().eigenvalues.iter().all(|v| *v > 0.0));
}

#[test]
fn a_view_at_the_horizon_keeps_a_finite_covariance() {
    let view = ViewGeometry {
        off_nadir_rad: core::f64::consts::FRAC_PI_2,
        ground_direction: [1.0, 0.0],
        range_m: 5_000.0,
    };
    let p = external_covariance_ned(&MAP, 1e-4, &view);
    assert!(p.iter().all(|v| v.is_finite()));
    assert!(p[(0, 0)] > 1_000.0, "{}", p[(0, 0)]);
}

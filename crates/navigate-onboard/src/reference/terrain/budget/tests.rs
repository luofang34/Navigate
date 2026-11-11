use super::*;

#[cfg(test)]
const MAP: MapSigmas = MapSigmas {
    horizontal_m: 3.0,
    height_m: 4.0,
    elevation_m: 0.0,
    elevation_length_m: 300.0,
};

#[test]
fn a_nadir_view_adds_variances_and_not_sigmas() {
    let view = ViewGeometry {
        off_nadir_rad: 0.0,
        ground_direction: [0.0, 0.0],
        range_m: 1_000.0,
        field_ratio: 1.0,
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
        field_ratio: 1.0,
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
        field_ratio: 1.0,
    };
    let p = external_covariance_ned(&MAP, 1e-4, &view);
    assert!(p.iter().all(|v| v.is_finite()));
    assert!(p[(0, 0)] > 1_000.0, "{}", p[(0, 0)]);
}

/// A view straight down at `range_m` with a narrower extent of `field` for
/// each metre of range.
#[cfg(test)]
fn nadir(range_m: f64, field: f64) -> ViewGeometry {
    ViewGeometry {
        off_nadir_rad: 0.0,
        ground_direction: [0.0, 0.0],
        range_m,
        field_ratio: field,
    }
}

#[test]
fn the_tilt_term_grows_with_the_height_and_falls_once_the_footprint_averages_the_error() {
    let map = MapSigmas {
        horizontal_m: 0.0,
        height_m: 50.0,
        elevation_m: 5.0,
        elevation_length_m: 300.0,
    };
    // A view 0.8 of the range across. At 200 m and 400 m above the ground
    // the footprint, 160 m and 320 m, is narrower than 1.6 L: the slope is
    // √2 · 5 / 300, and the term doubles with the height.
    let local = core::f64::consts::SQRT_2 * 5.0 / 300.0;
    for range_m in [200.0, 400.0] {
        let tilt = tilt_sigma_m(&map, &nadir(range_m, 0.8));
        assert!((tilt - range_m * local).abs() < 1e-9, "{tilt}");
    }
    // At 800 m and 1.6 km the footprint, 640 m and 1280 m, averages the
    // error: the slope is √12 · 5 · 300 / W², and the term halves when
    // the height doubles.
    let fitted = |w: f64| 12.0_f64.sqrt() * 5.0 * 300.0 / (w * w);
    let high = tilt_sigma_m(&map, &nadir(1_600.0, 0.8));
    let middle = tilt_sigma_m(&map, &nadir(800.0, 0.8));
    assert!((middle - 800.0 * fitted(640.0)).abs() < 1e-9, "{middle}");
    assert!((high - middle / 2.0).abs() < 1e-9, "{high} {middle}");
    // The geoid bound in `height_m` has no tilt; only `elevation_m` does.
    let p = external_covariance_ned(&map, 0.0, &nadir(400.0, 0.8));
    let tilt = 400.0 * local;
    assert!((p[(0, 0)] - tilt * tilt).abs() < 1e-9);
    assert!((p[(1, 1)] - tilt * tilt).abs() < 1e-9);
    assert!((p[(2, 2)] - 2_500.0).abs() < 1e-9);
}

#[test]
fn an_oblique_view_takes_the_tilt_at_its_height_above_the_ground() {
    let map = MapSigmas {
        horizontal_m: 0.0,
        height_m: 0.0,
        elevation_m: 4.0,
        elevation_length_m: 1_000.0,
    };
    let oblique = ViewGeometry {
        off_nadir_rad: core::f64::consts::FRAC_PI_3,
        ground_direction: [1.0, 0.0],
        range_m: 800.0,
        field_ratio: 0.5,
    };
    // 800 m of range at 60° off the nadir is 400 m above the ground, and
    // the 400 m footprint is narrower than the length.
    let tilt = tilt_sigma_m(&map, &oblique);
    assert!((tilt - 400.0 * core::f64::consts::SQRT_2 * 4.0 / 1_000.0).abs() < 1e-9);
}

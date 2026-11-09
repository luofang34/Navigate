use nalgebra::{UnitQuaternion, Vector3};

use super::*;
use crate::synthetic::{
    CATALOGUE_MAGNITUDE, RENDER_MAGNITUDE, catalogue, imager, mount, pair_table, pipeline, prior,
    render, render_with_shot_noise, run, run_with, streak_sources, truth,
};
use crate::{DetectConfig, IdentifyMode};

const ARCSEC: f64 = std::f64::consts::PI / (180.0 * 3600.0);

fn pair(body: Vector3<f64>, attitude: &UnitQuaternion<f64>) -> StarMatch {
    let body_unit = body.normalize();
    StarMatch {
        centroid_index: 0,
        hr: 1,
        body_unit,
        ecef_unit: attitude * body_unit,
        separation_rad: 0.0,
        direction_sigma_rad: ARCSEC,
    }
}

#[test]
fn davenport_recovers_an_exact_rotation() {
    let attitude = UnitQuaternion::from_euler_angles(0.3, -1.1, 2.4);
    let matches = [
        pair(Vector3::new(0.1, 0.2, -1.0), &attitude),
        pair(Vector3::new(-0.15, 0.05, -1.0), &attitude),
        pair(Vector3::new(0.02, -0.17, -1.0), &attitude),
    ];
    let solved = davenport(&matches).expect("solve");
    assert!(solved.angle_to(&attitude) < 1e-12);
    // One direction does not define the rotation about itself.
    let single = [matches[0], matches[0], matches[0]];
    assert!(matches!(
        davenport(&single),
        Err(StarTrackError::Degenerate)
    ));
}

#[test]
fn the_solved_attitude_is_within_a_few_arcseconds() {
    let run = pipeline(5.0, 5, &[]);
    let solution = &run.solution;
    let error = (truth().inverse() * solution.ecef_from_body).scaled_axis();
    let angle = error.norm() / ARCSEC;
    assert!(solution.matches.len() >= 10, "{}", solution.matches.len());
    assert!(solution.matches.len() >= run.matches.len());
    assert!(angle < 10.0, "attitude error {angle} arcsec");
    // The error is inside the stated covariance.
    let information = solution
        .covariance_body_rad2
        .try_inverse()
        .expect("inverse");
    let nees = error.dot(&(information * error));
    assert!(nees < 16.0, "normalized error {nees}");
    assert!(solution.rms_residual_rad < 0.2 / 2904.0 * 3.0);
}

#[test]
fn fewer_than_three_stars_is_an_error() {
    let run = pipeline(5.0, 5, &[]);
    let two = &run.solution.matches[..2];
    let result = solve_attitude(
        two,
        &run.centroids,
        &imager(),
        &mount(),
        &catalogue(CATALOGUE_MAGNITUDE),
        &SolveConfig::default(),
    );
    assert!(matches!(
        result,
        Err(StarTrackError::TooFewStars {
            found: 2,
            needed: 3
        })
    ));
    // Too few stars after the second match is also an error.
    let few = &run.centroids[..2];
    let result = solve_attitude(
        &run.solution.matches[..3],
        few,
        &imager(),
        &mount(),
        &catalogue(CATALOGUE_MAGNITUDE),
        &SolveConfig::default(),
    );
    assert!(matches!(
        result,
        Err(StarTrackError::TooFewStars { found, .. }) if found < 3
    ));
}

#[test]
fn an_outlier_is_removed() {
    let run = pipeline(5.0, 5, &[]);
    let mut centroids = run.centroids.clone();
    // Move one matched centroid by 1 pixel, many of its own deviations: a
    // wide second gate keeps it, and the residual test removes it.
    let moved = run.solution.matches[0].centroid_index;
    centroids[moved].pixel.x += 1.0;
    let config = SolveConfig {
        rematch_gate_sigmas: 400.0,
        ..SolveConfig::default()
    };
    let solution = solve_attitude(
        &run.matches,
        &centroids,
        &imager(),
        &mount(),
        &run.catalogue,
        &config,
    )
    .expect("solve");
    assert!(solution.rejected_outliers >= 1);
    assert!(solution.matches.iter().all(|m| m.centroid_index != moved));
    assert!(truth().angle_to(&solution.ecef_from_body) < 10.0 * ARCSEC);
}

/// The attitude error at the middle of an exposure of 0.1 s with the body
/// rate `deg_per_s` about a tilted axis, with and without a prior.
fn streak_error_arcsec(deg_per_s: f64, mode: IdentifyMode<'_>) -> f64 {
    let axis = Vector3::new(0.8, -0.5, 0.33).normalize();
    let rate = axis * deg_per_s.to_radians();
    let points = streak_sources(&catalogue(RENDER_MAGNITUDE), &truth(), rate, 0.1);
    let data = render(&points, 5.0, 13);
    let result = run(&data, mode).expect("solved");
    truth().angle_to(&result.solution.ecef_from_body) / ARCSEC
}

#[test]
fn a_streaked_image_solves_to_the_mid_exposure_attitude() {
    let one_degree = 1.0_f64.to_radians();
    for deg_per_s in [0.0, 1.0, 5.0] {
        let with_prior = streak_error_arcsec(
            deg_per_s,
            IdentifyMode::Prior(prior(one_degree, one_degree)),
        );
        let lost = streak_error_arcsec(deg_per_s, IdentifyMode::LostInSpace(pair_table()));
        assert!(with_prior < 10.0, "{deg_per_s} °/s: {with_prior} arcsec");
        assert!(lost < 10.0, "{deg_per_s} °/s: {lost} arcsec");
    }
}

/// The mean attitude NEES of `seeds` images with read and shot noise, a
/// gain of 2 e⁻/count, and the body rate `deg_per_s`, for the centroid
/// floor `floor_px`.
fn mean_nees(floor_px: f64, deg_per_s: f64, seeds: u64) -> f64 {
    let gain = 2.0;
    let detect = DetectConfig {
        electrons_per_count: gain,
        ..DetectConfig::default()
    };
    let solve = SolveConfig {
        min_centroid_sigma_px: floor_px,
        ..SolveConfig::default()
    };
    let axis = Vector3::new(0.8, -0.5, 0.33).normalize();
    let sources = streak_sources(
        &catalogue(RENDER_MAGNITUDE),
        &truth(),
        axis * deg_per_s.to_radians(),
        0.1,
    );
    let one_degree = 1.0_f64.to_radians();
    let nees: Vec<f64> = (1..=seeds)
        .map(|seed| {
            let data = render_with_shot_noise(&sources, 5.0, gain, seed);
            let mode = IdentifyMode::Prior(prior(one_degree, one_degree));
            let run = run_with(&data, mode, &detect, &solve).expect("solve");
            let error = (run.solution.ecef_from_body.inverse() * truth()).scaled_axis();
            let information = run
                .solution
                .covariance_body_rad2
                .try_inverse()
                .expect("inverse");
            error.dot(&(information * error))
        })
        .collect();
    nees.iter().sum::<f64>() / nees.len() as f64
}

/// The covariance follows the noise of each star: with a floor at the
/// sampling error of the point spread function, the mean NEES of a still
/// and of a streaked image is near its 3 degrees of freedom, and the
/// default floor stays on the safe side.
#[test]
fn the_attitude_covariance_is_consistent_with_the_pixel_noise() {
    const SEEDS: u64 = 24;
    // The mean of 24 χ²(3) draws has a deviation of 0.5.
    for deg_per_s in [0.0, 3.0] {
        let nees = mean_nees(0.005, deg_per_s, SEEDS);
        assert!((1.0..4.5).contains(&nees), "{deg_per_s} °/s: NEES {nees}");
    }
    let nees = mean_nees(SolveConfig::default().min_centroid_sigma_px, 0.0, SEEDS);
    assert!((0.3..3.0).contains(&nees), "default floor: NEES {nees}");
}

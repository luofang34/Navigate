use nalgebra::{UnitQuaternion, Vector3};

use super::*;
use crate::synthetic::{CATALOGUE_MAGNITUDE, catalogue, imager, mount, pipeline, truth};

const ARCSEC: f64 = std::f64::consts::PI / (180.0 * 3600.0);

fn pair(body: Vector3<f64>, attitude: &UnitQuaternion<f64>) -> StarMatch {
    let body_unit = body.normalize();
    StarMatch {
        centroid_index: 0,
        hr: 1,
        body_unit,
        ecef_unit: attitude * body_unit,
        separation_rad: 0.0,
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
    // Move one matched centroid by 4 pixels: it stays inside the second
    // gate and fails the residual test.
    let moved = run.solution.matches[0].centroid_index;
    centroids[moved].pixel.x += 4.0;
    let config = SolveConfig {
        rematch_gate_sigmas: 50.0,
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

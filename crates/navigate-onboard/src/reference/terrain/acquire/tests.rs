use nalgebra::UnitQuaternion;
use navigate_visual::CameraPose;

use super::*;

fn belief(sigma_h_m: f64) -> Belief {
    Belief {
        position: [41.7, -1.0, 2_000.0],
        body_to_ned: UnitQuaternion::identity(),
        sigma_h_m,
        sigma_attitude_rad: Some(0.002),
    }
}

fn pose() -> CameraPose {
    CameraPose {
        position: Vector3::new(0.0, 0.0, 2_000.0),
        orientation: UnitQuaternion::identity(),
    }
}

#[test]
fn a_belief_wider_than_a_view_reaches_is_acquired() {
    let config = TerrainConfig::default();
    assert_eq!(tier(&config, &belief(50.0), pose()), SearchTier::Local);
    assert_eq!(tier(&config, &belief(100.0), pose()), SearchTier::Local);
    assert_eq!(tier(&config, &belief(400.0), pose()), SearchTier::Region);
    assert_eq!(tier(&config, &belief(5_000.0), pose()), SearchTier::Global);
    assert_eq!(acquisition_radius_m(&config, 400.0, 0.0), 1_200.0);
    assert_eq!(acquisition_radius_m(&config, 5_000.0, 0.0), 3_000.0);
}

#[test]
fn a_moved_belief_is_offset_and_turned_by_the_move() {
    let from = belief(2_000.0);
    let offset = Vector2::new(-1_234.0, 987.0);
    let to = moved(&from, offset, 0.1).unwrap();
    let back = offset_m(&from, to.position).unwrap();
    assert!((back - offset).norm() < 1e-6, "{back:?}");
    assert_eq!(to.sigma_h_m, from.sigma_h_m);
    assert!((to.position[2] - from.position[2]).abs() < 1.0);
    // A turn to the left lowers the heading, measured clockwise from north.
    let (_, _, yaw) = to.body_to_ned.euler_angles();
    assert!((yaw + 0.1).abs() < 1e-12, "{yaw}");
}

#[test]
fn the_frame_is_laid_at_headings_out_to_the_belief_one_sigma() {
    let deg = std::f64::consts::PI / 180.0;
    let (step, max) = (2.0 * deg, 6.0 * deg);
    assert_eq!(rotations(Some(0.001), 3.0, step, max), [-step, 0.0, step]);
    assert_eq!(rotations(Some(0.0), 3.0, step, max), [0.0]);
    assert_eq!(rotations(Some(1.0), 3.0, step, max).len(), 7);
    assert_eq!(rotations(None, 3.0, step, max).len(), 7);
}

fn candidate(at_s: u64, correction: [f64; 2], belief_sigma_m: f64) -> Candidate {
    Candidate {
        at_ns: at_s * 1_000_000_000,
        correction_m: Vector2::new(correction[0], correction[1]),
        fix_variance_m2: 2.0 * 4.0_f64.powi(2),
        belief_sigma_m,
    }
}

#[test]
fn an_acquisition_is_confirmed_by_a_later_one_that_corrects_alike() {
    let config = TerrainConfig::default();
    // Two fixes of 4 m one-sigma on each axis: the difference has a
    // one-sigma of 8 m, and three of them make 24 m.
    let first = candidate(1, [800.0, -400.0], 1_000.0);
    assert!(first.confirmed_by(&candidate(2, [812.0, -390.0], 1_000.0), &config));
    assert!(!first.confirmed_by(&candidate(2, [830.0, -400.0], 1_000.0), &config));
    // A belief that grew by 20 m one-sigma in between widens the check.
    let grown = (1_000.0_f64.powi(2) + 20.0_f64.powi(2)).sqrt();
    assert!(first.confirmed_by(&candidate(2, [830.0, -400.0], grown), &config));
    assert!(!first.confirmed_by(&candidate(2, [1_100.0, -400.0], grown), &config));
    // Too late, or the same frame.
    assert!(!first.confirmed_by(&candidate(30, [800.0, -400.0], 1_000.0), &config));
    assert!(!first.confirmed_by(&first, &config));
}

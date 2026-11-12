use nalgebra::UnitQuaternion;

use super::*;

fn belief(sigma_h_m: f64) -> Belief {
    Belief {
        position: [41.7, -1.0, 2_000.0],
        body_to_ned: UnitQuaternion::identity(),
        sigma_h_m,
        sigma_attitude_rad: Some(0.002),
    }
}

#[test]
fn only_a_belief_wider_than_a_view_reaches_is_acquired() {
    let config = TerrainConfig::default();
    assert_eq!(acquisition_radius_m(&config, 50.0, 0.0), None);
    assert_eq!(acquisition_radius_m(&config, 100.0, 0.0), None);
    assert_eq!(acquisition_radius_m(&config, 400.0, 0.0), Some(1_200.0));
    assert_eq!(acquisition_radius_m(&config, 5_000.0, 0.0), Some(3_000.0));
    let off = TerrainConfig {
        max_acquisition_radius_m: config.max_search_radius_m,
        ..config
    };
    assert_eq!(acquisition_radius_m(&off, 5_000.0, 0.0), None);
}

#[test]
fn a_moved_belief_is_offset_by_the_move() {
    let from = belief(2_000.0);
    let offset = Vector2::new(-1_234.0, 987.0);
    let to = moved(&from, offset).unwrap();
    let back = offset_m(&from, to.position).unwrap();
    assert!((back - offset).norm() < 1e-6, "{back:?}");
    assert_eq!(to.sigma_h_m, from.sigma_h_m);
    assert!((to.position[2] - from.position[2]).abs() < 1.0);
}

#[test]
fn an_acquisition_is_confirmed_by_a_later_one_that_corrects_alike() {
    let config = TerrainConfig::default();
    let first = Candidate {
        at_ns: 1_000_000_000,
        correction_m: Vector2::new(800.0, -400.0),
    };
    let alike = Candidate {
        at_ns: 2_000_000_000,
        correction_m: Vector2::new(812.0, -390.0),
    };
    assert!(first.confirmed_by(&alike, &config));
    let elsewhere = Candidate {
        correction_m: Vector2::new(1_100.0, -400.0),
        ..alike
    };
    assert!(!first.confirmed_by(&elsewhere, &config));
    let late = Candidate {
        at_ns: 30_000_000_000,
        ..alike
    };
    assert!(!first.confirmed_by(&late, &config));
    // The same frame does not confirm itself.
    assert!(!first.confirmed_by(&first, &config));
}

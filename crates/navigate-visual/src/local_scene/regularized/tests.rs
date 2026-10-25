use super::*;
use crate::local_scene::tests::{camera, fixture, perturb};

#[test]
fn explicit_map_constraints_recover_known_geometry() {
    let truth = fixture();
    let priors: Vec<_> = truth
        .points
        .iter()
        .map(|p| ScenePointPrior {
            feature_id: p.feature_id,
            position: p.position,
            scale: 0.1,
        })
        .collect();
    let mut scene = truth.clone();
    perturb(&mut scene);
    let calibrations = vec![camera(); scene.cameras.len()];
    let result = refine_regularized_local_scene(&calibrations, &mut scene, &priors, 60)
        .expect("constrained refinement");
    assert!(result.final_cost < 1e-6);
    for (a, b) in scene.points.iter().zip(&truth.points) {
        assert!((a.position - b.position).norm() < 1e-5);
    }
    for (a, b) in scene.cameras.iter().zip(&truth.cameras) {
        assert!((a.pose.position - b.pose.position).norm() < 1e-5);
    }
}

#[test]
fn map_penalty_strength_controls_point_motion_without_recentering() {
    let truth = fixture();
    let mut displacements = Vec::new();
    for scale in [0.001, 100.0] {
        let mut scene = truth.clone();
        for c in &mut scene.cameras {
            c.fixed = true;
        }
        for p in &mut scene.points {
            for o in &mut p.observations {
                o.pixel.x += 2.0;
            }
        }
        let priors: Vec<_> = truth
            .points
            .iter()
            .map(|p| ScenePointPrior {
                feature_id: p.feature_id,
                position: p.position,
                scale,
            })
            .collect();
        let calibrations = vec![camera(); scene.cameras.len()];
        let result = refine_regularized_local_scene(&calibrations, &mut scene, &priors, 60)
            .expect("conflicting image and map costs");
        assert!(result.final_cost < result.initial_cost);
        let displacement: f64 = scene
            .points
            .iter()
            .zip(&truth.points)
            .map(|(a, b)| (a.position - b.position).norm())
            .sum();
        displacements.push(displacement);
        let before = scene.clone();
        refine_regularized_local_scene(&calibrations, &mut scene, &priors, 60)
            .expect("same immutable constraint centers");
        for (a, b) in scene.points.iter().zip(&before.points) {
            assert!((a.position - b.position).norm() < 1e-5);
        }
    }
    assert!(displacements[0] < displacements[1] * 0.25);
}

#[test]
fn invalid_constraints_leave_scene_unchanged() {
    let mut scene = fixture();
    let calibrations = vec![camera(); scene.cameras.len()];
    let unchanged = format!("{scene:?}");
    let valid = ScenePointPrior {
        feature_id: scene.points[0].feature_id,
        position: scene.points[0].position,
        scale: 1.0,
    };
    let mut invalid = Vec::new();
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MIN_POSITIVE] {
        invalid.push(vec![ScenePointPrior { scale, ..valid }]);
    }
    invalid.push(vec![valid, valid]);
    invalid.push(vec![ScenePointPrior {
        feature_id: u64::MAX,
        ..valid
    }]);
    invalid.push(vec![ScenePointPrior {
        position: Vector3::repeat(f64::NAN),
        ..valid
    }]);
    for priors in invalid {
        assert!(matches!(
            refine_regularized_local_scene(&calibrations, &mut scene, &priors, 20),
            Err(LocalSceneError::Point { .. })
        ));
        assert_eq!(format!("{scene:?}"), unchanged);
    }
}

#[test]
fn overflowing_map_cost_rejects_without_mutation() {
    let mut scene = fixture();
    let unchanged = format!("{scene:?}");
    let prior = ScenePointPrior {
        feature_id: scene.points[0].feature_id,
        position: Vector3::repeat(1e200),
        scale: 1.0,
    };
    let calibrations = vec![camera(); scene.cameras.len()];
    assert!(matches!(
        refine_regularized_local_scene(&calibrations, &mut scene, &[prior], 20),
        Err(LocalSceneError::Numerical { .. })
    ));
    assert_eq!(format!("{scene:?}"), unchanged);
}

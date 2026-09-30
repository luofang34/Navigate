use super::*;

fn camera() -> CameraModel {
    CameraModel {
        width: 960,
        height: 544,
        fx: 689.0,
        fy: 694.0,
        cx: 479.5,
        cy: 271.5,
    }
}

#[test]
fn similarity_seed_recovers_low_consensus_at_multiple_scales_and_headings() {
    let camera = camera();
    for height in [45.0, 110.0, 600.0] {
        for yaw in [-2.8, 0.7, 2.2] {
            let truth = CameraPose {
                position: Vector3::new(700.0, 540.0, 17.0 + height),
                orientation: UnitQuaternion::from_euler_angles(0.0, 0.0, yaw),
            };
            let pairs: Vec<_> = (0..160)
                .map(|i| {
                    let world = Vector3::new(
                        700.0 + f64::from(i % 16 - 8) * height / 20.0,
                        540.0 + f64::from(i / 16 - 5) * height / 20.0,
                        17.0,
                    );
                    let query = if i % 10 == 3 {
                        camera.project(&truth, world).expect("visible map point")
                    } else {
                        Vector2::new(
                            f64::from((i * 137 + 31) % 960),
                            f64::from((i * i * 19 + 17) % 544),
                        )
                    };
                    GroundCorrespondence { world, query }
                })
                .collect();
            let result = nadir_similarity_proposal(&camera, &pairs, 4.0)
                .expect("valid input")
                .expect("seed");
            assert!((result.pose.position - truth.position).norm() < 1e-6);
            assert!(result.pose.orientation.angle_to(&truth.orientation) < 1e-6);
            assert_eq!(result.inliers, 16);
        }
    }
}

#[test]
fn missing_degenerate_or_invalid_evidence_cannot_make_a_seed() {
    let camera = camera();
    assert!(
        nadir_similarity_proposal(&camera, &[], 4.0)
            .expect("empty")
            .is_none()
    );
    let repeated = vec![
        GroundCorrespondence {
            world: Vector3::new(0.0, 0.0, 17.0),
            query: Vector2::new(100.0, 200.0)
        };
        12
    ];
    assert!(
        nadir_similarity_proposal(&camera, &repeated, 4.0)
            .expect("degenerate")
            .is_none()
    );
    for threshold in [0.0, -1.0, 65.0, f64::NAN, f64::INFINITY] {
        assert!(nadir_similarity_proposal(&camera, &[], threshold).is_err());
    }
    let mut invalid = repeated.clone();
    invalid[0].world.z = f64::NAN;
    assert!(nadir_similarity_proposal(&camera, &invalid, 4.0).is_err());
    invalid[0] = GroundCorrespondence {
        world: Vector3::zeros(),
        query: Vector2::new(f64::INFINITY, 0.0),
    };
    assert!(nadir_similarity_proposal(&camera, &invalid, 4.0).is_err());
    assert!(nadir_similarity_proposal(&camera, &vec![repeated[0].clone(); 4097], 4.0).is_err());
}

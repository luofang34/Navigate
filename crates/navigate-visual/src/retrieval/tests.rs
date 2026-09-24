#![allow(clippy::panic)]
use super::*;
#[test]
fn rotated_oblique_planar_retrieval_keeps_pose_and_ignores_outliers() {
    let camera = CameraModel {
        width: 640,
        height: 360,
        fx: 450.0,
        fy: 450.0,
        cx: 319.5,
        cy: 179.5,
    };
    let truth = CameraPose {
        position: Vector3::new(27.0, -16.0, 180.0),
        orientation: UnitQuaternion::from_euler_angles(0.18, -0.08, 0.7),
    };
    let mut pairs = Vec::new();
    for y in 0..6 {
        for x in 0..9 {
            let world = Vector3::new(f64::from(x) * 16.0 - 60.0, f64::from(y) * 16.0 - 40.0, 30.0);
            if let Some(p) = camera.project(&truth, world) {
                pairs.push(GroundCorrespondence { world, query: p });
            }
        }
    }
    for i in 0..12 {
        pairs.push(GroundCorrespondence {
            world: Vector3::new(f64::from(i) * 8.0, 20.0, 30.0),
            query: Vector2::new(f64::from(i) * 25.0, 17.0),
        });
    }
    let result = planar_proposal(&camera, &pairs).unwrap_or_else(|e| panic!("{e}"));
    let estimated = result.unwrap_or_else(|| panic!("no proposal")).pose;
    assert!((estimated.position - truth.position).norm() < 1e-5);
    assert!(estimated.orientation.angle_to(&truth.orientation) < 1e-5);
}
#[test]
fn blank_and_degenerate_retrieval_do_not_propose_a_location() {
    let camera = CameraModel {
        width: 640,
        height: 360,
        fx: 450.0,
        fy: 450.0,
        cx: 319.5,
        cy: 179.5,
    };
    assert!(proposals(camera, &[]).is_none());
    let pairs = vec![
        GroundCorrespondence {
            world: Vector3::zeros(),
            query: Vector2::repeat(20.0)
        };
        12
    ];
    assert!(proposals(camera, &pairs).is_none());
}

#[test]
fn planar_retrieval_recovers_with_eighty_percent_outliers() {
    let camera = CameraModel {
        width: 640,
        height: 360,
        fx: 450.0,
        fy: 450.0,
        cx: 319.5,
        cy: 179.5,
    };
    let truth = CameraPose {
        position: Vector3::new(27.0, -16.0, 180.0),
        orientation: UnitQuaternion::from_euler_angles(0.18, -0.08, 0.7),
    };
    for offset in 0..4 {
        let mut pairs = Vec::new();
        for i in 0..150 {
            let world = Vector3::new(
                f64::from(i % 15) * 12.0 - 80.0,
                f64::from(i / 15) * 12.0 - 55.0,
                30.0,
            );
            let query = if i % 5 == offset {
                camera
                    .project(&truth, world)
                    .unwrap_or_else(|| panic!("visible point"))
            } else {
                Vector2::new(
                    f64::from((i * 137 + 31) % 640),
                    f64::from((i * i * 19 + 17) % 360),
                )
            };
            pairs.push(GroundCorrespondence { world, query });
        }
        let estimated = planar_proposal(&camera, &pairs)
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|| panic!("no proposal"));
        assert!(
            (estimated.pose.position - truth.position).norm() < 1.0,
            "offset {offset}"
        );
        assert!(
            estimated.pose.orientation.angle_to(&truth.orientation) < 0.01,
            "offset {offset}"
        );
        assert!(estimated.inliers >= 30);
    }
}

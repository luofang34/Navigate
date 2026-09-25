//! Verify the worker boundary without a graphics device.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use nalgebra::{UnitQuaternion, Vector2, Vector3};
use navigate_visual::{
    LocalSceneCamera, LocalScenePoint, LocalScenePose, MapRevision, ScenePointObservation,
};
fn fixture() -> (CameraModel, LocalScene, ReferenceView, Vec<PixelMatch>) {
    let camera = CameraModel {
        width: 96,
        height: 72,
        fx: 70.0,
        fy: 70.0,
        cx: 47.5,
        cy: 35.5,
    };
    let pose = LocalScenePose {
        position: Vector3::zeros(),
        orientation: UnitQuaternion::identity(),
    };
    let mut scene = LocalScene {
        cameras: (0..2)
            .map(|i| LocalSceneCamera {
                observation_sha256: format!("{i:064x}"),
                pose,
                fixed: i == 0,
            })
            .collect(),
        points: Vec::new(),
    };
    let mut reference = ReferenceView {
        map: MapRevision {
            release_id: "test".into(),
            manifest_sha256: "a".repeat(64),
        },
        frame: LocalFrame::anchor_mercator(40.0, -74.0).expect("valid anchor"),
        pose: CameraPose {
            position: Vector3::new(10.0, 20.0, 50.0),
            orientation: UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3),
        },
        image: image::GrayImage::new(96, 72),
        depth_m: vec![0.0; 96 * 72],
    };
    let mut pairs = Vec::new();
    for i in 0..48 {
        let pixel = Vector2::new(8.0 + (i % 8) as f64 * 11.0, 6.0 + (i / 8) as f64 * 12.0);
        let position = camera.unproject(
            &CameraPose {
                position: pose.position,
                orientation: pose.orientation,
            },
            pixel,
            5.0,
        );
        scene.points.push(LocalScenePoint {
            feature_id: i as u64,
            position,
            observations: (0..2)
                .map(|j| ScenePointObservation {
                    camera_index: j,
                    pixel,
                })
                .collect(),
        });
        reference.depth_m[pixel.y as usize * 96 + pixel.x as usize] = 35.0;
        pairs.push(PixelMatch {
            query: pixel,
            reference: pixel,
        });
    }
    (camera, scene, reference, pairs)
}
#[test]
fn registration_reports_conditional_geometry_and_render_provenance() {
    let (camera, scene, reference, pairs) = fixture();
    let result = register(
        &camera,
        &scene,
        &"0".repeat(64),
        &reference,
        &pairs,
        "test-adapter",
    )
    .expect("valid input");
    assert_eq!(result["geographic_acceptance"], false);
    assert_eq!(result["geographic_accuracy"], "not_independently_measured");
    assert_eq!(result["backend"], "test-adapter");
    assert_eq!(result["map_manifest_sha256"], "a".repeat(64));
    assert_eq!(
        result["reference_image_sha256"]
            .as_str()
            .expect("digest")
            .len(),
        64
    );
    assert_eq!(
        result["reference_depth_sha256"]
            .as_str()
            .expect("digest")
            .len(),
        64
    );
    assert_eq!(result["associations"].as_array().expect("links").len(), 48);
    let proposal = &result["candidates"][0];
    assert!((proposal["scale"].as_f64().expect("scale") - 7.0).abs() < 1e-9);
    assert_eq!(proposal["cameras"][0]["observation_sha256"], "0".repeat(64));
    assert!(
        (proposal["cameras"][0]["altitude_m"]
            .as_f64()
            .expect("altitude")
            - 50.0)
            .abs()
            < 1e-9
    );
    let registration: Registration =
        serde_json::from_value(proposal.clone()).expect("worker transform");
    let (transform, frame) = registration.model().expect("valid transform");
    let restored = map_cameras(&scene, transform, frame).expect("camera transform");
    let restored_position = restored[0]["position_enu_m"].as_array().expect("position");
    for (i, value) in restored_position.iter().enumerate() {
        assert!((value.as_f64().expect("coordinate") - reference.pose.position[i]).abs() < 1e-9);
    }
    assert_eq!(restored[1]["observation_sha256"], format!("{:064x}", 1));
}
#[test]
fn missing_depth_cannot_become_a_registration() {
    let (camera, scene, mut reference, pairs) = fixture();
    reference.depth_m.fill(0.0);
    assert!(matches!(
        register(&camera, &scene, &"0".repeat(64), &reference, &pairs, "test"),
        Err(PreviewError::SceneRegistration(_))
    ));
}
#[test]
fn map_transform_rejects_invalid_scale_rotation_and_cameras() {
    let mut registration = Registration {
        scale: -1.0,
        scene_to_map_xyzw: [0.0, 0.0, 0.0, 1.0],
        translation_enu_m: [0.0; 3],
        anchor_lat_lon: [40.0, -74.0],
    };
    assert!(registration.model().is_err());
    registration.scale = 1.0;
    registration.scene_to_map_xyzw = [0.0; 4];
    assert!(registration.model().is_err());
    registration.scene_to_map_xyzw = [0.0, 0.0, 0.0, 1.0];
    let (transform, frame) = registration.model().expect("valid transform");
    let (_, mut scene, _, _) = fixture();
    scene.cameras[0].pose.position.x = f64::NAN;
    assert!(map_cameras(&scene, transform, frame).is_err());
}

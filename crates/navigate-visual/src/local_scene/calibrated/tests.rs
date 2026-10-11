use super::*;
use crate::local_scene::{
    pose::Pose,
    tests::{fixture, perturb},
};

#[test]
fn different_camera_calibrations_recover_the_same_known_scene() {
    let mut truth = fixture();
    let calibrations: Vec<_> = (0..truth.cameras.len())
        .map(|i| {
            let (width, height) = if i % 2 == 0 { (640, 480) } else { (480, 640) };
            CameraModel {
                width,
                height,
                fx: 280.0 + i as f64 * 40.0,
                fy: 310.0 + i as f64 * 30.0,
                cx: f64::from(width) / 2.0 - 7.0,
                cy: f64::from(height) / 2.0 + 13.0,
            }
        })
        .collect();
    for point in &mut truth.points {
        for observation in &mut point.observations {
            let i = observation.camera_index;
            observation.pixel = Pose::from_scene(truth.cameras[i].pose)
                .project(&calibrations[i], point.position)
                .expect("visible point");
        }
        point.observations.retain(|observation| {
            let camera = calibrations[observation.camera_index];
            (0.0..f64::from(camera.width)).contains(&observation.pixel.x)
                && (0.0..f64::from(camera.height)).contains(&observation.pixel.y)
        });
    }
    let mut scene = truth.clone();
    perturb(&mut scene);
    let result =
        refine_calibrated_local_scene(&calibrations, &mut scene, 60, None).expect("mixed cameras");
    assert!(result.final_cost < 1e-5);
    for (actual, expected) in scene.cameras.iter().zip(&truth.cameras) {
        assert!((actual.pose.position - expected.pose.position).norm() < 1e-4);
        assert!(actual.pose.orientation.angle_to(&expected.pose.orientation) < 1e-5);
    }
    for (actual, expected) in scene.points.iter().zip(&truth.points) {
        assert!((actual.position - expected.position).norm() < 1e-4);
        for (a, b) in actual.observations.iter().zip(&expected.observations) {
            assert_eq!(a.pixel, b.pixel);
        }
    }
}

#[test]
fn mismatched_calibration_or_pixels_outside_their_own_camera_are_rejected() {
    let mut scene = fixture();
    let original = scene.cameras[2].pose;
    assert!(matches!(
        refine_calibrated_local_scene(&[], &mut scene, 20, None),
        Err(LocalSceneError::CalibrationCount { .. })
    ));
    let mut cameras = vec![
        CameraModel {
            width: 960,
            height: 544,
            fx: 700.0,
            fy: 690.0,
            cx: 479.5,
            cy: 271.5
        };
        scene.cameras.len()
    ];
    cameras[2].width = 64;
    assert!(matches!(
        refine_calibrated_local_scene(&cameras, &mut scene, 20, None),
        Err(LocalSceneError::Point { .. })
    ));
    assert_eq!(scene.cameras[2].pose.position, original.position);
    assert_eq!(scene.cameras[2].pose.orientation, original.orientation);
}

use super::*;
use crate::corridor::CorridorLandmarkBank;
use crate::corridor::tests::build_tile_landmarks;
use crate::{DEFAULT_KASHIN_SEED, HadamardProjector};
use nalgebra::{Vector2, Vector3};

fn oblique_camera() -> VirtualViewCamera {
    // Camera +X = north, +Y = east, +Z = down: a nadir view, a proper rotation.
    VirtualViewCamera {
        eye_enu: [130.0, -95.0, 1_700.0],
        world_to_cam: [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        intrinsics: [610.0, 590.0, 402.5, 297.25],
        image_size: [800, 600],
        max_landmarks: 256,
        min_cos_incidence: -1.0,
        min_cos_orientation: -1.0,
    }
}

#[test]
fn optical_pose_and_intrinsics_keep_every_field() {
    let cam = oblique_camera();
    let pose = cam.optical_pose().unwrap();
    assert_eq!(pose.to_row_major(), (cam.eye_enu, cam.world_to_cam));
    let intrinsics = cam.intrinsics();
    assert_eq!(
        [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy],
        cam.intrinsics
    );
    assert_eq!([intrinsics.width, intrinsics.height], cam.image_size);
}

#[test]
fn reflected_rotation_is_refused() {
    let mut cam = oblique_camera();
    cam.world_to_cam = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
    assert!(cam.optical_pose().is_err());
}

#[test]
fn virtual_view_pixels_equal_optical_projection() {
    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);
    let tiles = (-1..=1)
        .flat_map(|ty| (-1..=1).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| build_tile_landmarks(tx, ty, 2_000.0, 64, &projector))
        .collect();
    let bank =
        CorridorLandmarkBank::new([1; 32], [2; 32], DEFAULT_KASHIN_SEED, 2_000.0, tiles).unwrap();
    let cam = oblique_camera();
    let view = bank.project_virtual_view(&cam, &projector);
    assert!(
        view.features.len() > 10,
        "{} landmarks",
        view.features.len()
    );
    let pose = cam.optical_pose().unwrap();
    let intrinsics = cam.intrinsics();
    for (pixel, world) in view.features.pixels.iter().zip(&view.world_points_enu) {
        let ours = pose
            .project(&intrinsics, &Vector3::from(*world))
            .expect("landmark is in front");
        assert!(
            (ours - Vector2::from(*pixel)).norm() < 1e-6,
            "{ours} vs {pixel:?}"
        );
    }
}

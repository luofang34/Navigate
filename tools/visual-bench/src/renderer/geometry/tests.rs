use super::*;
use nalgebra::{UnitQuaternion, Vector3};

#[test]
fn coverage_removes_depth_outside_loaded_map() {
    let camera = CameraModel {
        width: 32,
        height: 32,
        fx: 100.0,
        fy: 100.0,
        cx: 15.5,
        cy: 15.5,
    };
    let pose = CameraPose {
        position: Vector3::new(0.0, 0.0, 100.0),
        orientation: UnitQuaternion::identity(),
    };
    let coverage = Coverage {
        imagery: vec![[-10.0, 10.0, -10.0, 10.0]],
        elevation: vec![[-10.0, 10.0, -10.0, 10.0]],
    };
    let mut depth = vec![100.0; 1024];
    coverage.mask(&mut depth, camera, pose, |_| true);
    assert_eq!(depth[0], 0.0);
    assert_eq!(depth[16 * 32 + 16], 100.0);
    let outside = CameraPose {
        position: Vector3::new(1000.0, 0.0, 100.0),
        ..pose
    };
    coverage.mask(&mut depth, camera, outside, |_| true);
    assert!(depth.iter().all(|d| *d == 0.0));
}

use super::*;
use nalgebra::{UnitQuaternion, Vector3};

#[test]
fn separation_is_required_in_each_image_and_is_order_invariant() {
    let camera = CameraModel {
        width: 640,
        height: 360,
        fx: 500.0,
        fy: 500.0,
        cx: 319.5,
        cy: 179.5,
    };
    let pose = CameraPose {
        position: Vector3::new(0.0, 0.0, 100.0),
        orientation: UnitQuaternion::identity(),
    };
    for reference_is_dense in [true, false] {
        let mut points = Vec::new();
        for i in 0..8 {
            let spread = Vector2::new(50.0 + f64::from(i) * 60.0, 100.0);
            let cluster = Vector2::new(100.0 + f64::from(i), 100.0);
            let (reference, query) = if reference_is_dense {
                (cluster, spread)
            } else {
                (spread, cluster)
            };
            points.push(Correspondence {
                world: camera.unproject(&pose, reference, 100.0),
                pixel: query,
            });
        }
        let (support, _) = separated(&camera, &pose, &points, 12.0);
        assert_eq!(support.len(), 1);
        points.reverse();
        let (reordered, _) = separated(&camera, &pose, &points, 12.0);
        assert_eq!(reordered.len(), 1);
        assert_eq!(support[0].pixel, reordered[0].pixel);
        assert_eq!(support[0].world, reordered[0].world);
    }
}

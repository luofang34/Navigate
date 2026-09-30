use super::*;
#[test]
fn resized_intrinsics_preserve_pixel_centres_and_aspect_mapping() -> Result<(), BenchError> {
    let original = CameraModel {
        width: 1920,
        height: 1080,
        fx: 1350.0,
        fy: 1350.0,
        cx: 959.5,
        cy: 539.5,
    };
    let resized = scaled_camera(original, 960)?;
    assert_eq!((resized.width, resized.height), (960, 544));
    let pose = navigate_visual::CameraPose {
        position: nalgebra::Vector3::new(0.0, 0.0, 100.0),
        orientation: nalgebra::UnitQuaternion::identity(),
    };
    let world = nalgebra::Vector3::new(10.0, 20.0, 0.0);
    let a = original
        .project(&pose, world)
        .ok_or_else(|| BenchError::Record {
            reason: "source projection absent".into(),
        })?;
    let b = resized
        .project(&pose, world)
        .ok_or_else(|| BenchError::Record {
            reason: "resized projection absent".into(),
        })?;
    assert!((b.x - ((a.x + 0.5) * 0.5 - 0.5)).abs() < 1e-8);
    assert!((b.y - ((a.y + 0.5) * 544.0 / 1080.0 - 0.5)).abs() < 1e-8);
    Ok(())
}

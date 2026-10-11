use super::*;
#[test]
fn nonidentity_fixture_cannot_silently_run_as_nadir() {
    let case = Case {
        id: "tilted".into(),
        anchor_deg: [50.0, 0.0],
        height_m: 6000.0,
        eye_to_world_rotation: Orientation::Named("tilted".into()),
        true_camera_enu_m: [0.0, 0.0, 6000.0],
        points: Vec::new(),
    };
    let camera = CameraModel {
        width: 1280,
        height: 960,
        fx: 900.0,
        fy: 900.0,
        cx: 639.5,
        cy: 479.5,
    };
    assert!(matches!(
        run_case(&case, camera, Geometry::Sphere),
        Err(Error::Input("fixture orientation must be identity"))
    ));
}

#[test]
fn fixture_rotation_rejects_reflection_and_accepts_oblique_axes() {
    assert!(
        Orientation::Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
            .rotation()
            .is_err()
    );
    let rotation = UnitQuaternion::from_euler_angles(0.4, -0.2, 0.3);
    let m = rotation.to_rotation_matrix();
    let rows = std::array::from_fn(|i| std::array::from_fn(|j| m[(i, j)]));
    assert!(
        Orientation::Matrix(rows)
            .rotation()
            .unwrap()
            .angle_to(&rotation)
            < 1e-12
    );
}

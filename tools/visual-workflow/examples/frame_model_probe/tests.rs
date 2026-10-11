use super::*;
#[test]
fn nonidentity_fixture_cannot_silently_run_as_nadir() {
    let case = Case {
        id: "tilted".into(),
        anchor_deg: [50.0, 0.0],
        height_m: 6000.0,
        eye_to_world_rotation: "tilted".into(),
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
        run_case(&case, camera, true),
        Err(Error::Input("fixture orientation must be identity"))
    ));
}

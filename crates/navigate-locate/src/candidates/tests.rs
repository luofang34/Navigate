use super::*;

#[test]
fn the_orientation_matches_the_oryol_candidate() {
    // h270-p45-a150 of the Oryol job: rotation [x, y, z, w] = [0.2706, 0.2706, 0.6533, 0.6533].
    let q = orientation(270.0, 45.0, 0.0);
    let expected =
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(0.6533, 0.2706, 0.2706, 0.6533));
    assert!(q.angle_to(&expected) < 1e-3, "{q:?}");
}

#[test]
fn the_camera_sits_back_along_the_heading() {
    // Same candidate: target (-100, 91, 190) gives camera (50, 91, 340).
    let pose = look_at(Vector3::new(-100.0, 91.0, 190.0), 270.0, 45.0, 150.0);
    assert!(
        (pose.position - Vector3::new(50.0, 91.0, 340.0)).norm() < 1e-9,
        "{:?}",
        pose.position
    );
}

#[test]
fn angles_invert_the_orientation() {
    for (h, p, r) in [(275.0, 39.0, 0.0), (10.0, 5.0, 3.0), (180.0, 80.0, -7.0)] {
        let [h2, p2, r2] = angles(&orientation(h, p, r));
        assert!(
            (h2 - h).abs() < 1e-9 && (p2 - p).abs() < 1e-9 && (r2 - r).abs() < 1e-9,
            "{h2} {p2} {r2}"
        );
    }
}

#[test]
fn the_standard_grid_has_the_oryol_count() {
    assert_eq!(LookAt::standard(282.0).poses(Vector3::zeros()).len(), 27);
}

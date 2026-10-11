use super::*;
use nalgebra::UnitQuaternion;
#[cfg(test)]
type Scene = (
    CameraModel,
    CameraPose,
    CameraPose,
    Vec<(Vector2<f64>, Vector2<f64>)>,
);
#[cfg(test)]
fn scene() -> Scene {
    let k = CameraModel {
        width: 640,
        height: 480,
        fx: 490.0,
        fy: 510.0,
        cx: 318.0,
        cy: 242.0,
    };
    let a = CameraPose {
        position: [10.0, 20.0, 100.0].into(),
        orientation: UnitQuaternion::from_euler_angles(0.05, -0.04, 0.12),
    };
    let b = CameraPose {
        position: [12.0, 19.0, 100.5].into(),
        orientation: UnitQuaternion::from_euler_angles(0.06, -0.03, 0.11),
    };
    let pairs = (0..50)
        .map(|i| {
            let p = Vector3::new(
                f64::from(i % 10) * 5.0 - 10.0,
                f64::from(i / 10) * 5.0,
                2.0 * (f64::from(i) * 0.3).sin(),
            );
            (k.project(&a, p).unwrap(), k.project(&b, p).unwrap())
        })
        .collect();
    (k, a, b, pairs)
}
#[test]
fn projected_points_pass_and_wrong_rotation_fails() {
    let (k, a, b, pairs) = scene();
    let valid = score(k, a, b, &pairs);
    assert_eq!(valid["epipolar"]["count"], 50);
    assert!(valid["epipolar"]["p90_px"].as_f64().unwrap() < 1e-9);
    let mut wrong = b;
    wrong.orientation *= UnitQuaternion::from_euler_angles(0.05, 0.07, 0.03);
    let changed = score(k, a, wrong, &pairs);
    assert!(
        changed["epipolar"]["median_px"].as_f64().unwrap() > 2.5,
        "{changed}"
    );
}
#[test]
fn translation_scale_and_global_frame_are_unobservable() {
    let (k, a, b, pairs) = scene();
    let transform = |p: CameraPose| {
        let q = UnitQuaternion::from_euler_angles(0.4, 0.1, 0.7);
        CameraPose {
            position: 2.0 * (q * p.position) + Vector3::new(300.0, -100.0, 40.0),
            orientation: q * p.orientation,
        }
    };
    assert!(
        score(k, transform(a), transform(b), &pairs)["epipolar"]["p90_px"]
            .as_f64()
            .unwrap()
            < 1e-9
    );
    let mut scaled = b;
    scaled.position = a.position + 2.0 * (b.position - a.position);
    assert!(
        score(k, a, scaled, &pairs)["epipolar"]["p90_px"]
            .as_f64()
            .unwrap()
            < 1e-9
    );
}
#[test]
fn zero_translation_is_not_a_perfect_epipolar_score() {
    let (k, a, _, pairs) = scene();
    let v = score(k, a, a, &pairs);
    assert_eq!(v["epipolar"]["count"], 0);
    assert!(v["epipolar"]["median_px"].is_null());
    assert!(v["rotation_only"]["median_px"].as_f64().unwrap() > 0.0);
}

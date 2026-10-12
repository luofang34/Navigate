use super::*;
use nalgebra::UnitQuaternion;
use navigate_visual::{CameraModel, FrameStamp, MapRevision, RasterGeometry};
fn fixture() -> (Frame, OrthographicReference) {
    let image = GrayImage::from_fn(160, 128, |x, y| {
        image::Luma([((x * 17 + y * 31 + x * y) % 180 + 30) as u8])
    });
    let reference = OrthographicReference {
        image: image.clone(),
        heights_m: vec![0.0; 160 * 128],
        map: MapRevision {
            release_id: "test".into(),
            manifest_sha256: "a".repeat(64),
        },
        geometry: RasterGeometry {
            origin: [-80.0, 64.0].into(),
            column: [1.0, 0.0].into(),
            row: [0.0, -1.0].into(),
        },
    };
    let frame = Frame {
        image,
        camera: CameraModel {
            width: 160,
            height: 128,
            fx: 120.0,
            fy: 120.0,
            cx: 80.0,
            cy: 64.0,
        },
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 0,
        },
    };
    (frame, reference)
}
#[test]
fn projection_scores_known_shift_and_never_changes_support_per_candidate() {
    let (frame, reference) = fixture();
    let good = CameraPose {
        position: [0.0, 0.0, 120.0].into(),
        orientation: UnitQuaternion::identity(),
    };
    let bad = CameraPose {
        position: [3.0, 0.0, 120.0].into(),
        ..good
    };
    let (patches, _) = compare(&frame, &reference, &[good, bad], &[[16.0, 16.0]]);
    assert!(patches.len() > 8);
    assert!(
        patches.iter().all(|p| p.centre != [16, 16]
            && p.scores.len() == 2
            && (p.scores[0] - 1.0).abs() < 1e-12)
    );
    assert!(patches.iter().map(|p| p.scores[1]).sum::<f64>() / (patches.len() as f64) < 0.9);
    let behind = CameraPose {
        position: [0.0, 0.0, -120.0].into(),
        ..good
    };
    assert!(
        compare(&frame, &reference, &[good, behind], &[])
            .0
            .is_empty()
    );
}
#[test]
fn correlation_handles_brightness_changes_inversion_and_flat_query_without_hiding_it() {
    let a = std::array::from_fn(|i| i as f64);
    let b = a.map(|x| 2.0 * x + 10.0);
    assert!((correlation(&a, &b) - 1.0).abs() < 1e-12);
    assert!((correlation(&a, &a.map(|x| -x)) + 1.0).abs() < 1e-12);
    assert_eq!(correlation(&a, &[5.0; 25]), 0.0);
    let image = GrayImage::new(32, 32);
    assert!(sample(&image, [-0.01, 2.0].into()).is_none());
    assert!(sample(&image, [31.0, 2.0].into()).is_none());
    assert!(sample(&image, [f64::NAN, 2.0].into()).is_none());
    assert_eq!(statistics(vec![])["median"], Value::Null);
}

#[test]
fn a_rotated_camera_keeps_pixel_axes_and_missing_surface_has_no_score() {
    let (mut frame, mut reference) = fixture();
    frame.image = GrayImage::from_fn(160, 128, |x, y| {
        reference.image[((160 - x) % 160, (128 - y) % 128)]
    });
    let pose = CameraPose {
        position: [0.0, 0.0, 120.0].into(),
        orientation: UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::z_axis(),
            std::f64::consts::PI,
        ),
    };
    let (patches, counts) = compare(&frame, &reference, &[pose], &[]);
    assert!(patches.len() > 8);
    assert!(patches.iter().all(|p| (p.scores[0] - 1.0).abs() < 1e-11));
    assert_eq!(counts.invalid_centres, 0);
    reference.heights_m.fill(f32::NAN);
    let (patches, counts) = compare(&frame, &reference, &[pose], &[]);
    assert!(patches.is_empty());
    assert_eq!(counts.centres, counts.invalid_centres);
}

use super::*;
use nalgebra::{UnitQuaternion, Vector3};
use navigate_visual::{CameraModel, CameraPose, FrameStamp, LocalFrame, MapRevision, PosePrior};

#[test]
fn a_real_tracked_shift_preserves_map_points_and_acceptance() {
    let (camera, reference) = fixture();
    let pose = reference.pose;
    let image = reference.image.clone();
    use navigate_visual::{PointTracker, PyramidalMatcher};
    let pairs: Vec<_> = PyramidalMatcher
        .features_blocking(&image)
        .unwrap()
        .into_iter()
        .map(|p| PixelMatch {
            reference: p,
            query: p,
        })
        .filter(|p| metrics::partition(p, 0) == Some(false))
        .collect();
    let prior = PosePrior {
        pose,
        position_radius_m: 100.0,
        attitude_radius_rad: 0.5,
    };
    let first = Frame {
        camera,
        image: image.clone(),
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 0,
        },
    };
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let estimate = verifier
        .verify(&first, &reference, &prior, &pairs, "test")
        .unwrap();
    let tracks = Tracks::update(first, pairs, (&reference, &camera), &estimate, None, true);
    let shifted = image::GrayImage::from_fn(320, 240, |x, y| *image.get_pixel((x + 1).min(319), y));
    let second = Frame {
        camera,
        image: shifted,
        stamp: FrameStamp {
            sequence: 1,
            capture_time_ns: 33_333_333,
        },
    };
    assert!(tracks.eligible(&second, 5));
    assert!(!tracks.eligible(&second, 1));
    let tracked = tracks.propagate_blocking(&second).unwrap();
    assert!(tracked.len() >= 20);
    assert!(
        tracked
            .iter()
            .all(|p| metrics::partition(p, 0) == Some(false))
    );
    let moved = verifier
        .verify(&second, &reference, &prior, &tracked, "test")
        .unwrap();
    assert!((moved.pose.position.x - 100.0 / 150.0).abs() < 0.1);
    assert!(moved.pose.position.y.abs() < 0.1);
    assert!(moved.pose.position.z.abs() < 0.1);
    let invalid = verifier
        .verify(&second, &reference, &prior, &tracked[..3], "test")
        .map_err(Error::from);
    assert!(!select_map(&Ok(moved.clone()), &invalid, None).0);
    assert!(!select_map(&Ok(estimate.clone()), &Ok(moved.clone()), Some(0.1)).0);
    assert!(select_map(&Ok(estimate), &Ok(moved), Some(2.0)).0);
    assert_gap(&tracks, second);
}

#[cfg(test)]
fn fixture() -> (CameraModel, ReferenceView) {
    let camera = CameraModel {
        width: 320,
        height: 240,
        fx: 150.0,
        fy: 150.0,
        cx: 160.0,
        cy: 120.0,
    };
    let pose = CameraPose {
        position: Vector3::zeros(),
        orientation: UnitQuaternion::identity(),
    };
    let image = image::GrayImage::from_fn(320, 240, |x, y| {
        let h = (x / 7).wrapping_mul(374761393) ^ (y / 9).wrapping_mul(668265263);
        image::Luma([40 + ((h ^ (h >> 13)).wrapping_mul(1274126177) % 160) as u8])
    });
    let reference = ReferenceView {
        map: MapRevision {
            release_id: "test".into(),
            manifest_sha256: "a".repeat(64),
        },
        frame: LocalFrame::anchor_mercator(35.0, -115.0).unwrap(),
        pose,
        image,
        depth_m: vec![100.0; 320 * 240],
    };
    (camera, reference)
}

#[cfg(test)]
fn assert_gap(tracks: &Tracks, second: Frame) {
    let delayed = Frame {
        stamp: FrameStamp {
            sequence: 2,
            capture_time_ns: 400_000_000,
        },
        ..second
    };
    assert!(!tracks.eligible(&delayed, 5));
}

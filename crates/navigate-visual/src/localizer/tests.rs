#![allow(clippy::expect_used)]

use super::*;
use crate::{CameraModel, PixelMatch, PyramidalMatcher};
use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector3};

struct KnownMatches(Vec<PixelMatch>);
impl ImageMatcher for KnownMatches {
    fn identity(&self) -> &str {
        "test-geometry"
    }
    fn match_images_blocking(
        &mut self,
        _: &GrayImage,
        _: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        Ok(self.0.clone())
    }
}

fn scene() -> (Frame, ReferenceView, PosePrior, Vec<PixelMatch>, CameraPose) {
    let camera = CameraModel {
        width: 320,
        height: 240,
        fx: 275.0,
        fy: 275.0,
        cx: 159.5,
        cy: 119.5,
    };
    let truth = CameraPose {
        position: Vector3::new(0.0, 0.0, 1000.0),
        orientation: UnitQuaternion::identity(),
    };
    let pose = CameraPose {
        position: truth.position + Vector3::new(30.0, -20.0, 15.0),
        orientation: UnitQuaternion::from_euler_angles(0.01, -0.01, 0.015),
    };
    let prior = PosePrior {
        pose,
        position_radius_m: 100.0,
        attitude_radius_rad: 0.1,
    };
    let frame = Frame {
        camera,
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 10,
        },
        image: GrayImage::new(320, 240),
    };
    let depth_m = (0..240)
        .flat_map(|y| {
            (0..320).map(move |x| (800.0 + 60.0 * (f64::from(x + y) * 0.01).sin()) as f32)
        })
        .collect::<Vec<_>>();
    let mut matches = Vec::new();
    for y in (30..210).step_by(20) {
        for x in (30..290).step_by(20) {
            let p = Vector2::new(x as f64, y as f64);
            let world = camera.unproject(&pose, p, f64::from(depth_m[y * 320 + x]));
            matches.push(PixelMatch {
                reference: p,
                query: camera.project(&truth, world).expect("visible point"),
            });
        }
    }
    let reference = ReferenceView {
        map: MapRevision {
            release_id: "fixture".into(),
            manifest_sha256: "a".repeat(64),
        },
        pose,
        image: GrayImage::new(320, 240),
        depth_m,
    };
    (frame, reference, prior, matches, truth)
}

#[test]
fn geometry_result_has_pose_identity_covariance_and_stamp() {
    let (frame, reference, prior, matches, truth) = scene();
    let mut localizer =
        Localizer::new(KnownMatches(matches), LocalizerConfig::default()).expect("valid config");
    let result = localizer
        .estimate_blocking(&frame, &reference, &prior)
        .expect("good geometry");
    assert!((result.pose.position - truth.position).norm() < 0.01);
    assert_eq!(result.map, reference.map);
    assert_eq!(result.stamp, frame.stamp);
    assert!(result.geometry_covariance.cholesky().is_some());
    assert!(result.quality.occupied_cells >= 8);
    assert!(matches!(
        localizer.estimate_blocking(&frame, &reference, &prior),
        Err(VisualError::FrameOrder { .. })
    ));
}

#[test]
fn prior_bounds_missing_depth_and_blank_frames_reject() {
    let (frame, mut reference, mut prior, matches, _) = scene();
    prior.position_radius_m = 1.0;
    let mut localizer = Localizer::new(KnownMatches(matches.clone()), LocalizerConfig::default())
        .expect("valid config");
    assert!(matches!(
        localizer.estimate_blocking(&frame, &reference, &prior),
        Err(VisualError::OutsidePrior { .. })
    ));
    prior.position_radius_m = 100.0;
    reference.depth_m.fill(0.0);
    let mut localizer =
        Localizer::new(KnownMatches(matches), LocalizerConfig::default()).expect("valid config");
    assert!(matches!(
        localizer.estimate_blocking(&frame, &reference, &prior),
        Err(VisualError::InsufficientMatches { found: 0, .. })
    ));
    let mut localizer =
        Localizer::new(PyramidalMatcher, LocalizerConfig::default()).expect("valid config");
    assert!(
        localizer
            .estimate_blocking(&frame, &reference, &prior)
            .is_err()
    );
}

#[test]
fn sequence_wrap_and_rejected_frame_consumption_are_explicit() {
    let (mut frame, reference, prior, _, _) = scene();
    let mut localizer =
        Localizer::new(PyramidalMatcher, LocalizerConfig::default()).expect("valid config");
    frame.stamp.sequence = u64::MAX;
    assert!(matches!(
        localizer.estimate_blocking(&frame, &reference, &prior),
        Err(VisualError::InsufficientMatches { .. })
    ));
    frame.stamp = FrameStamp {
        sequence: 0,
        capture_time_ns: 11,
    };
    assert!(matches!(
        localizer.estimate_blocking(&frame, &reference, &prior),
        Err(VisualError::InsufficientMatches { .. })
    ));
    frame.stamp.capture_time_ns = 12;
    assert!(matches!(
        localizer.estimate_blocking(&frame, &reference, &prior),
        Err(VisualError::Invalid {
            field: "frame sequence"
        })
    ));
}

#[test]
fn repeated_matches_do_not_inflate_evidence() {
    let (frame, reference, prior, matches, _) = scene();
    let repeated = matches[..8].iter().copied().cycle().take(160).collect();
    let mut localizer =
        Localizer::new(KnownMatches(repeated), LocalizerConfig::default()).expect("valid config");
    assert!(matches!(
        localizer.estimate_blocking(&frame, &reference, &prior),
        Err(VisualError::InsufficientMatches { found: 8, .. })
    ));
}

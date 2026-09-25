#![allow(clippy::expect_used)]
use super::*;
use crate::{CameraModel, FrameStamp, LocalizerConfig};
use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector2, Vector3};

fn scene() -> (
    Frame,
    Frame,
    ReferenceView,
    PosePrior,
    CameraPose,
    Vec<PixelMatch>,
) {
    let camera = CameraModel {
        width: 320,
        height: 240,
        fx: 290.0,
        fy: 290.0,
        cx: 159.5,
        cy: 119.5,
    };
    let previous = Frame {
        stamp: FrameStamp {
            sequence: 10,
            capture_time_ns: 100,
        },
        camera,
        image: GrayImage::new(320, 240),
    };
    let current = Frame {
        stamp: FrameStamp {
            sequence: 11,
            capture_time_ns: 200,
        },
        camera,
        image: GrayImage::new(320, 240),
    };
    let pose = CameraPose {
        position: Vector3::new(40.0, -20.0, 120.0),
        orientation: UnitQuaternion::from_euler_angles(0.2, -0.1, 0.8),
    };
    let expected = CameraPose {
        position: pose.position + Vector3::new(2.0, -1.0, 0.3),
        orientation: UnitQuaternion::from_euler_angles(0.21, -0.09, 0.82),
    };
    let surface = ReferenceView {
        map: MapRevision {
            release_id: "terrain".into(),
            manifest_sha256: "a".repeat(64),
        },
        frame: LocalFrame::anchor_mercator(40.0, -74.0).expect("anchor"),
        pose,
        image: GrayImage::new(320, 240),
        depth_m: vec![100.0; 320 * 240],
    };
    let prior = PosePrior {
        pose,
        position_radius_m: 50.0,
        attitude_radius_rad: 1.0,
    };
    let pairs: Vec<_> = (40..280)
        .step_by(30)
        .flat_map(|x| {
            (40..200).step_by(30).map(move |y| {
                let reference = Vector2::new(f64::from(x), f64::from(y));
                let world = camera.unproject(&pose, reference, 100.0);
                PixelMatch {
                    reference,
                    query: camera.project(&expected, world).expect("visible"),
                }
            })
        })
        .collect();
    (previous, current, surface, prior, expected, pairs)
}

#[test]
fn tracking_preserves_conditional_pose_and_observation_identity() {
    let (previous, mut current, mut surface, prior, expected, pairs) = scene();
    let camera = current.camera;
    let verifier = PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    let check = |frame: &Frame, view: &ReferenceView, matches: &[PixelMatch]| {
        verifier.track(
            frame,
            TrackingReference {
                observation: &previous,
                surface: view,
            },
            &prior,
            matches,
            "test",
        )
    };
    let first = check(&current, &surface, &pairs).expect("relative geometry");
    assert!((first.pose.position - expected.position).norm() < 0.01);
    assert!(first.pose.orientation.angle_to(&expected.orientation) < 0.0001);
    assert_eq!(
        first.reference_observation_sha256,
        previous.evidence_sha256()
    );
    assert_eq!(first.observation_sha256, current.evidence_sha256());
    let again = check(&current, &surface, &pairs).expect("repeat");
    assert_eq!(first.quality.inliers, again.quality.inliers);
    assert_eq!(first.pose.position, again.pose.position);
    assert!(check(&previous, &surface, &pairs).is_err());
    current.camera.fx += 1.0;
    assert!(check(&current, &surface, &pairs).is_err());
    current.camera = camera;
    assert!(check(&current, &surface, &[]).is_err());
    surface.depth_m.fill(0.0);
    assert!(check(&current, &surface, &pairs).is_err());
}

#[test]
fn fixed_tilt_is_explicit_and_preserves_oblique_attitude() {
    let (previous, current, mut surface, prior, _, _) = scene();
    let expected = CameraPose {
        position: surface.pose.position + Vector3::new(2.0, -1.0, 0.3),
        orientation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.04)
            * surface.pose.orientation,
    };
    let pairs: Vec<_> = (30..290)
        .step_by(25)
        .flat_map(|x| {
            (30..210).step_by(25).map(move |y| {
                let reference = Vector2::new(f64::from(x), f64::from(y));
                let world = current.camera.unproject(&surface.pose, reference, 100.0);
                PixelMatch {
                    reference,
                    query: current.camera.project(&expected, world).expect("visible"),
                }
            })
        })
        .collect();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    let check = |view: &ReferenceView, p: &[PixelMatch]| {
        verifier.track_with_motion(
            &current,
            TrackingReference {
                observation: &previous,
                surface: view,
            },
            &prior,
            p,
            "synthetic",
            TrackingMotion::FixedTilt,
        )
    };
    let mut with_outliers = pairs.clone();
    with_outliers.extend(pairs.iter().take(20).map(|p| PixelMatch {
        reference: p.reference + Vector2::new(1.0, 1.0),
        query: Vector2::new(25.0, 25.0),
    }));
    let result = check(&surface, &with_outliers).expect("constrained geometry");
    assert!((result.pose.position - expected.position).norm() < 0.01);
    assert!(result.pose.orientation.angle_to(&expected.orientation) < 0.0001);
    assert_eq!(result.observation_sha256, current.evidence_sha256());
    assert_eq!(result.motion, TrackingMotion::FixedTilt);
    assert!(check(&surface, &[]).is_err());
    surface.depth_m.fill(0.0);
    assert!(check(&surface, &pairs).is_err());
}

#[test]
fn sparse_camera_tracking_is_rejected_even_when_a_pose_can_be_fitted() {
    let (previous, current, surface, prior, _, pairs) = scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    let sparse: Vec<_> = pairs.iter().step_by(4).copied().collect();
    assert!((6..20).contains(&sparse.len()));
    let result = verifier.track(
        &current,
        TrackingReference {
            observation: &previous,
            surface: &surface,
        },
        &prior,
        &sparse,
        "sparse-adapter",
    );
    assert!(matches!(
        result,
        Err(VisualError::InsufficientMatches { required: 20, .. })
    ));
}

#[test]
fn supplied_pose_checks_expose_conflicts_without_refitting_the_candidate() {
    let (previous, mut current, mut surface, prior, expected, pairs) = scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    let check = |frame: &Frame, view: &ReferenceView, pose: &CameraPose| {
        verifier.check_tracking_pose(
            frame,
            TrackingReference {
                observation: &previous,
                surface: view,
            },
            &prior,
            pose,
            &pairs,
            "test-pairs",
        )
    };
    let first = check(&current, &surface, &expected).expect("supported pose");
    let repeat = check(&current, &surface, &expected).expect("same evidence");
    assert_eq!(first.inliers, repeat.inliers);
    assert!(first.reprojection_rms_px < 1e-8);
    let wrong = CameraPose {
        orientation: expected.orientation * UnitQuaternion::from_euler_angles(0.2, 0.1, -0.1),
        ..expected
    };
    assert!(check(&current, &surface, &wrong).is_err());
    surface.pose = wrong;
    assert!(check(&current, &surface, &expected).is_err());
    surface.pose = prior.pose;
    assert!(check(&previous, &surface, &expected).is_err());
    current.camera.fx += 1.0;
    assert!(check(&current, &surface, &expected).is_err());
    current.camera = previous.camera;
    surface.depth_m.fill(0.0);
    assert!(check(&current, &surface, &expected).is_err());
}

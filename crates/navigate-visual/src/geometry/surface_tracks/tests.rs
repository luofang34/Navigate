#![allow(clippy::expect_used)]
use super::*;
use crate::{CameraModel, FrameStamp, LocalizerConfig, PosePrior};
use image::GrayImage;
use nalgebra::UnitQuaternion;

fn scene() -> (Frame, ReferenceView, Vec<Vector2<f64>>, PosePrior) {
    let camera = CameraModel {
        width: 320,
        height: 240,
        fx: 280.0,
        fy: 280.0,
        cx: 159.5,
        cy: 119.5,
    };
    let pose = CameraPose {
        position: Vector3::new(0.0, 0.0, 100.0),
        orientation: UnitQuaternion::identity(),
    };
    let observation = Frame {
        camera,
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 0,
        },
        image: GrayImage::new(320, 240),
    };
    let surface = ReferenceView {
        pose,
        map: MapRevision {
            release_id: "fixture".into(),
            manifest_sha256: "a".repeat(64),
        },
        frame: LocalFrame::anchor_mercator(40.0, -74.0).expect("frame"),
        image: GrayImage::new(320, 240),
        depth_m: vec![83.0; 320 * 240],
    };
    let points = (25..220)
        .step_by(28)
        .flat_map(|y| {
            (25..300)
                .step_by(30)
                .map(move |x| Vector2::new(f64::from(x), f64::from(y)))
        })
        .collect();
    let prior = PosePrior {
        pose,
        position_radius_m: 200.0,
        attitude_radius_rad: 3.0,
    };
    (observation, surface, points, prior)
}
#[test]
fn world_points_stay_fixed_while_all_camera_rotations_change() {
    let (first, surface, pixels, prior) = scene();
    let mut tracks = SurfaceTracks::new(&first, &surface, &pixels).expect("tracks");
    let verifier = PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    let source = first.evidence_sha256();
    for step in 1..=20 {
        let t = f64::from(step);
        let pose = CameraPose {
            position: Vector3::new(t * 0.2, t * -0.1, 100.0 + t * 0.1),
            orientation: UnitQuaternion::from_euler_angles(t * 0.002, t * -0.001, t * 0.004),
        };
        let observation = Frame {
            camera: first.camera,
            stamp: FrameStamp {
                sequence: step as u64,
                capture_time_ns: step as u64 * 200_000_000,
            },
            image: GrayImage::new(320, 240),
        };
        let world: Vec<_> = tracks.points.iter().map(|p| p.world).collect();
        let locations: Vec<_> = world
            .iter()
            .map(|&w| first.camera.project(&pose, w))
            .collect();
        let update = tracks
            .update(
                &verifier,
                &observation,
                &prior,
                &locations,
                "oracle-point-tracker",
            )
            .expect("full pose fit");
        assert!((update.proposal.pose.position - pose.position).norm() < 1e-5);
        assert!(update.proposal.pose.orientation.angle_to(&pose.orientation) < 1e-7);
        assert_eq!(update.proposal.motion, super::super::TrackingMotion::Free);
        assert_eq!(
            update.depth_observations.as_slice(),
            std::slice::from_ref(&source)
        );
        assert_eq!(
            tracks.points.iter().map(|p| p.world).collect::<Vec<_>>(),
            world
        );
    }
}
#[test]
fn failed_updates_and_new_rendered_depth_cannot_move_existing_world_points() {
    let (first, mut surface, pixels, prior) = scene();
    let mut tracks = SurfaceTracks::new(&first, &surface, &pixels).expect("tracks");
    let world: Vec<_> = tracks.points.iter().map(|p| p.world).collect();
    surface.depth_m.fill(120.0);
    tracks
        .replenish(&surface, &pixels)
        .expect("same tracked features");
    assert_eq!(
        tracks.points.iter().map(|p| p.world).collect::<Vec<_>>(),
        world
    );
    let verifier = PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    let mut next = copy_frame(&first);
    next.stamp.sequence = 1;
    next.stamp.capture_time_ns = 200_000_000;
    assert!(
        tracks
            .update(
                &verifier,
                &next,
                &prior,
                &vec![None; tracks.len()],
                "oracle"
            )
            .is_err()
    );
    assert_eq!(
        tracks.observation().evidence_sha256(),
        first.evidence_sha256()
    );
    assert!(
        tracks
            .update(&verifier, &next, &prior, &[], "oracle")
            .is_err()
    );
    let same: Vec<_> = tracks.pixels().into_iter().map(Some).collect();
    assert!(
        tracks
            .update(&verifier, &first, &prior, &same, "oracle")
            .is_err()
    );
    next.camera.fx += 1.0;
    assert!(
        tracks
            .update(&verifier, &next, &prior, &same, "oracle")
            .is_err()
    );
    surface.map.manifest_sha256 = "b".repeat(64);
    assert!(tracks.replenish(&surface, &pixels).is_err());
    surface.depth_m.fill(0.0);
    assert!(SurfaceTracks::new(&first, &surface, &pixels).is_err());
}

fn map_matches(
    frame: &Frame,
    surface: &ReferenceView,
    pixels: &[Vector2<f64>],
) -> Vec<crate::PixelMatch> {
    let pose = CameraPose {
        position: Vector3::new(2.0, -1.0, 103.0),
        orientation: UnitQuaternion::from_euler_angles(0.01, 0.02, -0.03),
    };
    pixels
        .iter()
        .enumerate()
        .map(|(i, &reference)| {
            let world = frame.camera.unproject(&surface.pose, reference, 83.0);
            let noise = Vector2::new(0.4 * (i as f64).sin(), 0.3 * (i as f64 * 0.7).cos());
            crate::PixelMatch {
                reference,
                query: frame.camera.project(&pose, world).expect("visible") + noise,
            }
        })
        .collect()
}

#[test]
fn matched_map_points_do_not_move_to_fit_the_estimated_camera() {
    let (first, surface, pixels, prior) = scene();
    let matches = map_matches(&first, &surface, &pixels);
    let verifier = PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    let (mut tracks, anchor) = SurfaceTracks::from_map_matches(
        &first,
        &surface,
        &prior,
        &matches,
        &verifier,
        "map-adapter",
    )
    .expect("map geometry");
    assert!(anchor.quality.reprojection_rms_px > 0.1);
    let truth = CameraPose {
        position: Vector3::new(4.0, 3.0, 101.0),
        orientation: UnitQuaternion::from_euler_angles(0.04, -0.025, 0.09),
    };
    let locations: Vec<_> = tracks
        .pixels()
        .iter()
        .map(|pixel| {
            let pair = matches
                .iter()
                .find(|m| m.query == *pixel)
                .expect("matched pixel");
            let world = first.camera.unproject(&surface.pose, pair.reference, 83.0);
            first.camera.project(&truth, world)
        })
        .collect();
    let mut current = copy_frame(&first);
    current.stamp.sequence = 1;
    current.stamp.capture_time_ns = 200_000_000;
    let update = tracks
        .update(&verifier, &current, &prior, &locations, "point-adapter")
        .expect("free pose");
    assert!((update.proposal.pose.position - truth.position).norm() < 1e-5);
    assert!(
        update
            .proposal
            .pose
            .orientation
            .angle_to(&truth.orientation)
            < 1e-7
    );
    assert_eq!(update.proposal.motion, super::super::TrackingMotion::Free);
    assert_eq!(update.proposal.map, surface.map);
    assert_eq!(update.depth_observations, vec![first.evidence_sha256()]);
    assert_eq!(
        update.proposal.reference_observation_sha256,
        first.evidence_sha256()
    );
    assert!(
        tracks
            .update(&verifier, &current, &prior, &locations, "point-adapter")
            .is_err()
    );
}

#[test]
fn map_point_seeds_require_the_full_map_policy_and_valid_reference() {
    let (frame, mut surface, pixels, prior) = scene();
    let matches = map_matches(&frame, &surface, &pixels);
    let verifier = PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    for (pairs, backend) in [(&matches[..3], "adapter"), (&matches[..], "")] {
        assert!(
            SurfaceTracks::from_map_matches(&frame, &surface, &prior, pairs, &verifier, backend,)
                .is_err()
        );
    }
    surface.depth_m.fill(0.0);
    assert!(
        SurfaceTracks::from_map_matches(&frame, &surface, &prior, &matches, &verifier, "adapter",)
            .is_err()
    );
    surface.depth_m.fill(83.0);
    surface.map.manifest_sha256.clear();
    assert!(
        SurfaceTracks::from_map_matches(&frame, &surface, &prior, &matches, &verifier, "adapter",)
            .is_err()
    );
}

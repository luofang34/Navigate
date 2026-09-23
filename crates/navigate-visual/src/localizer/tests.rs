#![allow(clippy::expect_used)]

use super::*;
use crate::{CameraModel, PixelMatch, PyramidalMatcher};
use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector2, Vector3};

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
fn candidate_initializes_pose_without_replacing_prior_bounds() {
    let (frame, reference, mut prior, matches, truth) = scene();
    prior.pose.orientation = UnitQuaternion::from_euler_angles(0.0, 0.0, 2.8);
    prior.attitude_radius_rad = std::f64::consts::PI;
    let mut localizer = Localizer::new(KnownMatches(matches.clone()), LocalizerConfig::default())
        .expect("localizer");
    let estimate = localizer
        .estimate_blocking(&frame, &reference, &prior)
        .expect("candidate is within broad prior");
    assert!((estimate.pose.position - truth.position).norm() < 0.1);
    prior.attitude_radius_rad = 0.1;
    let mut restricted =
        Localizer::new(KnownMatches(matches), LocalizerConfig::default()).expect("localizer");
    assert!(matches!(
        restricted.estimate_blocking(&frame, &reference, &prior),
        Err(VisualError::OutsidePrior { .. })
    ));
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

#[test]
fn arbitrary_orientation_and_nonplanar_depth_use_shared_geometry() {
    let (frame, mut reference, mut prior, matches, mut truth) = scene();
    let rotation = UnitQuaternion::from_euler_angles(1.25, -0.35, 2.1);
    let turn = |pose: &mut CameraPose| {
        pose.position = rotation * pose.position;
        pose.orientation = rotation * pose.orientation;
    };
    turn(&mut reference.pose);
    turn(&mut prior.pose);
    turn(&mut truth);
    let verifier = crate::PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    let result = verifier
        .verify(
            &frame,
            &reference,
            &prior,
            &matches,
            "custom-correspondences",
        )
        .expect("nonplanar forward-looking geometry");
    assert!((result.pose.position - truth.position).norm() < 0.01);
    assert!(result.pose.orientation.angle_to(&truth.orientation) < 1e-5);
    assert_eq!(result.backend, "custom-correspondences");
}

#[test]
fn repeated_refinement_replaces_a_result_without_gaining_precision() {
    use crate::{CandidateDecision, CandidateId, CandidateResults, PoseVerifier};
    let (frame, reference, prior, matches, _) = scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    let mut candidates = CandidateResults::new(&frame);
    let first = verifier
        .verify(&frame, &reference, &prior, &matches, "custom")
        .expect("first fit");
    let covariance = first.geometry_covariance;
    let inliers = first.quality.inliers;
    candidates
        .record(CandidateId(7), Ok(first))
        .expect("record first fit");
    for _ in 0..4 {
        let again = verifier
            .verify(&frame, &reference, &prior, &matches, "custom")
            .expect("repeated fit");
        assert_eq!(again.geometry_covariance, covariance);
        assert_eq!(again.quality.inliers, inliers);
        candidates
            .record(CandidateId(7), Ok(again))
            .expect("replace same hypothesis");
    }
    assert_eq!(candidates.iter().count(), 1);
    assert_eq!(
        candidates.decision(),
        CandidateDecision::Unique(CandidateId(7))
    );
    candidates
        .record(CandidateId(7), Err(VisualError::DegenerateGeometry))
        .expect("record failed refinement");
    assert_eq!(candidates.decision(), CandidateDecision::Rejected);
}

#[test]
fn unresolved_places_remain_distinct_and_foreign_evidence_is_rejected() {
    use crate::{CandidateDecision, CandidateId, CandidateResults, PoseVerifier};
    let (mut frame, mut reference, mut prior, matches, _) = scene();
    prior.position_radius_m = 1000.0;
    let verifier = PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    let mut candidates = CandidateResults::new(&frame);
    let first = verifier
        .verify(&frame, &reference, &prior, &matches, "custom")
        .expect("first place");
    let first_position = first.pose.position;
    candidates
        .record(CandidateId(1), Ok(first))
        .expect("first hypothesis");
    reference.pose.position.x += 500.0;
    reference.map.release_id = "second-reference-release".into();
    reference.map.manifest_sha256 = "b".repeat(64);
    let second = verifier
        .verify(&frame, &reference, &prior, &matches, "custom")
        .expect("second place");
    assert!((second.pose.position - first_position).norm() > 499.0);
    candidates
        .record(CandidateId(2), Ok(second))
        .expect("second hypothesis");
    assert_eq!(
        candidates.decision(),
        CandidateDecision::Unresolved(vec![CandidateId(1), CandidateId(2)])
    );
    assert_eq!(candidates.iter().count(), 2);
    frame.image.put_pixel(0, 0, image::Luma([1]));
    let different = verifier
        .verify(&frame, &reference, &prior, &matches, "custom")
        .expect("different observation");
    assert!(candidates.record(CandidateId(3), Ok(different)).is_err());
    let identity = frame.evidence_sha256();
    frame.camera.fx += 0.1;
    assert_ne!(frame.evidence_sha256(), identity);
}

#[test]
fn majority_outliers_do_not_hide_nonplanar_oblique_consensus() {
    let (frame, mut reference, mut prior, mut pairs, mut truth) = scene();
    let rotation = UnitQuaternion::from_euler_angles(1.2, -0.3, 2.0);
    for pose in [&mut reference.pose, &mut prior.pose, &mut truth] {
        pose.position = rotation * pose.position;
        pose.orientation = rotation * pose.orientation;
    }
    let original = pairs.clone();
    let mut seed = 7123_u64;
    for (index, pair) in pairs.iter_mut().enumerate() {
        if index % 4 != 0 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            pair.query = Vector2::new(
                8.0 + ((seed >> 32) % 300) as f64,
                8.0 + ((seed >> 16) % 220) as f64,
            );
        }
    }
    let verifier = crate::PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    let result = verifier
        .verify(&frame, &reference, &prior, &pairs, "outlier-test")
        .expect("distributed surface consensus");
    assert!((result.pose.position - truth.position).norm() < 0.1);
    assert!(result.quality.inliers >= original.len() / 4);
    assert!(result.quality.inliers < pairs.len() / 2);
    let repeated = verifier
        .verify(&frame, &reference, &prior, &pairs, "outlier-test")
        .expect("repeat");
    assert_eq!(result.geometry_covariance, repeated.geometry_covariance);
    assert_eq!(result.quality.inliers, repeated.quality.inliers);
}

#[test]
fn random_correspondences_do_not_pass_consensus_acceptance() {
    let (frame, reference, prior, mut pairs, _) = scene();
    let mut seed = 9231_u64;
    for pair in &mut pairs {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        pair.query = Vector2::new(
            8.0 + ((seed >> 32) % 300) as f64,
            8.0 + ((seed >> 16) % 220) as f64,
        );
    }
    let verifier = crate::PoseVerifier::new(LocalizerConfig::default()).expect("policy");
    assert!(
        verifier
            .verify(&frame, &reference, &prior, &pairs, "negative-test")
            .is_err()
    );
}

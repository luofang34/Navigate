use super::*;
use crate::{LocalizerConfig, orthographic::tests::scene};
use nalgebra::Vector2;

#[test]
fn insufficient_unique_map_evidence_is_rejected_before_pose_fitting() {
    let (frame, mut reference, prior, pairs, _) = scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let mut selected = pairs[..19].to_vec();
    // These observations have no consistent pose, but their count is already decisive.
    for (i, pair) in selected.iter_mut().enumerate() {
        pair.query = Vector2::new(17.0 + i as f64 * 29.0, 41.0 + (i % 7) as f64 * 51.0);
    }
    selected.extend_from_within(..);
    let mut invalid = pairs[20];
    invalid.query.x = f64::NAN;
    selected.push(invalid);
    let missing = pairs[21];
    let index = missing.reference.y.round() as usize * reference.image.width() as usize
        + missing.reference.x.round() as usize;
    reference.heights_m[index] = f32::NAN;
    selected.push(missing);
    assert!(matches!(
        verifier.verify_orthographic(&frame, &reference, &prior, &selected, "test"),
        Err(VisualError::InsufficientMatches {
            found: 19,
            required: 20
        })
    ));
}

#[test]
fn early_rejection_uses_configured_count_and_validates_the_prior() {
    let (frame, reference, mut prior, pairs, _) = scene();
    let verifier = PoseVerifier::new(LocalizerConfig {
        min_inliers: 32,
        ..LocalizerConfig::default()
    })
    .unwrap();
    assert!(matches!(
        verifier.verify_orthographic(&frame, &reference, &prior, &pairs[..31], "test"),
        Err(VisualError::InsufficientMatches {
            found: 31,
            required: 32
        })
    ));
    prior.position_radius_m = f64::NAN;
    assert!(matches!(
        verifier.verify_orthographic(&frame, &reference, &prior, &[], "test"),
        Err(VisualError::Invalid { .. })
    ));
}

#[test]
fn sufficient_evidence_retains_the_full_fit_and_covariance() {
    let (frame, reference, prior, pairs, _) = scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    for count in [20, 40, pairs.len()] {
        let selected: Vec<_> = (0..count).map(|i| pairs[i * pairs.len() / count]).collect();
        let points = correspondences(&frame, &reference, &selected);
        let (pose, inliers, depth_matches) = verifier
            .fit_points(&frame, prior.pose, &prior, points, Motion::Free)
            .unwrap();
        let (points, cells) = support::separated_projected(
            reference.image.dimensions().into(),
            |world| reference.geometry.project(world),
            &inliers,
            verifier.config.support_separation_px,
        );
        let support = FitSupport::from_separated(&frame.camera, points, inliers.len(), cells);
        let (quality, covariance) = verifier
            .assess_support(&frame, &pose, &support, depth_matches)
            .unwrap();
        let result = verifier
            .verify_orthographic(&frame, &reference, &prior, &selected, "test")
            .unwrap();
        assert_eq!(result.pose.position, pose.position);
        assert_eq!(result.pose.orientation, pose.orientation);
        assert_eq!(result.quality.inliers, quality.inliers);
        assert_eq!(result.quality.spatial_support, quality.spatial_support);
        assert_eq!(
            result.quality.reprojection_rms_px,
            quality.reprojection_rms_px
        );
        assert_eq!(result.geometry_covariance, covariance);
    }
}

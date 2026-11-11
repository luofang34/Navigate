//! The evidence of a match: the matches as the matcher gave them, and the
//! inliers of the fit that made the estimate.

use super::{KnownMatches, scene};
use crate::{Localizer, LocalizerConfig, VisualError};
use nalgebra::Vector2;

#[test]
fn the_evidence_flags_the_inliers_of_the_fit_and_keeps_the_estimate() {
    let (frame, reference, prior, mut matches, _) = scene();
    let clean = matches.len();
    // Two matches far off the geometry, and one on a reference pixel
    // without depth.
    matches[3].query += Vector2::new(40.0, -35.0);
    matches[17].query += Vector2::new(-30.0, 25.0);
    let mut reference = reference;
    let hole = matches[40].reference;
    reference.depth_m[hole.y as usize * 320 + hole.x as usize] = 0.0;
    let mut plain =
        Localizer::new(KnownMatches(matches.clone()), LocalizerConfig::default()).expect("config");
    let mut shown =
        Localizer::new(KnownMatches(matches.clone()), LocalizerConfig::default()).expect("config");
    let expected = plain
        .estimate_blocking(&frame, &reference, &prior)
        .expect("good geometry");
    let (estimate, evidence) = shown.estimate_with_evidence_blocking(&frame, &reference, &prior);
    let estimate = estimate.expect("good geometry");
    assert_eq!(estimate.pose.position, expected.pose.position);
    assert_eq!(estimate.pose.orientation, expected.pose.orientation);
    assert_eq!(estimate.geometry_covariance, expected.geometry_covariance);
    assert_eq!(evidence.matches, matches);
    assert_eq!(evidence.inliers.len(), clean);
    let flagged = evidence.inliers.iter().filter(|inlier| **inlier).count();
    assert_eq!(flagged, estimate.quality.inliers);
    assert!(!evidence.inliers[3] && !evidence.inliers[17] && !evidence.inliers[40]);
    assert_eq!(flagged, clean - 3);
    // The stream state is that of `estimate_blocking`: the stamp is spent.
    assert!(matches!(
        shown
            .estimate_with_evidence_blocking(&frame, &reference, &prior)
            .0,
        Err(VisualError::FrameOrder { .. })
    ));
}

#[test]
fn a_frame_refused_before_the_fit_has_no_inlier_flags() {
    let (frame, mut reference, prior, matches, _) = scene();
    reference.depth_m.fill(0.0);
    let mut localizer =
        Localizer::new(KnownMatches(matches.clone()), LocalizerConfig::default()).expect("config");
    let (estimate, evidence) =
        localizer.estimate_with_evidence_blocking(&frame, &reference, &prior);
    assert!(matches!(
        estimate,
        Err(VisualError::InsufficientMatches { found: 0, .. })
    ));
    assert_eq!(evidence.matches, matches);
    assert!(evidence.inliers.is_empty());
}

//! The local geometry covariance of an estimate: the share of the residual
//! that is not independent pixel noise.

use super::{KnownMatches, scene};
use crate::{Localizer, LocalizerConfig};
use nalgebra::Vector2;

#[test]
fn a_residual_above_the_matcher_precision_widens_the_covariance() {
    let (frame, reference, prior, mut matches, _) = scene();
    // Residuals of about half a pixel, alternating in sign.
    for (i, pair) in matches.iter_mut().enumerate() {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        pair.query += Vector2::new(0.4 * sign, -0.3 * sign);
    }
    let config = LocalizerConfig {
        pixel_noise_floor: 0.01,
        ..LocalizerConfig::default()
    };
    let estimate = |config: LocalizerConfig| {
        Localizer::new(KnownMatches(matches.clone()), config)
            .expect("config")
            .estimate_blocking(&frame, &reference, &prior)
            .expect("good geometry")
    };
    let independent = estimate(config);
    let shared = estimate(LocalizerConfig {
        residual_correlation_px: 1.0,
        ..config
    });
    let rms = independent.quality.reprojection_rms_px;
    assert!(rms > 0.1, "{rms}");
    let ratio = shared.geometry_covariance[(0, 0)] / independent.geometry_covariance[(0, 0)];
    assert!((ratio - (1.0 + rms * rms)).abs() < 1e-9, "{ratio} {rms}");
    assert!(
        crate::PoseVerifier::new(LocalizerConfig {
            residual_correlation_px: 0.0,
            ..config
        })
        .is_err()
    );
}

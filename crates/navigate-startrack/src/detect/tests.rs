use nalgebra::Vector2;
use navigate_camera::{ImageView, Pixels};

use super::*;
use crate::synthetic::{
    BACKGROUND, RENDER_MAGNITUDE, SIZE, Source, catalogue, render, sources, truth, view,
};

/// The distance from each unsaturated source to the nearest centroid.
fn centroid_errors(sources: &[Source], stars: &[StarCentroid], min_flux: f64) -> Vec<f64> {
    sources
        .iter()
        .filter(|s| s.flux >= min_flux)
        .filter_map(|s| {
            let nearest = stars
                .iter()
                .filter(|c| !c.saturated)
                .map(|c| (c.pixel - s.pixel).norm())
                .fold(f64::INFINITY, f64::min);
            nearest.is_finite().then_some(nearest)
        })
        .filter(|e| *e < 3.0)
        .collect()
}

/// The sources farther than `px` from each other source and from the
/// image edge, so their images do not merge or leave the image.
fn isolated(sources: &[Source], px: f64) -> Vec<Source> {
    let inner = px..f64::from(SIZE) - 1.0 - px;
    sources
        .iter()
        .filter(|a| inner.contains(&a.pixel.x) && inner.contains(&a.pixel.y))
        .filter(|a| {
            sources
                .iter()
                .all(|b| std::ptr::eq(*a, b) || (a.pixel - b.pixel).norm() > px)
        })
        .copied()
        .collect()
}

#[test]
fn noiseless_centroids_are_within_five_hundredths_of_a_pixel() {
    let sources = isolated(&sources(&catalogue(RENDER_MAGNITUDE), &truth()), 12.0);
    assert!(sources.len() >= 20, "{} stars in view", sources.len());
    let data = render(&sources, 0.0, 1);
    let stars = detect_stars(&view(&data), &DetectConfig::default()).expect("detect");
    let errors = centroid_errors(&sources, &stars, 0.0);
    assert_eq!(errors.len(), sources.len());
    let worst = errors.iter().copied().fold(0.0, f64::max);
    assert!(worst < 0.05, "largest centroid error {worst} px");
    // Brightest first.
    assert!(stars.windows(2).all(|w| w[0].flux >= w[1].flux));
}

#[test]
fn noisy_bright_centroids_stay_near_the_truth() {
    let sources = isolated(&sources(&catalogue(RENDER_MAGNITUDE), &truth()), 12.0);
    let data = render(&sources, 5.0, 7);
    let stars = detect_stars(&view(&data), &DetectConfig::default()).expect("detect");
    let errors = centroid_errors(&sources, &stars, crate::synthetic::flux_of(6.0));
    assert!(errors.len() >= 10, "{} stars", errors.len());
    let rms = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
    assert!(rms < 0.1, "RMS centroid error {rms} px");
}

#[test]
fn a_hot_pixel_is_not_a_star() {
    let n = SIZE as usize;
    let mut data = vec![BACKGROUND as u16; n * n];
    data[500 * n + 300] = 5000;
    let star = Source {
        hr: 0,
        pixel: Vector2::new(700.3, 200.6),
        flux: 2.0e4,
    };
    let rendered = render(&[star], 0.0, 1);
    for (i, v) in rendered.iter().enumerate() {
        if *v > BACKGROUND as u16 {
            data[i] = *v;
        }
    }
    let stars = detect_stars(&view(&data), &DetectConfig::default()).expect("detect");
    assert_eq!(stars.len(), 1);
    assert!((stars[0].pixel - star.pixel).norm() < 0.05);
    assert!(stars[0].pixel_count >= 9);
    assert!(!stars[0].saturated);
    let single = DetectConfig {
        min_pixels: 1,
        ..DetectConfig::default()
    };
    assert_eq!(
        detect_stars(&view(&data), &single).expect("detect").len(),
        2
    );
}

#[test]
fn gray8_and_rgb8_give_the_same_centroid() {
    let (w, h) = (32_u32, 24_u32);
    let mut gray = vec![10_u8; (w * h) as usize];
    for (dx, dy, v) in [
        (0, 0, 200),
        (1, 0, 120),
        (0, 1, 90),
        (1, 1, 60),
        (-1, 0, 40),
    ] {
        gray[((12 + dy) * w as i32 + 15 + dx) as usize] = v;
    }
    let rgb: Vec<u8> = gray.iter().flat_map(|v| [*v, *v, *v]).collect();
    let config = DetectConfig {
        tile_px: 8,
        ..DetectConfig::default()
    };
    let g8 = ImageView::new(
        w,
        h,
        Pixels::Gray8 {
            data: &gray,
            stride: w as usize,
        },
    )
    .expect("gray");
    let c8 = ImageView::new(
        w,
        h,
        Pixels::Rgb8 {
            data: &rgb,
            stride: 3 * w as usize,
        },
    )
    .expect("rgb");
    let a = detect_stars(&g8, &config).expect("gray");
    let b = detect_stars(&c8, &config).expect("rgb");
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert!((a[0].pixel - b[0].pixel).norm() < 1e-3);
    assert!((a[0].pixel - Vector2::new(15.0, 12.0)).norm() < 0.5);
}

#[test]
fn saturation_and_the_component_limit() {
    let bright = Source {
        hr: 0,
        pixel: Vector2::new(400.0, 400.0),
        flux: 2.0e6,
    };
    let data = render(&[bright], 0.0, 1);
    let stars = detect_stars(&view(&data), &DetectConfig::default()).expect("detect");
    assert!(stars[0].saturated);
    let sources = sources(&catalogue(RENDER_MAGNITUDE), &truth());
    let data = render(&sources, 0.0, 1);
    let tight = DetectConfig {
        max_components: 5,
        ..DetectConfig::default()
    };
    assert!(matches!(
        detect_stars(&view(&data), &tight),
        Err(StarTrackError::TooManyComponents { limit: 5 })
    ));
}

#[test]
fn invalid_settings_are_refused() {
    let data = vec![0_u16; (SIZE * SIZE) as usize];
    for config in [
        DetectConfig {
            grow_sigmas: 6.0,
            ..DetectConfig::default()
        },
        DetectConfig {
            tile_px: 0,
            ..DetectConfig::default()
        },
        DetectConfig {
            min_pixels: 0,
            ..DetectConfig::default()
        },
    ] {
        assert!(matches!(
            detect_stars(&view(&data), &config),
            Err(StarTrackError::Invalid { .. })
        ));
    }
}

#[test]
fn a_streak_is_kept_with_its_length_and_a_long_trail_is_not() {
    // 5 °/s about the body pitch axis for 0.1 s moves each star 0.5°.
    let rate = nalgebra::Vector3::new(0.0, 5.0_f64.to_radians(), 0.0);
    let catalogue = catalogue(crate::synthetic::CATALOGUE_MAGNITUDE);
    let points = crate::synthetic::streak_sources(&catalogue, &truth(), rate, 0.1);
    let data = render(&points, 5.0, 17);
    let stars = detect_stars(&view(&data), &DetectConfig::default()).expect("detect");
    let expected = 0.5_f64.to_radians() * crate::synthetic::imager().intrinsics.fx;
    // A streak that leaves the image is shorter.
    let inner = 30.0..f64::from(SIZE) - 30.0;
    let bright: Vec<&StarCentroid> = stars
        .iter()
        .filter(|s| s.flux > 1.0e4 && inner.contains(&s.pixel.x) && inner.contains(&s.pixel.y))
        .collect();
    assert!(bright.len() >= 5);
    for star in &bright {
        assert!(
            (star.length_px - expected).abs() < 0.1 * expected,
            "length {} px, travel {expected} px",
            star.length_px
        );
    }
    let short = DetectConfig {
        max_length_px: 0.5 * expected,
        ..DetectConfig::default()
    };
    let kept = detect_stars(&view(&data), &short).expect("detect");
    assert!(kept.iter().all(|s| s.length_px <= 0.5 * expected));
    assert!(kept.len() < stars.len());
    // Stars without motion have almost no length.
    let still = render(&sources(&catalogue, &truth()), 5.0, 17);
    let round = detect_stars(&view(&still), &DetectConfig::default()).expect("detect");
    let mut lengths: Vec<f64> = round
        .iter()
        .filter(|s| s.flux > 1.0e4)
        .map(|s| s.length_px)
        .collect();
    lengths.sort_by(f64::total_cmp);
    // Close double stars merge into one longer group.
    assert!(lengths[lengths.len() / 2] < 2.0, "{lengths:?}");
}

/// The nearest centroid to `source`, if it is within 3 pixels.
fn nearest<'a>(stars: &'a [StarCentroid], source: &Source) -> Option<&'a StarCentroid> {
    stars
        .iter()
        .filter(|c| (c.pixel - source.pixel).norm() < 3.0)
        .min_by(|a, b| {
            (a.pixel - source.pixel)
                .norm()
                .total_cmp(&(b.pixel - source.pixel).norm())
        })
}

#[test]
fn the_centroid_covariance_predicts_the_scatter_of_the_centroids() {
    let all = sources(&catalogue(RENDER_MAGNITUDE), &truth());
    let stars = isolated(&all, 12.0);
    let config = DetectConfig {
        electrons_per_count: 2.0,
        ..DetectConfig::default()
    };
    // Each star's squared error over its predicted variance, pooled over
    // the stars of 30 images; the pooled mean is 1 for a matching model.
    let (mut normalized, mut count) = (0.0, 0_u32);
    let mut dim_over_bright = Vec::new();
    for seed in 1..=30_u64 {
        let data = crate::synthetic::render_with_shot_noise(&stars, 5.0, 2.0, seed);
        let found = detect_stars(&view(&data), &config).expect("detect");
        let mut sigmas = Vec::new();
        for source in &stars {
            let Some(star) = nearest(&found, source).filter(|c| !c.saturated) else {
                continue;
            };
            let error = star.pixel - source.pixel;
            let information = star.covariance_px2.try_inverse().expect("inverse");
            normalized += error.dot(&(information * error)) / 2.0;
            count += 1;
            sigmas.push((source.flux, star.covariance_px2.trace().sqrt()));
        }
        sigmas.sort_by(|a, b| a.0.total_cmp(&b.0));
        if let (Some(dim), Some(bright)) = (sigmas.first(), sigmas.last()) {
            dim_over_bright.push(dim.1 / bright.1);
        }
    }
    let mean = normalized / f64::from(count);
    assert!(count > 100, "{count} centroids");
    // The PSF sampling bias of about 0.004 px adds a little to the noise.
    assert!(
        (0.7..1.5).contains(&mean),
        "normalized squared error {mean}"
    );
    assert!(dim_over_bright.iter().all(|r| *r > 1.0));
}

#[test]
fn a_streak_is_uncertain_along_its_length() {
    let rate = nalgebra::Vector3::new(0.0, 3.0_f64.to_radians(), 0.0);
    let catalogue = catalogue(crate::synthetic::CATALOGUE_MAGNITUDE);
    let points = crate::synthetic::streak_sources(&catalogue, &truth(), rate, 0.1);
    let data = render(&points, 5.0, 23);
    let stars = detect_stars(&view(&data), &DetectConfig::default()).expect("detect");
    let ratios: Vec<f64> = stars
        .iter()
        .filter(|s| s.length_px > 10.0)
        .map(|s| {
            let eigen = s.covariance_px2.symmetric_eigenvalues();
            eigen.max() / eigen.min()
        })
        .collect();
    assert!(ratios.len() >= 5);
    // A uniform line of 15 px over a point spread of 1.3 px; a close
    // double star merged into one group is rounder, so take the median.
    let mut sorted = ratios.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    assert!(median > 5.0, "{ratios:?}");
}

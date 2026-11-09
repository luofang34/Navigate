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

/// Whether two sources are closer than `px`, so their images merge.
fn isolated(sources: &[Source], px: f64) -> Vec<Source> {
    sources
        .iter()
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

use nalgebra::Vector2;

use super::*;
use crate::synthetic::{
    CATALOGUE_MAGNITUDE, RENDER_MAGNITUDE, Source, catalogue, imager, mount, pair_table, pipeline,
    prior, render, sources, truth, view,
};
use crate::{DetectConfig, detect_stars};

/// The HR number of the source nearest to `pixel`.
fn true_hr(sources: &[Source], pixel: &Vector2<f64>) -> Option<u16> {
    sources
        .iter()
        .min_by(|a, b| {
            (a.pixel - pixel)
                .norm()
                .total_cmp(&(b.pixel - pixel).norm())
        })
        .filter(|s| (s.pixel - pixel).norm() < 2.0)
        .map(|s| s.hr)
}

fn detected() -> (Vec<Source>, Vec<StarCentroid>) {
    let sources = sources(&catalogue(RENDER_MAGNITUDE), &truth());
    let data = render(&sources, 5.0, 3);
    let centroids = detect_stars(&view(&data), &DetectConfig::default()).expect("detect");
    (sources, centroids)
}

#[test]
fn a_one_degree_prior_error_identifies_the_true_stars() {
    let (sources, centroids) = detected();
    let one_degree = 1.0_f64.to_radians();
    let matches = identify_stars(
        &centroids,
        &imager(),
        &mount(),
        IdentifyMode::Prior(prior(one_degree, one_degree)),
        &catalogue(CATALOGUE_MAGNITUDE),
        &IdentifyConfig::default(),
    )
    .expect("identify");
    assert!(matches.len() >= 6, "{} matches", matches.len());
    for m in &matches {
        let centroid = &centroids[m.centroid_index];
        assert_eq!(true_hr(&sources, &centroid.pixel), Some(m.hr));
        // The prior error moves each predicted star by about 1°.
        assert!(m.separation_rad < 3.0 * one_degree);
    }
}

#[test]
fn identification_without_a_prior_finds_the_true_stars() {
    let (sources, centroids) = detected();
    let matches = identify_stars(
        &centroids,
        &imager(),
        &mount(),
        IdentifyMode::LostInSpace(pair_table()),
        &catalogue(CATALOGUE_MAGNITUDE),
        &IdentifyConfig::default(),
    )
    .expect("identify");
    assert!(matches.len() >= 8, "{} matches", matches.len());
    for m in &matches {
        let centroid = &centroids[m.centroid_index];
        assert_eq!(true_hr(&sources, &centroid.pixel), Some(m.hr));
    }
}

#[test]
fn a_zero_prior_deviation_is_refused() {
    let (_, centroids) = detected();
    let result = identify_stars(
        &centroids,
        &imager(),
        &mount(),
        IdentifyMode::Prior(prior(0.0, 0.0)),
        &catalogue(CATALOGUE_MAGNITUDE),
        &IdentifyConfig::default(),
    );
    assert!(matches!(result, Err(StarTrackError::Invalid { .. })));
}

#[test]
fn a_false_blob_does_not_match() {
    // A bright point source at least 40 pixels from every rendered star.
    let rendered = sources(&catalogue(RENDER_MAGNITUDE), &truth());
    let pixel = (2..10)
        .flat_map(|i| {
            (2..10).map(move |j| Vector2::new(100.0 * f64::from(i), 100.0 * f64::from(j)))
        })
        .find(|p| rendered.iter().all(|s| (s.pixel - p).norm() > 40.0))
        .expect("an empty area");
    let blob = Source {
        hr: 0,
        pixel,
        flux: 3.0e5,
    };
    let run = pipeline(5.0, 11, &[blob]);
    let index = run
        .centroids
        .iter()
        .position(|c| (c.pixel - blob.pixel).norm() < 0.5)
        .expect("the blob is detected");
    let stars = sources(&run.catalogue, &truth());
    assert!(
        stars.iter().all(|s| (s.pixel - blob.pixel).norm() > 30.0),
        "the blob is near a catalogue star"
    );
    assert!(run.matches.iter().all(|m| m.centroid_index != index));
    assert!(
        run.solution
            .matches
            .iter()
            .all(|m| m.centroid_index != index)
    );
}

#[test]
fn pair_consistency_removes_a_wrong_match() {
    let (_, centroids) = detected();
    let one_degree = 1.0_f64.to_radians();
    let mut matches = identify_stars(
        &centroids,
        &imager(),
        &mount(),
        IdentifyMode::Prior(prior(one_degree, one_degree)),
        &catalogue(CATALOGUE_MAGNITUDE),
        &IdentifyConfig::default(),
    )
    .expect("identify");
    let count = matches.len();
    let mut wrong = matches[0];
    wrong.centroid_index = usize::MAX;
    wrong.ecef_unit = (truth() * nalgebra::Vector3::new(0.1, 0.0, -1.0)).normalize();
    matches.insert(1, wrong);
    let kept = consistent_pairs(matches, 7e-4);
    assert_eq!(kept.len(), count);
    assert!(kept.iter().all(|m| m.centroid_index != usize::MAX));
}

#[test]
fn invalid_settings_are_refused() {
    let (_, centroids) = detected();
    let bad = IdentifyConfig {
        ambiguity_ratio: 1.5,
        ..IdentifyConfig::default()
    };
    let result = identify_stars(
        &centroids,
        &imager(),
        &mount(),
        IdentifyMode::Prior(prior(0.0, 0.01)),
        &catalogue(CATALOGUE_MAGNITUDE),
        &bad,
    );
    assert!(matches!(result, Err(StarTrackError::Invalid { .. })));
}

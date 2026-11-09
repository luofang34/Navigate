use nalgebra::Vector2;

use crate::synthetic::{
    CATALOGUE_MAGNITUDE, RENDER_MAGNITUDE, SIZE, Source, catalogue, pair_table, random_attitude,
    render, run, sources,
};
use crate::{IdentifyMode, PairCatalogue, StarTrackError};

const ARCSEC: f64 = std::f64::consts::PI / (180.0 * 3600.0);
/// Random attitudes of the lost-in-space test.
const ATTITUDES: u64 = 20;

#[test]
fn lost_in_space_solves_random_attitudes_without_a_wrong_answer() {
    let stars = catalogue(RENDER_MAGNITUDE);
    let mut solved = 0;
    for seed in 1..=ATTITUDES {
        let truth = random_attitude(seed);
        let data = render(&sources(&stars, &truth), 5.0, seed);
        let Ok(result) = run(&data, IdentifyMode::LostInSpace(pair_table())) else {
            continue;
        };
        let error = truth.angle_to(&result.solution.ecef_from_body) / ARCSEC;
        assert!(error < 20.0, "seed {seed}: wrong attitude, {error} arcsec");
        solved += 1;
    }
    assert_eq!(solved, ATTITUDES);
}

#[test]
fn a_mirrored_sky_is_not_identified() {
    let truth = random_attitude(3);
    let data = render(&sources(&catalogue(RENDER_MAGNITUDE), &truth), 5.0, 3);
    let n = SIZE as usize;
    let mirrored: Vec<u16> = (0..n * n)
        .map(|i| data[(i / n) * n + (n - 1 - i % n)])
        .collect();
    assert!(run(&data, IdentifyMode::LostInSpace(pair_table())).is_ok());
    assert!(matches!(
        run(&mirrored, IdentifyMode::LostInSpace(pair_table())),
        Err(StarTrackError::NotIdentified)
    ));
}

#[test]
fn bright_false_stars_do_not_stop_or_corrupt_the_identification() {
    let truth = random_attitude(5);
    let mut points = sources(&catalogue(RENDER_MAGNITUDE), &truth);
    let brightest = points.iter().map(|s| s.flux).fold(0.0, f64::max);
    let mut false_pixels = Vec::new();
    for (x, y) in [(150.0, 170.0), (830.0, 610.0), (420.0, 880.0)] {
        let pixel = Vector2::new(x, y);
        if points.iter().all(|s| (s.pixel - pixel).norm() > 20.0) {
            false_pixels.push(pixel);
            points.push(Source {
                hr: 0,
                pixel,
                flux: 1.5 * brightest.min(3.0e5),
            });
        }
    }
    assert!(false_pixels.len() >= 2);
    let data = render(&points, 5.0, 5);
    let result = run(&data, IdentifyMode::LostInSpace(pair_table())).expect("identified");
    assert!(truth.angle_to(&result.solution.ecef_from_body) < 20.0 * ARCSEC);
    for m in &result.solution.matches {
        let pixel = result.centroids[m.centroid_index].pixel;
        assert!(false_pixels.iter().all(|p| (p - pixel).norm() > 2.0));
    }
}

#[test]
fn three_stars_are_not_enough() {
    let truth = random_attitude(7);
    let mut points = sources(&catalogue(CATALOGUE_MAGNITUDE), &truth);
    points.truncate(3);
    let data = render(&points, 5.0, 7);
    assert!(matches!(
        run(&data, IdentifyMode::LostInSpace(pair_table())),
        Err(StarTrackError::NotIdentified)
    ));
}

#[test]
fn the_pair_table_is_sorted_and_complete() {
    let stars = catalogue(4.0);
    let max = 20.0_f64.to_radians();
    let table = PairCatalogue::new(&stars, max, 4.0).expect("table");
    let brute = stars
        .iter()
        .enumerate()
        .flat_map(|(i, a)| stars.iter().skip(i + 1).map(move |b| (a, b)))
        .filter(|(a, b)| {
            let dot: f64 = a
                .direction
                .iter()
                .zip(&b.direction)
                .map(|(x, y)| x * y)
                .sum();
            dot >= max.cos()
        })
        .count();
    assert_eq!(table.star_count(), stars.len());
    assert_eq!(table.pair_count(), brute);
    let band = table.within(0.1, 0.01);
    assert!(!band.is_empty());
    assert!(
        band.iter()
            .all(|p| (f64::from(p.angle_rad) - 0.1).abs() <= 0.010_001)
    );
    assert!(band.windows(2).all(|w| w[0].angle_rad <= w[1].angle_rad));
    assert!(matches!(
        PairCatalogue::new(&stars, 0.0, 4.0),
        Err(StarTrackError::Invalid { .. })
    ));
}

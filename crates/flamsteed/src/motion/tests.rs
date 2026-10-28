use crate::Catalog;
#[cfg(feature = "all-stars")]
use std::vec::Vec;

#[cfg(feature = "all-stars")]
const REFERENCE: &str = include_str!("../../fixtures/star-space-motion.txt");

fn separation_arcsec(a: [f64; 3], b: [f64; 3]) -> f64 {
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let sin = cross.iter().map(|c| c * c).sum::<f64>().sqrt();
    let cos: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    sin.atan2(cos).to_degrees() * 3600.0
}

// The reference includes faint stars.
#[cfg(feature = "all-stars")]
#[test]
fn space_motion_matches_the_reference_from_minus_2000_to_3000() {
    let stars: Vec<_> = Catalog::bright_star_catalogue().stars().collect();
    let mut rows = 0;
    for line in REFERENCE.lines().filter(|line| !line.starts_with('#')) {
        let v: Vec<f64> = line.split(' ').map(|x| x.parse().unwrap()).collect();
        let star = stars
            .iter()
            .find(|star| star.hr().map(f64::from) == Some(v[0]))
            .unwrap();
        let error = separation_arcsec(star.direction_at(v[1]), [v[2], v[3], v[4]]);
        // The packed positions round to 0.3 milliarcsecond.
        assert!(error < 0.01, "HR {} at {} years: {error}\"", v[0], v[1]);
        rows += 1;
    }
    assert_eq!(rows, 150);
}

#[test]
fn at_j2000_the_direction_is_the_catalogue_position() {
    for star in Catalog::bright_star_catalogue().stars().take(50) {
        let (sin_ra, cos_ra) = star.right_ascension_rad().sin_cos();
        let (sin_dec, cos_dec) = star.declination_rad().sin_cos();
        let expected = [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec];
        assert!(separation_arcsec(star.direction_at(0.0), expected) < 1e-9);
    }
}

#[test]
fn a_prepared_motion_starts_at_the_reference_position() {
    for star in Catalog::bright_star_catalogue().brighter_than(3.0).stars() {
        let motion = star.motion();
        let length = motion.position.iter().map(|c| c * c).sum::<f64>().sqrt();
        // The table stores the unit vector in steps of 2^-31.
        assert!((length - 1.0).abs() < 1e-9, "{length}");
        assert_eq!(motion.reference_epoch_years, star.reference_epoch_years);
        assert!(
            separation_arcsec(
                motion.direction_at(motion.reference_epoch_years),
                motion.position
            ) < 1e-9
        );
        assert_eq!(motion.direction_at(-2330.0), star.direction_at(-2330.0));
    }
}

#[test]
fn f32_motion_holds_a_direction_to_0_03_arcsecond() {
    let mut worst: f64 = 0.0;
    for star in Catalog::bright_star_catalogue().stars() {
        let motion = star.motion();
        let single = motion.cast::<f32>();
        for years in [-4000.0, -2330.0, 0.0, 26.0, 1000.0] {
            let reference = motion.direction_at(years);
            let direction = single.direction_at(years as f32).map(f64::from);
            worst = worst.max(separation_arcsec(direction, reference));
        }
    }
    assert!(worst < 0.03, "{worst}\"");
}

#[test]
fn position_at_is_the_direction_before_normalization() {
    for star in Catalog::bright_star_catalogue().brighter_than(2.0).stars() {
        let motion = star.motion();
        for years in [-4000.0, 0.0, 1000.0] {
            let position = motion.position_at(years);
            let length = position.iter().map(|c| c * c).sum::<f64>().sqrt();
            let direction = motion.direction_at(years);
            assert!(
                position
                    .iter()
                    .zip(direction)
                    .all(|(p, d)| (p / length - d).abs() < 1e-15)
            );
        }
    }
}

#[test]
fn cast_to_f64_and_back_keeps_the_motion() {
    let motion = Catalog::bright_star_catalogue()
        .stars()
        .next()
        .unwrap()
        .motion();
    assert_eq!(motion.cast::<f64>(), motion);
    let single = motion.cast::<f32>();
    assert_eq!(single.cast::<f64>().cast::<f32>(), single);
}

#[test]
fn f32_motions_decode_without_f64_to_the_same_directions() {
    let catalog = Catalog::bright_star_catalogue();
    let mut worst: f64 = 0.0;
    for (star, single) in catalog.stars().zip(catalog.motions::<f32>()) {
        for years in [-4000.0, 0.0, 1000.0] {
            let direction = single.direction_at(years as f32).map(f64::from);
            worst = worst.max(separation_arcsec(direction, star.direction_at(years)));
        }
    }
    assert!(worst < 0.03, "{worst}\"");
    assert_eq!(catalog.motions::<f64>().len(), catalog.len());
    assert!(
        catalog
            .motions::<f64>()
            .zip(catalog.stars())
            .all(|(motion, star)| motion == star.motion())
    );
}

#[test]
fn prepared_motions_have_the_documented_size() {
    // The README gives these sizes for RAM planning.
    assert_eq!(core::mem::size_of::<crate::Motion<f32>>(), 28);
    assert_eq!(core::mem::size_of::<crate::Motion<f64>>(), 56);
}

#[cfg(feature = "all-stars")]
#[test]
fn barycentric_positions_match_the_reference_from_minus_2000_to_3000() {
    let stars: Vec<_> = Catalog::bright_star_catalogue().stars().collect();
    let mut rows = 0;
    for line in REFERENCE.lines().filter(|line| !line.starts_with('#')) {
        let v: Vec<f64> = line.split(' ').map(|x| x.parse().unwrap()).collect();
        let star = stars
            .iter()
            .find(|star| star.hr().map(f64::from) == Some(v[0]))
            .unwrap();
        let position = star.barycentric_position_pc_at(v[1]);
        if v[5].is_nan() {
            assert_eq!(position, None, "HR {}", v[0]);
            continue;
        }
        let position = position.unwrap();
        let expected = [v[5], v[6], v[7]];
        let error: f64 = position
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let length: f64 = expected.iter().map(|c| c * c).sum::<f64>().sqrt();
        // The packed vectors round to about 1e-9 of the distance.
        assert!(
            error / length < 1e-8,
            "HR {} at {}: {}",
            v[0],
            v[1],
            error / length
        );
        rows += 1;
    }
    assert_eq!(rows, 135);
}

#[test]
fn alpha_centauri_recedes_from_the_past_at_its_radial_velocity() {
    let star = Catalog::bright_star_catalogue()
        .stars()
        .find(|star| star.hr() == Some(5459))
        .unwrap();
    let distance = star.distance_pc().unwrap();
    assert!((distance - 1000.0 / 751.0).abs() < 1e-12);
    let now = star.barycentric_position_pc_at(0.0).unwrap();
    assert!((now.iter().map(|c| c * c).sum::<f64>().sqrt() - distance).abs() < 1e-8);
    // -22 km/s for 4000 years is 0.090 pc; the transverse motion adds about
    // 0.003 pc to the distance.
    let then = star.barycentric_position_pc_at(-4000.0).unwrap();
    let change = then.iter().map(|c| c * c).sum::<f64>().sqrt() - distance;
    assert!((0.090..0.096).contains(&change), "{change}");
}

#[test]
fn a_star_without_a_parallax_has_no_position() {
    let star = Catalog::bright_star_catalogue()
        .stars()
        .find(|star| star.parallax_mas.is_none())
        .unwrap();
    assert_eq!(star.distance_pc(), None);
    assert_eq!(star.barycentric_position_pc_at(0.0), None);
}

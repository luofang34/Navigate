use super::*;
use crate::Catalog;
use std::vec::Vec;

#[cfg(feature = "all-stars")]
const EPHEMERIDES: &str = include_str!("../../fixtures/orb6-ephemerides.txt");

/// Julian years from J2000 of a Besselian epoch.
#[cfg(feature = "all-stars")]
fn besselian(year: f64) -> f64 {
    (2_415_020.313_52 + (year - 1900.0) * 365.242_198_781 - 2_451_545.0) / 365.25
}

fn star(hr: u16) -> Star {
    Catalog::bright_star_catalogue()
        .stars()
        .find(|star| star.hr() == Some(hr))
        .unwrap()
}

#[test]
fn embedded_table_has_a_valid_header_and_checksum() {
    let records = RECORDS.unwrap();
    assert_eq!(records.len(), 133);
    let stored = u32::from_le_bytes(ORBITS[24..28].try_into().unwrap());
    assert_eq!(crate::catalog::crc32(&ORBITS[HEADER_BYTES..]), stored);
    let numbers: Vec<u16> = records
        .iter()
        .map(|record| u16::from_le_bytes([record[0], record[1]]))
        .collect();
    assert!(numbers.windows(2).all(|pair| pair[0] < pair[1]), "HR order");
}

// The fixture covers stars to magnitude 6.5.
#[cfg(feature = "all-stars")]
#[test]
fn orbits_match_the_orb6_ephemerides_from_2025_to_2029() {
    let mut worst: f64 = 0.0;
    let mut rows = 0;
    for line in EPHEMERIDES.lines().filter(|line| !line.starts_with('#')) {
        let v: Vec<f64> = line.split(' ').map(|x| x.parse().unwrap()).collect();
        let hr = v[0] as u16;
        let orbit = orbit_of(hr).unwrap();
        let star = star(hr);
        for (index, year) in [2025.0, 2026.0, 2027.0, 2028.0, 2029.0]
            .into_iter()
            .enumerate()
        {
            // The ephemerides give the position angle for the equinox of
            // date, which turns from the J2000 equinox by the precession
            // 0.00557 degree * sin(ra) / cos(dec) per year.
            let precession = (0.00557 * star.right_ascension_rad().sin()
                / star.declination_rad().cos()
                * (year - 2000.0))
                .to_radians();
            let theta = v[1 + 2 * index].to_radians() - precession;
            let rho = v[2 + 2 * index];
            let [east, north] = orbit.relative_arcsec(besselian(year));
            let error = (east - rho * theta.sin()).hypot(north - rho * theta.cos());
            // The ephemerides round the separation to 0.001 arcsecond and the
            // position angle to 0.1 degree.
            let tolerance = 0.001 + rho * 0.06_f64.to_radians();
            worst = worst.max(error / tolerance);
            assert!(error < tolerance, "HR {hr} {year}: {error} > {tolerance}");
        }
        rows += 1;
    }
    assert_eq!(rows, 133);
    assert!(worst < 1.0);
}

#[test]
fn alpha_centauri_components_move_oppositely_about_the_barycentre() {
    let (a, b) = (star(5459), star(5460));
    for years in [-2000.0, -330.0, 26.0, 1000.0] {
        let offset_a = a.orbital_offset_rad_at(years).unwrap();
        let offset_b = b.orbital_offset_rad_at(years).unwrap();
        // Opposite directions, in the ratio of the mass fractions.
        let cross = offset_a[0] * offset_b[1] - offset_a[1] * offset_b[0];
        let dot = offset_a[0] * offset_b[0] + offset_a[1] * offset_b[1];
        assert!(dot < 0.0 && cross.abs() < 1e-12, "{years}");
        let ratio = offset_a[0].hypot(offset_a[1]) / offset_b[0].hypot(offset_b[1]);
        let orbit_a = orbit_of(5459).unwrap();
        let orbit_b = orbit_of(5460).unwrap();
        // The dynamical masses of Akeson et al. (2021): 1.0788 and 0.9092.
        assert!((ratio - 0.9092 / 1.0788).abs() < 1e-9, "{ratio}");
        // Each offset stays within twice the star's share of the apastron.
        let reach = |orbit: Orbit| {
            2.0 * orbit.fraction.abs() * orbit.semi_major_arcsec * (1.0 + orbit.eccentricity)
        };
        assert!(offset_a[0].hypot(offset_a[1]) / ARCSEC <= reach(orbit_a));
        assert!(offset_b[0].hypot(offset_b[1]) / ARCSEC <= reach(orbit_b));
    }
}

#[test]
fn white_dwarf_companions_move_sirius_and_procyon_by_arcseconds() {
    // With the dynamical masses, the photocentre of Sirius A moves on an
    // orbit of semi-major axis 2.48 arcseconds and Procyon A on one of 1.23.
    for (hr, semi_major) in [(2491, 2.476), (2943, 1.232)] {
        let orbit = orbit_of(hr).unwrap();
        let photocentre = -orbit.fraction * orbit.semi_major_arcsec;
        assert!(
            (photocentre - semi_major).abs() < 0.002,
            "HR {hr}: {photocentre}"
        );
        let largest = (0..400)
            .map(|step| {
                let [east, north] = star(hr)
                    .orbital_offset_rad_at(f64::from(step) / 8.0)
                    .unwrap();
                east.hypot(north) / ARCSEC
            })
            .fold(0.0, f64::max);
        assert!(largest > semi_major, "HR {hr}: {largest}");
    }
}

#[test]
fn offsets_are_zero_at_j2000_and_absent_without_an_orbit() {
    let alpha = star(5459);
    let [east, north] = alpha.orbital_offset_rad_at(0.0).unwrap();
    assert_eq!((east, north), (0.0, 0.0));
    assert_eq!(alpha.component_direction_at(0.0), alpha.direction_at(0.0));
    // Arcturus has no orbit.
    let arcturus = star(5340);
    assert_eq!(arcturus.orbital_offset_rad_at(-2330.0), None);
    assert_eq!(
        arcturus.component_direction_at(-2330.0),
        arcturus.direction_at(-2330.0)
    );
}

#[test]
fn the_component_direction_carries_the_offset() {
    let alpha = star(5459);
    let years = -2330.0;
    let [east, north] = alpha.orbital_offset_rad_at(years).unwrap();
    let a = alpha.direction_at(years);
    let b = alpha.component_direction_at(years);
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let angle = cross.iter().map(|c| c * c).sum::<f64>().sqrt();
    assert!((angle - east.hypot(north)).abs() < 1e-12, "{angle}");
}

#[test]
fn largest_offsets_from_2000_bc_to_ad_3000() {
    // The README quotes these values.
    for (hr, expected) in [(5459, 11.00), (5460, 13.06), (2491, 4.07), (2943, 2.25)] {
        let star = star(hr);
        let largest = (0..=50_000)
            .map(|step| {
                let years = -2000.0 + f64::from(step) * 0.1;
                let [east, north] = star.orbital_offset_rad_at(years).unwrap();
                east.hypot(north) / ARCSEC
            })
            .fold(0.0, f64::max);
        assert!((largest - expected).abs() < 0.01, "HR {hr}: {largest}");
    }
}

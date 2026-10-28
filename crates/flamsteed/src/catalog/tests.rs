use std::vec::Vec;

use super::*;

const MAS: f64 = core::f64::consts::PI / 648_000_000.0;

fn star(hr: u16) -> Star {
    Catalog::bright_star_catalogue()
        .stars()
        .find(|star| star.hr() == Some(hr))
        .unwrap()
}

/// The stars that the features embed: 518 to magnitude 4.0, 1630 to 5.0,
/// and all 9096.
#[cfg(feature = "all-stars")]
const EXPECTED: (usize, f64) = (9096, 7.96);
#[cfg(all(feature = "magnitude-5", not(feature = "all-stars")))]
const EXPECTED: (usize, f64) = (1630, 5.0);
#[cfg(not(feature = "magnitude-5"))]
const EXPECTED: (usize, f64) = (518, 4.0);

#[test]
fn embedded_file_has_a_valid_header_and_checksum() {
    let catalog = Catalog::bright_star_catalogue();
    assert_eq!(catalog.len(), EXPECTED.0);
    assert!(catalog.stars().all(|star| star.v_magnitude <= EXPECTED.1));
    assert_eq!(catalog.reference_epoch_years, 0.0);
    let stored = u32::from_le_bytes(STARS[24..28].try_into().unwrap());
    assert_eq!(crc32(&STARS[HEADER_BYTES..]), stored);
}

#[test]
fn malformed_files_are_rejected() {
    assert!(parse(STARS).is_ok());
    let rejected = |file: &[u8]| Catalog::from_bytes(file).err();
    let mut file = STARS.to_vec();
    file[0] = b'X';
    assert_eq!(rejected(&file), Some(FormatError::Magic));
    let mut file = STARS.to_vec();
    file[8] = 2;
    assert_eq!(
        rejected(&file),
        Some(FormatError::Version {
            version: 2,
            record_bytes: 34
        })
    );
    let mut file = STARS.to_vec();
    file[28] = 3;
    assert_eq!(rejected(&file), Some(FormatError::Source { source: 3 }));
    let mut file = STARS.to_vec();
    file[12] ^= 1;
    assert!(matches!(rejected(&file), Some(FormatError::Length { .. })));
    assert!(matches!(
        rejected(&STARS[..STARS.len() - 1]),
        Some(FormatError::Length { .. })
    ));
    assert_eq!(
        rejected(&STARS[..HEADER_BYTES - 1]),
        Some(FormatError::TooShort {
            length: HEADER_BYTES - 1
        })
    );
    let mut file = STARS.to_vec();
    file[HEADER_BYTES + 5] ^= 1;
    assert!(matches!(
        rejected(&file),
        Some(FormatError::Checksum { .. })
    ));
}

/// A format version 1 file of `records`, with a correct header.
fn file_of(records: &[[u8; RECORD_BYTES]], epoch: f64) -> Vec<u8> {
    let body: Vec<u8> = records.concat();
    let mut file = STARS[..HEADER_BYTES].to_vec();
    file[12..16].copy_from_slice(&u32::try_from(records.len()).unwrap().to_le_bytes());
    file[16..24].copy_from_slice(&epoch.to_le_bytes());
    file[24..28].copy_from_slice(&crc32(&body).to_le_bytes());
    file.extend_from_slice(&body);
    file
}

#[test]
fn from_bytes_reads_imported_data_without_copying() {
    let embedded = Catalog::bright_star_catalogue();
    let imported = Catalog::from_bytes(STARS).unwrap();
    assert!(imported.stars().eq(embedded.stars()));
    assert_eq!(imported.len(), embedded.len());

    // A subset written by another tool, at another epoch, keeps its records.
    let records: Vec<[u8; RECORD_BYTES]> = embedded.records[..3].to_vec();
    let file = file_of(&records, 16.0);
    let subset = Catalog::from_bytes(&file).unwrap();
    assert_eq!(subset.len(), 3);
    assert!(
        subset
            .stars()
            .all(|star| star.reference_epoch_years == 16.0)
    );
    assert!(
        subset
            .stars()
            .zip(embedded.stars())
            .all(|(a, b)| a.position == b.position && a.id == b.id)
    );
    assert_eq!(
        subset
            .brighter_than(subset.stars().nth(1).unwrap().v_magnitude)
            .len(),
        2
    );
}

#[test]
fn from_bytes_needs_magnitude_order_and_a_finite_epoch() {
    let records = &Catalog::bright_star_catalogue().records;
    let reversed = [records[1], records[0]];
    assert_eq!(
        Catalog::from_bytes(&file_of(&reversed, 0.0)).err(),
        Some(FormatError::Order { index: 1 })
    );
    assert_eq!(
        Catalog::from_bytes(&file_of(&records[..2], f64::NAN)).err(),
        Some(FormatError::Epoch)
    );
    assert!(Catalog::from_bytes(&file_of(&[], 0.0)).unwrap().is_empty());
}

#[test]
fn stars_are_brightest_first_and_brighter_than_reads_a_prefix() {
    let catalog = Catalog::bright_star_catalogue();
    let stars: Vec<Star> = catalog.stars().collect();
    assert!(
        stars
            .windows(2)
            .all(|pair| pair[0].v_magnitude <= pair[1].v_magnitude)
    );
    assert_eq!(stars[0].hr(), Some(2491), "Sirius is the brightest");
    for limit in [-2.0, 0.0, 1.0, 3.5, 6.0, 9.0] {
        let prefix: Vec<Star> = catalog.brighter_than(limit).stars().collect();
        let filtered: Vec<Star> = stars
            .iter()
            .copied()
            .filter(|star| star.v_magnitude <= limit)
            .collect();
        assert_eq!(prefix, filtered, "limit {limit}");
    }
}

#[test]
fn sirius_decodes_field_by_field() {
    let sirius = star(2491);
    assert_eq!(sirius.id, StarId::Hr(2491));
    assert_eq!(sirius.v_magnitude, -1.46);
    assert_eq!(sirius.b_minus_v, Some(0.0));
    // 06h45m08.9s -16°42'58" J2000, proper motion -0.553 and -1.205 arcsec/yr.
    assert!((sirius.right_ascension_rad().to_degrees() - 101.287_08).abs() < 1e-5);
    assert!((sirius.declination_rad().to_degrees() + 16.716_11).abs() < 1e-5);
    // The light-time factor for -8 km/s changes the rates by 3e-5.
    let [pm_ra, pm_dec] = sirius.proper_motion_rad_per_year().map(|rate| rate / MAS);
    assert!((pm_ra + 553.0).abs() < 0.03, "{pm_ra}");
    assert!((pm_dec + 1205.0).abs() < 0.05, "{pm_dec}");
    assert_eq!(sirius.parallax_mas, Some(375.0));
    assert_eq!(sirius.radial_velocity_km_s, Some(-8.0));
}

#[cfg(feature = "all-stars")]
#[test]
fn absent_values_are_none() {
    let catalog = Catalog::bright_star_catalogue();
    assert!(catalog.stars().any(|star| star.b_minus_v.is_none()));
    let with_parallax = catalog
        .stars()
        .filter(|star| star.parallax_mas.is_some())
        .count();
    // STARS gives a trigonometric parallax for 3289 entries; the dynamical ones stay absent.
    assert!((2500..3300).contains(&with_parallax), "{with_parallax}");
    // A zero radial velocity is a measurement, not an absent value.
    assert!(
        catalog
            .stars()
            .any(|star| star.radial_velocity_km_s == Some(0.0))
    );
    assert!(
        catalog
            .stars()
            .any(|star| star.radial_velocity_km_s.is_none())
    );
    let alpha_centauri = star(5459);
    assert_eq!(alpha_centauri.parallax_mas, Some(751.0));
    assert_eq!(alpha_centauri.radial_velocity_km_s, Some(-22.0));
}

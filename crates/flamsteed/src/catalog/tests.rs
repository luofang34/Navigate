use super::*;

fn star(hr: u16) -> Star {
    Catalog::bright_star_catalogue()
        .stars()
        .find(|star| star.hr() == Some(hr))
        .unwrap()
}

#[test]
fn embedded_file_has_a_valid_header_and_checksum() {
    let catalog = Catalog::bright_star_catalogue();
    assert_eq!(catalog.len(), 9096);
    assert_eq!(catalog.reference_epoch_years, 0.0);
    let stored = u32::from_le_bytes(BSC5[24..28].try_into().unwrap());
    assert_eq!(crc32(&BSC5[HEADER_BYTES..]), stored);
}

#[test]
fn malformed_files_are_rejected() {
    assert!(parse(BSC5).is_some());
    let mut file = BSC5.to_vec();
    file[0] = b'X';
    assert!(parse(&file).is_none(), "magic");
    let mut file = BSC5.to_vec();
    file[8] = 2;
    assert!(parse(&file).is_none(), "format version");
    let mut file = BSC5.to_vec();
    file[12] ^= 1;
    assert!(parse(&file).is_none(), "record count");
    assert!(parse(&BSC5[..BSC5.len() - 1]).is_none(), "partial record");
    assert!(parse(&BSC5[..HEADER_BYTES - 1]).is_none(), "short header");
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
        let prefix: Vec<Star> = catalog.brighter_than(limit).collect();
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
    assert!((sirius.right_ascension_rad.to_degrees() - 101.287_08).abs() < 1e-5);
    assert!((sirius.declination_rad.to_degrees() + 16.716_11).abs() < 1e-5);
    assert!((sirius.proper_motion_ra_rad_per_year / MAS + 553.0).abs() < 1e-9);
    assert!((sirius.proper_motion_dec_rad_per_year / MAS + 1205.0).abs() < 1e-9);
    assert_eq!(sirius.parallax_mas, Some(375.0));
    assert_eq!(sirius.radial_velocity_km_s, Some(-8.0));
}

#[test]
fn absent_values_are_none() {
    let catalog = Catalog::bright_star_catalogue();
    assert!(catalog.stars().any(|star| star.b_minus_v.is_none()));
    let with_parallax = catalog
        .stars()
        .filter(|star| star.parallax_mas.is_some())
        .count();
    // BSC5 gives a trigonometric parallax for 3289 entries; the dynamical ones stay absent.
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

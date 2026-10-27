use super::*;

fn star(hr: u16) -> Star {
    Catalog::bright_star_catalogue()
        .stars()
        .find(|star| star.hr == hr)
        .unwrap()
}

#[test]
fn every_embedded_record_decodes() {
    let (records, rest) = BSC5.as_chunks::<RECORD_BYTES>();
    assert!(rest.is_empty());
    let catalog = Catalog::bright_star_catalogue();
    assert_eq!(catalog.len(), 9096);
    assert_eq!(records.len(), catalog.len());
    assert!(
        catalog
            .stars()
            .all(|star| star.declination_rad.abs() <= std::f64::consts::FRAC_PI_2)
    );
    assert!(catalog.stars().any(|star| star.b_minus_v.is_none()));
}

#[test]
fn sirius_decodes_field_by_field() {
    let sirius = star(2491);
    assert_eq!(sirius.v_magnitude, -1.46);
    assert_eq!(sirius.b_minus_v, Some(0.0));
    // 06h45m08.9s -16°42'58" J2000, proper motion -0.553 and -1.205 arcsec/yr.
    assert!((sirius.right_ascension_rad.to_degrees() - 101.287_08).abs() < 1e-5);
    assert!((sirius.declination_rad.to_degrees() + 16.716_11).abs() < 1e-5);
    assert!((sirius.proper_motion_ra_rad_per_year / MAS + 553.0).abs() < 1e-9);
    assert!((sirius.proper_motion_dec_rad_per_year / MAS + 1205.0).abs() < 1e-9);
}

#[test]
fn parallax_and_radial_velocity_decode() {
    // Alpha Centauri A: parallax 0.751 arcsec, radial velocity -22 km/s.
    let alpha_centauri = star(5459);
    assert_eq!(alpha_centauri.parallax_mas, 751.0);
    assert_eq!(alpha_centauri.radial_velocity_km_s, -22.0);
    // A star with only a dynamical parallax keeps none.
    assert!(
        Catalog::bright_star_catalogue()
            .stars()
            .filter(|star| star.parallax_mas > 0.0)
            .count()
            < 3300
    );
}

use super::*;

const SKYFIELD: &str = include_str!("../../fixtures/skyfield-stars.txt");
// The measured maximum over the Skyfield table is 8.1 arcseconds. The bound
// keeps the UT1 - UTC margin of the Sun.
const BOUND_DEG: f64 = 15.0 / 3600.0;

fn separation_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (a, b) = (Vector3::from(a), Vector3::from(b));
    a.cross(&b).norm().atan2(a.dot(&b)).to_degrees()
}

#[test]
fn every_embedded_record_decodes() {
    let (records, rest) = CATALOG.as_chunks::<RECORD_BYTES>();
    assert!(rest.is_empty());
    let records: Vec<Record> = records.iter().map(decode).collect();
    assert_eq!(records.len(), 9096);
    let sirius = records.iter().find(|star| star.hr == 2491).unwrap();
    assert_eq!(sirius.v_magnitude, -1.46);
    assert_eq!(sirius.b_minus_v, Some(0.0));
    // 06h45m08.9s -16°42'58" J2000, proper motion -0.553 and -1.205 arcsec/yr.
    assert!((sirius.ra.to_degrees() - 101.287_08).abs() < 1e-5);
    assert!((sirius.dec.to_degrees() + 16.716_11).abs() < 1e-5);
    assert!((sirius.pm_ra / MAS + 553.0).abs() < 1e-9);
    assert!((sirius.pm_dec / MAS + 1205.0).abs() < 1e-9);
    assert!(
        records
            .iter()
            .all(|star| star.dec.abs() <= std::f64::consts::FRAC_PI_2)
    );
    assert!(records.iter().any(|star| star.b_minus_v.is_none()));
}

#[test]
fn stars_match_skyfield_from_1975_to_2026() {
    let mut worst = 0.0_f64;
    let mut rows = 0;
    let mut cached: Option<(f64, Vec<StarSight>)> = None;
    for line in SKYFIELD.lines().filter(|line| !line.starts_with('#')) {
        let v: Vec<f64> = line.split(' ').map(|x| x.parse().unwrap()).collect();
        if cached.as_ref().is_none_or(|(tdb_s, _)| *tdb_s != v[1]) {
            cached = Some((
                v[1],
                stars_ecef(Epoch::from_tdb_seconds(v[1]), 99.0).unwrap(),
            ));
        }
        let stars = &cached.as_ref().unwrap().1;
        let star = stars
            .iter()
            .find(|star| f64::from(star.hr) == v[0])
            .unwrap();
        worst = worst.max(separation_deg(star.direction, [v[2], v[3], v[4]]));
        rows += 1;
    }
    assert_eq!(rows, 294);
    assert!(worst < BOUND_DEG, "{}\"", worst * 3600.0);
}

#[test]
fn magnitude_limit_selects_bright_stars() {
    let epoch: Epoch = "2026-06-14T09:19:41 UTC".parse().unwrap();
    let bright = stars_ecef(epoch, 1.0).unwrap();
    assert!(bright.iter().all(|star| star.v_magnitude <= 1.0));
    assert!(bright.iter().any(|star| star.hr == 2491));
    assert!((10..=20).contains(&bright.len()), "{}", bright.len());
    let all = stars_ecef(epoch, 6.0).unwrap();
    assert!(all.len() > 5000 && all.len() < 9096, "{}", all.len());
    assert!(
        all.iter()
            .all(|star| (Vector3::from(star.direction).norm() - 1.0).abs() < 1e-12)
    );
}

#[test]
fn rejects_epochs_before_the_ephemeris_range() {
    let early: Epoch = "1899-06-01T00:00:00 UTC".parse().unwrap();
    assert_eq!(
        stars_ecef(early, 6.0),
        Err(AnalyticError::EpochOutOfRange(early))
    );
    assert!(stars_ecef("1943-05-16T21:28:00 UTC".parse().unwrap(), 6.0).is_ok());
}

use super::*;

const SKYFIELD: &str = include_str!("../../fixtures/skyfield-stars.txt");
// The measured maximum over the Skyfield table is 1.4 arcseconds.
const BOUND_DEG: f64 = 3.0 / 3600.0;

fn separation_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (a, b) = (Vector3::from(a), Vector3::from(b));
    a.cross(&b).norm().atan2(a.dot(&b)).to_degrees()
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
    // The stars have the range of the ephemeris: 1800 to 2200, or -2000 to
    // 3000 with the long-range feature.
    #[cfg(not(feature = "long-range"))]
    let early: Epoch = "1799-12-01T00:00:00 UTC".parse().unwrap();
    #[cfg(feature = "long-range")]
    let early = Epoch::from_gregorian(-2001, 1, 1, 0, 0, 0, 0, crate::TimeScale::TT);
    assert!(matches!(
        stars_ecef_tt(early, 6.0),
        Err(AnalyticError::EpochOutOfRange(_))
    ));
    assert!(stars_ecef("1943-05-16T21:28:00 UTC".parse().unwrap(), 6.0).is_ok());
}

#[cfg(feature = "long-range")]
#[test]
fn long_range_stars_cover_the_ancient_sky() {
    for year in [-1999, -330] {
        let epoch = Epoch::from_gregorian(year, 9, 20, 18, 0, 0, 0, crate::TimeScale::TT);
        let stars = stars_ecef_tt(epoch, 6.0).unwrap();
        assert!(stars.len() > 5000, "{year}: {}", stars.len());
        assert!(
            stars
                .iter()
                .all(|star| (Vector3::from(star.direction).norm() - 1.0).abs() < 1e-12)
        );
    }
}

#[test]
fn stars_tt_takes_the_instant_and_stars_reads_a_clock() {
    // Before 1972 a civil reading is UT1, so the instant is Delta T later.
    let reading: Epoch = "1900-05-28T14:00:00 UTC".parse().unwrap();
    let instant = civil_time(reading).terrestrial;
    let sirius = |stars: Vec<StarSight>| stars.into_iter().find(|star| star.hr == 2491).unwrap();
    let from_clock = sirius(stars_ecef(reading, 0.0).unwrap());
    let from_instant = sirius(stars_ecef_tt(instant, 0.0).unwrap());
    assert_eq!(from_clock, from_instant);

    // The documented pipeline at the instant: space motion, aberration, and
    // the Earth rotation that the Sun and Moon use at the same TT.
    let record = flamsteed::Catalog::bright_star_catalogue()
        .stars()
        .find(|star| star.hr == 2491)
        .unwrap();
    let years = instant.to_tdb_centuries_since_j2000() * 100.0;
    let moved = Vector3::from(record.direction_at(years));
    let beta = Vector3::from(
        AnalyticEphemeris
            .body_state(399, instant)
            .unwrap()
            .velocity_m_s,
    ) / LIGHT_SPEED_M_S;
    let expected = (earth_rotation::j2000_to_earth_fixed(instant)
        * observation::aberrate(moved, beta))
    .normalize();
    // Reading the instant as a UT1 clock would turn it by Delta T, about 40
    // arcseconds in 1900.
    assert!(separation_deg(from_instant.direction, expected.into()) < 1e-9);
}

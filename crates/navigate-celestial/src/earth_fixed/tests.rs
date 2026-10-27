use super::*;
use crate::{Duration, Ephemeris, TimeScale};
use navigate_contract::GeodeticPosition;
use navigate_geodesy::geodetic_to_ecef;

const ITRF93: &str = include_str!("../../fixtures/de421-itrf93.txt");
const BEFORE_1972: &str = include_str!("../../fixtures/skyfield-before-1972.txt");
const ARCSECOND: f64 = 1.0 / 3600.0;
// Measured over the ITRF93 table with UT1 from Delta T: 0.97 arcsecond for
// the Sun, and 11.1 arcseconds and 9.4 km for the Moon. The Moon error comes
// from the truncated lunar series.
const SUN_BOUND_DEG: f64 = 2.0 * ARCSECOND;
const MOON_BOUND_DEG: f64 = 15.0 * ARCSECOND;
// Measured over the Skyfield table from 1800 to 1971: 0.45 arcsecond for the
// Sun and 6.4 arcseconds for the Moon.
const MOON_BEFORE_1972_BOUND_DEG: f64 = 10.0 * ARCSECOND;
const MOON_RANGE_BOUND_M: f64 = 15_000.0;
/// The sunrise test observer: 41.07204° N, 71.85733° W, 0 m on WGS84.
const HORIZON_OBSERVER: (f64, f64) = (41.072_04, -71.857_33);

struct Row {
    label: String,
    epoch: Epoch,
    observer: [f64; 3],
    sun: [f64; 3],
    moon: [f64; 3],
    moon_range_m: f64,
}

fn rows() -> Vec<Row> {
    ITRF93
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.split(' ');
            let label = fields.next().unwrap().to_owned();
            let v: Vec<f64> = fields.map(|x| x.parse().unwrap()).collect();
            Row {
                label,
                epoch: Epoch::from_tdb_seconds(v[0]),
                observer: [v[1], v[2], v[3]],
                sun: [v[4], v[5], v[6]],
                moon: [v[8], v[9], v[10]],
                moon_range_m: v[11],
            }
        })
        .collect()
}

fn separation_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (a, b) = (Vector3::from(a), Vector3::from(b));
    a.cross(&b).norm().atan2(a.dot(&b)).to_degrees()
}

fn surface((latitude_deg, longitude_deg): (f64, f64)) -> [f64; 3] {
    let ecef = geodetic_to_ecef(&GeodeticPosition::new(
        latitude_deg.to_radians(),
        longitude_deg.to_radians(),
        0.0,
    ));
    [ecef.x_m, ecef.y_m, ecef.z_m]
}

/// Elevation and azimuth in degrees, from the local geodetic east, north, and up.
fn horizontal((latitude_deg, longitude_deg): (f64, f64), direction: [f64; 3]) -> (f64, f64) {
    let (sin_lat, cos_lat) = latitude_deg.to_radians().sin_cos();
    let (sin_lon, cos_lon) = longitude_deg.to_radians().sin_cos();
    let d = Vector3::from(direction);
    let east = Vector3::new(-sin_lon, cos_lon, 0.0).dot(&d);
    let north = Vector3::new(-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat).dot(&d);
    let up = Vector3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat).dot(&d);
    (
        up.asin().to_degrees(),
        east.atan2(north).to_degrees().rem_euclid(360.0),
    )
}

fn utc(text: &str) -> Epoch {
    text.parse().unwrap()
}

#[test]
fn sun_and_moon_match_de421_with_itrf93_from_2000_to_2026() {
    let mut worst = [0.0_f64; 3];
    let rows = rows();
    for row in &rows {
        let observer = Some(row.observer);
        let sun = sun_ecef(row.epoch, observer).unwrap();
        let moon = moon_ecef(row.epoch, observer).unwrap();
        let errors = [
            separation_deg(sun.direction, row.sun),
            separation_deg(moon.direction, row.moon),
            (moon.distance_m - row.moon_range_m).abs(),
        ];
        for (worst, error) in worst.iter_mut().zip(errors) {
            *worst = worst.max(error);
        }
    }
    assert_eq!(rows.len(), 307);
    assert!(
        worst[0] < SUN_BOUND_DEG,
        "Sun error {}\"",
        worst[0] * 3600.0
    );
    assert!(
        worst[1] < MOON_BOUND_DEG,
        "Moon error {}\"",
        worst[1] * 3600.0
    );
    assert!(
        worst[2] < MOON_RANGE_BOUND_M,
        "Moon range error {} m",
        worst[2]
    );
}

/// Find when an elevation function rises through 0°, by bisection.
fn crossing(elevation: impl Fn(Epoch) -> f64, mut below: Epoch, mut above: Epoch) -> Epoch {
    assert!(elevation(below) < 0.0 && elevation(above) > 0.0);
    while (above - below) > Duration::from_milliseconds(1.0) {
        let middle = below + (above - below) / 2;
        if elevation(middle) < 0.0 {
            below = middle;
        } else {
            above = middle;
        }
    }
    below
}

/// The Sun's centre crosses the geometric horizon at 41.07204° N, 71.85733° W.
///
/// The horizon is geometric: no refraction and no semidiameter. The Sun
/// direction is apparent, with light time and aberration, as a camera sees
/// it. Without those corrections the direction moves by about 20 arcseconds
/// and the crossing by about 2 s. The expected 09:19:41Z and 58.40° come from
/// a low-precision NOAA calculation, good to about 0.01° (about 4 s).
#[test]
fn sun_centre_crosses_the_geometric_horizon_at_the_2026_06_14_sunrise() {
    let observer = Some(surface(HORIZON_OBSERVER));
    let elevation = |epoch| {
        horizontal(
            HORIZON_OBSERVER,
            sun_ecef(epoch, observer).unwrap().direction,
        )
        .0
    };
    let rise = crossing(
        elevation,
        utc("2026-06-14T09:10:00 UTC"),
        utc("2026-06-14T09:30:00 UTC"),
    );
    let expected = utc("2026-06-14T09:19:41 UTC");
    let sun = sun_ecef(rise, observer).unwrap();
    let azimuth = horizontal(HORIZON_OBSERVER, sun.direction).1;
    assert!(
        (rise - expected).abs() < Duration::from_seconds(10.0),
        "{rise}"
    );
    assert!((azimuth - 58.40).abs() < 0.05, "{azimuth}");

    // JPL DE421 with ITRF93: interpolate the elevation between the bracketing rows.
    let rows: Vec<Row> = rows()
        .into_iter()
        .filter(|row| row.label == "horizon")
        .collect();
    let anise = |row: &Row| horizontal(HORIZON_OBSERVER, row.sun).0;
    let pair = rows
        .windows(2)
        .find(|pair| anise(&pair[0]) < 0.0 && anise(&pair[1]) >= 0.0)
        .unwrap();
    let span = (pair[1].epoch - pair[0].epoch).to_seconds();
    let fraction = -anise(&pair[0]) / (anise(&pair[1]) - anise(&pair[0]));
    let reference = pair[0].epoch + Duration::from_seconds(fraction * span);
    assert!(
        (rise - reference).abs() < Duration::from_seconds(1.0),
        "{reference}"
    );

    for row in &rows {
        let moon = moon_ecef(row.epoch, observer).unwrap();
        assert!(separation_deg(moon.direction, row.moon) < MOON_BOUND_DEG);
    }
}

#[test]
fn subsolar_point_has_the_sun_overhead() {
    let epoch = utc("2026-06-14T09:19:41 UTC");
    let point = subsolar_point(epoch).unwrap();
    assert!((point.latitude_deg - 23.271).abs() < 0.02, "{point:?}");
    assert!((point.longitude_deg - 40.153).abs() < 0.02, "{point:?}");
    let place = (point.latitude_deg, point.longitude_deg);
    let sun = sun_ecef(epoch, Some(surface(place))).unwrap();
    // Solar parallax only: less than 9 arcseconds from the zenith.
    assert!(horizontal(place, sun.direction).0 > 90.0 - 9.0 * ARCSECOND);
}

#[test]
fn disc_radius_follows_the_distance() {
    let epoch = utc("2026-06-14T09:19:41 UTC");
    let sun = sun_ecef(epoch, None).unwrap();
    // Near aphelion the Sun's radius is about 15.73 arcminutes.
    assert!((sun.angular_radius_rad.to_degrees() * 60.0 - 15.73).abs() < 0.02);
    let geocentric = moon_ecef(epoch, None).unwrap();
    // An observer under the Moon is one Earth radius closer than the geocentre.
    let [x, y, z] = geocentric.direction;
    let below = moon_ecef(epoch, Some([x * 6.371e6, y * 6.371e6, z * 6.371e6])).unwrap();
    assert!((geocentric.distance_m - below.distance_m - 6.371e6).abs() < 2e4);
    assert!(below.angular_radius_rad > geocentric.angular_radius_rad * 1.015);
}

#[test]
fn an_earth_fixed_observer_moves_with_the_earth_rotation() {
    let epoch = utc("2026-06-14T09:19:41 UTC");
    let equator = State {
        position_m: [6_378_137.0, 0.0, 0.0],
        velocity_m_s: [0.0; 3],
        epoch,
        frame: FrameId::EARTH_ITRF93,
    };
    let state = AnalyticEphemeris.barycentric_state(equator).unwrap();
    let earth = AnalyticEphemeris.body_state(399, epoch).unwrap();
    let offset = Vector3::from(state.position_m) - Vector3::from(earth.position_m);
    let speed = (Vector3::from(state.velocity_m_s) - Vector3::from(earth.velocity_m_s)).norm();
    assert!((offset.norm() - 6_378_137.0).abs() < 1e-3);
    assert!((speed - 465.1).abs() < 0.1, "{speed}");
}

#[test]
fn sun_and_moon_match_skyfield_with_historical_delta_t_from_1800_to_1971() {
    let mut worst = [0.0_f64; 2];
    for line in BEFORE_1972.lines().filter(|line| !line.starts_with('#')) {
        let mut fields = line.split(' ');
        let epoch = utc(&format!("{} UTC", fields.next().unwrap()));
        let v: Vec<f64> = fields.map(|x| x.parse().unwrap()).collect();
        let observer = (v[..3] != [0.0; 3]).then(|| [v[0], v[1], v[2]]);
        // The civil functions read the UT1 clock. The TT functions take the
        // instant, which is that UT1 plus Delta T.
        let instant = crate::civil_time(epoch).terrestrial;
        let suns = [sun_ecef(epoch, observer), sun_ecef_tt(instant, observer)];
        let moons = [moon_ecef(epoch, observer), moon_ecef_tt(instant, observer)];
        for (sun, moon) in suns.into_iter().zip(moons) {
            let (sun, moon) = (sun.unwrap().direction, moon.unwrap().direction);
            worst[0] = worst[0].max(separation_deg(sun, [v[3], v[4], v[5]]));
            worst[1] = worst[1].max(separation_deg(moon, [v[6], v[7], v[8]]));
        }
    }
    assert!(worst[0] < SUN_BOUND_DEG, "{worst:?}");
    assert!(worst[1] < MOON_BEFORE_1972_BOUND_DEG, "{worst:?}");
}

#[test]
fn rejects_epochs_before_the_ephemeris_range() {
    assert!(matches!(
        sun_ecef(utc("1799-12-01T00:00:00 UTC"), None),
        Err(ObservationError::Ephemeris {
            source: AnalyticError::EpochOutOfRange(_),
            ..
        })
    ));
    assert!(moon_ecef(utc("1800-01-02T00:00:00 UTC"), None).is_ok());
}

#[test]
fn tt_entry_points_take_the_instant_and_civil_ones_read_a_clock() {
    let observer = Some(surface(HORIZON_OBSERVER));
    // Before 1972 the civil reading is UT1, so the instant is UT1 + Delta T.
    for reading in [
        utc("1900-05-28T14:00:00 UTC"),
        utc("2026-06-14T09:19:41 UTC"),
    ] {
        let instant = crate::civil_time(reading).terrestrial;
        assert_eq!(
            sun_ecef(reading, observer).unwrap(),
            sun_ecef_tt(instant, observer).unwrap()
        );
        assert_eq!(
            moon_ecef(reading, None).unwrap(),
            moon_ecef_tt(instant, None).unwrap()
        );
    }
    // From 1972 the reading is UTC, which is the instant itself.
    let modern = utc("2026-06-14T09:19:41 UTC");
    assert_eq!(
        sun_ecef(modern, None).unwrap(),
        sun_ecef_tt(modern.to_time_scale(TimeScale::TT), None).unwrap()
    );
}

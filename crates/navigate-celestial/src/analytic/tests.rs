use super::*;
use crate::{Correction, ObservationError, observe_body};

const CSPICE: &str = include_str!("../../fixtures/celestial-cspice.json");
const DE421: &str = include_str!("../../fixtures/de421-geocentric.txt");
const DE440: &str = include_str!("../../fixtures/de440-geocentric.txt");
#[cfg(feature = "long-range")]
const DE431: &str = include_str!("../../fixtures/de431-geocentric.txt");
const ARCSECOND: f64 = 1.0 / 3600.0;
// About 1.5 times the largest error measured: 0.73 arcsecond for the Sun,
// 12.1 arcseconds and 9.8 km for the Moon over the DE421 table, and 0.83
// arcsecond, 10.1 arcseconds and 10.1 km over the DE440 table.
#[cfg(not(feature = "long-range"))]
const SUN_BOUND_DEG: f64 = 1.5 * ARCSECOND;
#[cfg(not(feature = "long-range"))]
const MOON_BOUND_DEG: f64 = 18.0 * ARCSECOND;
#[cfg(not(feature = "long-range"))]
const MOON_RANGE_BOUND_M: f64 = 15_000.0;
// With the series fitted to DE431: 0.18 arcsecond for the Sun, 0.30
// arcsecond and 150 m for the Moon over DE421, and 0.21 arcsecond, 0.28
// arcsecond and 160 m over DE440.
#[cfg(feature = "long-range")]
const SUN_BOUND_DEG: f64 = 0.3 * ARCSECOND;
#[cfg(feature = "long-range")]
const MOON_BOUND_DEG: f64 = 0.4 * ARCSECOND;
#[cfg(feature = "long-range")]
const MOON_RANGE_BOUND_M: f64 = 250.0;

fn geocentre(epoch: Epoch) -> State {
    State {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        epoch,
        frame: FrameId {
            center: EARTH,
            orientation: J2000,
        },
    }
}

fn separation_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (a, b) = (Vector3::from(a), Vector3::from(b));
    a.cross(&b).norm().atan2(a.dot(&b)).to_degrees()
}

fn observe(observer: State, body: i32) -> crate::Sight {
    observe_body(
        &AnalyticEphemeris,
        observer,
        body,
        Correction::VacuumReception,
    )
    .unwrap()
}

#[test]
fn sun_matches_the_cspice_reception_fixture() {
    let fixture: serde_json::Value = serde_json::from_str(CSPICE).unwrap();
    let epoch = Epoch::from_tdb_seconds(fixture["epoch_tdb_seconds"].as_f64().unwrap());
    for case in fixture["cases"].as_array().unwrap() {
        let numbers = |key: &str| -> Vec<f64> {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        };
        let state = numbers("state_m");
        let observer = State {
            position_m: [state[0], state[1], state[2]],
            velocity_m_s: [state[3], state[4], state[5]],
            epoch,
            frame: FrameId {
                center: case["center"].as_i64().unwrap() as i32,
                orientation: case["orientation"].as_i64().unwrap() as i32,
            },
        };
        let result = observe_body(
            &AnalyticEphemeris,
            observer,
            SUN,
            Correction::VacuumReception,
        );
        match case["name"].as_str().unwrap() {
            "vleo" | "geo" | "earth_fixed" => {
                let sight = result.unwrap();
                let expected = numbers("direction");
                let error =
                    separation_deg(sight.direction, [expected[0], expected[1], expected[2]]);
                assert!(error < SUN_BOUND_DEG, "{}: {error}°", case["name"]);
                let range = case["range_m"].as_f64().unwrap();
                assert!((sight.coordinate_range_m - range).abs() < 2e6);
            }
            // The provider has no Mars model.
            _ => assert!(matches!(
                result,
                Err(ObservationError::Ephemeris {
                    role: "observer",
                    source: AnalyticError::UnsupportedFrame(_),
                    ..
                })
            )),
        }
    }
}

/// The largest Sun direction, Moon direction, Moon range, and Moon-Sun
/// elongation errors over a table that `examples/reference_table.rs` writes.
fn worst_errors(table: &str) -> [f64; 4] {
    let worst = worst_errors_of(table);
    assert_eq!(rows_in(table), 400);
    worst
}

fn rows_in(table: &str) -> usize {
    table.lines().filter(|line| !line.starts_with('#')).count()
}

/// The largest errors over the rows of `table`, which may be a subset.
fn worst_errors_of(table: &str) -> [f64; 4] {
    let mut worst = [0.0_f64; 4];
    for line in table.lines().filter(|line| !line.starts_with('#')) {
        let v: Vec<f64> = line.split(' ').map(|x| x.parse().unwrap()).collect();
        let observer = geocentre(Epoch::from_tdb_seconds(v[0]));
        let sun = observe(observer, SUN);
        let moon = observe(observer, MOON);
        let errors = [
            separation_deg(sun.direction, [v[1], v[2], v[3]]),
            separation_deg(moon.direction, [v[5], v[6], v[7]]),
            (moon.coordinate_range_m - v[8]).abs(),
            (separation_deg(sun.direction, moon.direction)
                - separation_deg([v[1], v[2], v[3]], [v[5], v[6], v[7]]))
            .abs(),
        ];
        for (worst, error) in worst.iter_mut().zip(errors) {
            *worst = worst.max(error);
        }
    }
    worst
}

#[test]
fn sun_and_moon_match_de421_from_1900_to_2050() {
    let worst = worst_errors(DE421);
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

#[cfg(feature = "long-range")]
#[test]
fn long_range_sun_and_moon_match_de431_from_minus_2000_to_3000() {
    // About 1.3 times the measured maximum of each millennium: Sun and Moon
    // directions and the Moon-Sun elongation in arcseconds, the Moon range
    // in metres. The elongation sets the time of an eclipse. The Moon error
    // grows into the past, where the fit to DE431 leaves periodic terms
    // that the cubic corrections do not reach.
    let eras = [
        (-2000, -1000, 1.15, 5.5, 6.5, 1_050.0),
        (-1000, 0, 0.5, 2.9, 3.3, 520.0),
        (0, 1000, 0.4, 1.2, 1.4, 190.0),
        (1000, 2000, 0.32, 0.38, 0.4, 175.0),
        (2000, 3000, 0.21, 0.48, 0.37, 180.0),
    ];
    for (first, last, sun_bound, moon_bound, elongation_bound, range_bound) in eras {
        let rows: String = DE431
            .lines()
            .filter(|line| {
                let tdb_s: f64 = line.split(' ').next().unwrap().parse().unwrap_or(f64::NAN);
                let year = 2000.0 + tdb_s / (365.25 * 86_400.0);
                (f64::from(first)..f64::from(last)).contains(&year)
            })
            .map(|line| format!("{line}\n"))
            .collect();
        let worst = worst_errors_of(&rows);
        assert!(rows_in(&rows) > 50, "{first}..{last}");
        assert!(
            worst[0] < sun_bound * ARCSECOND,
            "{first}: Sun {}\"",
            worst[0] * 3600.0
        );
        assert!(
            worst[1] < moon_bound * ARCSECOND,
            "{first}: Moon {}\"",
            worst[1] * 3600.0
        );
        assert!(worst[2] < range_bound, "{first}: Moon range {} m", worst[2]);
        assert!(
            worst[3] < elongation_bound * ARCSECOND,
            "{first}: elongation {}\"",
            worst[3] * 3600.0
        );
    }
}

#[test]
fn sun_and_moon_match_de440_from_1800_to_2200() {
    let worst = worst_errors(DE440);
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

#[test]
fn rejects_unknown_bodies_barycentric_origins_and_distant_epochs() {
    let epoch = Epoch::from_tdb_seconds(0.0);
    assert_eq!(
        AnalyticEphemeris.body_state(499, epoch),
        Err(AnalyticError::UnsupportedBody(499))
    );
    let barycentric = State {
        frame: FrameId::SSB_J2000,
        ..geocentre(epoch)
    };
    assert_eq!(
        AnalyticEphemeris.barycentric_state(barycentric),
        Err(AnalyticError::UnsupportedFrame(FrameId::SSB_J2000))
    );
    #[cfg(not(feature = "long-range"))]
    let late: Epoch = "2200-01-03T00:00:00 TDB".parse().unwrap();
    #[cfg(feature = "long-range")]
    let late: Epoch = "3000-01-20T00:00:00 TDB".parse().unwrap();
    assert_eq!(
        AnalyticEphemeris.body_state(SUN, late),
        Err(AnalyticError::EpochOutOfRange(late))
    );
    #[cfg(not(feature = "long-range"))]
    let early: Epoch = "1799-12-01T00:00:00 TDB".parse().unwrap();
    #[cfg(feature = "long-range")]
    let early = Epoch::from_gregorian(-2001, 1, 1, 0, 0, 0, 0, crate::TimeScale::TDB);
    assert!(AnalyticEphemeris.body_state(MOON, early).is_err());
}

#[test]
fn earth_orbital_velocity_is_the_derivative_of_position() {
    let epoch: Epoch = "2026-10-26T00:00:00 UTC".parse().unwrap();
    let state = AnalyticEphemeris.body_state(EARTH, epoch).unwrap();
    let later = AnalyticEphemeris
        .body_state(EARTH, epoch + crate::Duration::from_seconds(1.0))
        .unwrap();
    let difference = Vector3::from(later.position_m) - Vector3::from(state.position_m);
    let speed = Vector3::from(state.velocity_m_s).norm();
    assert!((29_000.0..30_500.0).contains(&speed));
    assert!((difference - Vector3::from(state.velocity_m_s)).norm() < 1e-2);
}

#[test]
fn velocity_holds_at_the_ends_of_the_range() {
    let modern: Vec<Epoch> = ["1800-01-02T00:00:00 TDB", "2199-12-30T00:00:00 TDB"]
        .iter()
        .map(|text| text.parse().unwrap())
        .collect();
    // Measured: 4 mm/s, and 0.074 m/s at -1999 with long-range.
    #[cfg(not(feature = "long-range"))]
    let (epochs, bound) = (modern, 0.01);
    #[cfg(feature = "long-range")]
    let (epochs, bound) = {
        let mut epochs = modern;
        epochs.push("2999-12-30T00:00:00 TDB".parse().unwrap());
        epochs.push(Epoch::from_gregorian(
            -1999,
            1,
            2,
            0,
            0,
            0,
            0,
            crate::TimeScale::TDB,
        ));
        (epochs, 0.15)
    };
    for epoch in epochs {
        for body in [EARTH, MOON] {
            let state = AnalyticEphemeris.body_state(body, epoch).unwrap();
            let step = crate::Duration::from_seconds(30.0);
            let before = AnalyticEphemeris.body_state(body, epoch - step).unwrap();
            let after = AnalyticEphemeris.body_state(body, epoch + step).unwrap();
            let difference =
                (Vector3::from(after.position_m) - Vector3::from(before.position_m)) / 60.0;
            let error = (difference - Vector3::from(state.velocity_m_s)).norm();
            assert!(error < bound, "{epoch} {body}: {error} m/s");
        }
    }
}

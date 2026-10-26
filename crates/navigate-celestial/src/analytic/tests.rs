use super::*;
use crate::{Correction, ObservationError, observe_body};

const CSPICE: &str = include_str!("../../fixtures/celestial-cspice.json");
const DE421: &str = include_str!("../../fixtures/de421-geocentric.txt");
const ARCSECOND: f64 = 1.0 / 3600.0;
// About 1.5 times the largest error measured over the DE421 table:
// 0.73 arcsecond for the Sun, 12.1 arcseconds and 9.8 km for the Moon.
const SUN_BOUND_DEG: f64 = 1.5 * ARCSECOND;
const MOON_BOUND_DEG: f64 = 18.0 * ARCSECOND;
const MOON_RANGE_BOUND_M: f64 = 15_000.0;

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

#[test]
fn sun_and_moon_match_de421_from_1900_to_2050() {
    let mut worst = [0.0_f64; 3];
    let mut rows = 0;
    for line in DE421.lines().filter(|line| !line.starts_with('#')) {
        let v: Vec<f64> = line.split(' ').map(|x| x.parse().unwrap()).collect();
        let observer = geocentre(Epoch::from_tdb_seconds(v[0]));
        let sun = observe(observer, SUN);
        let moon = observe(observer, MOON);
        let errors = [
            separation_deg(sun.direction, [v[1], v[2], v[3]]),
            separation_deg(moon.direction, [v[5], v[6], v[7]]),
            (moon.coordinate_range_m - v[8]).abs(),
        ];
        for (worst, error) in worst.iter_mut().zip(errors) {
            *worst = worst.max(error);
        }
        rows += 1;
    }
    assert_eq!(rows, 400);
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
    let late: Epoch = "2100-01-02T00:00:00 TDB".parse().unwrap();
    assert_eq!(
        AnalyticEphemeris.body_state(SUN, late),
        Err(AnalyticError::EpochOutOfRange(late))
    );
    let early: Epoch = "1899-12-31T00:00:00 TDB".parse().unwrap();
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

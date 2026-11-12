//! A source whose fixes come less often than the assessment window is
//! long declares a hold time, and its last fix counts toward the
//! assessment of the solution until that time ends.

use navigate_contract::{DurationNanos, FaultDetection, NavigationSolution, Redundancy};

use std::vec::Vec;

use super::*;
use crate::observation::{Observation, ObservationValue};

const VISUAL: u32 = 7;
const BARO: u32 = 8;
const GNSS: u32 = 9;

fn visual() -> SourceComposition {
    SourceComposition::of(SensorClass::VisualLandmark)
}

/// A position fix at the origin from `source`.
fn fix(source: u32, sequence: u32, at_ms: u64, composition: SourceComposition) -> Observation {
    let value = ObservationValue::PositionFix {
        position: earth_origin(),
        covariance: SymmetricCov3::from_diagonal(4.0, 4.0, 4.0),
    };
    Observation::new(stamp(source, sequence, at_ms), value, composition)
}

/// A vehicle at rest for `seconds` with a barometric altitude each
/// second. `aid` gives the other observations of each 100 ms step. The
/// result is the solution at the end of each second.
fn rest(seconds: u64, aid: impl Fn(u64) -> Option<Observation>) -> Vec<NavigationSolution> {
    let origin = earth_origin();
    let mut filter = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        origin,
    );
    let rates = rest_rates(&filter);
    let mut solutions = Vec::new();
    for step in 1..=seconds * 10 {
        let at_ms = step * 100;
        assert_accepted(filter.ingest_imu(&imu(step as u32, at_ms, rates, 0.1), t(at_ms)));
        if let Some(observation) = aid(step) {
            assert_accepted(filter.ingest(&observation, t(at_ms)));
        }
        if step.is_multiple_of(10) {
            let altitude = ObservationValue::Altitude {
                altitude_m: origin.altitude_m,
                variance_m2: 1.0,
            };
            let composition = SourceComposition::of(SensorClass::Barometric);
            let observation =
                Observation::new(stamp(BARO, step as u32, at_ms), altitude, composition);
            assert_accepted(filter.ingest(&observation, t(at_ms)));
            solutions.push(filter.tick(t(at_ms)).expect("published"));
        }
    }
    solutions
}

#[test]
fn a_fix_each_eleven_seconds_with_a_hold_contributes_without_a_gap() {
    // A terrain fix at 1 s, 12 s, 23 s, and 34 s, with a hold of 30 s;
    // the assessment window is 5 s.
    let hold = DurationNanos::from_millis(30_000);
    let solutions = rest(80, |step| {
        let fixes = [10, 120, 230, 340];
        fixes
            .contains(&step)
            .then(|| fix(VISUAL, step as u32, step * 100, visual()).with_hold(hold))
    });
    let holds = |s: &NavigationSolution| {
        s.integrity
            .contributing
            .contains(SensorClass::VisualLandmark)
    };
    // From the first fix to 30 s after the last one, at 64 s.
    for (index, solution) in solutions.iter().enumerate() {
        let second = index as u64 + 1;
        assert_eq!(holds(solution), second <= 64, "{second}");
        let (redundancy, detection) = if second <= 64 {
            (Redundancy::IndependentClasses, FaultDetection::Monitoring)
        } else {
            (Redundancy::None, FaultDetection::Unavailable)
        };
        assert_eq!(solution.integrity.redundancy, redundancy, "{second}");
        assert_eq!(solution.integrity.fault_detection, detection, "{second}");
    }
}

#[test]
fn the_same_fixes_without_a_hold_contribute_for_the_window_only() {
    let solutions = rest(20, |step| {
        (step == 10 || step == 120).then(|| fix(VISUAL, step as u32, step * 100, visual()))
    });
    let seconds: Vec<_> = solutions
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            s.integrity
                .contributing
                .contains(SensorClass::VisualLandmark)
        })
        .map(|(index, _)| index + 1)
        .collect();
    assert_eq!(seconds, [1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17]);
}

#[test]
fn a_receiver_at_ten_hertz_drops_out_one_window_after_its_last_fix() {
    // Fixes at each step for 10 s, then none.
    let solutions = rest(20, |step| {
        (step <= 100).then(|| fix(GNSS, step as u32, step * 100, gnss()))
    });
    let seconds: Vec<_> = solutions
        .iter()
        .enumerate()
        .filter(|(_, s)| s.integrity.contributing.contains(SensorClass::Gnss))
        .map(|(index, _)| index + 1)
        .collect();
    assert_eq!(seconds, (1..=15).collect::<Vec<_>>());
}

//! The range of heading errors from which the filter converges after an
//! alignment in motion, as a measurement.

use navigate_contract::NavigationSolution;

use super::moving::{CRAB_SIGMA_RAD, attitude_error, cross_wind, fly, navigator};
use super::path::{Profile, Truth, Vehicle, ramp};
use super::rig::{self, at};
use crate::{AlignmentStatus, Navigator};

/// A flight at 55 m/s that is straight for 20 s and then weaves: the
/// heading swings by 17° each way with a period of 20 s, with the bank of
/// a coordinated turn, to 28°.
fn straight_then_weave(heading: f64) -> impl Fn(f64) -> Profile {
    move |t| {
        let weave = |t: f64| 0.3 * ramp(t, 20.0, 25.0) * (core::f64::consts::TAU * t / 20.0).sin();
        let rate = (weave(t + 1e-3) - weave(t - 1e-3)) / 2e-3;
        Profile {
            speed_m_s: 55.0,
            heading_rad: heading + weave(t),
            roll_rad: (55.0 * rate / 9.806_65).atan(),
            pitch_rad: 0.05,
            climb_m_s: 0.0,
        }
    }
}

/// The heading error and the heading one-sigma of a solution, radians.
fn heading_state(truth: &Truth, solution: &NavigationSolution) -> (f64, f64) {
    let (_, _, variance) = solution.attitude_cov.unwrap().diagonal();
    (attitude_error(truth, solution).z, variance.sqrt())
}

/// One flight of the envelope measurement.
struct Flown {
    /// The heading error and its one-sigma, radians, at the alignment,
    /// after the straight part, and at the end.
    states: [(f64, f64); 3],
    /// Count of GNSS values that the filter refused.
    refused: usize,
    /// The navigator said at one time or more that its alignment is
    /// suspect.
    suspect: bool,
}

fn envelope_flight(crab: f64, seed: u64, steps: [u64; 2]) -> Flown {
    let heading = 0.9 + seed as f64;
    let wind = cross_wind(heading, crab);
    let mut vehicle = Vehicle::new(straight_then_weave(heading), wind, seed, 1.0);
    let mut navigator = navigator(Some(CRAB_SIGMA_RAD));
    let truth = fly(&mut navigator, &mut vehicle, 1_000);
    let mut states = [heading_state(&truth, &navigator.solution(at(vehicle.step)).unwrap()); 3];
    let mut sink = Vec::new();
    let mut suspect = false;
    for (state, until) in states.iter_mut().skip(1).zip(steps) {
        while vehicle.step < until {
            navigator.ingest(&vehicle.increment(), &mut sink).unwrap();
            if vehicle.step.is_multiple_of(100) {
                navigator.ingest(&vehicle.gnss(0), &mut sink).unwrap();
                suspect |= matches!(
                    navigator.alignment_status(),
                    Some(AlignmentStatus::Aligned { suspect: true, .. })
                );
            }
        }
        let solution = navigator.solution(at(vehicle.step)).unwrap();
        *state = heading_state(&vehicle.truth(), &solution);
    }
    let refused = sink
        .iter()
        .filter(|r| matches!(r.outcome, crate::Outcome::Rejected(_)))
        .count();
    Flown {
        states,
        refused,
        suspect,
    }
}

/// MEASURE
#[test]
fn the_filter_finds_the_heading_in_a_weave_from_each_crab_angle() {
    rig::show_events();
    let quick = cfg!(debug_assertions);
    let crabs_deg: &[f64] = if quick {
        &[5.0]
    } else {
        &[0.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    };
    let steps = if quick { [300, 600] } else { [2_000, 6_000] };
    for crab_deg in crabs_deg {
        for seed in 0..if quick { 1 } else { 32 } {
            let sign = if seed % 2 == 0 { 1.0 } else { -1.0 };
            let flown = envelope_flight(sign * crab_deg.to_radians(), seed, steps);
            let [start, straight, end] =
                flown.states.map(|(e, s)| (e.to_degrees(), s.to_degrees()));
            let (refused, suspect) = (flown.refused, flown.suspect);
            tracing::info!(
                crab_deg,
                seed,
                ?start,
                ?straight,
                ?end,
                refused,
                suspect,
                "heading error and one-sigma, degrees"
            );
        }
    }
}

//! Star lines of sight on a surface vehicle with a tactical IMU: the
//! distribution of the horizontal error with and without stars over
//! many seeds, at rest and on a route with coordinated turns. The truth
//! runs through the mechanization of the filter, so the IMU, the
//! alignment, and the star errors are exactly the errors that the
//! filter models. A correct filter that receives a correct
//! extra measurement must not get worse on average, and its horizontal
//! NEES must stay in its band.

use nalgebra::{Rotation3, Vector3};

use super::flight_truth::{Alignment, FREE_INDEX, Flight, Route, draw3, tactical};
use super::*;
use crate::observation::{LineOfSightReference, Observation, ObservationValue};

/// Independent runs for each case. The debug build of `cargo test`
/// runs a small sample to keep the run short; the release build that
/// `ci.sh` runs takes the full sample.
const SEEDS: u32 = if cfg!(debug_assertions) { 3 } else { 20 };
/// The band of the mean horizontal NEES. The sum of the NEES of
/// [`SEEDS`] consistent runs is χ² with `2·SEEDS` degrees of freedom.
/// For 3 runs, the 0.1 % and 99.9 % points of χ²(6) give a mean in
/// `[0.13, 7.5]`. For 20 runs, the band is the `[0.5, 6]` of the trial
/// acceptance, which is wider than the χ²(40) interval.
const NEES_BAND: (f64, f64) = if cfg!(debug_assertions) {
    (0.13, 7.5)
} else {
    (0.5, 6.0)
};
/// The vehicle at rest for the ten minutes of the measured case in the
/// release build.
const AT_REST: Route = Route {
    speed_mps: 0.0,
    bank_deg: 0.0,
    duration_s: if cfg!(debug_assertions) { 90.0 } else { 600.0 },
    step_s: 1.0,
    turns: &[],
};
/// A route with coordinated 90° turns at 45° bank and 55 m/s.
const TURNING: Route = if cfg!(debug_assertions) {
    Route {
        speed_mps: 55.0,
        bank_deg: 45.0,
        duration_s: 90.0,
        step_s: 0.5,
        turns: &[(30.0, 1.0)],
    }
} else {
    Route {
        speed_mps: 55.0,
        bank_deg: 45.0,
        duration_s: 600.0,
        step_s: 0.1,
        turns: &[(100.0, 1.0), (250.0, -1.0), (400.0, 1.0), (520.0, 1.0)],
    }
};
/// The in-flight alignment of the measured case.
const ALIGNMENT: Alignment = Alignment {
    position_m: 2.0,
    velocity_mps: 0.1,
    attitude_rad: 1e-3,
};
/// Stars in each 1 Hz frame of a roof camera.
const STARS_PER_FRAME: u32 = 8;
/// Centroid direction 1-sigma, radians.
const STAR_SIGMA_RAD: f64 = 5e-6;
/// Camera mount alignment 1-sigma, radians: a constant error of each
/// run, declared in the variance of each line of sight.
const MOUNT_SIGMA_RAD: f64 = 2e-5;

/// One case of the distribution.
#[derive(Clone, Copy)]
struct Case {
    stars: bool,
    route: Route,
}

/// The horizontal NEES and the horizontal error in meters at the end of
/// one run of a tactical IMU.
fn run(seed: u32, case: Case) -> (f64, f64) {
    let mut flight = Flight::new(case.route, seed, tactical(), ALIGNMENT);
    let mount = Rotation3::new(draw3(seed, FREE_INDEX, MOUNT_SIGMA_RAD));
    let variance = STAR_SIGMA_RAD.powi(2) + MOUNT_SIGMA_RAD.powi(2);
    let per_frame = (1.0 / case.route.step_s).round() as u32;
    let mut star = 0_u32;
    let mut at_ms = 0;
    for k in 1..=case.route.steps() {
        at_ms = flight.step();
        if !case.stars || k % per_frame != 0 {
            continue;
        }
        let c = flight.truth_body_to_frame();
        for _ in 0..STARS_PER_FRAME {
            star = star.wrapping_add(1);
            let tilt = draw3(seed, FREE_INDEX + 3 + 6 * star, 0.25);
            let body_unit = Vector3::new(tilt.x, tilt.y, -1.0).normalize();
            let noise = draw3(seed, FREE_INDEX + 6 + 6 * star, STAR_SIGMA_RAD);
            let measured_unit = (mount * body_unit + noise).normalize();
            let observation = Observation::in_frame(
                stamp(7, star, at_ms),
                ObservationValue::LineOfSight {
                    measured_body_unit: measured_unit.into(),
                    reference: LineOfSightReference::InertialDirection {
                        direction_in_frame_unit: (c * body_unit).into(),
                    },
                    angular_variance_rad2: variance,
                },
                SourceComposition::of(SensorClass::Celestial),
                flight.filter.frame(),
            );
            assert_accepted(flight.filter.ingest(&observation, t(at_ms)));
        }
    }
    let (nees, _, error) = flight.horizontal(at_ms);
    (nees, error)
}

/// Mean horizontal NEES, RMS horizontal error, and the errors of each
/// seed.
fn distribution(case: Case) -> (f64, f64, std::vec::Vec<f64>) {
    let runs: std::vec::Vec<(f64, f64)> = (0..SEEDS).map(|seed| run(seed, case)).collect();
    let n = f64::from(SEEDS);
    let nees = runs.iter().map(|r| r.0).sum::<f64>() / n;
    let rms = (runs.iter().map(|r| r.1 * r.1).sum::<f64>() / n).sqrt();
    (nees, rms, runs.into_iter().map(|r| r.1).collect())
}

#[test]
fn monte_carlo_stars_do_not_make_the_horizontal_estimate_worse() {
    for (label, route) in [("at rest", AT_REST), ("turning", TURNING)] {
        let (free_nees, free_rms, free) = distribution(Case {
            stars: false,
            route,
        });
        let (star_nees, star_rms, starred) = distribution(Case { stars: true, route });
        // The horizontal NEES has two degrees of freedom: its mean is 2.
        let (low, high) = NEES_BAND;
        for nees in [free_nees, star_nees] {
            assert!(
                (low..=high).contains(&nees),
                "{label}: NEES free {free_nees} stars {star_nees}"
            );
        }
        // Three runs are too few to compare two error distributions;
        // the release sample compares them.
        if cfg!(debug_assertions) {
            continue;
        }
        assert!(
            star_rms < free_rms,
            "{label}: RMS stars {star_rms} m vs free {free_rms} m"
        );
        // A single seed can be worse with stars and still be correct. For
        // two independent horizontal errors with the 1-sigma σ_s and σ_f,
        // the chance that the one with stars is larger is
        // σ_s² / (σ_s² + σ_f²): a quarter at σ_s = 0.58·σ_f. So the test
        // asks for more than half of the seeds, not for each one.
        let better = free.iter().zip(&starred).filter(|(f, s)| s < f).count();
        assert!(
            better * 2 > free.len(),
            "{label}: stars better on {better} of {} seeds",
            free.len()
        );
    }
}

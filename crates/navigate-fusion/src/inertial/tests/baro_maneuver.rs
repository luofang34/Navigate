//! A barometric altimeter on a route with coordinated turns and
//! kilometers of horizontal error (defect 5 of the simulator trial).
//!
//! The truth runs through the mechanization of the filter, so the IMU,
//! the alignment, and the altimeter errors are exactly the errors that
//! the filter models: a tactical IMU, an in-flight alignment, and an
//! altimeter with a bias and white noise. Without the second-order
//! height term and with a first-order update of the horizontal states,
//! the horizontal NEES of this route rose to 80 after the third turn.

use std::vec::Vec;

use navigate_geodesy::BodyEllipsoid;

use super::flight_truth::{Alignment, FREE_INDEX, Flight, Route, gauss, tactical};
use super::*;
use crate::inertial::local_level::LocalLevel;
use crate::inertial::nominal::{POS, VEL};
use crate::observation::{Observation, ObservationValue};

/// The route of the trial: 40 m/s, three turns at 25° bank, 1400 s.
const TRIAL_ROUTE: Route = Route {
    speed_mps: 40.0,
    bank_deg: 25.0,
    duration_s: 1_400.0,
    step_s: 0.1,
    turns: &[(507.5, 1.0), (784.0, -1.0), (1_203.9, 1.0)],
};

/// A short route for the debug build: 55 m/s, two turns at 45° bank.
const SHORT_ROUTE: Route = Route {
    speed_mps: 55.0,
    bank_deg: 45.0,
    duration_s: 300.0,
    step_s: 0.5,
    turns: &[(100.0, 1.0), (200.0, -1.0)],
};

/// The debug build of `cargo test` runs the short route with two seeds;
/// the release build that `ci.sh` runs takes the trial route with
/// twelve.
const ROUTE: Route = if cfg!(debug_assertions) {
    SHORT_ROUTE
} else {
    TRIAL_ROUTE
};
const SEEDS: u32 = if cfg!(debug_assertions) { 2 } else { 12 };

/// The band of the mean horizontal NEES: the 0.1 % and 99.9 % points of
/// χ²(4) over 2 for two runs, and the `[0.5, 6]` of the trial acceptance
/// for twelve.
const NEES_BAND: (f64, f64) = if cfg!(debug_assertions) {
    (0.045, 9.2)
} else {
    (0.5, 6.0)
};

/// The band of the mean NEES of the geodetic height error (one degree
/// of freedom per run): the 0.1 % and 99.9 % points of χ²(2)/2 for two
/// runs and of χ²(12)/12 for twelve.
const HEIGHT_BAND: (f64, f64) = if cfg!(debug_assertions) {
    (0.001, 6.9)
} else {
    (0.18, 2.74)
};

/// The in-flight alignment of the trial.
const ALIGNMENT: Alignment = Alignment {
    position_m: 20.0,
    velocity_mps: 0.2,
    attitude_rad: 2e-3,
};

/// The horizontal NEES, the horizontal 1-sigma, the horizontal error,
/// and the NEES of the geodetic height error every 100 s of one run,
/// with a 1 m altimeter that has the filter's bias model at each IMU
/// step when `baro` holds.
fn run(seed: u32, baro: bool) -> Vec<[f64; 4]> {
    let config = tactical();
    let mut flight = Flight::<22>::new(ROUTE, seed, config, ALIGNMENT);
    let baro_bias = config.baro_bias.sigma() * gauss(seed, FREE_INDEX);
    let steps = ROUTE.steps();
    let per_report = (100.0 / ROUTE.step_s).round() as u32;
    let mut samples = Vec::new();
    let mut refused = 0_u32;
    for k in 1..=steps {
        let at_ms = flight.step();
        if baro {
            let obs = Observation::new(
                stamp(3, k, at_ms),
                ObservationValue::Altitude {
                    altitude_m: flight.truth_height_m() + baro_bias + gauss(seed, FREE_INDEX + k),
                    variance_m2: 1.0,
                },
                SourceComposition::of(SensorClass::Barometric),
            );
            if !flight.filter.ingest(&obs, t(at_ms)).is_accepted() {
                refused = refused.wrapping_add(1);
            }
        }
        if k % per_report == 0 {
            let (nees, sigma, error) = flight.horizontal(at_ms);
            samples.push([nees, sigma, error, flight.height_nees()]);
        }
    }
    // The 0.1 % innovation gate refuses a few true draws.
    assert!(refused * 100 <= steps, "{refused} of {steps} refused");
    samples
}

/// Mean horizontal NEES, mean horizontal 1-sigma, RMS horizontal error,
/// and mean height NEES at each report over the seeds.
fn statistics(baro: bool) -> Vec<[f64; 4]> {
    let runs: Vec<Vec<[f64; 4]>> = (0..SEEDS).map(|seed| run(seed, baro)).collect();
    let n = f64::from(SEEDS);
    let reports = runs.first().map_or(0, Vec::len);
    (0..reports)
        .map(|i| {
            let at = |run: &Vec<[f64; 4]>| run.get(i).copied().unwrap_or_default();
            let mean = |j: usize| runs.iter().map(|r| at(r)[j]).sum::<f64>() / n;
            let rms = (runs.iter().map(|r| at(r)[2].powi(2)).sum::<f64>() / n).sqrt();
            [mean(0), mean(1), rms, mean(3)]
        })
        .collect()
}

#[test]
fn monte_carlo_an_altimeter_in_turns_keeps_the_horizontal_estimate_consistent() {
    let free = statistics(false);
    let baro = statistics(true);
    let (low, high) = NEES_BAND;
    for (index, ([free_nees, _, free_rms, _], [nees, sigma, rms, _])) in
        free.iter().zip(&baro).enumerate()
    {
        let at_s = (index + 1) * 100;
        assert!(
            (low..=high).contains(nees),
            "{at_s} s: NEES {nees}, σ_h {sigma} m, RMS {rms} m"
        );
        assert!(
            (low..=high).contains(free_nees),
            "{at_s} s: INS NEES {free_nees}"
        );
        // The altimeter does not make the horizontal error worse.
        assert!(
            *rms <= 1.05 * free_rms,
            "{at_s} s: RMS {rms} vs INS {free_rms}"
        );
    }
    // The height is consistent too: the residual of an altitude is the
    // geodetic height difference, which is the vertical error state.
    let (low, high) = HEIGHT_BAND;
    for (index, report) in baro.iter().enumerate() {
        let height = report[3];
        assert!(
            (low..=high).contains(&height),
            "{} s: height NEES {height}",
            (index + 1) * 100
        );
    }
}

#[test]
fn an_altitude_update_does_not_change_the_horizontal_covariance() {
    // After a turn the error state correlates the vertical with the
    // horizontal; an altitude update leaves the horizontal position and
    // velocity covariance as it was, and still reduces the vertical.
    let mut filter = surface_filter(tactical(), PlanetaryBody::EARTH_WGS84, earth_origin());
    let rates = rest_rates(&filter);
    for k in 1..=120_u32 {
        let at_ms = u64::from(k) * 1_000;
        assert_accepted(filter.ingest_imu(&imu(k, at_ms, rates, 1.0), t(at_ms)));
    }
    let before = *core(&filter);
    let obs = Observation::new(
        stamp(3, 1, 120_000),
        ObservationValue::Altitude {
            altitude_m: earth_origin().altitude_m + 2.0,
            variance_m2: 1.0,
        },
        SourceComposition::of(SensorClass::Barometric),
    );
    assert_accepted(filter.ingest(&obs, t(120_000)));
    let after = *core(&filter);
    let local = LocalLevel::at_body_fixed(&before.nominal.p, BodyEllipsoid::WGS84);
    for offset in [POS, VEL] {
        let block = |c: &crate::inertial::estimate::InertialCore| {
            local.bf_to_ned_cov(&c.p.fixed_view::<3, 3>(offset, offset).into_owned())
        };
        let (b, a) = (block(&before), block(&after));
        let horizontal = (b.fixed_view::<2, 2>(0, 0) - a.fixed_view::<2, 2>(0, 0))
            .abs()
            .max();
        assert!(horizontal <= 1e-9 * b.abs().max(), "{offset}: {horizontal}");
        assert!(a[(2, 2)] < b[(2, 2)], "{offset}: vertical");
    }
    let shift = local.bf_to_ned() * (after.nominal.p - before.nominal.p);
    assert!(shift.x.hypot(shift.y) < 1e-6, "{shift}");
}

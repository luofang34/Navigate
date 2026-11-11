//! GNSS position fixes with a time-correlated error on a turning route
//! (ADR-0023): the 3-D position NEES over many seeds when the receiver
//! declares its error as white, when it declares the correlated part
//! and the layout has a block for it, and when it declares it and the
//! layout has no block, so the filter inflates the fixes. The truth runs
//! through the mechanization of the filter, and the fix error is a
//! first-order Gauss–Markov process plus white noise, as the receiver
//! declares it.

use nalgebra::Vector3;
use navigate_geodesy::{BodyEllipsoid, BodyFixedPosition, body_fixed_to_geodetic};

use super::flight_truth::{Alignment, FREE_INDEX, Flight, Route, gauss, tactical};
use super::*;
use crate::inertial::StateLayout;
use crate::inertial::local_level::LocalLevel;
use crate::observation::{CorrelatedError, Observation, ObservationValue};
use nalgebra::{Const, DimMin};

/// Independent runs for each case: one short run in the debug build,
/// the full sample in the release build that `ci.sh` runs.
const SEEDS: u32 = if cfg!(debug_assertions) { 1 } else { 12 };
/// The band of the mean 3-D NEES: the 0.1 % and 99.9 % points of
/// χ²(3·SEEDS)/SEEDS. χ²(36)/12 gives `[1.28, 5.67]`, χ²(3) gives
/// `[0.024, 16.3]`.
pub(super) const POSITION_BAND: (f64, f64) = if cfg!(debug_assertions) {
    (0.024, 16.3)
} else {
    (1.28, 5.67)
};
/// 40 m/s with 25° turns, twenty minutes: four correlation times.
const ROUTE: Route = if cfg!(debug_assertions) {
    Route {
        speed_mps: 40.0,
        bank_deg: 25.0,
        duration_s: 120.0,
        step_s: 0.5,
        turns: &[(40.0, 1.0)],
    }
} else {
    Route {
        speed_mps: 40.0,
        bank_deg: 25.0,
        duration_s: 1_200.0,
        step_s: 0.1,
        turns: &[(300.0, 1.0), (620.0, -1.0), (900.0, 1.0)],
    }
};
/// The times of the NEES reports, seconds.
const REPORTS_S: &[u64] = if cfg!(debug_assertions) {
    &[60, 120]
} else {
    &[300, 600, 900, 1_200]
};
const ALIGNMENT: Alignment = Alignment {
    position_m: 2.0,
    velocity_mps: 0.1,
    attitude_rad: 1e-3,
};
/// The receiver of the simulator trial: 1 Hz fixes, white 0.6 m
/// horizontal and 1.0 m vertical, and a correlated 1.2 m horizontal and
/// 2.0 m vertical with a correlation time of 300 s.
pub(super) const WHITE_SIGMA_NED_M: [f64; 3] = [0.6, 0.6, 1.0];
pub(super) const CORRELATED_SIGMA_NED_M: [f64; 3] = [1.2, 1.2, 2.0];
pub(super) const TAU_S: f64 = 300.0;
const FIX_PERIOD_S: f64 = 1.0;
const SOURCE: u32 = 3;

/// What the receiver declares.
#[derive(Clone, Copy, Debug)]
pub(super) enum Declared {
    /// The white and the correlated variance together, as white.
    AllWhite,
    /// The white variance, and the correlated process with the
    /// correlation time `tau_s`.
    Correlated { tau_s: f64 },
}

/// The correlated fix error of one seed: a first-order Gauss–Markov
/// process in the local NED axes, drawn from its stationary
/// distribution at the first fix.
pub(super) struct FixError {
    seed: u32,
    state: Vector3<f64>,
    fix: u32,
}

impl FixError {
    pub(super) fn new(seed: u32) -> Self {
        let mut error = Self {
            seed,
            state: Vector3::zeros(),
            fix: 0,
        };
        error.state = error.draw(CORRELATED_SIGMA_NED_M);
        error
    }

    fn draw(&mut self, sigma: [f64; 3]) -> Vector3<f64> {
        self.fix = self.fix.wrapping_add(1);
        let base = FREE_INDEX + 1_000_000 + 3 * self.fix;
        Vector3::from_fn(|axis, _| {
            let index = base + axis as u32;
            gauss(self.seed, index) * sigma.get(axis).copied().unwrap_or_default()
        })
    }

    /// The error of the next fix: the correlated state one period on,
    /// plus white noise.
    pub(super) fn next(&mut self) -> Vector3<f64> {
        let rho = (-FIX_PERIOD_S / TAU_S).exp();
        let scale = (1.0 - rho * rho).sqrt();
        let innovation = self.draw(CORRELATED_SIGMA_NED_M.map(|s| s * scale));
        self.state = self.state * rho + innovation;
        self.state + self.draw(WHITE_SIGMA_NED_M)
    }
}

/// The fix of the truth of `flight` with the NED error `error`, with
/// the covariance and the declaration of `declared`.
pub(super) fn fix<const N: usize>(
    flight: &Flight<N>,
    error: &Vector3<f64>,
    declared: Declared,
    sequence: u32,
    at_ms: u64,
) -> Observation
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let level = LocalLevel::at_body_fixed(&flight.truth.p, BodyEllipsoid::WGS84);
    let p = flight.truth.p + level.ned_to_bf * error;
    let position =
        body_fixed_to_geodetic(&BodyFixedPosition::new(p.x, p.y, p.z), BodyEllipsoid::WGS84);
    let white = WHITE_SIGMA_NED_M.map(|s| s * s);
    let correlated = CORRELATED_SIGMA_NED_M.map(|s| s * s);
    let variance = match declared {
        Declared::AllWhite => [0, 1, 2].map(|i| {
            white.get(i).copied().unwrap_or_default()
                + correlated.get(i).copied().unwrap_or_default()
        }),
        Declared::Correlated { .. } => white,
    };
    let [n, e, d] = variance;
    let observation = Observation::new(
        stamp(SOURCE, sequence, at_ms),
        ObservationValue::PositionFix {
            position,
            covariance: SymmetricCov3::from_diagonal(n, e, d),
        },
        gnss(),
    );
    match declared {
        Declared::AllWhite => observation,
        Declared::Correlated { tau_s } => {
            observation.with_correlated_error(CorrelatedError::new(tau_s, CORRELATED_SIGMA_NED_M))
        }
    }
}

/// The 3-D position NEES of one run at each report time.
fn run<const N: usize>(seed: u32, layout: StateLayout, declared: Declared) -> std::vec::Vec<f64>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let config = InertialConfig {
        layout,
        ..tactical()
    };
    let mut flight = Flight::<N>::new(ROUTE, seed, config, ALIGNMENT);
    let mut error = FixError::new(seed);
    let per_fix = (FIX_PERIOD_S / ROUTE.step_s).round() as u32;
    let per_report = |s: u64| (s as f64 / ROUTE.step_s).round() as u32;
    let mut nees = std::vec::Vec::new();
    let (mut sequence, mut refused) = (0_u32, 0_u32);
    for k in 1..=ROUTE.steps() {
        let at_ms = flight.step();
        if k % per_fix == 0 {
            sequence = sequence.wrapping_add(1);
            let observation = fix(&flight, &error.next(), declared, sequence, at_ms);
            // The 0.1 % gate refuses a few true draws.
            if !flight.filter.ingest(&observation, t(at_ms)).is_accepted() {
                refused = refused.wrapping_add(1);
            }
        }
        if REPORTS_S.iter().any(|s| per_report(*s) == k) {
            nees.push(flight.position_nees());
        }
    }
    assert!(
        refused * 100 <= sequence.max(1),
        "{refused} of {sequence} refused"
    );
    nees
}

/// The mean 3-D NEES over the seeds at each report time.
pub(super) fn mean_nees<const N: usize>(
    layout: StateLayout,
    declared: Declared,
) -> std::vec::Vec<f64>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let mut sum = std::vec![0.0; REPORTS_S.len()];
    for seed in 0..SEEDS {
        for (total, nees) in sum.iter_mut().zip(run::<N>(seed, layout, declared)) {
            *total += nees;
        }
    }
    sum.into_iter()
        .map(|total| total / f64::from(SEEDS))
        .collect()
}

fn in_band(nees: &[f64]) -> bool {
    let (low, high) = POSITION_BAND;
    nees.iter().all(|value| (low..=high).contains(value))
}

/// The layout with one correlated-error block: 25 states.
const BOUND: StateLayout = StateLayout::FULL.with_correlated_blocks(1);

#[test]
fn monte_carlo_a_correlated_gnss_error_declared_as_white_makes_the_position_over_confident() {
    let nees = mean_nees::<22>(StateLayout::FULL, Declared::AllWhite);
    // The white model averages the correlated error over many fixes it
    // takes as independent, so the covariance shrinks below the error:
    // the mean NEES is 16–28 after the first correlation time in the
    // release sample. One debug run cannot show it.
    if !cfg!(debug_assertions) {
        assert!(
            nees.iter().skip(1).all(|value| *value > POSITION_BAND.1),
            "NEES {nees:?}"
        );
    }
}

#[test]
fn monte_carlo_a_declared_correlated_gnss_error_with_a_block_keeps_the_position_nees_in_band() {
    let nees = mean_nees::<25>(BOUND, Declared::Correlated { tau_s: TAU_S });
    assert!(in_band(&nees), "NEES {nees:?}");
}

#[test]
fn monte_carlo_a_declared_correlated_gnss_error_without_a_block_is_inflated_into_band() {
    let nees = mean_nees::<22>(StateLayout::FULL, Declared::Correlated { tau_s: TAU_S });
    // The inflation gives each fix the information of its share of the
    // independent samples of the process, without the knowledge of the
    // block, so its covariance is larger than the error: the mean NEES
    // is near 0.3 in the release sample. It must not be above the band,
    // and a floor keeps the fixes useful.
    assert!(
        nees.iter()
            .all(|value| (INFLATED_FLOOR..=POSITION_BAND.1).contains(value)),
        "NEES {nees:?}"
    );
}

/// The lowest mean NEES of the inflated fixes.
const INFLATED_FLOOR: f64 = if cfg!(debug_assertions) { 0.001 } else { 0.1 };

#[test]
fn monte_carlo_a_correlation_time_declared_three_times_too_short_or_too_long_stays_in_band() {
    // The block model is not sensitive to a factor of three in the
    // correlation time on this route: the fixes estimate the block, and
    // its covariance follows the declared stationary variance.
    for tau_s in [TAU_S / 3.0, TAU_S * 3.0] {
        let nees = mean_nees::<25>(BOUND, Declared::Correlated { tau_s });
        assert!(in_band(&nees), "tau {tau_s} s: NEES {nees:?}");
    }
}

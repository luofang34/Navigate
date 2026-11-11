//! Star lines of sight on a surface vehicle with a tactical IMU: the
//! distribution of the horizontal error with and without stars over
//! many seeds, at rest and on a route with coordinated turns. The truth
//! runs through the mechanization of the filter, so the IMU, the
//! alignment, and the star errors are exactly the errors that the
//! filter models. A correct filter that receives a correct
//! extra measurement must not get worse on average, and its horizontal
//! NEES must stay in its band.

use nalgebra::{Const, DimMin, Rotation3, Vector3};

use super::flight_truth::{Alignment, FREE_INDEX, Flight, Route, draw3, tactical};
use super::*;
use crate::inertial::StateLayout;
use crate::observation::{CorrelatedError, LineOfSightReference, Observation, ObservationValue};

/// Independent runs for each case. The debug build of `cargo test`
/// runs one short run of each case as a smoke test; the release build
/// that `ci.sh` runs takes the full sample.
const SEEDS: u32 = if cfg!(debug_assertions) { 1 } else { 20 };
/// The bands of the mean NEES: the 0.1 % and 99.9 % points of
/// χ²(k·SEEDS)/SEEDS for k degrees of freedom. Horizontal (k = 2):
/// χ²(40)/20 gives `[0.90, 3.67]`, χ²(2) gives `[0.002, 13.8]`.
/// Attitude (k = 3): χ²(60)/20 gives `[1.59, 4.98]`, χ²(3) gives
/// `[0.024, 16.3]`.
const HORIZONTAL_BAND: (f64, f64) = if cfg!(debug_assertions) {
    (0.002, 13.8)
} else {
    (0.90, 3.67)
};
const ATTITUDE_BAND: (f64, f64) = if cfg!(debug_assertions) {
    (0.024, 16.3)
} else {
    (1.59, 4.98)
};
/// The vehicle at rest for the ten minutes of the measured case in the
/// release build.
const AT_REST: Route = Route {
    speed_mps: 0.0,
    bank_deg: 0.0,
    duration_s: if cfg!(debug_assertions) { 60.0 } else { 600.0 },
    step_s: 1.0,
    turns: &[],
};
/// A route with coordinated 90° turns at 45° bank and 55 m/s.
const TURNING: Route = if cfg!(debug_assertions) {
    Route {
        speed_mps: 55.0,
        bank_deg: 45.0,
        duration_s: 60.0,
        step_s: 0.5,
        turns: &[(20.0, 1.0)],
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
/// The constant camera mount error of the mount cases, radians about
/// each axis.
const MOUNT_SIGMA_RAD: f64 = 2e-5;

/// One case of the distribution.
#[derive(Clone, Copy)]
struct Case {
    stars: bool,
    route: Route,
    /// 1-sigma of the constant mount error of each run, radians.
    mount_rad: f64,
    /// Whether the camera declares the mount error as a constant
    /// correlated error. Otherwise each line of sight declares it as
    /// white variance.
    declare_mount: bool,
}

/// One frame of the roof camera: `STARS_PER_FRAME` lines of sight in a
/// cone around the body zenith, with centroid noise.
fn offer_stars<const N: usize>(
    flight: &mut Flight<N>,
    star: &mut u32,
    mount: &Rotation3<f64>,
    case: Case,
    at_ms: u64,
) -> u32
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let seed = flight.seed;
    let mount_rad = case.mount_rad;
    let variance = if case.declare_mount {
        STAR_SIGMA_RAD.powi(2)
    } else {
        STAR_SIGMA_RAD.powi(2) + mount_rad.powi(2)
    };
    let c = flight.truth_body_to_frame();
    let mut refused = 0_u32;
    for _ in 0..STARS_PER_FRAME {
        *star = star.wrapping_add(1);
        let tilt = draw3(seed, FREE_INDEX + 3 + 6 * *star, 0.25);
        let body_unit = Vector3::new(tilt.x, tilt.y, -1.0).normalize();
        let noise = draw3(seed, FREE_INDEX + 6 + 6 * *star, STAR_SIGMA_RAD);
        let measured_unit = (mount * body_unit + noise).normalize();
        let observation = Observation::in_frame(
            stamp(7, *star, at_ms),
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
        let observation = if case.declare_mount {
            observation.with_correlated_error(CorrelatedError::new(f64::INFINITY, [mount_rad; 3]))
        } else {
            observation
        };
        // The 0.1 % gate refuses a few true draws.
        if !flight.filter.ingest(&observation, t(at_ms)).is_accepted() {
            refused = refused.wrapping_add(1);
        }
    }
    refused
}

/// The horizontal NEES, the attitude NEES, and the horizontal error in
/// meters at the end of one run of a tactical IMU.
fn run<const N: usize>(seed: u32, case: Case, layout: StateLayout) -> (f64, f64, f64)
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let config = InertialConfig {
        layout,
        ..tactical()
    };
    let mut flight = Flight::<N>::new(case.route, seed, config, ALIGNMENT);
    let mount = Rotation3::new(draw3(seed, FREE_INDEX, case.mount_rad));
    let per_frame = (1.0 / case.route.step_s).round() as u32;
    let (mut star, mut refused) = (0_u32, 0_u32);
    let mut at_ms = 0;
    for k in 1..=case.route.steps() {
        at_ms = flight.step();
        if case.stars && k % per_frame == 0 {
            refused =
                refused.wrapping_add(offer_stars(&mut flight, &mut star, &mount, case, at_ms));
        }
    }
    assert!(refused * 100 <= star.max(1), "{refused} of {star} refused");
    let (nees, _, error) = flight.horizontal(at_ms);
    (nees, flight.attitude_nees(), error)
}

/// Mean horizontal NEES, mean attitude NEES, RMS horizontal error, and
/// the errors of each seed.
fn distribution(case: Case) -> (f64, f64, f64, std::vec::Vec<f64>) {
    let runs: std::vec::Vec<(f64, f64, f64)> = (0..SEEDS)
        .map(|seed| {
            if case.declare_mount {
                run::<25>(seed, case, StateLayout::FULL.with_correlated_blocks(1))
            } else {
                run::<22>(seed, case, StateLayout::FULL)
            }
        })
        .collect();
    let n = f64::from(SEEDS);
    let nees = runs.iter().map(|r| r.0).sum::<f64>() / n;
    let attitude = runs.iter().map(|r| r.1).sum::<f64>() / n;
    let rms = (runs.iter().map(|r| r.2 * r.2).sum::<f64>() / n).sqrt();
    (nees, attitude, rms, runs.into_iter().map(|r| r.2).collect())
}

#[test]
fn monte_carlo_stars_do_not_make_the_horizontal_estimate_worse() {
    for (label, route) in [("at rest", AT_REST), ("turning", TURNING)] {
        let (free_nees, free_att, free_rms, free) = distribution(Case {
            stars: false,
            route,
            mount_rad: 0.0,
            declare_mount: false,
        });
        let (star_nees, star_att, star_rms, starred) = distribution(Case {
            stars: true,
            route,
            mount_rad: 0.0,
            declare_mount: false,
        });
        let (low, high) = HORIZONTAL_BAND;
        for nees in [free_nees, star_nees] {
            assert!(
                (low..=high).contains(&nees),
                "{label}: NEES free {free_nees} stars {star_nees}"
            );
        }
        // Stars fix the attitude: its covariance must follow the error.
        let (low, high) = ATTITUDE_BAND;
        for nees in [free_att, star_att] {
            assert!(
                (low..=high).contains(&nees),
                "{label}: attitude NEES free {free_att} stars {star_att}"
            );
        }
        // One run cannot compare two error distributions; the release
        // sample compares them.
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

#[test]
fn monte_carlo_a_constant_mount_error_needs_its_correlated_declaration() {
    // A constant camera mount error that each line of sight declares as
    // white variance: the filter averages it over the lines of sight and
    // the frames as if it were noise, so the attitude covariance falls
    // below the error, and the mean attitude NEES is above its band.
    // Declared as a constant correlated error (ADR-0023), the mount has a
    // block that every line of sight of the camera reads, and the
    // attitude covariance keeps the mount uncertainty.
    let case = Case {
        stars: true,
        route: AT_REST,
        mount_rad: MOUNT_SIGMA_RAD,
        declare_mount: false,
    };
    let (_, white, _, _) = distribution(case);
    assert!(white > ATTITUDE_BAND.1, "attitude NEES {white}");
    let (_, declared, _, _) = distribution(Case {
        declare_mount: true,
        ..case
    });
    let (low, high) = ATTITUDE_BAND;
    assert!((low..=high).contains(&declared), "attitude NEES {declared}");
}

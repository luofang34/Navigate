//! The filter flown on from a declared start at 55 m/s: with GNSS, with a
//! barometer and position fixes, and with a barometer alone. The vehicle
//! accelerates along its track over the levelling window by a draw of the
//! allowance that the host states, so the levelled tilt is wrong by what
//! that allowance says.

use navigate_contract::{NavigationSolution, SourceId, SymmetricCov3};
use navigate_fusion::FogmBias;

use super::declared::{CRAB_SIGMA_RAD, STATED_AT, declare, statement};
use super::moving::{attitude_error, normalized};
use super::path::{Profile, Truth, Vehicle, ramp};
use super::rig::{self, Noise, Rig, at};
use crate::{
    AlignmentConfig, AlignmentStatus, ErrorBudget, Navigator, OnboardError, ReferenceNavigator,
    SensorData, SensorInput, SensorKind, SensorSpec, SensorSuite,
};

const OPERATOR: SourceId = SourceId::new(5);
const BARO: SourceId = SourceId::new(6);
/// Seconds after the start at which the state is judged.
const CHECKS_S: [u64; 8] = [0, 5, 10, 20, 30, 60, 120, 180];
/// The along-track acceleration allowance that the host states for its
/// start in steady flight, m/s².
const ACCEL_SIGMA: f64 = 0.3;
const SEEDS: u64 = 12;
/// The largest mean of twelve normalized errors of three degrees of
/// freedom: chi-square of 36, 3.3 sigma above.
const NEES_BAND: f64 = 5.3;

/// What aids the filter after the start.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Aiding {
    /// GNSS position and velocity each second.
    Gnss,
    /// A barometer each second and a position fix of 15 m every 5 s, as a
    /// terrain fix gives it.
    BaroAndPosition,
    /// A barometer each second.
    Baro,
}

/// Along-track acceleration `accel` until the statement, the stated speed
/// at it; then straight to 30 s and a weave of ±17° of heading with a
/// period of 40 s, at the bank of a coordinated turn. The swing grows
/// smoothly over 10 s, so the motion has a continuous rate.
fn weave(heading: f64, accel: f64) -> impl Fn(f64) -> Profile {
    let stated_s = STATED_AT as f64 / 100.0;
    move |t| {
        let omega = core::f64::consts::TAU / 40.0;
        let swing = 0.3 * ramp(t, 30.0, 40.0);
        let phase = omega * (t - 30.0);
        // The rate of the heading; the growth of the swing adds little.
        let rate = swing * omega * phase.cos();
        Profile {
            speed_m_s: 55.0 + accel * (t - stated_s).min(0.0),
            heading_rad: heading + swing * phase.sin(),
            roll_rad: (55.0 * rate / 9.806_65).atan(),
            pitch_rad: 0.04,
            climb_m_s: 0.0,
        }
    }
}

/// A navigator with an inertial unit, an operator, a receiver and a
/// barometer, whose course is its heading, and that states `accel` as
/// the acceleration allowance of its start.
fn navigator(accel: Option<f64>) -> Result<ReferenceNavigator, OnboardError> {
    let rig = Rig::new([0.0; 3], 0, 1.0);
    let declared = rig.suite(None);
    let of = |kind| declared.of_kind(kind).next().unwrap().clone();
    let imu = of(SensorKind::Imu);
    let operator = SensorSpec {
        id: OPERATOR,
        kind: SensorKind::Operator,
        budget: ErrorBudget::Declared,
        ..imu.clone()
    };
    let baro = SensorSpec {
        id: BARO,
        kind: SensorKind::Barometer,
        budget: ErrorBudget::Barometer {
            white_sigma_m: 2.0,
            bias: FogmBias::new(600.0, 5.0),
        },
        ..imu.clone()
    };
    let suite = SensorSuite::new(vec![imu, of(SensorKind::Gnss), operator, baro]).unwrap();
    let mut navigator = ReferenceNavigator::new(suite, rig::CLOCK, None).unwrap();
    navigator.set_alignment_config(AlignmentConfig {
        course_heading_sigma_rad: Some(CRAB_SIGMA_RAD),
        declared_acceleration_sigma_m_s2: accel,
        ..AlignmentConfig::default()
    })?;
    Ok(navigator)
}

/// The normalized attitude error and the stated tilt one-sigma.
fn judge(truth: &Truth, solution: &NavigationSolution) -> [f64; 2] {
    let error = attitude_error(truth, solution);
    let cov = solution.attitude_cov.unwrap();
    let (north, east, _) = cov.diagonal();
    [normalized(error, &cov), ((north + east) / 2.0).sqrt()]
}

/// Gives the aiding of `aiding` that is due at the present step.
fn aid<F: Fn(f64) -> Profile>(
    navigator: &mut ReferenceNavigator,
    vehicle: &mut Vehicle<F>,
    aiding: Aiding,
    baro_noise: &mut Noise,
) {
    let mut sink = Vec::new();
    let second = vehicle.step.is_multiple_of(100);
    if second && aiding != Aiding::Gnss {
        let height = vehicle.truth().position[2] + baro_noise.normal() * 2.0;
        let input = SensorInput {
            sensor: BARO,
            acquired_at: at(vehicle.step),
            delivered_at: at(vehicle.step),
            data: SensorData::BaroAltitude { altitude_m: height },
        };
        navigator.ingest(&input, &mut sink).unwrap();
    }
    match aiding {
        Aiding::Gnss if second => {
            navigator.ingest(&vehicle.gnss(0), &mut sink).unwrap();
        }
        Aiding::BaroAndPosition if vehicle.step.is_multiple_of(500) => {
            let mut fix = vehicle.gnss(0);
            if let SensorData::GnssFix {
                velocity,
                position_cov,
                ..
            } = &mut fix.data
            {
                *velocity = None;
                *position_cov = Some(SymmetricCov3::from_diagonal(225.0, 225.0, 225.0));
            }
            navigator.ingest(&fix, &mut sink).unwrap();
        }
        _ => {}
    }
}

/// The vehicle and navigator of seed `seed` at a declared start: the
/// tilt one-sigma of the alignment and [`judge`] of its state.
fn start(
    seed: u64,
    accel: Option<f64>,
) -> (
    Vehicle<impl Fn(f64) -> Profile>,
    ReferenceNavigator,
    f64,
    [f64; 2],
) {
    let mut noise = Noise::new(seed.wrapping_add(9_000));
    let heading = noise.normal() * 3.0;
    let wind = [noise.normal() * 5.0, noise.normal() * 5.0];
    let along = noise.normal() * ACCEL_SIGMA;
    let mut vehicle = Vehicle::new(weave(heading, along), wind, seed, 1.0);
    let mut navigator = navigator(accel).unwrap();
    let (truth, declared) = declare(
        &mut navigator,
        &mut vehicle,
        |truth| statement(truth, 55.0, &mut Noise::new(seed), 1.0),
        0,
    );
    let Some(AlignmentStatus::Aligned { record, .. }) = navigator.alignment_status() else {
        panic!("a declared start aligns");
    };
    let judged = judge(&truth, &declared.unwrap());
    (vehicle, navigator, record.tilt_sigma_rad, judged)
}

/// One flight of seed `seed` from a declared start: the tilt one-sigma of
/// the alignment, and [`judge`] at each of [`CHECKS_S`].
fn flight(aiding: Aiding, seed: u64, accel: Option<f64>) -> (f64, Vec<[f64; 2]>) {
    let (mut vehicle, mut navigator, tilt, judged) = start(seed, accel);
    let mut checks = vec![judged];
    let mut sink = Vec::new();
    let mut baro_noise = Noise::new(seed.wrapping_add(11_000));
    let last = STATED_AT + CHECKS_S[CHECKS_S.len() - 1] * 100;
    while vehicle.step < last {
        navigator.ingest(&vehicle.increment(), &mut sink).unwrap();
        aid(&mut navigator, &mut vehicle, aiding, &mut baro_noise);
        let elapsed = vehicle.step.saturating_sub(STATED_AT);
        if elapsed.is_multiple_of(100) && CHECKS_S.contains(&(elapsed / 100)) {
            let solution = navigator.solution(at(vehicle.step)).unwrap();
            checks.push(judge(&vehicle.truth(), &solution));
        }
    }
    (tilt, checks)
}

/// The mean over [`SEEDS`] flights of the tilt one-sigma of the alignment
/// and of [`judge`] at each check.
fn mean_flight(aiding: Aiding, accel: Option<f64>) -> (f64, Vec<[f64; 2]>) {
    let n = SEEDS as f64;
    let mut tilt = 0.0;
    let mut mean = vec![[0.0; 2]; CHECKS_S.len()];
    for seed in 0..SEEDS {
        let (sigma, checks) = flight(aiding, seed, accel);
        tilt += sigma / n;
        for (total, check) in mean.iter_mut().zip(checks) {
            total[0] += check[0] / n;
            total[1] += check[1] / n;
        }
    }
    for (seconds, [nees, sigma]) in CHECKS_S.iter().zip(&mean) {
        tracing::info!(
            ?aiding,
            ?accel,
            seconds,
            nees,
            sigma,
            "after a declared start"
        );
    }
    (tilt, mean)
}

/// With the host's statement of a steady start, the levelled tilt is
/// stated to about 0.03 rad, and from there the attitude of each aiding
/// stays as good as it states through three minutes: GNSS brings the
/// tilt to a few milliradians in seconds, position fixes in two minutes;
/// a barometer alone holds it near 25 mrad. Without the statement the
/// tilt one-sigma is the bound of the size test, about 0.2 rad, and a
/// barometer alone leaves the band: the filter is outside its linear
/// range there. A step of the filter is slow without optimization, so
/// the test runs in an optimized build only.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "needs an optimized build: cargo test --release -p navigate-onboard --lib aligner"
)]
fn the_filter_flies_on_from_a_steady_declared_start() {
    rig::show_events();
    for aiding in [Aiding::Gnss, Aiding::BaroAndPosition, Aiding::Baro] {
        let (tilt, mean) = mean_flight(aiding, Some(ACCEL_SIGMA));
        assert!(tilt < 0.035, "{aiding:?}: {tilt}");
        for (seconds, [nees, _]) in CHECKS_S.iter().zip(&mean) {
            assert!(*nees < NEES_BAND, "{aiding:?} at {seconds} s: {nees}");
        }
        let settled = match aiding {
            Aiding::Gnss => 0.005,
            Aiding::BaroAndPosition => 0.01,
            Aiding::Baro => 0.03,
        };
        let last = mean.last().map_or(f64::NAN, |[_, sigma]| *sigma);
        assert!(last < settled, "{aiding:?}: {last}");
    }
    let (tilt, mean) = mean_flight(Aiding::Baro, None);
    assert!(tilt > 0.15, "{tilt}");
    let worst = mean.iter().map(|[nees, _]| *nees).fold(0.0, f64::max);
    assert!(worst > NEES_BAND, "{worst}");
}

/// An acceleration allowance that is not finite and positive is refused,
/// and one larger than the bound of the size test leaves that bound.
#[test]
fn the_acceleration_allowance_only_lowers_the_tilt_bound() {
    for accel in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(navigator(Some(accel)).is_err(), "{accel}");
    }
    let unbounded = start(0, None).2;
    let large = start(0, Some(100.0)).2;
    let stated = start(0, Some(ACCEL_SIGMA)).2;
    assert!((unbounded - large).abs() < 1e-12, "{unbounded} {large}");
    assert!(stated < unbounded / 4.0, "{stated} {unbounded}");
}

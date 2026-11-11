//! The checkpointed history: its depth at a high IMU rate, the replay
//! from a checkpoint, and the pools of the large events.

use std::vec::Vec;

use navigate_contract::{DurationNanos, NavigationSolution, SymmetricCov3};
use navigate_geodesy::BodyEllipsoid;

use super::*;
use crate::inertial::MAX_INERTIAL_HISTORY;
use crate::inertial::history::CHECKPOINT_STRIDE;
use crate::observation::{Observation, ObservationValue};

/// IMU interval of a 500 Hz unit, in microseconds.
const IMU_STEP_US: u64 = 2_000;

fn t_us(us: u64) -> MonotonicNanos {
    MonotonicNanos::from_nanos(us.saturating_mul(1_000))
}

fn altitude(sequence: u32, at: MonotonicNanos, altitude_m: f64) -> Observation {
    Observation::new(
        ObservationStamp::new(
            SourceId::new(3),
            SourceEpoch::new(1),
            WrappingSequence::new(sequence),
            at,
            CLOCK,
        ),
        ObservationValue::Altitude {
            altitude_m,
            variance_m2: 4.0,
        },
        SourceComposition::of(SensorClass::Barometric),
    )
}

fn imu_at(sequence: u32, at: MonotonicNanos, rates: ([f64; 3], [f64; 3])) -> ImuSample {
    ImuSample::new(
        ObservationStamp::new(
            SourceId::new(IMU_SOURCE),
            SourceEpoch::new(1),
            WrappingSequence::new(sequence),
            at,
            CLOCK,
        ),
        rates.0,
        rates.1,
        2e-3,
        inertial(),
    )
}

/// Events that one second of 500 Hz IMU data and 100 Hz aiding gives.
const EVENTS_PER_SECOND: usize = 600;

/// A history capacity of three checkpoint strides: small, so a short
/// debug run fills it, and with the same eviction rule as the full one.
const SMALL_CAPACITY: usize = 3 * CHECKPOINT_STRIDE;

/// Runs `millis` milliseconds of 500 Hz IMU data with a 100 Hz altimeter
/// 1 ms after every fifth sample, with a history of `capacity` events.
/// Returns the filter and the solution published after each event, with
/// its time.
fn run_500_hz(
    millis: u64,
    capacity: usize,
) -> (
    InertialNavigationFilter,
    Vec<(MonotonicNanos, NavigationSolution)>,
) {
    let mut config = InertialConfig::default();
    config.fusion.history_capacity = capacity;
    let mut filter = surface_filter(config, PlanetaryBody::EARTH_WGS84, earth_origin());
    let (force, rate) = rest_rates(&filter);
    let rates = (force, [rate[0], rate[1], rate[2] + 0.2]);
    let mut published = Vec::new();
    let mut altimeter = 0_u32;
    let samples = u32::try_from(millis * 1_000 / IMU_STEP_US).expect("short run");
    for k in 1..=samples {
        let at = t_us(u64::from(k) * IMU_STEP_US);
        assert_accepted(filter.ingest_imu(&imu_at(k, at, rates), at));
        published.push((at, filter.tick(at).expect("published")));
        if k % 5 == 0 {
            altimeter = altimeter.wrapping_add(1);
            let off_grid = t_us(u64::from(k) * IMU_STEP_US + 1_000);
            let offset = f64::from(altimeter % 7) - 3.0;
            let obs = altitude(altimeter, off_grid, earth_origin().altitude_m + offset);
            assert_accepted(filter.ingest(&obs, off_grid));
            published.push((off_grid, filter.tick(off_grid).expect("published")));
        }
    }
    (filter, published)
}

#[test]
fn the_full_history_spans_one_second_at_500_hz_with_100_hz_aiding() {
    // The history drops a whole stride at a time, so it keeps at least
    // its capacity less one stride. A run of the full capacity takes
    // minutes in the debug build; the small run below shows the rule.
    let kept = MAX_INERTIAL_HISTORY - CHECKPOINT_STRIDE;
    assert!(kept >= EVENTS_PER_SECOND, "{kept} events");
}

#[test]
fn a_full_history_keeps_between_one_stride_less_than_its_capacity_and_its_capacity() {
    // 0.5 s is 300 events, more than the small history keeps.
    let (filter, published) = run_500_hz(500, SMALL_CAPACITY);
    let depth = filter.history_depth().expect("initialized");
    let event_ns = 1e9 / EVENTS_PER_SECOND as f64;
    let events = depth.as_nanos() as f64 / event_ns;
    let low = (SMALL_CAPACITY - CHECKPOINT_STRIDE) as f64;
    assert!(
        (low - 1.0..=SMALL_CAPACITY as f64).contains(&events),
        "{events} events"
    );
    // Every kept time gives the state that the filter had then. The
    // checkpoints are 64 events apart, so most of these states come from
    // a replay of up to 63 events, which include off-grid observations
    // that the in-order path propagated with the last rates.
    let state_at = published.last().expect("run").0;
    let oldest = state_at.as_nanos() - depth.as_nanos();
    let mut checked = 0_u32;
    for (at, solution) in published
        .iter()
        .filter(|(at, _)| at.as_nanos() >= oldest)
        .step_by(7)
    {
        let past = filter.state_at(*at).expect("kept");
        let mut expected = *solution;
        expected.stamp.sequence = past.solution.stamp.sequence;
        assert_eq!(past.solution, expected, "at {at:?}");
        checked = checked.wrapping_add(1);
    }
    assert!(checked >= 18, "{checked}");
}

#[test]
fn many_regime_updates_shorten_the_history_and_keep_it_valid() {
    let mut filter = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    );
    let rates = rest_rates(&filter);
    for k in 1..=400_u32 {
        let at_ms = u64::from(k) * 10;
        assert_accepted(filter.ingest_imu(&imu(k, at_ms, rates, 0.01), t(at_ms)));
        if k % 20 == 0 {
            // A regime update is a history event in the pool of large
            // events, which holds four.
            filter
                .attach_surface_ellipsoid(Some(BodyEllipsoid::WGS84))
                .expect("same ellipsoid");
        }
    }
    let depth = filter.history_depth().expect("initialized");
    // Twenty updates, one each 200 ms: four fit in 0.8 s at most.
    assert!(depth <= DurationNanos::from_millis(800), "{depth:?}");
    assert!(depth >= DurationNanos::from_millis(100), "{depth:?}");
    let oldest = t(4_000 - depth.as_nanos() / 1_000_000);
    assert!(filter.state_at(oldest).is_ok());
    let fix = Observation::new(
        stamp(1, 1, 3_950),
        ObservationValue::PositionFix {
            position: earth_origin(),
            covariance: SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
        },
        gnss(),
    );
    assert_accepted(filter.ingest(&fix, t(4_000)));
}

#[test]
fn the_history_is_smaller_than_one_core_per_event() {
    // The size of the filter is reported, not optimized for: the old
    // history of 32 full cores was 0.18 MB.
    let size = core::mem::size_of::<InertialNavigationFilter>();
    assert!(size < 260 * 1024, "{size} bytes");
}

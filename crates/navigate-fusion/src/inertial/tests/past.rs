//! The state at a past time from the history.

use navigate_contract::SymmetricCov3;

use super::*;
use crate::inertial::{PastState, PastStateRefusal};
use crate::observation::{Observation, ObservationValue};

/// IMU interval of these tests: 100 Hz.
const STEP_MS: u64 = 10;

/// A filter at rest at [`earth_origin`] that turns about the body z
/// axis at 0.5 rad/s, with IMU samples up to `last_ms`.
fn turning_filter(last_ms: u64) -> (InertialNavigationFilter, ([f64; 3], [f64; 3])) {
    let mut filter = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    );
    let (force, rate) = rest_rates(&filter);
    let rates = (force, [rate[0], rate[1], rate[2] + 0.5]);
    for k in 1..=(last_ms / STEP_MS) {
        let at_ms = k * STEP_MS;
        let sequence = u32::try_from(k).expect("small");
        assert_accepted(filter.ingest_imu(&imu(sequence, at_ms, rates, 0.01), t(at_ms)));
    }
    (filter, rates)
}

fn past(filter: &InertialNavigationFilter, at_ms: u64) -> PastState {
    filter.state_at(t(at_ms)).expect("inside the history")
}

#[test]
fn the_past_state_at_an_imu_time_is_the_solution_published_then() {
    let (mut filter, rates) = turning_filter(200);
    let published = filter.tick(t(200)).expect("published");
    // Five more samples: 200 ms is now in the past.
    for k in 21..=25_u32 {
        let at_ms = u64::from(k) * STEP_MS;
        assert_accepted(filter.ingest_imu(&imu(k, at_ms, rates, 0.01), t(at_ms)));
    }
    let state = past(&filter, 200);
    let mut expected = published;
    expected.stamp.sequence = state.solution.stamp.sequence;
    assert_eq!(state.solution, expected);
    assert_eq!(state.solution.stamp.solved_at, t(200));
    // The rate of the interval, without the (zero) estimated bias.
    assert_eq!(state.angular_rate_body_rad_s, rates.1);
}

#[test]
fn a_time_between_samples_propagates_the_earlier_state() {
    let (filter, _) = turning_filter(200);
    let before = past(&filter, 150);
    let between = past(&filter, 155);
    let after = past(&filter, 160);
    let yaw = |s: &PastState| {
        let q = s.solution.attitude.expect("attitude");
        2.0 * q.z.atan2(q.w)
    };
    // Half of one 10 ms step at 0.5 rad/s: 2.5 mrad either side.
    assert!((yaw(&between) - yaw(&before) - 0.0025).abs() < 1e-6);
    assert!((yaw(&after) - yaw(&between) - 0.0025).abs() < 1e-6);
    let (_, _, before_var) = before.solution.position_cov.diagonal();
    let (_, _, between_var) = between.solution.position_cov.diagonal();
    assert!(between_var >= before_var, "{between_var} < {before_var}");
}

#[test]
fn the_query_does_not_change_the_filter() {
    let (mut filter, _) = turning_filter(200);
    let first = filter.tick(t(200)).expect("published");
    let copy = filter.clone();
    let state = past(&filter, 120);
    assert_eq!(state.solution.stamp.sequence, first.stamp.sequence);
    assert_eq!(filter.tick(t(200)), Some(first));
    assert_eq!(copy.state_at(t(120)), Ok(state));
}

#[test]
fn outside_the_history_is_refused_by_name() {
    let unready = InertialNavigationFilter::new_surface(
        InertialConfig::default(),
        CLOCK,
        PlanetaryBody::EARTH_WGS84,
    );
    assert_eq!(
        unready.state_at(t(0)),
        Err(PastStateRefusal::NotInitialized)
    );
    let (filter, _) = turning_filter(1_000);
    assert_eq!(
        filter.state_at(t(1_010)),
        Err(PastStateRefusal::AfterStateTime {
            at: t(1_010),
            state_at: t(1_000),
        })
    );
    // 32 entries of 10 ms: the oldest kept state is 320 ms back.
    let oldest = t(1_000 - 32 * STEP_MS);
    assert!(filter.state_at(oldest).is_ok());
    assert_eq!(
        filter.state_at(t(600)),
        Err(PastStateRefusal::BeforeHistory { at: t(600), oldest })
    );
}

#[test]
fn a_delayed_fix_conditions_a_later_past_state() {
    let (mut filter, _) = turning_filter(200);
    let before = past(&filter, 150);
    let mut moved = earth_origin();
    moved.latitude_rad += 3.0 / 6_371_000.0;
    let fix = Observation::new(
        stamp(1, 1, 120),
        ObservationValue::PositionFix {
            position: moved,
            covariance: SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
        },
        gnss(),
    );
    assert_accepted(filter.ingest(&fix, t(200)));
    let after = past(&filter, 150);
    assert!(after.solution.position.latitude_rad > before.solution.position.latitude_rad);
    let (n_before, _, _) = before.solution.position_cov.diagonal();
    let (n_after, _, _) = after.solution.position_cov.diagonal();
    assert!(n_after < n_before, "{n_after} vs {n_before}");
}

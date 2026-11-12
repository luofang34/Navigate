//! The GNSS check of the reference navigator on a suite without another
//! position sensor: a tactical inertial unit with its noise and biases on
//! a turning route, and a 10 Hz receiver with a time-correlated error
//! that an emitter moves off the truth at a constant rate.

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{AttitudeQuaternion, NavigationSolution, NedVelocity, SymmetricCov3};

use super::correlated_gnss::{
    CLOCK, CORRELATED_M, Draws, GNSS, IMU, STEP_S, TAU_S, Truth, VELOCITY_SIGMA, WHITE_M, at,
    imu_budget, moved, ned_offset, start, suite,
};
use crate::{
    AcceptanceRecord, Alignment, GnssIntegrityConfig, GnssIntegrityState, GnssIntegrityStatus,
    Navigator, ReferenceNavigator, SensorData, SensorInput,
};

/// The receiver error that an emitter adds: from `from_s` the position
/// moves north at `rate_m_s`, and the Doppler velocity with it. At `to_s`
/// the offset stays, or goes back to zero.
#[derive(Debug, Clone, Copy)]
struct Ramp {
    from_s: f64,
    to_s: f64,
    rate_m_s: f64,
    drops: bool,
}

impl Ramp {
    /// The north offset of the position and of the velocity at `t_s`.
    fn at(&self, t_s: f64) -> (f64, f64) {
        if t_s < self.from_s {
            (0.0, 0.0)
        } else if t_s < self.to_s {
            (self.rate_m_s * (t_s - self.from_s), self.rate_m_s)
        } else if self.drops {
            (0.0, 0.0)
        } else {
            (self.rate_m_s * (self.to_s - self.from_s), 0.0)
        }
    }
}

/// One step of a run, after its GNSS fix.
struct Step {
    t_s: f64,
    status: Option<GnssIntegrityStatus>,
    solution: NavigationSolution,
    /// Horizontal error of the solution, metres.
    error_m: f64,
}

/// A navigator of the suite with a correlated receiver error, aligned
/// near the start with the errors of the seed.
fn started(draws: &mut Draws, check: Option<GnssIntegrityConfig>) -> ReferenceNavigator {
    let mut navigator = ReferenceNavigator::new(suite(true), CLOCK, None).unwrap();
    navigator.set_gnss_integrity(check).unwrap();
    let start_error = draws.vec3([2.0; 3]);
    let dv = draws.vec3([0.1; 3]);
    let tilt = UnitQuaternion::from_scaled_axis(draws.vec3([1e-3; 3]));
    navigator
        .align(&Alignment {
            position: moved(&start(), &start_error),
            velocity: NedVelocity::new(40.0 + dv.x, dv.y, dv.z),
            attitude: AttitudeQuaternion::new(tilt.w, tilt.i, tilt.j, tilt.k),
            position_cov: SymmetricCov3::from_diagonal(4.0, 4.0, 4.0),
            velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            at: at(0),
            imu_bias_prior: None,
        })
        .unwrap();
    navigator
}

/// Flies `duration_s` with the seed `seed`, with or without the check,
/// and gives each step to `see`.
fn fly(
    seed: u32,
    duration_s: f64,
    ramp: Option<Ramp>,
    check: Option<GnssIntegrityConfig>,
    sink: &mut Vec<AcceptanceRecord>,
    see: impl FnMut(&Step),
) {
    fly_with_lag(seed, duration_s, ramp, check, 0, sink, see);
}

/// [`fly`] with each GNSS fix delivered `lag_steps` inertial steps after
/// its acquisition.
fn fly_with_lag(
    seed: u32,
    duration_s: f64,
    ramp: Option<Ramp>,
    check: Option<GnssIntegrityConfig>,
    lag_steps: u32,
    sink: &mut Vec<AcceptanceRecord>,
    mut see: impl FnMut(&Step),
) {
    let mut late = std::collections::VecDeque::new();
    let mut draws = Draws {
        seed: u64::from(seed),
        index: 0,
    };
    let mut truth = Truth::new();
    let mut navigator = started(&mut draws, check);
    let (accel, gyro, accel_bias, gyro_bias) = imu_budget();
    let b_a = draws.vec3([accel_bias.sigma(); 3]);
    let b_g = draws.vec3([gyro_bias.sigma(); 3]);
    let mut correlated_error = draws.vec3(CORRELATED_M);
    let rho = (-STEP_S / TAU_S).exp();
    let innovation = CORRELATED_M.map(|s| s * (1.0 - rho * rho).sqrt());
    let steps = (duration_s / STEP_S).round() as u32;
    for step in 1..=steps {
        let t_s = f64::from(step) * STEP_S;
        let (theta, dv) = truth.step(step, t_s - STEP_S);
        let theta = Vector3::from(theta) + b_g * STEP_S + draws.vec3([gyro * STEP_S.sqrt(); 3]);
        let dv = Vector3::from(dv) + b_a * STEP_S + draws.vec3([accel * STEP_S.sqrt(); 3]);
        let input = |sensor, data| SensorInput {
            sensor,
            acquired_at: at(step),
            delivered_at: at(step),
            data,
        };
        let imu = SensorData::ImuIncrement {
            dt_s: STEP_S,
            delta_theta_rad: theta.into(),
            delta_velocity_m_s: dv.into(),
        };
        navigator.ingest(&input(IMU, imu), sink).unwrap();
        correlated_error = correlated_error * rho + draws.vec3(innovation);
        let (north_m, north_m_s) = ramp.map_or((0.0, 0.0), |r| r.at(t_s));
        let error = correlated_error + draws.vec3(WHITE_M) + Vector3::new(north_m, 0.0, 0.0);
        let v = truth.state.velocity;
        let dv = draws.vec3([VELOCITY_SIGMA; 3]);
        let fix = SensorData::GnssFix {
            position: moved(&truth.state.position, &error),
            velocity: Some(NedVelocity::new(
                v.north_mps + north_m_s + dv.x,
                v.east_mps + dv.y,
                v.down_mps + dv.z,
            )),
            position_cov: None,
            velocity_cov: None,
        };
        late.push_back(input(GNSS, fix));
        if late.len() > lag_steps as usize {
            let mut fix = late.pop_front().unwrap();
            fix.delivered_at = at(step);
            navigator.ingest(&fix, sink).unwrap();
        }
        let solution = navigator.solution(at(step)).unwrap();
        let error = ned_offset(&truth.state.position, &solution.position);
        see(&Step {
            t_s,
            status: navigator.gnss_integrity(),
            solution,
            error_m: error.xy().norm(),
        });
    }
}

/// A time of seed 0 a few seconds after the reference started again:
/// its bound is then near its lowest value, 8.4 m.
const FRESH_S: f64 = 440.0;

#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn monte_carlo_an_honest_receiver_is_never_excluded_and_the_check_changes_no_solution() {
    let (mut fixes, mut doubted, mut worst_chi2) = (0_u32, 0_u32, 0.0_f64);
    for seed in 0..4 {
        let mut plain = Vec::new();
        fly(seed, 1_200.0, None, None, &mut Vec::new(), |step| {
            assert!(step.status.is_none());
            plain.push(step.solution);
        });
        let mut plain = plain.into_iter();
        let check = Some(GnssIntegrityConfig::default());
        fly(seed, 1_200.0, None, check, &mut Vec::new(), |step| {
            let status = step.status.unwrap();
            assert_ne!(status.state, GnssIntegrityState::Excluded, "{}", step.t_s);
            fixes += 1;
            worst_chi2 = worst_chi2.max(status.last.map_or(0.0, |last| last.chi2));
            let unchanged = plain.next();
            if status.state == GnssIntegrityState::Consistent {
                assert_eq!(Some(step.solution), unchanged, "{}", step.t_s);
            } else {
                doubted += 1;
            }
        });
    }
    // 48 000 fixes over 80 minutes. The caution probability of one fix is
    // 1e-3; the fixes of one reference are not independent, and the
    // largest statistic was 11.6 against the caution threshold 13.8.
    assert_eq!(fixes, 48_000);
    assert!(doubted * 200 <= fixes, "{doubted} of {fixes}, {worst_chi2}");
}

/// What one drift did: the times from its start to the first doubt and to
/// the exclusion, the largest error of the solution up to the exclusion,
/// and the one-sigma at the exclusion and at the end.
#[derive(Debug, Default)]
struct Found {
    doubted_s: Option<f64>,
    excluded_s: Option<f64>,
    worst_m: f64,
    sigma_then_m: f64,
    sigma_end_m: f64,
    doubted_quality_good: bool,
    lowest_rate_m_s: f64,
    witnessed: bool,
    end_error_m: f64,
}

fn drift(rate_m_s: f64, duration_s: f64) -> Found {
    drift_from(FRESH_S, rate_m_s, duration_s)
}

/// A drift of `rate_m_s` that starts at `from_s` and goes on.
fn drift_from(from_s: f64, rate_m_s: f64, duration_s: f64) -> Found {
    let ramp = Ramp {
        from_s,
        to_s: f64::INFINITY,
        rate_m_s,
        drops: false,
    };
    let mut found = Found {
        lowest_rate_m_s: f64::INFINITY,
        ..Found::default()
    };
    let check = Some(GnssIntegrityConfig::default());
    let end_s = from_s + duration_s;
    fly(0, end_s, Some(ramp), check, &mut Vec::new(), |step| {
        let status = step.status.unwrap();
        let since_s = step.t_s - from_s;
        found.witnessed |= status.reference_position_fix_at.is_some();
        if since_s < 0.0 {
            assert_eq!(status.state, GnssIntegrityState::Consistent);
            return;
        }
        if let Some(rate) = status.inertial_detectable_rate_m_s {
            found.lowest_rate_m_s = found.lowest_rate_m_s.min(rate);
        }
        let sigma_m = step.solution.integrity.horizontal_1sigma_m;
        match status.state {
            GnssIntegrityState::Disagreeing if found.doubted_s.is_none() => {
                found.doubted_s = Some(since_s);
                found.doubted_quality_good =
                    step.solution.integrity.quality == navigate_contract::SolutionQuality::Good;
            }
            GnssIntegrityState::Excluded if found.excluded_s.is_none() => {
                found.excluded_s = Some(since_s);
                found.sigma_then_m = sigma_m;
            }
            _ => {}
        }
        if found.excluded_s.is_none() {
            found.worst_m = found.worst_m.max(step.error_m);
        }
        found.sigma_end_m = sigma_m;
        found.end_error_m = step.error_m;
    });
    found
}

#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn a_drift_that_begins_on_a_fresh_reference_is_excluded_before_it_moves_the_solution() {
    // Measured: 0.5 m/s is excluded after 17.0 s with at most 5.4 m of
    // error, 1 m/s after 8.4 s with 2.2 m, and 2 m/s after 4.4 s with
    // 2.0 m.
    for (rate_m_s, within_s, error_m) in [(0.5, 25.0, 8.0), (1.0, 14.0, 5.0), (2.0, 8.0, 5.0)] {
        let found = drift(rate_m_s, 90.0);
        let (doubted_s, excluded_s) = (found.doubted_s.unwrap(), found.excluded_s.unwrap());
        assert!(doubted_s < excluded_s && excluded_s < within_s, "{found:?}");
        assert!(!found.doubted_quality_good, "{found:?}");
        assert!(found.worst_m < error_m, "{rate_m_s}: {found:?}");
        // Nothing but the inertial data holds the position after the
        // exclusion, and the solution says so.
        assert!(!found.witnessed);
        assert!(found.sigma_end_m > 5.0 * found.sigma_then_m, "{found:?}");
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn a_drift_slower_than_the_stated_rate_is_not_found_and_the_status_states_that_rate() {
    let found = drift(0.1, 120.0);
    assert!(found.excluded_s.is_none(), "{found:?}");
    // The receiver took the solution 12 m off.
    assert!(found.end_error_m > 8.0, "{found:?}");
    // The status says that the reference has only inertial data, and
    // gives the slowest drift that it finds: 0.28 m/s with this inertial
    // unit, above the 0.1 m/s of the receiver.
    assert!(!found.witnessed);
    assert!((0.15..0.6).contains(&found.lowest_rate_m_s), "{found:?}");
}

/// The largest separation of a fix from a reference in its first twenty
/// seconds, when its drift is small, over 120 s with each fix delivered
/// `lag_steps` steps late. A fix that is older than the start of the
/// reference was compared with the reference before it.
fn worst_fresh_separation_m(lag_steps: u32) -> f64 {
    let check = Some(GnssIntegrityConfig::default());
    let (mut worst_m, mut fixes) = (0.0_f64, 0_u32);
    fly_with_lag(0, 120.0, None, check, lag_steps, &mut Vec::new(), |step| {
        let status = step.status.unwrap();
        assert_eq!(status.state, GnssIntegrityState::Consistent, "{}", step.t_s);
        let Some((from, last)) = status.reference_anchored_at.zip(status.last) else {
            return;
        };
        assert_eq!(last.at, at((step.t_s / STEP_S).round() as u32 - lag_steps));
        let fresh = last
            .at
            .elapsed_since(from)
            .is_some_and(|d| (1..20_000_000_000).contains(&d.as_nanos()));
        if fresh {
            worst_m = worst_m.max(last.separation_m);
            fixes += 1;
        }
    });
    assert!(fixes > 200, "{fixes}");
    worst_m
}

#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn a_late_fix_meets_the_reference_at_its_acquisition_time() {
    // The vehicle moves 8 m in the 0.2 s from the acquisition to the
    // delivery. A reference of the delivery time would be that much
    // farther from each fix.
    let (on_time_m, late_m) = (worst_fresh_separation_m(0), worst_fresh_separation_m(2));
    assert!((late_m - on_time_m).abs() < 1.0, "{on_time_m} {late_m}");
}

/// How the start time of a drift in the life of the reference decides
/// whether it is found: ten start times over one life of about 96 s, for
/// each rate, with the count of drifts excluded in 120 s and the largest
/// error of the solution at the end. Printed for a report, not asserted.
#[test]
#[ignore = "a sweep for a report: about a minute in an optimized build"]
fn sweep_the_start_time_of_a_drift_on_inertial_data_alone() {
    tracing_subscriber::fmt().with_test_writer().try_init().ok();
    for rate_m_s in [0.5, 1.0, 2.0] {
        let (mut excluded, mut worst_m, mut latest_s) = (0, 0.0_f64, 0.0_f64);
        for start in 0..10 {
            let from_s = 400.0 + 10.0 * f64::from(start);
            let found = drift_from(from_s, rate_m_s, 120.0);
            if let Some(excluded_s) = found.excluded_s {
                excluded += 1;
                latest_s = latest_s.max(excluded_s);
                worst_m = worst_m.max(found.worst_m);
            } else {
                let end_error_m = found.end_error_m;
                tracing::info!(rate_m_s, from_s, end_error_m, "a drift was not found");
            }
        }
        tracing::info!(
            rate_m_s,
            excluded,
            latest_s,
            worst_m,
            "drifts of ten start times"
        );
    }
}

//! The GNSS check of the reference navigator on a vehicle at rest with
//! exact inertial increments: the states that it goes through, what it
//! records, and when it takes the receiver again.

use super::*;
use crate::reference::monitor::GNSS_EXCLUDED;
use crate::{GnssFault, GnssIntegrityConfig, GnssIntegrityState, GnssIntegrityStatus};
use navigate_contract::SolutionQuality;

/// The receiver error of a test: the position moves north at `rate_m_s`
/// from `from_s` to `to_s`, then keeps its offset or drops it.
#[derive(Debug, Clone, Copy)]
struct Drift {
    from_s: u64,
    to_s: u64,
    rate_m_s: f64,
    drops: bool,
}

impl Drift {
    fn north_m(&self, second: u64) -> f64 {
        if second < self.from_s || (second >= self.to_s && self.drops) {
            0.0
        } else {
            self.rate_m_s * (second.min(self.to_s) - self.from_s) as f64
        }
    }
}

/// A position fix of another sensor at the truth, with a one-sigma of
/// 3 m: what a terrain camera gives.
fn witness(second: u64) -> SensorInput {
    let stamp = ObservationStamp::new(
        PASS,
        SourceEpoch::new(0),
        WrappingSequence::new(second as u32),
        at(second * RATE),
        CLOCK,
    );
    let value = ObservationValue::PositionFix {
        position: GeodeticPosition::new(LAT.to_radians(), LON.to_radians(), HEIGHT),
        covariance: SymmetricCov3::from_diagonal(9.0, 9.0, 9.0),
    };
    let composition = SourceComposition::of(SensorClass::VisualLandmark);
    SensorInput {
        sensor: PASS,
        acquired_at: at(second * RATE),
        delivered_at: at(second * RATE),
        data: SensorData::Observation(Observation::new(stamp, value, composition)),
    }
}

/// What a run saw at the end of each second.
struct Second {
    second: u64,
    status: GnssIntegrityStatus,
    solution: NavigationSolution,
}

/// A navigator with a receiver and the check, and with a second position
/// sensor when `witnessed`.
fn checked(witnessed: bool) -> ReferenceNavigator {
    let extra: &[SensorKind] = if witnessed {
        &[SensorKind::Gnss, SensorKind::ObservationSource]
    } else {
        &[SensorKind::Gnss]
    };
    let mut navigator = aligned(extra);
    navigator.set_gnss_integrity(Some(config())).unwrap();
    navigator
}

/// The default check with a shorter time of agreement, so that a test
/// of some tens of seconds sees a receiver taken again.
fn config() -> GnssIntegrityConfig {
    GnssIntegrityConfig {
        readmission_s: 10.0,
        ..GnssIntegrityConfig::default()
    }
}

/// Runs `seconds` at rest: a GNSS fix each second with the error of
/// `drift`, and a fix of the second position sensor each fifth second
/// when `witnessed`.
fn watch(
    navigator: &mut ReferenceNavigator,
    seconds: u64,
    drift: &Drift,
    witnessed: bool,
    sink: &mut Vec<AcceptanceRecord>,
) -> Vec<Second> {
    let mut seen = Vec::new();
    for step in 1..=seconds * RATE {
        navigator.ingest(&rest_increment(step), sink).unwrap();
        if !step.is_multiple_of(RATE) {
            continue;
        }
        let second = step / RATE;
        if witnessed && second.is_multiple_of(5) {
            navigator.ingest(&witness(second), sink).unwrap();
        }
        let fix = gnss_fix(step, drift.north_m(second));
        navigator.ingest(&fix, sink).unwrap();
        seen.push(Second {
            second,
            status: navigator.gnss_integrity().unwrap(),
            solution: navigator.solution(at(step)).unwrap(),
        });
    }
    seen
}

fn first(seen: &[Second], state: GnssIntegrityState) -> Option<&Second> {
    seen.iter().find(|s| s.status.state == state)
}

fn north_m(solution: &NavigationSolution) -> f64 {
    (solution.position.latitude_rad - LAT.to_radians()) * 6_364_000.0
}

const RAMP: Drift = Drift {
    from_s: 5,
    to_s: 20,
    rate_m_s: 2.0,
    drops: true,
};

#[test]
fn a_receiver_that_leaves_the_reference_is_doubted_then_excluded_and_each_value_is_recorded() {
    let mut navigator = checked(true);
    let mut sink = Vec::new();
    let seen = watch(&mut navigator, 20, &RAMP, true, &mut sink);
    let doubted = first(&seen, GnssIntegrityState::Disagreeing).unwrap();
    let excluded = first(&seen, GnssIntegrityState::Excluded).unwrap();
    assert!(
        seen.iter().take(5).all(|s| {
            s.status.state == GnssIntegrityState::Consistent && s.status.fault.is_none()
        })
    );
    assert!(doubted.second < excluded.second, "{}", doubted.second);
    assert_ne!(doubted.solution.integrity.quality, SolutionQuality::Good);
    // The status names the receiver, the fault, the time, and a separation
    // above its bound.
    let status = excluded.status;
    let last = status.last.unwrap();
    assert_eq!(last.source, GNSS);
    assert_eq!(status.fault, Some(GnssFault::HorizontalSeparation));
    assert_eq!(status.since, Some(at(excluded.second * RATE)));
    assert!(last.separation_m > last.exclusion_bound_m, "{last:?}");
    assert!(status.within_alert_limit && status.reference_position_fix_at.is_some());
    // The receiver went 2 m/s for 15 s; the solution stayed at the other
    // position sensor.
    let worst = seen
        .iter()
        .map(|s| north_m(&s.solution).abs())
        .fold(0.0, f64::max);
    assert!(worst < 6.0, "{worst}");
    // The solution says which sensors it holds: the receiver until the
    // exclusion, and not from it.
    let holds_gnss = |s: &Second| {
        s.solution
            .integrity
            .contributing
            .contains(SensorClass::Gnss)
    };
    assert!(holds_gnss(&seen[3]) && holds_gnss(doubted));
    assert!(
        seen.iter()
            .filter(|s| s.second >= excluded.second)
            .all(|s| !holds_gnss(s))
    );
    // Each value of the excluded receiver has its record, the Doppler
    // velocity with the position.
    for measurement in ["gnss_position", "gnss_velocity"] {
        let unused = sink
            .iter()
            .filter(|r| r.measurement == measurement)
            .filter(|r| r.outcome == Outcome::Unused(GNSS_EXCLUDED))
            .count() as u64;
        assert_eq!(unused, 20 - excluded.second + 1, "{measurement}");
    }
}

#[test]
fn the_solutions_stay_in_one_epoch_and_in_order_over_an_exclusion() {
    let mut navigator = checked(true);
    let seen = watch(&mut navigator, 20, &RAMP, true, &mut Vec::new());
    assert!(first(&seen, GnssIntegrityState::Excluded).is_some());
    for pair in seen.windows(2) {
        let (a, b) = (pair[0].solution.stamp, pair[1].solution.stamp);
        assert_eq!(a.epoch, b.epoch);
        assert!(b.sequence.get() > a.sequence.get(), "{a:?} {b:?}");
    }
}

#[test]
fn a_receiver_that_agrees_again_for_the_configured_time_is_taken_again() {
    let mut navigator = checked(true);
    let mut sink = Vec::new();
    let seen = watch(&mut navigator, 45, &RAMP, true, &mut sink);
    let back = seen
        .iter()
        .skip(20)
        .find(|s| s.status.state == GnssIntegrityState::Consistent)
        .unwrap();
    // The receiver is at the truth again from second 20; the first fix of
    // the agreeing run is that one.
    assert_eq!(back.second, 20 + config().readmission_s as u64);
    let before = &seen[back.second as usize - 2];
    assert_eq!(before.status.agreeing_since, Some(at(20 * RATE)));
    let last = sink.iter().rev().find(|r| r.measurement == "gnss_position");
    assert_eq!(last.map(|r| r.outcome.clone()), Some(Outcome::Accepted));
    assert!(north_m(&seen[44].solution).abs() < 1.0);
}

#[test]
fn a_receiver_that_keeps_its_offset_is_not_taken_again() {
    let mut navigator = checked(true);
    let frozen = Drift {
        drops: false,
        ..RAMP
    };
    let seen = watch(&mut navigator, 60, &frozen, true, &mut Vec::new());
    let excluded = first(&seen, GnssIntegrityState::Excluded).unwrap().second;
    assert!(seen.iter().skip(excluded as usize).all(|s| s.status.state
        == GnssIntegrityState::Excluded
        && s.status.agreeing_since.is_none()));
    assert!(north_m(&seen[59].solution).abs() < 6.0);
}

#[test]
fn without_another_position_sensor_the_receiver_is_not_taken_again_and_the_uncertainty_grows() {
    let mut navigator = checked(false);
    // The reference has only inertial data, so the receiver must leave
    // faster to be found: 3 m/s.
    let fast = Drift {
        rate_m_s: 3.0,
        ..RAMP
    };
    let seen = watch(&mut navigator, 60, &fast, false, &mut Vec::new());
    let excluded = first(&seen, GnssIntegrityState::Excluded).unwrap();
    assert!(excluded.status.reference_position_fix_at.is_none());
    let end = &seen[59];
    // The receiver agrees again from second 20, and the status says so,
    // but nothing but the receiver says where the vehicle is.
    assert_eq!(end.status.state, GnssIntegrityState::Excluded);
    assert!(end.status.agreeing_since.is_some());
    let (then, now) = (
        excluded.solution.integrity.horizontal_1sigma_m,
        end.solution.integrity.horizontal_1sigma_m,
    );
    assert!(now > 3.0 * then, "{then} {now}");
    assert_ne!(end.solution.integrity.quality, SolutionQuality::Good);
    // A new alignment forgets the exclusion.
    navigator.realign().unwrap();
    assert_eq!(
        navigator
            .gnss_integrity()
            .map(|s| (s.state, s.last, s.reference_anchored_at)),
        Some((GnssIntegrityState::Consistent, None, None))
    );
}

#[test]
fn the_check_is_absent_until_the_host_starts_it_and_refuses_an_unusable_configuration() {
    let mut navigator = aligned(&[SensorKind::Gnss]);
    assert_eq!(navigator.gnss_integrity(), None);
    let unusable = [
        GnssIntegrityConfig {
            caution_probability: 0.0,
            ..GnssIntegrityConfig::default()
        },
        GnssIntegrityConfig {
            exclusion_probability: 0.5,
            ..GnssIntegrityConfig::default()
        },
        GnssIntegrityConfig {
            confirming_fixes: 0,
            ..GnssIntegrityConfig::default()
        },
        GnssIntegrityConfig {
            readmission_s: f64::NAN,
            ..GnssIntegrityConfig::default()
        },
        GnssIntegrityConfig {
            alert_limit_m: -1.0,
            ..GnssIntegrityConfig::default()
        },
        GnssIntegrityConfig {
            restart_rate_ratio: 0.5,
            ..GnssIntegrityConfig::default()
        },
    ];
    for config in unusable {
        assert!(
            navigator.set_gnss_integrity(Some(config)).is_err(),
            "{config:?}"
        );
        assert_eq!(navigator.gnss_integrity(), None);
    }
    let config = GnssIntegrityConfig::default();
    // −2·ln(p) for 2 degrees of freedom.
    assert!((config.caution_chi2() - 13.816).abs() < 1e-3);
    assert!((config.exclusion_chi2() - 27.631).abs() < 1e-3);
    navigator.set_gnss_integrity(Some(config)).unwrap();
    assert!(navigator.gnss_integrity().is_some());
    navigator.set_gnss_integrity(None).unwrap();
    assert_eq!(navigator.gnss_integrity(), None);
}

#[test]
fn the_check_changes_no_solution_while_the_receiver_agrees() {
    let honest = Drift {
        rate_m_s: 0.0,
        ..RAMP
    };
    let mut plain = aligned(&[SensorKind::Gnss, SensorKind::ObservationSource]);
    let mut navigator = checked(true);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let seen = watch(&mut navigator, 45, &honest, true, &mut a);
    for step in 1..=45 * RATE {
        plain.ingest(&rest_increment(step), &mut b).unwrap();
        if step.is_multiple_of(RATE) {
            if (step / RATE).is_multiple_of(5) {
                plain.ingest(&witness(step / RATE), &mut b).unwrap();
            }
            plain.ingest(&gnss_fix(step, 0.0), &mut b).unwrap();
            let second = &seen[(step / RATE) as usize - 1];
            assert_eq!(Some(second.solution), plain.solution(at(step)));
            assert_eq!(second.status.state, GnssIntegrityState::Consistent);
        }
    }
    assert_eq!(a, b);
}

#[test]
fn an_observation_of_the_gnss_class_is_not_offered_while_gnss_is_excluded() {
    let mut navigator = checked(true);
    let mut sink = Vec::new();
    watch(&mut navigator, 20, &RAMP, true, &mut sink);
    let status = navigator.gnss_integrity().unwrap();
    assert_eq!(status.state, GnssIntegrityState::Excluded);
    // The second sensor now gives a position of a receiver.
    let mut input = witness(21);
    if let SensorData::Observation(observation) = &mut input.data {
        observation.composition = SourceComposition::of(SensorClass::Gnss);
    }
    navigator.ingest(&input, &mut sink).unwrap();
    let last = sink.last().unwrap();
    assert_eq!(last.measurement, "observation");
    assert_eq!(last.outcome, Outcome::Unused(GNSS_EXCLUDED));
}

/// The time of one inertial step with and without the check, for a
/// report: a vehicle at rest, 50 Hz increments, one GNSS fix each second.
#[test]
#[ignore = "a timing for a report"]
fn time_one_inertial_step_with_and_without_the_check() {
    tracing_subscriber::fmt().with_test_writer().try_init().ok();
    let honest = Drift {
        rate_m_s: 0.0,
        ..RAMP
    };
    for with_check in [false, true] {
        let mut navigator = aligned(&[SensorKind::Gnss]);
        if with_check {
            navigator.set_gnss_integrity(Some(config())).unwrap();
        }
        let mut sink = Vec::new();
        let seconds = 60;
        let started = std::time::Instant::now();
        for step in 1..=seconds * RATE {
            navigator.ingest(&rest_increment(step), &mut sink).unwrap();
            if step.is_multiple_of(RATE) {
                let fix = gnss_fix(step, honest.north_m(step / RATE));
                navigator.ingest(&fix, &mut sink).unwrap();
            }
        }
        let step_us = started.elapsed().as_secs_f64() * 1e6 / (seconds * RATE) as f64;
        tracing::info!(with_check, step_us, "one inertial step");
    }
}

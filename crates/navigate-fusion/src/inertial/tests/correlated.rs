//! The bookkeeping of time-correlated source errors (ADR-0023): the
//! white path, the inflation factor, the refusals, a new source epoch,
//! a delayed fix, and a fork.

use nalgebra::{Const, DimMin, Vector3};

use super::correlated_gnss::{Declared, FixError, TAU_S, fix};
use super::flight_truth::{Alignment, Flight, Route, tactical};
use super::*;
use crate::inertial::StateLayout;
use crate::inertial::correlated::{MAX_UNBOUND_SOURCES, Plan};
use crate::inertial::nominal::GYRO_BIAS;
use crate::observation::{CorrelatedError, Observation, ObservationValue};
use crate::rejection::RejectionReason;

const ROUTE: Route = Route {
    speed_mps: 40.0,
    bank_deg: 25.0,
    duration_s: 30.0,
    step_s: 0.5,
    turns: &[(10.0, 1.0)],
};
const ALIGNMENT: Alignment = Alignment {
    position_m: 2.0,
    velocity_mps: 0.1,
    attitude_rad: 1e-3,
};
const BOUND: StateLayout = StateLayout::FULL.with_correlated_blocks(1);

/// A boxed flight: two filters with their histories do not fit on the
/// stack of a test thread.
fn flight<const N: usize>(layout: StateLayout) -> std::boxed::Box<Flight<N>>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let config = InertialConfig {
        layout,
        ..tactical()
    };
    std::boxed::Box::new(Flight::<N>::new(ROUTE, 5, config, ALIGNMENT))
}

/// Flies `steps` steps with a fix at each step.
fn fly<const N: usize>(flight: &mut Flight<N>, error: &mut FixError, declared: Declared, steps: u32)
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    for _ in 0..steps {
        let at_ms = flight.step();
        let sequence = (at_ms / 500) as u32;
        let observation = fix(flight, &error.next(), declared, sequence, at_ms);
        assert_accepted(flight.filter.ingest(&observation, t(at_ms)));
    }
}

#[test]
fn white_fixes_give_the_same_core_estimate_with_and_without_a_correlated_block() {
    let mut full = flight::<22>(StateLayout::FULL);
    let mut bound = flight::<25>(BOUND);
    let (mut a, mut b) = (FixError::new(5), FixError::new(5));
    fly(&mut full, &mut a, Declared::AllWhite, 40);
    fly(&mut bound, &mut b, Declared::AllWhite, 40);
    let (x, y) = (core(&full.filter), core(&bound.filter));
    assert!((x.nominal.p - y.nominal.p).norm() < 1e-9);
    assert!((x.nominal.v - y.nominal.v).norm() < 1e-9);
    assert!(x.nominal.q.angle_to(&y.nominal.q) < 1e-9);
    let n = GYRO_BIAS + 3;
    let difference = x.p.view((0, 0), (n, n)) - y.p.view((0, 0), (n, n));
    let largest = difference.amax();
    let scale = x.p.view((0, 0), (n, n)).amax();
    assert!(largest <= 1e-9 * scale, "{largest} of {scale}");
}

#[test]
fn a_source_without_a_block_inflates_by_the_correlation_of_its_last_fix() {
    let mut full = flight::<22>(StateLayout::FULL);
    let mut error = FixError::new(5);
    let declared = Declared::Correlated { tau_s: TAU_S };
    fly(&mut full, &mut error, declared, 1);
    let at_ms = full.step();
    let observation = fix(&full, &error.next(), declared, 2, at_ms);
    let mut candidate = *core(&full.filter);
    let correlated = observation.correlated.expect("declared");
    let pending = candidate
        .prepare_correlated(&observation.stamp, &observation.value, &correlated)
        .expect("planned");
    let rho = (-0.5 / TAU_S).exp();
    let Plan::Inflated { factor, .. } = pending.plan else {
        panic!("a layout without a block inflates: {pending:?}");
    };
    assert!((factor - (1.0 + rho) / (1.0 - rho)).abs() < 1e-9 * factor);
}

#[test]
fn a_constant_error_without_a_block_is_used_once() {
    let mut full = flight::<22>(StateLayout::FULL);
    let constant = CorrelatedError::new(f64::INFINITY, [1.0; 3]);
    let mut outcomes = std::vec::Vec::new();
    for sequence in 1..=2 {
        let at_ms = full.step();
        let observation = fix(
            &full,
            &Vector3::zeros(),
            Declared::AllWhite,
            sequence,
            at_ms,
        )
        .with_correlated_error(constant);
        outcomes.push(full.filter.ingest(&observation, t(at_ms)));
    }
    assert!(outcomes.first().is_some_and(IngestOutcome::is_accepted));
    assert!(matches!(
        outcomes.get(1),
        Some(IngestOutcome::Rejected(
            RejectionReason::CorrelatedErrorRepeated { .. }
        ))
    ));
}

#[test]
fn declarations_without_a_model_or_a_place_are_refused() {
    let mut full = flight::<22>(StateLayout::FULL);
    let at_ms = full.step();
    let declared = CorrelatedError::new(TAU_S, [1.0; 3]);
    let range = Observation::new(
        stamp(40, 1, at_ms),
        ObservationValue::Range {
            station: earth_origin(),
            range_m: 100.0,
            variance_m2: 1.0,
        },
        gnss(),
    )
    .with_correlated_error(declared);
    assert!(matches!(
        full.filter.ingest(&range, t(at_ms)),
        IngestOutcome::Rejected(RejectionReason::CorrelationNotModelled { .. })
    ));
    let implausible = fix(&full, &Vector3::zeros(), Declared::AllWhite, 1, at_ms)
        .with_correlated_error(CorrelatedError::new(-1.0, [1.0; 3]));
    assert!(matches!(
        full.filter.ingest(&implausible, t(at_ms)),
        IngestOutcome::Rejected(RejectionReason::NonFiniteValue)
    ));
    let sources = u32::try_from(MAX_UNBOUND_SOURCES).expect("small");
    for source in 0..=sources {
        let observation = fix(&full, &Vector3::zeros(), Declared::AllWhite, 1, at_ms);
        let observation = Observation {
            stamp: stamp(50 + source, 1, at_ms),
            ..observation
        }
        .with_correlated_error(declared);
        let outcome = full.filter.ingest(&observation, t(at_ms));
        if source < sources {
            assert_accepted(outcome);
        } else {
            assert!(matches!(
                outcome,
                IngestOutcome::Rejected(RejectionReason::CorrelatedSourceUntracked)
            ));
        }
    }
}

#[test]
fn a_new_source_epoch_starts_a_new_block() {
    let mut bound = flight::<25>(BOUND);
    let mut error = FixError::new(5);
    let declared = Declared::Correlated { tau_s: TAU_S };
    fly(&mut bound, &mut error, declared, 20);
    let at_ms = bound.step();
    let observation = fix(&bound, &error.next(), declared, 21, at_ms);
    let restarted = Observation {
        stamp: ObservationStamp::new(
            SourceId::new(3),
            SourceEpoch::new(2),
            WrappingSequence::new(1),
            t(at_ms),
            CLOCK,
        ),
        ..observation
    };
    let correlated = observation.correlated.expect("declared");
    let block = 22;
    for (obs, new_block) in [(observation, false), (restarted, true)] {
        let mut candidate = *core(&bound.filter);
        let pending = candidate
            .prepare_correlated(&obs.stamp, &obs.value, &correlated)
            .expect("planned");
        let Plan::Bound { reset, offset, .. } = pending.plan else {
            panic!("the source has a block: {pending:?}");
        };
        assert_eq!((reset, offset), (new_block, block));
        // A new block has the stationary covariance and no correlation
        // with the core states; the block of the same epoch keeps both.
        let cross = candidate.p.view((0, block), (15, 3)).amax();
        assert_eq!(cross == 0.0, new_block, "cross-covariance {cross}");
    }
}

/// The core after a fix at each of the first `steps` steps, with the
/// fix of step `late` offered one step after its acquisition.
fn with_late_fix<const N: usize>(
    layout: StateLayout,
    declared: Declared,
    late: Option<u32>,
) -> InertialCore<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let mut flight = flight::<N>(layout);
    let mut error = FixError::new(5);
    let mut held = None;
    for k in 1..=20_u32 {
        let at_ms = flight.step();
        if let Some(observation) = held.take() {
            assert_accepted(flight.filter.ingest(&observation, t(at_ms)));
        }
        let observation = fix(&flight, &error.next(), declared, k, at_ms);
        if late == Some(k) {
            held = Some(observation);
        } else {
            assert_accepted(flight.filter.ingest(&observation, t(at_ms)));
        }
    }
    *core(&flight.filter)
}

#[test]
fn a_delayed_correlated_fix_replays_to_the_estimate_of_the_fix_in_order() {
    let declared = Declared::Correlated { tau_s: TAU_S };
    // The late fix arrives after the fix of the next step, so the replay
    // reapplies that fix with the correlation state of the history.
    let in_order = with_late_fix::<25>(BOUND, declared, None);
    let delayed = with_late_fix::<25>(BOUND, declared, Some(10));
    assert!((in_order.nominal.p - delayed.nominal.p).norm() < 1e-6);
    let in_order = with_late_fix::<22>(StateLayout::FULL, declared, None);
    let delayed = with_late_fix::<22>(StateLayout::FULL, declared, Some(10));
    assert!((in_order.nominal.p - delayed.nominal.p).norm() < 1e-6);
}

/// Whether a fork of a filter taken after 20 fixes continues as the
/// original over 20 more.
fn fork_continues<const N: usize>(layout: StateLayout) -> bool
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let declared = Declared::Correlated { tau_s: TAU_S };
    let (mut original, mut forked) = (flight::<N>(layout), flight::<N>(layout));
    let (mut a, mut b) = (FixError::new(5), FixError::new(5));
    fly(&mut original, &mut a, declared, 20);
    fly(&mut forked, &mut b, declared, 20);
    forked.filter = original.filter.clone();
    fly(&mut original, &mut a, declared, 20);
    fly(&mut forked, &mut b, declared, 20);
    core(&original.filter) == core(&forked.filter)
}

#[test]
fn a_fork_continues_as_the_original_with_its_bound_and_inflated_sources() {
    // The clone of a filter with its history passes through the stack,
    // which in the debug build is larger than the stack of a test
    // thread.
    let forks = std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            (
                fork_continues::<25>(BOUND),
                fork_continues::<22>(StateLayout::FULL),
            )
        })
        .expect("a test thread");
    assert_eq!(forks.join().expect("no panic"), (true, true));
}

#[test]
fn a_source_that_changes_its_declaration_keeps_its_process() {
    // A receiver that scales its declaration with the satellite geometry:
    // its block keeps its estimate, and its inflation keeps the time of
    // its last fix.
    let declared = |sequence: u32| {
        let scale = if sequence.is_multiple_of(2) {
            1.0
        } else {
            1.05
        };
        CorrelatedError::new(TAU_S, [1.2 * scale, 1.2 * scale, 2.0 * scale])
    };
    let mut bound = flight::<25>(BOUND);
    let mut full = flight::<22>(StateLayout::FULL);
    for sequence in 1..=10_u32 {
        let at_ms = bound.step();
        let obs = fix(
            &bound,
            &Vector3::zeros(),
            Declared::AllWhite,
            sequence,
            at_ms,
        )
        .with_correlated_error(declared(sequence));
        assert_accepted(bound.filter.ingest(&obs, t(at_ms)));
        let at_ms = full.step();
        let obs = fix(
            &full,
            &Vector3::zeros(),
            Declared::AllWhite,
            sequence,
            at_ms,
        )
        .with_correlated_error(declared(sequence));
        assert_accepted(full.filter.ingest(&obs, t(at_ms)));
    }
    let at_ms = bound.step();
    let obs = fix(&bound, &Vector3::zeros(), Declared::AllWhite, 11, at_ms)
        .with_correlated_error(declared(11));
    let mut candidate = *core(&bound.filter);
    let pending = candidate
        .prepare_correlated(&obs.stamp, &obs.value, &declared(11))
        .expect("planned");
    assert!(
        matches!(pending.plan, Plan::Bound { reset: false, .. }),
        "{pending:?}"
    );
    let at_ms_full = full.step();
    assert_eq!(at_ms, at_ms_full);
    let obs = fix(&full, &Vector3::zeros(), Declared::AllWhite, 11, at_ms)
        .with_correlated_error(declared(11));
    let mut candidate = *core(&full.filter);
    let pending = candidate
        .prepare_correlated(&obs.stamp, &obs.value, &declared(11))
        .expect("planned");
    let Plan::Inflated { factor, .. } = pending.plan else {
        panic!("a layout without a block inflates: {pending:?}");
    };
    assert!(factor > 100.0, "factor {factor}");
}

/// The core after fixes of two sources on a layout with one block: a
/// constant error from the steps 5 to 12, and one fix of a second source
/// acquired at step 3 that arrives at step 3 or, delayed, at step 8.
fn two_sources(delayed: bool) -> InertialCore<25> {
    let constant = CorrelatedError::new(f64::INFINITY, [1.0; 3]);
    let other = CorrelatedError::new(TAU_S, [1.0; 3]);
    // The delayed fix is older than the default staleness bound.
    let config = InertialConfig {
        layout: BOUND,
        fusion: crate::FusionConfig {
            staleness_bound: navigate_contract::DurationNanos::from_millis(5_000),
            ..tactical().fusion
        },
        ..tactical()
    };
    let mut flight = std::boxed::Box::new(Flight::<25>::new(ROUTE, 5, config, ALIGNMENT));
    let mut held = None;
    for k in 1..=12_u32 {
        let at_ms = flight.step();
        if k == 3 {
            let obs = fix(&flight, &Vector3::zeros(), Declared::AllWhite, 1, at_ms);
            let obs = Observation {
                stamp: stamp(4, 1, at_ms),
                ..obs
            }
            .with_correlated_error(other);
            if delayed {
                held = Some(obs);
            } else {
                assert_accepted(flight.filter.ingest(&obs, t(at_ms)));
            }
        }
        if k == 8
            && let Some(obs) = held.take()
        {
            assert_accepted(flight.filter.ingest(&obs, t(at_ms)));
        }
        if k >= 5 {
            let obs = fix(&flight, &Vector3::zeros(), Declared::AllWhite, k, at_ms)
                .with_correlated_error(constant);
            // In order, the second source holds the block, and the
            // constant error is used once.
            let outcome = flight.filter.ingest(&obs, t(at_ms));
            assert_eq!(
                outcome.is_accepted(),
                k == 5 || (delayed && k < 8),
                "step {k}"
            );
        }
    }
    *core(&flight.filter)
}

#[test]
fn a_delayed_fix_that_takes_a_block_first_replays_as_in_order() {
    let in_order = two_sources(false);
    let delayed = two_sources(true);
    assert!((in_order.nominal.p - delayed.nominal.p).norm() < 1e-6);
    assert_eq!(in_order.correlated, delayed.correlated);
}

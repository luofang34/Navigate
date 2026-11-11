//! Keyed correlated processes (ADR-0023): fixes of several sources that
//! share one process, the takeover of a block by a newer constant
//! process, and its replay.

use nalgebra::Vector3;

use super::correlated::flight;
use super::correlated_gnss::{CORRELATED_SIGMA_NED_M, Declared, FixError, fix};
use super::flight_truth::Flight;
use super::*;
use crate::inertial::StateLayout;
use crate::observation::{CorrelatedError, Observation};
use crate::rejection::RejectionReason;

const BOUND: StateLayout = StateLayout::FULL.with_correlated_blocks(1);

/// The fix of `source` at `at_ms` with the constant error of the cell
/// `key`, and the white covariance.
fn anchor(
    flight: &Flight<25>,
    error: &Vector3<f64>,
    source: u32,
    key: u64,
    at_ms: u64,
) -> Observation {
    let observation = fix(flight, error, Declared::Correlated { tau_s: 1.0 }, 1, at_ms);
    Observation {
        stamp: stamp(source, (at_ms / 500) as u32, at_ms),
        correlated: Some(CorrelatedError::new(f64::INFINITY, CORRELATED_SIGMA_NED_M).with_key(key)),
        ..observation
    }
}

#[test]
fn one_fix_per_cell_with_a_block_equals_the_white_fix_with_the_map_variance() {
    // A fix whose map error has a fresh block of variance Σ updates the
    // core as a white fix of variance R + Σ, and a block that a new cell
    // takes over drops only the old error.
    let mut white = flight::<22>(StateLayout::FULL);
    let mut keyed = flight::<25>(BOUND);
    let (mut a, mut b) = (FixError::new(5), FixError::new(5));
    for cell in 1..=20_u64 {
        let at_ms = white.step();
        assert_eq!(keyed.step(), at_ms);
        let (ea, eb) = (a.next(), b.next());
        let sequence = (at_ms / 500) as u32;
        let obs = fix(&white, &ea, Declared::AllWhite, sequence, at_ms);
        assert_accepted(white.filter.ingest(&obs, t(at_ms)));
        let obs = Observation {
            stamp: stamp(3, sequence, at_ms),
            ..anchor(&keyed, &eb, 3, cell, at_ms)
        };
        assert_accepted(keyed.filter.ingest(&obs, t(at_ms)));
    }
    let (x, y) = (core(&white.filter), core(&keyed.filter));
    assert!(
        (x.nominal.p - y.nominal.p).norm() < 1e-6,
        "{}",
        (x.nominal.p - y.nominal.p).norm()
    );
    let n = 15;
    let difference = x.p.view((0, 0), (n, n)) - y.p.view((0, 0), (n, n));
    assert!(difference.amax() <= 1e-9 * x.p.view((0, 0), (n, n)).amax());
}

#[test]
fn two_sources_on_one_cell_share_its_block() {
    let mut keyed = flight::<25>(BOUND);
    let block = 22;
    let at_ms = keyed.step();
    assert_accepted(
        keyed
            .filter
            .ingest(&anchor(&keyed, &Vector3::zeros(), 3, 7, at_ms), t(at_ms)),
    );
    let after_first = core(&keyed.filter).p[(block, block)];
    let at_ms = keyed.step();
    assert_accepted(
        keyed
            .filter
            .ingest(&anchor(&keyed, &Vector3::zeros(), 4, 7, at_ms), t(at_ms)),
    );
    // The second camera reads the block of the first: its variance keeps
    // falling and is not reset to the stationary variance.
    assert!(core(&keyed.filter).p[(block, block)] < after_first);
    assert!(keyed.filter.binds_correlated_key(7));
}

#[test]
fn a_new_cell_takes_over_the_block_and_the_old_cell_is_refused() {
    let mut keyed = flight::<25>(BOUND);
    for (key, accepted) in [(1, true), (2, true), (1, false), (2, true)] {
        let at_ms = keyed.step();
        let outcome = keyed
            .filter
            .ingest(&anchor(&keyed, &Vector3::zeros(), 3, key, at_ms), t(at_ms));
        if accepted {
            assert_accepted(outcome);
        } else {
            assert_eq!(
                outcome,
                IngestOutcome::Rejected(RejectionReason::CorrelatedProcessRetired)
            );
        }
    }
    assert!(keyed.filter.binds_correlated_key(2));
    assert!(!keyed.filter.binds_correlated_key(1));
}

#[test]
fn a_takeover_that_the_gate_refuses_leaves_the_held_cell_intact() {
    let mut keyed = flight::<25>(BOUND);
    let at_ms = keyed.step();
    let first = anchor(&keyed, &Vector3::new(1.0, -2.0, 0.5), 3, 1, at_ms);
    assert_accepted(keyed.filter.ingest(&first, t(at_ms)));
    let at_ms = keyed.step();
    let before = *core(&keyed.filter);
    // A fix of cell 2 kilometres off: the gate refuses it after its plan
    // took over the block of cell 1.
    let far = anchor(&keyed, &Vector3::new(4_000.0, 0.0, 0.0), 3, 2, at_ms);
    assert!(matches!(
        keyed.filter.ingest(&far, t(at_ms)),
        IngestOutcome::Rejected(RejectionReason::InnovationGate { .. })
    ));
    assert!(keyed.filter.binds_correlated_key(1));
    assert!(!keyed.filter.binds_correlated_key(2));
    let after = core(&keyed.filter);
    assert_eq!(after.p, before.p);
    assert_eq!(after.nominal.correlated, before.nominal.correlated);
    // Cell 1 still reads its block.
    let at_ms = keyed.step();
    let again = anchor(&keyed, &Vector3::new(1.0, -2.0, 0.5), 3, 1, at_ms);
    assert_accepted(keyed.filter.ingest(&again, t(at_ms)));
}

/// The core after fixes of cell 1 at each step, and a fix of cell 2
/// acquired at step 3, after the fix of cell 1 of that step, that
/// arrives at step 3 or, delayed, at step 6.
fn takeover(delayed: bool) -> InertialCore<25> {
    let config = InertialConfig {
        layout: BOUND,
        fusion: crate::FusionConfig {
            staleness_bound: navigate_contract::DurationNanos::from_millis(5_000),
            ..super::flight_truth::tactical().fusion
        },
        ..super::flight_truth::tactical()
    };
    let route = super::flight_truth::Route {
        speed_mps: 40.0,
        bank_deg: 25.0,
        duration_s: 30.0,
        step_s: 0.5,
        turns: &[],
    };
    let alignment = super::flight_truth::Alignment {
        position_m: 2.0,
        velocity_mps: 0.1,
        attitude_rad: 1e-3,
    };
    let mut flight = std::boxed::Box::new(Flight::<25>::new(route, 5, config, alignment));
    let mut held = None;
    for k in 1..=8_u32 {
        let at_ms = flight.step();
        let new_cell = Observation {
            stamp: stamp(4, 1, at_ms),
            ..anchor(&flight, &Vector3::zeros(), 4, 2, at_ms)
        };
        if k == 6
            && let Some(obs) = held.take()
        {
            assert_accepted(flight.filter.ingest(&obs, t(at_ms)));
        }
        let old_cell = anchor(&flight, &Vector3::zeros(), 3, 1, at_ms);
        // In order, the fixes of cell 1 after the takeover are refused.
        let outcome = flight.filter.ingest(&old_cell, t(at_ms));
        assert_eq!(
            outcome.is_accepted(),
            k <= 3 || (delayed && k < 6),
            "step {k}"
        );
        if k == 3 {
            if delayed {
                held = Some(new_cell);
            } else {
                assert_accepted(flight.filter.ingest(&new_cell, t(at_ms)));
            }
        }
    }
    *core(&flight.filter)
}

#[test]
fn a_delayed_takeover_replays_as_in_order() {
    let in_order = takeover(false);
    let delayed = takeover(true);
    assert!((in_order.nominal.p - delayed.nominal.p).norm() < 1e-6);
    assert_eq!(in_order.correlated, delayed.correlated);
}

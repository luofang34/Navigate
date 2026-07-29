//! End-to-end assertions over the deterministic scripted scenario:
//! plan completion, admission health, lateral containment, and
//! single-source integrity honesty visible at the far end of the chain.

#![allow(clippy::expect_used, clippy::panic)]

use navigate_contract::{FaultDetection, Redundancy};
use navigate_fpl::SequenceReason;
use navrun::run_scenario;

#[test]
fn plan_completes_within_the_step_budget() {
    let summary = run_scenario().expect("scenario runs");
    assert!(
        summary.plan_completed,
        "plan must complete within the budget: {summary:?}"
    );
}

#[test]
fn every_observation_is_admitted_and_every_step_publishes() {
    let summary = run_scenario().expect("scenario runs");
    assert_eq!(
        summary.observations_rejected, 0,
        "no gate should fire on the scripted feed: {summary:?}"
    );
    assert_eq!(
        summary.solutions_published,
        u64::from(summary.steps_run),
        "the first fix initializes the filter, so every step publishes: {summary:?}"
    );
}

#[test]
fn lateral_deviation_stays_consistent_with_the_offset_table() {
    let summary = run_scenario().expect("scenario runs");
    assert!(
        summary.max_lateral_dev_m > 0.0,
        "the offset table must produce some cross-track deviation: {summary:?}"
    );
    // Offsets are under 3 m; the filter's velocity-convergence transient
    // adds a little, but nothing should approach 10 m.
    assert!(
        summary.max_lateral_dev_m < 10.0,
        "lateral deviation must stay near the offset magnitude: {summary:?}"
    );
}

#[test]
fn single_source_honesty_is_visible_end_to_end() {
    let summary = run_scenario().expect("scenario runs");
    assert_eq!(
        summary.final_redundancy,
        Redundancy::None,
        "one GNSS source can claim no redundancy: {summary:?}"
    );
    assert_eq!(
        summary.final_fault_detection,
        FaultDetection::Unavailable,
        "a single source agreeing with itself proves nothing: {summary:?}"
    );
}

#[test]
fn nav_tt_005_the_collinear_route_sequences_by_capture_not_anticipation() {
    let summary = run_scenario().expect("scenario runs");
    assert_eq!(
        summary.last_sequence_reason,
        Some(SequenceReason::Overflown),
        "collinear legs have no turn to anticipate: {summary:?}"
    );
}

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
    // The fly-by corner cut at W1 IS the dominant deviation, honestly
    // reported against the new leg (NAV-HN-001): about the 282 m DTA,
    // never runaway. Straight-leg tracking stays within the offset
    // table's magnitude underneath it.
    assert!(
        summary.max_lateral_dev_m < 320.0,
        "deviation is bounded by the corner geometry: {summary:?}"
    );
    assert!(
        summary.max_lateral_dev_m > 100.0,
        "the anticipated corner really was cut: {summary:?}"
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
fn nav_tt_005_the_dogleg_fix_sequences_by_anticipation() {
    let summary = run_scenario().expect("scenario runs");
    assert_eq!(
        summary.last_sequence_reason,
        Some(SequenceReason::Anticipated),
        "a 90° corner at 30 m/s anticipates well beyond the capture radius: {summary:?}"
    );
}

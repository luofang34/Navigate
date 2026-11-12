use navigate_contract::{
    ClockDomainId, DurationNanos, MonotonicNanos, ObservationStamp, SensorClass, SourceComposition,
    SourceEpoch, SourceId, WrappingSequence,
};

use super::{ConstrainedState, MAX_SOURCES, MAX_WINDOW_ADMISSIONS, ProvenanceLedger};
use crate::rejection::RejectionReason;

fn stamp(source: u32, sequence: u32, at_ns: u64) -> ObservationStamp {
    ObservationStamp::new(
        SourceId::new(source),
        SourceEpoch::new(0),
        WrappingSequence::new(sequence),
        MonotonicNanos::from_nanos(at_ns),
        ClockDomainId::new(0),
    )
}

#[test]
fn a_full_source_table_refuses_a_new_source_and_keeps_the_known_ones() {
    let mut ledger = ProvenanceLedger::default();
    let gnss = SourceComposition::of(SensorClass::Gnss);
    let window = DurationNanos::from_millis(1_000);
    let now = MonotonicNanos::from_nanos(10);
    for source in 0..MAX_SOURCES as u32 {
        let first = stamp(source, 0, 1);
        assert_eq!(ledger.check_source_ordering(&first), Ok(()));
        ledger.commit(&first, gnss, ConstrainedState::POSITION, now, window);
    }
    let newcomer = stamp(MAX_SOURCES as u32, 0, 1);
    assert_eq!(
        ledger.check_source_ordering(&newcomer),
        Err(RejectionReason::SourceCapacityExceeded {
            capacity: MAX_SOURCES
        })
    );
    // A tracked source keeps its sequence check.
    assert_eq!(ledger.check_source_ordering(&stamp(3, 1, 2)), Ok(()));
    assert!(matches!(
        ledger.check_source_ordering(&stamp(3, 0, 2)),
        Err(RejectionReason::SequenceNotAdmitted { .. })
    ));
    // A new filter epoch has room again.
    ledger.clear();
    assert_eq!(ledger.check_source_ordering(&newcomer), Ok(()));
}

#[test]
fn a_full_window_drops_its_oldest_admission() {
    let mut ledger = ProvenanceLedger::default();
    let window = DurationNanos::from_millis(1_000_000);
    let composition = SourceComposition::of(SensorClass::Gnss);
    let total = MAX_WINDOW_ADMISSIONS + 3;
    for index in 0..total as u32 {
        let at = MonotonicNanos::from_nanos(u64::from(index) + 1);
        ledger.commit(
            &stamp(index % 4, index / 4, at.as_nanos()),
            composition,
            ConstrainedState::POSITION,
            at,
            window,
        );
    }
    assert_eq!(ledger.window().count(), MAX_WINDOW_ADMISSIONS);
    let oldest = ledger.window().map(|c| c.admitted_at).min();
    assert_eq!(oldest, Some(MonotonicNanos::from_nanos(4)));
    // The window drops the admissions older than its duration.
    let later = MonotonicNanos::from_nanos(1_000_000_000_000);
    ledger.prune_window(later, DurationNanos::from_nanos(0));
    assert_eq!(ledger.window().count(), 0);
}

#[test]
fn a_held_admission_counts_until_its_hold_ends_and_a_fast_source_does_not_push_it_out() {
    let mut ledger = ProvenanceLedger::default();
    let window = DurationNanos::from_millis(5_000);
    let hold = DurationNanos::from_millis(30_000);
    let ms = |ms: u64| MonotonicNanos::from_nanos(ms * 1_000_000);
    let visual = SourceComposition::of(SensorClass::VisualLandmark);
    let gnss = SourceComposition::of(SensorClass::Gnss);
    let held = |ledger: &ProvenanceLedger| ledger.window().any(|c| c.composition == visual);
    // One fix of a slow source at 1 s, then a fast source at 100 Hz for
    // 20 s: more admissions than the window holds.
    ledger.commit_held(
        &stamp(9, 0, ms(1_000).as_nanos()),
        visual,
        ConstrainedState::POSITION,
        ms(1_000),
        (window, Some(hold)),
    );
    for index in 0..2_000_u32 {
        let at = ms(1_010 + 10 * u64::from(index));
        ledger.commit_held(
            &stamp(1, index, at.as_nanos()),
            gnss,
            ConstrainedState::POSITION,
            at,
            (window, None),
        );
    }
    ledger.prune_window(ms(21_000), window);
    assert!(held(&ledger));
    // The fast source fills the window, and its admissions are not held.
    assert_eq!(ledger.window().count(), MAX_WINDOW_ADMISSIONS + 1);
    // The hold ends 30 s after the admission.
    ledger.prune_window(ms(31_000), window);
    assert!(held(&ledger));
    ledger.prune_window(ms(31_001), window);
    assert!(!held(&ledger));
    assert_eq!(ledger.window().count(), 0);
}

#[test]
fn a_later_admission_of_a_source_replaces_its_hold_or_ends_it() {
    let mut ledger = ProvenanceLedger::default();
    let window = DurationNanos::from_millis(5_000);
    let ms = |ms: u64| MonotonicNanos::from_nanos(ms * 1_000_000);
    let visual = SourceComposition::of(SensorClass::VisualLandmark);
    let admit = |ledger: &mut ProvenanceLedger, sequence: u32, at_ms: u64, hold_ms: Option<u64>| {
        ledger.commit_held(
            &stamp(9, sequence, ms(at_ms).as_nanos()),
            visual,
            ConstrainedState::POSITION,
            ms(at_ms),
            (window, hold_ms.map(DurationNanos::from_millis)),
        );
    };
    admit(&mut ledger, 0, 1_000, Some(30_000));
    admit(&mut ledger, 1, 11_000, Some(30_000));
    // The second fix holds until 41 s; the first one alone ended at 31 s.
    ledger.prune_window(ms(40_000), window);
    assert_eq!(ledger.window().count(), 1);
    // A fix without a declared hold counts for the window only.
    admit(&mut ledger, 2, 40_000, None);
    ledger.prune_window(ms(45_001), window);
    assert_eq!(ledger.window().count(), 0);
    // A new filter epoch forgets a hold.
    admit(&mut ledger, 3, 50_000, Some(30_000));
    ledger.clear();
    assert_eq!(ledger.window().count(), 0);
}

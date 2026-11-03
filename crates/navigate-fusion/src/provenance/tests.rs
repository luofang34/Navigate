use navigate_contract::{
    ClockDomainId, DurationNanos, MonotonicNanos, ObservationStamp, SensorClass, SourceComposition,
    SourceEpoch, SourceId, WrappingSequence,
};

use super::{MAX_SOURCES, MAX_WINDOW_ADMISSIONS, ProvenanceLedger};
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
        ledger.commit(&first, gnss, now, window);
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

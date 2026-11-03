use std::string::ToString;

use navigate_contract::{DurationNanos, FrameId};

use super::{RejectionCounters, RejectionReason};

#[test]
fn frame_refusals_each_increment_one_counter_and_wrap() {
    let mut counters = RejectionCounters {
        missing_frame: u64::MAX,
        ..RejectionCounters::default()
    };
    counters.record(&RejectionReason::FrameMismatch {
        expected: FrameId::MOON_ME,
        got: FrameId::MOON_PA,
    });
    counters.record(&RejectionReason::MissingFrame {
        expected: FrameId::MOON_ME,
    });
    assert_eq!(
        counters,
        RejectionCounters {
            frame_mismatch: 1,
            missing_frame: 0,
            ..RejectionCounters::default()
        }
    );
}

#[test]
fn frame_refusals_name_both_naif_codes() {
    let text = RejectionReason::FrameMismatch {
        expected: FrameId::MOON_ME,
        got: FrameId::MOON_PA,
    }
    .to_string();
    assert!(text.contains("301/31000"), "{text}");
    assert!(text.contains("301/31001"), "{text}");
    let missing = RejectionReason::MissingFrame {
        expected: FrameId::MARS_IAU,
    }
    .to_string();
    assert!(missing.contains("499/10014"), "{missing}");
}

#[test]
fn counters_wrap_instead_of_overflowing() {
    let mut counters = RejectionCounters {
        stale: u64::MAX,
        ..RejectionCounters::default()
    };
    counters.record(&RejectionReason::Stale {
        age: DurationNanos::from_nanos(1),
        bound: DurationNanos::from_nanos(0),
    });
    assert_eq!(counters.stale, 0);
    counters.record(&RejectionReason::NotInitialized);
    assert_eq!(counters.not_initialized, 1);
    counters.record(&RejectionReason::NoInertialData);
    assert_eq!(counters.no_inertial_data, 1);
    counters.record(&RejectionReason::SourceCapacityExceeded { capacity: 32 });
    assert_eq!(counters.source_capacity_exceeded, 1);
}

#[test]
fn clock_and_mode_refusals_each_increment_one_counter_and_wrap() {
    let mut counters = RejectionCounters {
        propagation_mode_mismatch: u64::MAX,
        ..RejectionCounters::default()
    };
    counters.record(&RejectionReason::ClockStateDisabled {
        kind: crate::MeasurementKind::Pseudorange,
    });
    counters.record(&RejectionReason::PropagationModeMismatch);
    assert_eq!(
        counters,
        RejectionCounters {
            clock_state_disabled: 1,
            propagation_mode_mismatch: 0,
            ..RejectionCounters::default()
        }
    );
    let text = RejectionReason::ClockStateDisabled {
        kind: crate::MeasurementKind::PseudorangeRate,
    }
    .to_string();
    assert!(text.contains("PseudorangeRate"), "{text}");
}

#[test]
fn map_and_history_refusals_each_increment_one_counter() {
    let mut counters = RejectionCounters::default();
    counters.record(&RejectionReason::LinearizationRegionExceeded {
        distance_plus_ksigma_m: 600.0,
        valid_radius_m: 500.0,
    });
    counters.record(&RejectionReason::UninformativeGradient {
        information_ratio: 1e-6,
        floor: 1e-3,
    });
    counters.record(&RejectionReason::UninformativeGradient {
        information_ratio: f64::NAN,
        floor: 1e-3,
    });
    counters.record(&RejectionReason::OutsideHistory {
        age: DurationNanos::from_millis(300),
        depth: DurationNanos::from_millis(100),
    });
    assert_eq!(
        counters,
        RejectionCounters {
            linearization_region_exceeded: 1,
            uninformative_gradient: 2,
            outside_history: 1,
            ..RejectionCounters::default()
        }
    );
}

#[test]
fn new_counters_wrap_instead_of_overflowing() {
    let mut counters = RejectionCounters {
        linearization_region_exceeded: u64::MAX,
        uninformative_gradient: u64::MAX,
        outside_history: u64::MAX,
        ..RejectionCounters::default()
    };
    counters.record(&RejectionReason::LinearizationRegionExceeded {
        distance_plus_ksigma_m: 1.0,
        valid_radius_m: 0.5,
    });
    counters.record(&RejectionReason::UninformativeGradient {
        information_ratio: 0.0,
        floor: 1e-3,
    });
    counters.record(&RejectionReason::OutsideHistory {
        age: DurationNanos::from_nanos(1),
        depth: DurationNanos::from_nanos(0),
    });
    assert_eq!(counters.linearization_region_exceeded, 0);
    assert_eq!(counters.uninformative_gradient, 0);
    assert_eq!(counters.outside_history, 0);
}

#[test]
fn new_reasons_name_their_values() {
    let text = RejectionReason::OutsideHistory {
        age: DurationNanos::from_millis(300),
        depth: DurationNanos::from_millis(100),
    }
    .to_string();
    assert!(text.contains("300000000"), "{text}");
    assert!(text.contains("100000000"), "{text}");
}

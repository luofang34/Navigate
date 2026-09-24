use super::*;

fn range(sequence: u32, at_ms: u64, station: GeodeticPosition, range_m: f64) -> Observation {
    Observation::new(
        stamp(20, 1, sequence, at_ms),
        ObservationValue::Range {
            station,
            range_m,
            variance_m2: 4.0,
        },
        SourceComposition::of(SensorClass::RadioNavigation),
    )
}

#[test]
fn a_range_before_initialization_is_refused() {
    let mut f = filter();
    assert_eq!(
        f.ingest(
            &range(1, 1000, position_north_m(10_000.0), 10_000.0),
            t(1000)
        ),
        IngestOutcome::Rejected(RejectionReason::NotInitialized)
    );
}

#[test]
fn a_range_pulls_the_position_along_the_line_of_sight() {
    let mut f = filter();
    assert!(
        f.ingest(&gnss_fix(1, 1, 1000, origin()), t(1000))
            .is_accepted()
    );
    let before = f.tick(t(1000)).expect("solution");
    // The station is 10 km north. A measured range of 10 002 m says the
    // vehicle is about 2 m further south than the fix.
    let outcome = f.ingest(
        &range(1, 1000, position_north_m(10_000.0), 10_002.0),
        t(1000),
    );
    assert!(outcome.is_accepted(), "{outcome:?}");
    let after = f.tick(t(1000)).expect("solution");
    assert!(after.position.latitude_rad < before.position.latitude_rad);
    let (north_before, _, _) = before.position_cov.diagonal();
    let (north_after, east_after, _) = after.position_cov.diagonal();
    assert!(
        north_after < north_before,
        "the range reduces north variance"
    );
    assert!(
        (east_after - north_before).abs() < 1e-6,
        "east is unobserved"
    );
}

#[test]
fn a_range_far_outside_the_gate_is_refused_and_invalid_values_are_screened() {
    let mut f = filter();
    assert!(
        f.ingest(&gnss_fix(1, 1, 1000, origin()), t(1000))
            .is_accepted()
    );
    assert!(matches!(
        f.ingest(
            &range(1, 1000, position_north_m(10_000.0), 11_000.0),
            t(1000)
        ),
        IngestOutcome::Rejected(RejectionReason::InnovationGate { .. })
    ));
    assert_eq!(
        f.ingest(
            &range(2, 1000, position_north_m(10_000.0), f64::NAN),
            t(1000)
        ),
        IngestOutcome::Rejected(RejectionReason::NonFiniteValue)
    );
}

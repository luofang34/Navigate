//! Tests of the dual-layer time vocabulary: exact normalization, exact
//! fixed-offset conversions, and refusals across scales and clocks.

use super::{
    ATTOSECONDS_PER_SECOND, ClockAnchor, ClockDomainId, MonotonicNanos, PhysicalInstant,
    SignedDuration, TimeScale,
};

const QUARTER_S: i64 = ATTOSECONDS_PER_SECOND / 4;

/// 1969-07-20T20:17:40 (the Apollo 11 landing) as a label on one scale:
/// 11121.654398148 days before J2000.0.
const APOLLO_11_OFFSET_S: i64 = -960_910_940;

/// −2000-01-01T12:00:00 (2001 BC): 4000 Julian years of 365.25 days
/// before J2000.0.
const MINUS_2000_OFFSET_S: i64 = -126_230_400_000;

fn duration(seconds: i64, attoseconds: i64) -> SignedDuration {
    SignedDuration::new(seconds, attoseconds).expect("in range")
}

#[test]
fn a_negative_fraction_normalizes_to_a_floor_and_a_positive_remainder() {
    let minus_quarter = duration(0, -QUARTER_S);
    assert_eq!(minus_quarter.seconds(), -1);
    assert_eq!(minus_quarter.subsec_attoseconds(), 3 * QUARTER_S);
    assert!(minus_quarter.is_negative());
    // A carry of several whole seconds, both ways.
    let carried = duration(10, 5 * ATTOSECONDS_PER_SECOND + 7);
    assert_eq!((carried.seconds(), carried.subsec_attoseconds()), (15, 7));
    let borrowed = duration(10, -3 * ATTOSECONDS_PER_SECOND - 1);
    assert_eq!(
        (borrowed.seconds(), borrowed.subsec_attoseconds()),
        (6, ATTOSECONDS_PER_SECOND - 1)
    );
    assert_eq!(SignedDuration::new(i64::MAX, ATTOSECONDS_PER_SECOND), None);
}

#[test]
fn one_attosecond_is_kept_through_add_sub_and_negation() {
    let tick = duration(0, 1);
    let sum = SignedDuration::from_seconds(1)
        .checked_add(tick)
        .expect("sum");
    assert_eq!((sum.seconds(), sum.subsec_attoseconds()), (1, 1));
    let back = sum.checked_sub(tick).expect("difference");
    assert_eq!(back, SignedDuration::from_seconds(1));
    let negative = tick.checked_neg().expect("negation");
    assert_eq!(
        (negative.seconds(), negative.subsec_attoseconds()),
        (-1, ATTOSECONDS_PER_SECOND - 1)
    );
    assert_eq!(negative.abs(), tick);
    assert_eq!(negative.checked_neg(), Some(tick));
    assert_eq!(SignedDuration::MIN.checked_neg(), None);
    assert_eq!(SignedDuration::MIN.abs(), SignedDuration::MAX);
    assert_eq!(SignedDuration::MAX.checked_add(tick), None);
}

#[test]
fn the_order_is_numeric_for_negative_and_positive_spans() {
    let ascending = [
        duration(-2, 0),
        duration(0, -3 * QUARTER_S),
        duration(0, -QUARTER_S),
        SignedDuration::ZERO,
        duration(0, 1),
        duration(1, 0),
    ];
    for pair in ascending.windows(2) {
        assert!(pair[0] < pair[1], "{:?} < {:?}", pair[0], pair[1]);
    }
}

#[test]
fn nanosecond_and_float_conversions_are_floor_exact() {
    let minus_one_ns = SignedDuration::from_nanos(-1).expect("in range");
    assert_eq!(minus_one_ns.seconds(), -1);
    assert_eq!(
        minus_one_ns.subsec_attoseconds(),
        ATTOSECONDS_PER_SECOND - 1_000_000_000
    );
    assert_eq!(minus_one_ns.as_nanos_floor(), -1);
    assert_eq!(
        duration(0, -1).as_nanos_floor(),
        -1,
        "floor, not truncation"
    );
    let parsed = SignedDuration::from_seconds_f64(-0.25).expect("finite");
    assert_eq!(parsed, duration(0, -QUARTER_S));
    assert!((duration(3, QUARTER_S).to_seconds_f64() - 3.25).abs() < 1e-15);
    assert_eq!(SignedDuration::from_seconds_f64(f64::NAN), None);
    assert_eq!(SignedDuration::from_seconds_f64(1e19), None);
}

#[test]
fn a_ratio_applies_exactly_to_a_span_of_millennia() {
    // 5000 Julian years times 7e-10, which overflows a direct i128
    // product of the attosecond count and 7·10^9.
    let span = SignedDuration::from_seconds(157_788_000_000);
    let scaled = span
        .checked_mul_ratio(7_000_000_000, 10_000_000_000_000_000_000)
        .expect("in range");
    assert_eq!(scaled, duration(110, 451_600_000_000_000_000));
    let negative = span.checked_neg().expect("negation");
    let scaled_negative = negative
        .checked_mul_ratio(7_000_000_000, 10_000_000_000_000_000_000)
        .expect("in range");
    assert_eq!(scaled_negative, scaled.checked_neg().expect("negation"));
    assert_eq!(span.checked_mul_ratio(1, 0), None);
}

#[test]
fn negative_j2000_epochs_keep_exact_labels() {
    let apollo = PhysicalInstant::from_j2000(TimeScale::Tt, APOLLO_11_OFFSET_S, -QUARTER_S)
        .expect("in range");
    assert_eq!(apollo.seconds_since_j2000(), APOLLO_11_OFFSET_S - 1);
    assert_eq!(apollo.subsec_attoseconds(), 3 * QUARTER_S);
    let ancient =
        PhysicalInstant::from_j2000(TimeScale::Tt, MINUS_2000_OFFSET_S, 0).expect("in range");
    let span = apollo.elapsed_since(ancient).expect("one scale");
    assert_eq!(
        span,
        duration(APOLLO_11_OFFSET_S - MINUS_2000_OFFSET_S, -QUARTER_S)
    );
    assert_eq!(
        ancient.elapsed_since(apollo),
        span.checked_neg(),
        "the reverse span is the negation"
    );
    assert_eq!(
        PhysicalInstant::j2000(TimeScale::Tt).offset_from_j2000(),
        SignedDuration::ZERO
    );
}

#[test]
fn tai_and_tt_differ_by_exactly_32_184_seconds() {
    let tai = PhysicalInstant::from_j2000(TimeScale::Tai, 1, 1).expect("in range");
    let tt = PhysicalInstant::from_tai_fixed(tai, TimeScale::Tt).expect("fixed offset");
    assert_eq!(tt.scale(), TimeScale::Tt);
    assert_eq!(tt.seconds_since_j2000(), 33);
    assert_eq!(tt.subsec_attoseconds(), 184_000_000_000_000_001);
    assert_eq!(tt.to_tai_fixed(), Some(tai), "the round trip is exact");
}

#[test]
fn gpst_is_tai_minus_19_seconds() {
    let tai = PhysicalInstant::from_j2000(TimeScale::Tai, APOLLO_11_OFFSET_S, 5).expect("range");
    let gpst = PhysicalInstant::from_tai_fixed(tai, TimeScale::Gpst).expect("fixed offset");
    assert_eq!(gpst.seconds_since_j2000(), APOLLO_11_OFFSET_S - 19);
    assert_eq!(gpst.subsec_attoseconds(), 5);
    assert_eq!(gpst.to_tai_fixed(), Some(tai));
}

#[test]
fn model_scales_have_no_fixed_offset_and_are_refused() {
    let tai = PhysicalInstant::j2000(TimeScale::Tai);
    for scale in [
        TimeScale::Tdb,
        TimeScale::Tcg,
        TimeScale::Tcb,
        TimeScale::Tcl,
        TimeScale::Ut1,
        TimeScale::Utc,
    ] {
        assert_eq!(scale.fixed_offset_from_tai(), None, "{scale:?}");
        assert_eq!(PhysicalInstant::from_tai_fixed(tai, scale), None);
        assert_eq!(PhysicalInstant::j2000(scale).to_tai_fixed(), None);
    }
    let tt = PhysicalInstant::j2000(TimeScale::Tt);
    assert_eq!(
        PhysicalInstant::from_tai_fixed(tt, TimeScale::Gpst),
        None,
        "the source must be on TAI"
    );
}

#[test]
fn a_span_across_two_scales_is_refused() {
    let tai = PhysicalInstant::j2000(TimeScale::Tai);
    let tt = PhysicalInstant::j2000(TimeScale::Tt);
    assert_eq!(tt.elapsed_since(tai), None);
    let tt_from_tai = PhysicalInstant::from_tai_fixed(tai, TimeScale::Tt).expect("fixed");
    assert_eq!(
        tt_from_tai.elapsed_since(tt),
        Some(duration(32, 184_000_000_000_000_000)),
        "after an explicit conversion the span exists"
    );
}

#[test]
fn a_clock_anchor_maps_its_own_domain_with_the_rate_offset() {
    let clock = ClockDomainId::new(3);
    let reference = PhysicalInstant::from_j2000(TimeScale::Tai, 100, 0).expect("in range");
    let anchor = ClockAnchor::new(
        clock,
        MonotonicNanos::from_nanos(5_000_000_000),
        reference,
        1e-6,
        20.0,
    );
    assert!(anchor.is_plausible());
    let later = anchor
        .map_monotonic(MonotonicNanos::from_nanos(6_000_000_000), clock)
        .expect("same domain");
    // One second of the clock is 1 s + 1 µs on the scale.
    assert_eq!(
        later.elapsed_since(reference),
        Some(duration(1, 1_000_000_000_000))
    );
    let earlier = anchor
        .map_monotonic(MonotonicNanos::from_nanos(4_000_000_000), clock)
        .expect("same domain");
    assert_eq!(
        earlier.elapsed_since(reference),
        duration(1, 1_000_000_000_000).checked_neg()
    );
}

#[test]
fn a_clock_anchor_refuses_another_domain_and_an_implausible_rate() {
    let reference = PhysicalInstant::j2000(TimeScale::Tai);
    let anchor = ClockAnchor::new(
        ClockDomainId::new(1),
        MonotonicNanos::from_nanos(0),
        reference,
        0.0,
        1.0,
    );
    let reading = MonotonicNanos::from_nanos(10);
    assert_eq!(anchor.map_monotonic(reading, ClockDomainId::new(2)), None);
    assert!(
        anchor
            .map_monotonic(reading, ClockDomainId::new(1))
            .is_some()
    );
    for (rate, uncertainty_ns) in [
        (f64::NAN, 1.0),
        (0.5, 1.0),
        (0.0, -1.0),
        (0.0, f64::INFINITY),
    ] {
        let bad = ClockAnchor::new(
            ClockDomainId::new(1),
            MonotonicNanos::from_nanos(0),
            reference,
            rate,
            uncertainty_ns,
        );
        assert!(
            !bad.is_plausible(),
            "rate {rate} uncertainty {uncertainty_ns}"
        );
        assert_eq!(bad.map_monotonic(reading, ClockDomainId::new(1)), None);
    }
}

//! The physical-time bridge: exact fixed offsets, the IAU linear scales,
//! the lunar drift, and the nanosecond round trip through `hifitime`.

use navigate_contract::{ATTOSECONDS_PER_SECOND, PhysicalInstant, SignedDuration, TimeScale};

use super::*;
use crate::{DeltaT, DeltaTBasis};

const SECONDS_PER_DAY: i64 = 86_400;

/// 56.021 760 µs: 86 400 s times 6.484·10^-10, in attoseconds.
const TCL_DRIFT_PER_DAY_AS: i128 = 56_021_760_000_000;

fn tt(seconds: i64, attoseconds: i64) -> PhysicalInstant {
    PhysicalInstant::from_j2000(TimeScale::Tt, seconds, attoseconds).expect("in range")
}

fn epoch(text: &str) -> Epoch {
    text.parse().expect("a valid epoch")
}

/// The signed span `a − b` in attoseconds across two scales.
fn label_gap_as(a: PhysicalInstant, b: PhysicalInstant) -> i128 {
    a.offset_from_j2000().as_attoseconds() - b.offset_from_j2000().as_attoseconds()
}

#[test]
fn tcl_gains_56_021_microseconds_per_day_on_tt() {
    let day0 = tt(0, 0);
    let day1 = tt(SECONDS_PER_DAY, 0);
    let gap0 = label_gap_as(tt_to_tcl(day0).expect("TT"), day0);
    let gap1 = label_gap_as(tt_to_tcl(day1).expect("TT"), day1);
    assert!(
        (gap1 - gap0 - TCL_DRIFT_PER_DAY_AS).abs() <= 1,
        "drift {} as per day",
        gap1 - gap0
    );
    // 8400.4996275 days after T0 the lunar scale leads by about 0.4706 s.
    assert!((gap0 as f64 / 1e18 - 0.470_61).abs() < 1e-5, "{gap0}");
    // TCL equals TT at T0.
    let t0 = tt(-725_803_168, 184_000_000_000_000_000);
    assert_eq!(label_gap_as(tt_to_tcl(t0).expect("TT"), t0), 0);
}

#[test]
fn tcl_and_tt_round_trip_within_two_attoseconds() {
    for instant in [
        tt(0, 0),
        tt(-725_803_168, 184_000_000_000_000_000),
        tt(3_155_760_000, 123_456_789_012_345_678),
        tt(-126_230_400_000, 1),
    ] {
        let tcl = tt_to_tcl(instant).expect("TT");
        assert_eq!(tcl.scale(), TimeScale::Tcl);
        let back = tcl_to_tt(tcl).expect("TCL");
        assert!(
            label_gap_as(back, instant).abs() <= 2,
            "{instant:?} -> {back:?}"
        );
    }
}

#[test]
fn tcg_and_tcb_lead_tt_and_tdb_by_the_iau_values_at_j2000() {
    let tcg = tt_to_tcg(tt(0, 0)).expect("TT");
    let tcg_lead_s = label_gap_as(tcg, tt(0, 0)) as f64 / 1e18;
    assert!(
        (tcg_lead_s - 0.505_833).abs() < 1e-6,
        "TCG - TT {tcg_lead_s} s"
    );
    let tdb = PhysicalInstant::j2000(TimeScale::Tdb);
    let tcb = tdb_to_tcb(tdb).expect("TDB");
    let tcb_lead_s = label_gap_as(tcb, tdb) as f64 / 1e18;
    assert!(
        (tcb_lead_s - 11.253_7).abs() < 1e-4,
        "TCB - TDB {tcb_lead_s} s"
    );
    for instant in [tt(0, 0), tt(1_000_000_000, 7), tt(-2_000_000_000, 0)] {
        let back = tcg_to_tt(tt_to_tcg(instant).expect("TT")).expect("TCG");
        assert!(label_gap_as(back, instant).abs() <= 2);
        let tdb = PhysicalInstant::new(TimeScale::Tdb, instant.offset_from_j2000());
        let back = tcb_to_tdb(tdb_to_tcb(tdb).expect("TDB")).expect("TCB");
        assert!(label_gap_as(back, tdb).abs() <= 2);
    }
}

#[test]
fn tcg_agrees_with_the_hifitime_model() {
    for text in [
        "2000-01-01T12:00:00 TT",
        "2026-11-03T00:00:00 TAI",
        "1977-01-01T00:00:32.184 TT",
    ] {
        let e = epoch(text);
        let ours = physical_from_epoch(e, TimeScale::Tcg, None).expect("in range");
        let theirs = e.to_tcg_duration().total_nanoseconds();
        let t0_as = -725_803_167_816_000_000_000_000_000_i128;
        let ours_ns = (ours.offset_from_j2000().as_attoseconds() - t0_as) / 1_000_000_000;
        assert!(
            (ours_ns - theirs).abs() <= 2,
            "{text}: {ours_ns} vs {theirs} ns"
        );
    }
}

#[test]
fn fixed_scales_follow_hifitime_exactly() {
    let j2000_tt = epoch("2000-01-01T12:00:00 TT");
    let on_tt = physical_from_epoch(j2000_tt, TimeScale::Tt, None).expect("in range");
    assert_eq!(on_tt, PhysicalInstant::j2000(TimeScale::Tt));
    let on_tai = physical_from_epoch(j2000_tt, TimeScale::Tai, None).expect("in range");
    assert_eq!(
        on_tai.offset_from_j2000(),
        SignedDuration::new(-33, 816_000_000_000_000_000).expect("range")
    );
    let on_gpst = physical_from_epoch(j2000_tt, TimeScale::Gpst, None).expect("in range");
    assert_eq!(
        on_gpst.elapsed_since(PhysicalInstant::new(
            TimeScale::Gpst,
            on_tai.offset_from_j2000()
        )),
        Some(SignedDuration::from_seconds(-19))
    );
    let later = epoch("2024-03-05T07:08:09.123456789 UTC");
    let on_tt = physical_from_epoch(later, TimeScale::Tt, None).expect("in range");
    assert_eq!(
        on_tt.offset_from_j2000().as_nanos_floor(),
        later.to_tt_since_j2k().total_nanoseconds()
    );
}

#[test]
fn utc_counts_labels_and_tai_counts_the_leap_second() {
    let before = epoch("2016-12-31T23:59:59 UTC");
    let after = epoch("2017-01-01T00:00:00 UTC");
    let span = |scale| {
        let a = physical_from_epoch(after, scale, None).expect("in range");
        let b = physical_from_epoch(before, scale, None).expect("in range");
        a.elapsed_since(b).expect("one scale")
    };
    assert_eq!(span(TimeScale::Utc), SignedDuration::from_seconds(1));
    assert_eq!(span(TimeScale::Tai), SignedDuration::from_seconds(2));
}

#[test]
fn every_scale_round_trips_an_epoch_to_the_nanosecond() {
    let ut1_minus_tai = Some(-36.85);
    for text in [
        "2024-03-05T07:08:09.123456789 UTC",
        "1969-07-20T20:17:40 TT",
        "2100-06-30T00:00:00.000000001 TAI",
    ] {
        let e = epoch(text);
        for scale in [
            TimeScale::Tai,
            TimeScale::Tt,
            TimeScale::Gpst,
            TimeScale::Utc,
            TimeScale::Tdb,
            TimeScale::Tcg,
            TimeScale::Tcb,
            TimeScale::Tcl,
            TimeScale::Ut1,
        ] {
            let physical = physical_from_epoch(e, scale, ut1_minus_tai).expect("in range");
            assert_eq!(physical.scale(), scale);
            let back = epoch_from_physical(physical, ut1_minus_tai).expect("in range");
            let gap_ns = (back.to_tai_duration() - e.to_tai_duration()).total_nanoseconds();
            // The TDB model of hifitime inverts within a nanosecond.
            let bound_ns = if matches!(scale, TimeScale::Tdb | TimeScale::Tcb) {
                1
            } else {
                0
            };
            assert!(gap_ns.abs() <= bound_ns, "{text} {scale:?}: {gap_ns} ns");
        }
    }
}

#[test]
fn ut1_needs_a_finite_offset_from_tai() {
    let e = epoch("2026-11-03T00:00:00 TAI");
    assert_eq!(
        physical_from_epoch(e, TimeScale::Ut1, None),
        Err(TimeBridgeError::MissingUt1Offset)
    );
    assert!(matches!(
        physical_from_epoch(e, TimeScale::Ut1, Some(f64::NAN)),
        Err(TimeBridgeError::NonFiniteUt1Offset(_))
    ));
    let ut1 = physical_from_epoch(e, TimeScale::Ut1, Some(-37.25)).expect("in range");
    let tai = physical_from_epoch(e, TimeScale::Tai, None).expect("in range");
    assert_eq!(
        label_gap_as(ut1, tai),
        -37 * i128::from(ATTOSECONDS_PER_SECOND) - i128::from(ATTOSECONDS_PER_SECOND) / 4
    );
    let delta_t = DeltaT {
        seconds: 69.184,
        uncertainty_seconds: 0.1,
        basis: DeltaTBasis::Observed,
    };
    assert!((ut1_minus_tai_s(delta_t) + 37.0).abs() < 1e-12);
}

#[test]
fn a_wrong_scale_and_an_out_of_range_instant_are_refused() {
    let tai = PhysicalInstant::j2000(TimeScale::Tai);
    assert_eq!(
        tt_to_tcl(tai),
        Err(TimeBridgeError::WrongScale {
            expected: TimeScale::Tt,
            got: TimeScale::Tai
        })
    );
    assert!(matches!(
        tcb_to_tdb(tai),
        Err(TimeBridgeError::WrongScale { .. })
    ));
    let far =
        PhysicalInstant::from_j2000(TimeScale::Tai, 100_000_000_000_000_000, 0).expect("range");
    assert_eq!(
        epoch_from_physical(far, None),
        Err(TimeBridgeError::OutOfRange)
    );
    let edge = PhysicalInstant::new(TimeScale::Tt, SignedDuration::MAX);
    assert_eq!(tt_to_tcg(edge), Err(TimeBridgeError::OutOfRange));
}

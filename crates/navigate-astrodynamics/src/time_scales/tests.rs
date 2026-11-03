//! Exact fixed offsets, the IAU secular rates, the lunar drift, the
//! periodic TDB term, and UT1 from the Earth orientation.

use navigate_contract::{PhysicalInstant, SignedDuration, TimeScale};

use super::*;

const SECONDS_PER_DAY: i64 = 86_400;

/// 56.021 760 µs: 86 400 s times 6.484·10^-10, in attoseconds.
const TCL_DRIFT_PER_DAY_AS: i128 = 56_021_760_000_000;

fn on(scale: TimeScale, seconds: i64, attoseconds: i64) -> PhysicalInstant {
    PhysicalInstant::from_j2000(scale, seconds, attoseconds).expect("in range")
}

fn tt(seconds: i64, attoseconds: i64) -> PhysicalInstant {
    on(TimeScale::Tt, seconds, attoseconds)
}

/// The signed span `a − b` in attoseconds across two scales.
fn label_gap_as(a: PhysicalInstant, b: PhysicalInstant) -> i128 {
    a.offset_from_j2000().as_attoseconds() - b.offset_from_j2000().as_attoseconds()
}

#[test]
fn fixed_offsets_are_exact_in_both_directions() {
    let tai = on(TimeScale::Tai, 1_000, 5);
    let tt = tai_to_tt(tai).expect("TAI");
    assert_eq!(tt, on(TimeScale::Tt, 1_032, 184_000_000_000_000_005));
    assert_eq!(tt_to_tai(tt), Ok(tai));
    let gpst = tai_to_gpst(tai).expect("TAI");
    assert_eq!(gpst, on(TimeScale::Gpst, 981, 5));
    assert_eq!(gpst_to_tai(gpst), Ok(tai));
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
fn the_periodic_tdb_term_has_the_fairhead_bretagnon_amplitude_and_period() {
    // At J2000.0 TT, g = 357.53°, so TDB − TT = 0.001657·sin g +
    // 0.000014·sin 2g = −72.6 µs.
    let g = 357.53_f64.to_radians();
    let expected_s = 0.001_657 * g.sin() + 0.000_014 * (2.0 * g).sin();
    let gap_s = label_gap_as(tt_to_tdb(tt(0, 0)).expect("TT"), tt(0, 0)) as f64 / 1e18;
    assert!(
        (gap_s - expected_s).abs() < 1e-15,
        "{gap_s} vs {expected_s}"
    );
    assert!((gap_s + 72.6e-6).abs() < 0.1e-6, "{gap_s}");
    let mut peak_s: f64 = 0.0;
    for day in 0..366 {
        let instant = tt(day * SECONDS_PER_DAY, 0);
        let gap = label_gap_as(tt_to_tdb(instant).expect("TT"), instant) as f64 / 1e18;
        peak_s = peak_s.max(gap.abs());
    }
    assert!((1.65e-3..=1.671e-3).contains(&peak_s), "{peak_s}");
    // One anomalistic year later the term repeats.
    let year_s = (360.0 / 0.985_600_28 * 86_400.0) as i64;
    let a = label_gap_as(tt_to_tdb(tt(0, 0)).expect("TT"), tt(0, 0));
    let b = label_gap_as(tt_to_tdb(tt(year_s, 0)).expect("TT"), tt(year_s, 0));
    assert!((a - b).abs() < 1_000_000_000, "{a} vs {b} as");
}

#[test]
fn tdb_and_tt_round_trip_within_one_picosecond() {
    for instant in [
        tt(0, 0),
        tt(86_400 * 91, 0),
        tt(3_155_760_000, 123_456_789_012_345_678),
        tt(-3_155_760_000, 1),
    ] {
        let tdb = tt_to_tdb(instant).expect("TT");
        assert_eq!(tdb.scale(), TimeScale::Tdb);
        let back = tdb_to_tt(tdb).expect("TDB");
        assert!(
            label_gap_as(back, instant).abs() <= 1_000_000,
            "{instant:?} -> {back:?}"
        );
    }
}

#[test]
fn ut1_follows_the_earth_orientation_offset() {
    let orientation = EarthOrientation {
        ut1_minus_tai_s: -37.25,
        polar_motion_xp_rad: 1e-6,
        polar_motion_yp_rad: 2e-6,
    };
    let tai = on(TimeScale::Tai, 800_000_000, 0);
    let ut1 = tai_to_ut1(tai, &orientation).expect("TAI");
    assert_eq!(
        ut1,
        on(TimeScale::Ut1, 799_999_962, 750_000_000_000_000_000)
    );
    assert_eq!(ut1_to_tai(ut1, &orientation), Ok(tai));
    for implausible in [
        EarthOrientation {
            ut1_minus_tai_s: f64::NAN,
            ..orientation
        },
        EarthOrientation {
            polar_motion_xp_rad: 0.5,
            ..orientation
        },
        EarthOrientation {
            polar_motion_yp_rad: f64::INFINITY,
            ..orientation
        },
    ] {
        assert!(!implausible.is_plausible());
        assert_eq!(
            tai_to_ut1(tai, &implausible),
            Err(TimeScaleError::ImplausibleEarthOrientation)
        );
    }
}

#[test]
fn a_wrong_scale_and_an_out_of_range_instant_are_refused() {
    let tai = PhysicalInstant::j2000(TimeScale::Tai);
    assert_eq!(
        tt_to_tcl(tai),
        Err(TimeScaleError::WrongScale {
            expected: TimeScale::Tt,
            got: TimeScale::Tai
        })
    );
    for refused in [
        tcb_to_tdb(tai),
        tdb_to_tt(tai),
        tt_to_tai(tai),
        gpst_to_tai(tai),
        tai_to_tt(tt(0, 0)),
    ] {
        assert!(matches!(refused, Err(TimeScaleError::WrongScale { .. })));
    }
    let edge = PhysicalInstant::new(TimeScale::Tt, SignedDuration::MAX);
    assert_eq!(tt_to_tcg(edge), Err(TimeScaleError::OutOfRange));
    assert_eq!(tt_to_tai(edge).map(|_| ()), Ok(()));
    let top = PhysicalInstant::new(TimeScale::Tai, SignedDuration::MAX);
    assert_eq!(tai_to_tt(top), Err(TimeScaleError::OutOfRange));
}

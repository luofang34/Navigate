use super::CalendarDate;

fn jd(year: i16, month: u8, day: u8) -> f64 {
    CalendarDate { year, month, day }.julian_date_at_midnight()
}

#[test]
fn julian_dates_match_known_days() {
    assert_eq!(jd(2000, 1, 1), 2_451_544.5);
    assert_eq!(jd(1858, 11, 17), 2_400_000.5);
    // The first Gregorian day follows the last Julian day.
    assert_eq!(jd(1582, 10, 15), 2_299_160.5);
    assert_eq!(jd(1582, 10, 4), 2_299_159.5);
    // Julian date 0 is noon of -4712 January 1 in the Julian calendar.
    assert_eq!(jd(-4712, 1, 1), -0.5);
    assert_eq!(jd(0, 1, 1), 1_721_057.5);
    assert_eq!(jd(-1999, 1, 1), 990_923.5);
    assert_eq!(jd(-1999, 6, 12), 991_085.5);
}

#[test]
fn the_calendar_changes_on_1582_october_15() {
    let date = |year, month, day| CalendarDate { year, month, day };
    assert!(!date(1582, 10, 4).is_gregorian());
    assert!(date(1582, 10, 15).is_gregorian());
    assert!(!date(-1999, 12, 31).is_gregorian());
    assert!(date(1583, 1, 1).is_gregorian());
}

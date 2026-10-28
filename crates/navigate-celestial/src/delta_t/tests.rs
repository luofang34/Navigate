use super::*;

fn utc(text: &str) -> Epoch {
    text.parse().unwrap()
}

#[test]
fn utc_keeps_37_leap_seconds_after_leap_seconds_end() {
    // CGPM 2026, Resolution 5: UTC has no leap seconds from 20 May 2027.
    for date in [
        "2017-01-01T00:00:00 UTC",
        "2030-06-01T00:00:00 UTC",
        "2099-12-31T00:00:00 UTC",
    ] {
        let epoch = utc(date);
        let tt_minus_utc = (epoch.to_tt_duration() - epoch.to_utc_duration()).to_seconds();
        assert!(
            (tt_minus_utc - 69.184).abs() < 1e-9,
            "{date}: {tt_minus_utc}"
        );
    }
}

#[test]
fn ut1_minus_utc_is_continuous_where_utc_starts_to_follow_tai() {
    let before = civil_time(utc("1971-12-31T23:59:59 UTC"));
    let after = civil_time(utc("1972-01-01T00:00:00 UTC"));
    assert_eq!(before.ut1_minus_utc_seconds, 0.0);
    // IERS gives UT1 - UTC = -0.04 s on 1 January 1972.
    assert!(after.ut1_minus_utc_seconds.abs() < 0.1, "{after:?}");
    let tt_step = (after.terrestrial - before.terrestrial).to_seconds();
    assert!((tt_step - 1.0).abs() < 0.1, "{tt_step}");
}

#[test]
fn tt_before_1972_is_ut1_plus_delta_t_not_the_utc_of_that_time() {
    // UTC ran slow against UT2 by 0.002592 s per day in 1966 to 1967;
    // TT must not follow it. Delta T was 38.3 s at the start of 1968.
    let reading = utc("1968-01-01T00:00:00 UTC");
    let time = civil_time(reading);
    assert_eq!(time.delta_t.basis, DeltaTBasis::Observed);
    assert!((time.delta_t.seconds - 38.3).abs() < 1.5, "{time:?}");
    let ut1 = time.terrestrial.to_mjd_tt_days() - time.delta_t.seconds / SECONDS_PER_DAY;
    assert!((ut1 - reading.to_mjd_utc_days()).abs() * SECONDS_PER_DAY < 1e-3);
}

#[test]
fn the_basis_follows_measurement_prediction_and_extrapolation() {
    let modelled = civil_time(utc("1943-05-16T20:25:00 UTC")).delta_t;
    assert_eq!(modelled.basis, DeltaTBasis::Modelled);
    assert!((modelled.seconds - 25.6).abs() < 1.0, "{modelled:?}");
    let observed = civil_time(utc("1951-05-16T20:25:00 UTC")).delta_t;
    assert_eq!(observed.basis, DeltaTBasis::Observed);
    assert!((observed.seconds - 29.7).abs() < 0.2, "{observed:?}");
    assert_eq!(
        delta_t(utc("2020-01-01T00:00:00 UTC")).basis,
        DeltaTBasis::Observed
    );
    assert_eq!(
        delta_t(utc("2027-03-01T00:00:00 UTC")).basis,
        DeltaTBasis::Predicted
    );
    let later = delta_t(utc("2040-01-01T00:00:00 UTC"));
    let much_later = delta_t(utc("2099-01-01T00:00:00 UTC"));
    assert_eq!(later.basis, DeltaTBasis::Extrapolated);
    assert!(later.uncertainty_seconds > 1.0);
    assert!(much_later.uncertainty_seconds > later.uncertainty_seconds);
}

#[test]
fn extrapolation_starts_at_the_last_prediction() {
    let [last_mjd, last_value, last_error] = *table::ROWS.last().unwrap();
    let inside = at_mjd(last_mjd);
    let outside = at_mjd(last_mjd + 1.0);
    assert_eq!(inside.seconds, last_value);
    assert!((outside.seconds - last_value).abs() < 0.01, "{outside:?}");
    assert!(outside.uncertainty_seconds >= last_error);
}

#[test]
fn extrapolation_stays_inside_its_stated_error_over_the_table_from_1900() {
    let [last_mjd, _, _] = *table::ROWS.last().unwrap();
    let mut base = 51_544.5 - 100.0 * DAYS_PER_YEAR;
    let span = (table::EXTRAPOLATION_ERROR.len() - 1) as f64;
    let mut checked = 0;
    while base + DAYS_PER_YEAR <= last_mjd {
        let mut years = 0.5;
        while years <= span && base + years * DAYS_PER_YEAR <= last_mjd {
            let mjd = base + years * DAYS_PER_YEAR;
            let error = (extrapolate(base, mjd) - interpolate(mjd).0).abs();
            assert!(
                error <= extrapolation_error(years) + 0.01,
                "from MJD {base}, {years} years: {error} s"
            );
            checked += 1;
            years += 0.5;
        }
        base += DAYS_PER_YEAR / 12.0;
    }
    assert!(checked > 100_000, "{checked}");
}

#[test]
fn the_extrapolation_error_reaches_past_2200() {
    let [last_mjd, _, _] = *table::ROWS.last().unwrap();
    let years_to_2200 =
        (utc("2200-01-01T00:00:00 UTC").to_mjd_tt_days() - last_mjd) / DAYS_PER_YEAR;
    assert!((table::EXTRAPOLATION_ERROR.len() as f64) > years_to_2200);
    assert!(
        table::EXTRAPOLATION_ERROR
            .windows(2)
            .all(|pair| pair[0] <= pair[1])
    );
}

#[test]
fn the_model_meets_the_table_at_its_first_row() {
    let [first_mjd, first_seconds, _] = table::ROWS[0];
    let before = at_mjd(first_mjd - 1.0);
    let at = at_mjd(first_mjd);
    assert_eq!(before.basis, DeltaTBasis::Modelled);
    assert_eq!(at.basis, DeltaTBasis::Observed);
    assert_eq!(at.seconds, first_seconds);
    // The spline meets the historical value within 0.02 s.
    assert!((before.seconds - first_seconds).abs() < 0.02, "{before:?}");
}

#[test]
fn ancient_delta_t_is_hours_with_hours_of_uncertainty() {
    // An epoch in 585 BC (astronomical year -584).
    let delta_t = delta_t(Epoch::from_gregorian(
        -584,
        5,
        28,
        12,
        0,
        0,
        0,
        TimeScale::UTC,
    ));
    assert_eq!(delta_t.basis, DeltaTBasis::Modelled);
    assert!(
        (17_000.0..23_000.0).contains(&delta_t.seconds),
        "{delta_t:?}"
    );
    assert!(
        (400.0..700.0).contains(&delta_t.uncertainty_seconds),
        "{delta_t:?}"
    );
}

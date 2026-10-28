use super::*;

/// Delta T in seconds from Table S6 (version 2020) of the addendum
/// supplement: year, value, and half its rounding step. The table rounds
/// to 10 s before 1700, to 1 s at 1700, and to 0.01 s after.
#[rustfmt::skip]
const PUBLISHED: [[f64; 3]; 22] = [
    [-720.0, 20_370.0, 5.0], [-700.0, 20_050.0, 5.0], [-500.0, 16_940.0, 5.0],
    [-300.0, 14_080.0, 5.0], [-100.0, 11_560.0, 5.0], [0.0, 10_440.0, 5.0],
    [200.0, 8_420.0, 5.0], [400.0, 6_540.0, 5.0], [600.0, 4_650.0, 5.0],
    [800.0, 2_940.0, 5.0], [1000.0, 1_650.0, 5.0], [1200.0, 910.0, 5.0],
    [1400.0, 480.0, 5.0], [1600.0, 110.0, 5.0], [1700.0, 14.0, 0.5],
    [1800.0, 18.37, 0.005], [1810.0, 15.68, 0.005], [1850.0, 9.34, 0.005],
    [1880.0, -3.21, 0.005], [1900.0, -1.98, 0.005], [1920.0, 21.62, 0.005],
    [1940.0, 24.43, 0.005],
];

#[test]
fn spline_matches_the_published_values() {
    for [year, published, half_step] in PUBLISHED {
        let error = (value(year) - published).abs();
        assert!(error <= half_step, "{year}: {} vs {published}", value(year));
    }
}

#[test]
fn spline_rows_join_without_steps() {
    for pair in SPLINE.windows(2) {
        let (end, start) = (pair[0][1], pair[1][0]);
        assert_eq!(end, start);
        let [_, _, a0, a1, a2, a3] = pair[0];
        // The coefficients are rounded to 0.001 s.
        assert!((a0 + a1 + a2 + a3 - pair[1][2]).abs() < 0.004, "{end}");
    }
}

#[test]
fn the_parabola_joins_the_spline_with_value_and_slope() {
    let first = SPLINE[0][0];
    let step = 1e-6;
    for year in [first, JOIN_START] {
        let (before, at, after) = (value(year - step), value(year), value(year + step));
        assert!((after - before).abs() < 0.01, "{year}: value step");
        let slope_change = (after - at) / step - (at - before) / step;
        assert!(slope_change.abs() < 0.01, "{year}: slope step");
    }
    // Before the join, the value is the long-term parabola.
    let u: f64 = (-2000.0 - 1825.0) / 100.0;
    assert!((value(-2000.0) - (-320.0 + 32.5 * u * u)).abs() < 1e-9);
    // The parabola is above the spline at -720.
    let gap = parabola(first).0 - spline(first).0;
    assert!((350.0..370.0).contains(&gap), "{gap}");
}

#[test]
fn uncertainty_follows_morrison_and_stephenson() {
    assert_eq!(uncertainty(-500.0), 430.0);
    assert_eq!(uncertainty(-450.0), 410.0);
    let u: f64 = (-2000.0 - 1820.0) / 100.0;
    assert!((uncertainty(-2000.0) - 0.8 * u * u).abs() < 1e-9);
    // The fit agrees with the first table row within a few percent.
    assert!((uncertainty(-500.0 - 1e-9) - 430.0).abs() < 15.0);
}

#[test]
fn uncertainty_covers_the_usno_historical_values_from_1800() {
    // USNO historic_deltat.data: the two values in each half century that
    // differ most from the spline.
    let usno = [
        [1801.0, 12.0],
        [1801.5, 11.9],
        [1853.0, 7.03],
        [1854.0, 7.15],
        [1906.5, 5.73],
        [1907.0, 6.29],
    ];
    for [year, usno] in usno {
        let (seconds, uncertainty) = at_year(year);
        let difference = (seconds - usno).abs();
        assert!(
            difference < uncertainty,
            "{year}: {difference} > {uncertainty}"
        );
    }
    // The constants change at the half centuries.
    assert_eq!(uncertainty(1849.9), 6.1);
    assert_eq!(uncertainty(1850.0), 3.3);
    assert_eq!(uncertainty(1900.0), 1.3);
}

//! Delta T before 1948, from Stephenson, Morrison and Hohenkerk (2016) with
//! the 2020 addendum of Morrison, Stephenson, Hohenkerk and Zawilski (2021).
//!
//! From -720 the value is their cubic spline, Table S15 (version 2020) of
//! the addendum supplement, CC BY 4.0. The spline meets the USNO historical
//! values at 1948.0 within 0.02 s; from 1948 the measured table is better,
//! because the spline smooths the atomic-time values after 1955 by up to
//! 0.7 s. Before -720 it is their long-term
//! parabola -320 + 32.5 u^2 seconds, where u is centuries from 1825. A cubic
//! from -1520 to -720 joins the two with the value and the slope of each,
//! as in Skyfield, because the parabola is 360 s above the spline at -720.
//!
//! The spline comes from eclipse records before 1600 reduced with the
//! analytical lunar ephemeris j=2 (tidal acceleration -26.00 arcseconds per
//! century squared) and from occultations after 1600 reduced with JPL DE430.
//! The authors found DE430 within about 10 arcseconds of j=2 back to 700 BC,
//! about 20 s of Delta T, and give the values for use with DE430 or a Moon
//! with a tidal acceleration near -25.82 arcseconds per century squared,
//! without a tidal correction. JPL DE441 is not such a Moon: before 500 it
//! departs from DE431, the long version of DE430, by 65 arcseconds at -762.
//!
//! The uncertainty is the standard error of Morrison and Stephenson (2004)
//! to 1750: before -500 it is 0.8 u^2 seconds with u in centuries from 1820,
//! and from -500 it interpolates their table. From 1800 to 1948 it is the
//! largest difference of the spline from the USNO historical values in each
//! half century, which is larger than either standard error: 6.1 s from
//! 1800, 3.3 s from 1850, and 1.3 s from 1900.

/// Row: first and last year, then the coefficients a0 to a3 of
/// a0 + a1 t + a2 t^2 + a3 t^3, with t from 0 to 1 over the row.
#[rustfmt::skip]
const SPLINE: [[f64; 6]; 35] = [
    [-720.0, -100.0, 20_371.848, -9_999.586, 776.247, 409.160],
    [-100.0, 400.0, 11_557.668, -5_822.270, 1_303.151, -503.433],
    [400.0, 1000.0, 6_535.116, -5_671.519, -298.291, 1_085.087],
    [1000.0, 1150.0, 1_650.393, -753.210, 184.811, -25.346],
    [1150.0, 1300.0, 1_056.647, -459.628, 108.771, -24.641],
    [1300.0, 1500.0, 681.149, -421.345, 61.953, -29.414],
    [1500.0, 1600.0, 292.343, -192.841, -6.572, 16.197],
    [1600.0, 1650.0, 109.127, -78.697, 10.505, 3.018],
    [1650.0, 1720.0, 43.952, -68.089, 38.333, -2.127],
    [1720.0, 1800.0, 12.068, 2.507, 41.731, -37.939],
    [1800.0, 1810.0, 18.367, -3.481, -1.126, 1.918],
    [1810.0, 1820.0, 15.678, 0.021, 4.629, -3.812],
    [1820.0, 1830.0, 16.516, -2.157, -6.806, 3.250],
    [1830.0, 1840.0, 10.804, -6.018, 2.944, -0.096],
    [1840.0, 1850.0, 7.634, -0.416, 2.658, -0.539],
    [1850.0, 1855.0, 9.338, 1.642, 0.261, -0.883],
    [1855.0, 1860.0, 10.357, -0.486, -2.389, 1.558],
    [1860.0, 1865.0, 9.040, -0.591, 2.284, -2.477],
    [1865.0, 1870.0, 8.255, -3.456, -5.148, 2.720],
    [1870.0, 1875.0, 2.371, -5.593, 3.011, -0.914],
    [1875.0, 1880.0, -1.126, -2.314, 0.269, -0.039],
    [1880.0, 1885.0, -3.210, -1.893, 0.152, 0.563],
    [1885.0, 1890.0, -4.388, 0.101, 1.842, -1.438],
    [1890.0, 1895.0, -3.884, -0.531, -2.474, 1.871],
    [1895.0, 1900.0, -5.017, 0.134, 3.138, -0.232],
    [1900.0, 1905.0, -1.977, 5.715, 2.443, -1.257],
    [1905.0, 1910.0, 4.923, 6.828, -1.329, 0.720],
    [1910.0, 1915.0, 11.142, 6.330, 0.831, -0.825],
    [1915.0, 1920.0, 17.479, 5.518, -1.643, 0.262],
    [1920.0, 1925.0, 21.617, 3.020, -0.856, 0.008],
    [1925.0, 1930.0, 23.789, 1.333, -0.831, 0.127],
    [1930.0, 1935.0, 24.418, 0.052, -0.449, 0.142],
    [1935.0, 1940.0, 24.164, -0.419, -0.022, 0.702],
    [1940.0, 1945.0, 24.426, 1.645, 2.086, -1.106],
    [1945.0, 1950.0, 27.050, 2.499, -1.232, 0.614],
];
/// The first year of the join from the parabola to the spline.
const JOIN_START: f64 = -1520.0;

/// Year and standard error in seconds: Morrison and Stephenson (2004) to
/// 1750, then from 1800 a constant in each half century, the largest
/// difference of the spline from the USNO historical values there rounded
/// up. A repeated year starts a new constant.
#[rustfmt::skip]
const STANDARD_ERROR: [[f64; 2]; 30] = [
    [-500.0, 430.0], [-400.0, 390.0], [-300.0, 360.0], [-200.0, 330.0],
    [-100.0, 290.0], [0.0, 260.0], [100.0, 240.0], [200.0, 210.0],
    [300.0, 180.0], [400.0, 160.0], [500.0, 140.0], [600.0, 120.0],
    [700.0, 100.0], [800.0, 80.0], [900.0, 70.0], [1000.0, 55.0],
    [1100.0, 40.0], [1200.0, 30.0], [1300.0, 20.0], [1400.0, 20.0],
    [1500.0, 20.0], [1600.0, 20.0], [1700.0, 5.0], [1750.0, 2.0],
    [1800.0, 6.1], [1850.0, 6.1], [1850.0, 3.3], [1900.0, 3.3],
    [1900.0, 1.3], [1948.0, 1.3],
];

/// Delta T and its uncertainty in seconds at the decimal year `year`.
pub(super) fn at_year(year: f64) -> (f64, f64) {
    (value(year), uncertainty(year))
}

fn value(year: f64) -> f64 {
    let first = SPLINE.first().map_or(f64::NEG_INFINITY, |row| row[0]);
    if year >= first {
        return spline(year).0;
    }
    if year >= JOIN_START {
        let (end, end_slope) = spline(first);
        let (start, start_slope) = parabola(JOIN_START);
        return hermite(
            JOIN_START,
            first,
            [start, start_slope],
            [end, end_slope],
            year,
        );
    }
    parabola(year).0
}

/// The spline value and its slope in seconds per year.
fn spline(year: f64) -> (f64, f64) {
    let row = SPLINE
        .iter()
        .find(|row| year <= row[1])
        .or(SPLINE.last())
        .copied()
        .unwrap_or_default();
    let [start, end, a0, a1, a2, a3] = row;
    let width = end - start;
    let t = (year - start) / width;
    (
        a0 + t * (a1 + t * (a2 + t * a3)),
        (a1 + t * (2.0 * a2 + t * 3.0 * a3)) / width,
    )
}

/// The long-term parabola and its slope in seconds per year.
fn parabola(year: f64) -> (f64, f64) {
    let u = (year - 1825.0) / 100.0;
    (-320.0 + 32.5 * u * u, 0.65 * u)
}

/// The cubic from `start` to `end` with the given values and slopes.
fn hermite(start: f64, end: f64, [y0, slope0]: [f64; 2], [y1, slope1]: [f64; 2], year: f64) -> f64 {
    let width = end - start;
    let t = (year - start) / width;
    let (m0, m1) = (slope0 * width, slope1 * width);
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * m1
}

fn uncertainty(year: f64) -> f64 {
    let after = STANDARD_ERROR.partition_point(|&[row_year, _]| row_year <= year);
    match (
        after.checked_sub(1).and_then(|i| STANDARD_ERROR.get(i)),
        STANDARD_ERROR.get(after),
    ) {
        (Some(&[year0, error0]), Some(&[year1, error1])) => {
            error0 + (error1 - error0) * (year - year0) / (year1 - year0)
        }
        (None, _) => 0.8 * ((year - 1820.0) / 100.0).powi(2),
        (Some(&[_, error]), None) => error,
    }
}

#[cfg(test)]
mod tests;

//! Delta T before the USNO table, from the polynomials of Espenak and Meeus
//! (NASA Five Millennium Canon of Solar Eclipses, 2006).
//!
//! The polynomials fit the Morrison and Stephenson (2004) values. Those
//! values assume a lunar tidal acceleration of -26 arcseconds per century
//! squared. The Canon uses -25.858, and so does this crate's long-term Moon,
//! so each value gets the Canon correction c = -0.91072 (-25.858 + 26) u^2
//! seconds, where u is centuries from 1955. Before -500 the uncertainty is
//! 0.8 u^2 seconds with u in centuries from 1820, the fit of Morrison and
//! Stephenson; from -500 to 1750 it interpolates their standard errors.

/// Year, Morrison and Stephenson (2004) standard error in seconds.
#[rustfmt::skip]
const STANDARD_ERROR: [[f64; 2]; 25] = [
    [-500.0, 430.0], [-400.0, 390.0], [-300.0, 360.0], [-200.0, 330.0],
    [-100.0, 290.0], [0.0, 260.0], [100.0, 240.0], [200.0, 210.0],
    [300.0, 180.0], [400.0, 160.0], [500.0, 140.0], [600.0, 120.0],
    [700.0, 100.0], [800.0, 80.0], [900.0, 70.0], [1000.0, 55.0],
    [1100.0, 40.0], [1200.0, 30.0], [1300.0, 20.0], [1400.0, 20.0],
    [1500.0, 20.0], [1600.0, 20.0], [1700.0, 5.0], [1750.0, 2.0],
    [1800.0, 1.0],
];

/// Delta T and its uncertainty in seconds at the decimal year `year`.
pub(super) fn at_year(year: f64) -> (f64, f64) {
    let correction = -0.910_72 * (-25.858 + 26.0) * ((year - 1955.0) / 100.0).powi(2);
    (polynomial(year) + correction, uncertainty(year))
}

fn polynomial(year: f64) -> f64 {
    let horner = |u: f64, coefficients: &[f64]| {
        coefficients
            .iter()
            .rev()
            .fold(0.0, |total, coefficient| total * u + coefficient)
    };
    if year < -500.0 {
        let u = (year - 1820.0) / 100.0;
        -20.0 + 32.0 * u * u
    } else if year < 500.0 {
        horner(
            year / 100.0,
            &[
                10_583.6,
                -1_014.41,
                33.783_11,
                -5.952_053,
                -0.179_845_2,
                0.022_174_192,
                0.009_031_652_1,
            ],
        )
    } else if year < 1600.0 {
        horner(
            (year - 1000.0) / 100.0,
            &[
                1_574.2,
                -556.01,
                71.234_72,
                0.319_781,
                -0.850_346_3,
                -0.005_050_998,
                0.008_357_207_3,
            ],
        )
    } else if year < 1700.0 {
        horner(year - 1600.0, &[120.0, -0.980_8, -0.015_32, 1.0 / 7_129.0])
    } else {
        horner(
            year - 1700.0,
            &[
                8.83,
                0.160_3,
                -0.005_928_5,
                0.000_133_36,
                -1.0 / 1_174_000.0,
            ],
        )
    }
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

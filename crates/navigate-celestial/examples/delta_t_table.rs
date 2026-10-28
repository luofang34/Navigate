//! Write `src/delta_t/table.rs`: Delta T (TT - UT1) from 1948.0 to the IERS
//! prediction horizon.
//!
//! ```text
//! cargo run -p navigate-celestial --example delta_t_table -- \
//!     historic_deltat.data deltat.data finals2000A.all \
//!     crates/navigate-celestial/src/delta_t/table.rs
//! ```
//!
//! The inputs are USNO files from <https://maia.usno.navy.mil/ser7/>:
//! `historic_deltat.data` gives Delta T twice a year with its error up to
//! 1984, `deltat.data` gives the measured Delta T on the first day of each
//! month from 1973, and `finals2000A.all` gives IERS UT1 - UTC with its error,
//! including the predictions for about one year. A predicted row becomes
//! Delta T with TAI - UTC = 37 s, which stays fixed because UTC has no leap
//! seconds from 20 May 2027 (CGPM 2026, Resolution 5).
//!
//! The table uses the historical rows from 1948.0 to 1973.0, the monthly
//! rows from 1973-02-01, and then the predicted rows on the first day of
//! each month. A monthly row gets an uncertainty of 0.01 s: linear
//! interpolation between monthly rows differs from the daily IERS values by
//! at most 0.0068 s from 1973 to 2026. Before 1948.0 the crate uses the
//! spline of Stephenson, Morrison and Hohenkerk, which meets the historical
//! rows there within 0.02 s.
//!
//! After the last row, Delta T continues with the slope of the last year.
//! The error of that rule after n years is the largest error it gave: from
//! each month, extrapolated in half-year steps, and compared with the
//! historical and later rows from 1799.5. Up to 80 years the months start in
//! 1900. From 81 to 175 years, which reach past 2200, only the months from
//! 1802 give enough history.

use std::{fmt::Write as _, path::Path};

type Error = Box<dyn std::error::Error>;

const YEAR: f64 = 365.25;
const J2000_MJD: f64 = 51_544.5;
const HISTORY_START_YEAR: f64 = 1799.5;
const MONTHLY_START_YEAR: f64 = 1973.0;
const TABLE_START_YEAR: f64 = 1948.0;
/// TT - TAI plus TAI - UTC, in seconds, for a predicted row.
const TT_MINUS_UTC: f64 = 32.184 + 37.0;
const MONTHLY_UNCERTAINTY: f64 = 0.01;

/// Modified Julian date, Delta T in seconds, and its uncertainty in seconds.
type Row = (f64, f64, f64);

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [historic, monthly, finals, out] = args.as_slice() else {
        return Err("usage: delta_t_table <historic_deltat.data> <deltat.data> <finals2000A.all> <table.rs>".into());
    };
    let mut rows = historic_rows(&read(historic)?)?;
    rows.extend(monthly_rows(&read(monthly)?)?);
    let first_predicted = append_predictions(&mut rows, &read(finals)?)?;
    if !rows.windows(2).all(|pair| pair[0].0 < pair[1].0) {
        return Err("the rows are not in time order".into());
    }
    let errors = extrapolation_errors(&rows)?;
    let start = J2000_MJD + (TABLE_START_YEAR - 2000.0) * YEAR;
    let table: Vec<Row> = rows.into_iter().filter(|row| row.0 >= start).collect();
    std::fs::write(out, text(first_predicted, &errors, &table)?)
        .map_err(|error| format!("{out}: {error}").into())
}

fn read(path: &str) -> Result<String, Error> {
    std::fs::read_to_string(Path::new(path)).map_err(|error| format!("{path}: {error}").into())
}

fn mjd(year: i64, month: i64, day: i64) -> f64 {
    let a = (14 - month).div_euclid(12);
    let y = year + 4800 - a;
    let m = month + 12 * a - 3;
    let jdn = day + (153 * m + 2).div_euclid(5) + 365 * y + y.div_euclid(4) - y.div_euclid(100)
        + y.div_euclid(400)
        - 32045;
    jdn as f64 - 2_400_001.0
}

fn historic_rows(text: &str) -> Result<Vec<Row>, Error> {
    let mut rows = Vec::new();
    for line in text.lines().skip(2) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let [year, delta_t, error, ..] = fields.as_slice() else {
            return Err(format!("historic row {line:?} has too few fields").into());
        };
        let year: f64 = year.parse()?;
        if (HISTORY_START_YEAR..=MONTHLY_START_YEAR).contains(&year) {
            rows.push((
                J2000_MJD + (year - 2000.0) * YEAR,
                delta_t.parse()?,
                error.parse()?,
            ));
        }
    }
    Ok(rows)
}

fn monthly_rows(text: &str) -> Result<Vec<Row>, Error> {
    let mut rows = Vec::new();
    for line in text.lines() {
        if let [year, month, day, delta_t] = line.split_whitespace().collect::<Vec<_>>().as_slice()
        {
            rows.push((
                mjd(year.parse()?, month.parse()?, day.parse()?),
                delta_t.parse()?,
                MONTHLY_UNCERTAINTY,
            ));
        }
    }
    Ok(rows)
}

/// Columns `start..end` of a finals line, trimmed, as far as the line goes.
fn column(line: &str, start: usize, end: usize) -> &str {
    line.get(start..end.min(line.len())).unwrap_or("").trim()
}

/// Append the predicted rows on the first day of each month after the last
/// row, and return the date of the first one.
fn append_predictions(rows: &mut Vec<Row>, finals: &str) -> Result<f64, Error> {
    let mut first_predicted = None;
    // The length test counts the line end, as the format does.
    for line in finals.split_inclusive('\n') {
        if line.len() < 78 || line.get(57..58) != Some("P") || line.get(4..6) != Some(" 1") {
            continue;
        }
        let year: i64 = column(line, 0, 2).parse()?;
        let century = if year < 73 { 2000 } else { 1900 };
        let day = mjd(year + century, column(line, 2, 4).parse()?, 1);
        if rows.last().is_some_and(|last| day <= last.0) {
            continue;
        }
        first_predicted.get_or_insert(day);
        let ut1_minus_utc: f64 = column(line, 58, 68).parse()?;
        rows.push((
            day,
            TT_MINUS_UTC - ut1_minus_utc,
            column(line, 68, 78).parse()?,
        ));
    }
    first_predicted.ok_or_else(|| "finals2000A.all has no predicted rows".into())
}

fn interpolate(rows: &[Row], day: f64) -> Result<f64, Error> {
    rows.windows(2)
        .find_map(|pair| {
            let ((d0, v0, _), (d1, v1, _)) = (pair[0], pair[1]);
            (d0 <= day && day <= d1).then(|| v0 + (v1 - v0) * (day - d0) / (d1 - d0))
        })
        .ok_or_else(|| format!("no rows around MJD {day}").into())
}

/// The largest error of extrapolation after each whole number of years, up
/// to `years`, from the months that start at `first_base`.
fn backtest(rows: &[Row], first_base: f64, years: usize) -> Result<Vec<f64>, Error> {
    let last = rows.last().map_or(f64::NEG_INFINITY, |row| row.0);
    let mut errors = vec![0.0_f64; years + 1];
    let mut base = first_base;
    while base + YEAR <= last {
        let value = interpolate(rows, base)?;
        let slope = value - interpolate(rows, base - YEAR)?;
        for step in 1..2 * errors.len() - 1 {
            let half_years = step as f64;
            let day = base + half_years * YEAR / 2.0;
            if day > last {
                break;
            }
            let error = (value + slope * half_years / 2.0 - interpolate(rows, day)?).abs();
            let whole = step.div_ceil(2);
            errors[whole] = errors[whole].max(error);
        }
        base += YEAR / 12.0;
    }
    Ok(errors)
}

fn extrapolation_errors(rows: &[Row]) -> Result<Vec<f64>, Error> {
    let mut errors = backtest(rows, J2000_MJD - 100.0 * YEAR, 80)?;
    errors.extend(
        backtest(rows, J2000_MJD - 198.0 * YEAR, 175)?
            .into_iter()
            .skip(81),
    );
    for n in 1..errors.len() {
        errors[n] = errors[n].max(errors[n - 1]);
    }
    // Round up to 0.01 s.
    Ok(errors
        .into_iter()
        .map(|error| -(-error * 100.0).floor() / 100.0)
        .collect())
}

fn text(first_predicted: f64, errors: &[f64], rows: &[Row]) -> Result<String, Error> {
    let mut text = String::from(
        "//! Delta T (TT - UT1) rows, generated by `examples/delta_t_table.rs`. Do not edit.\n\n\
         /// Modified Julian date of the first row that is an IERS prediction.\n",
    );
    writeln!(
        text,
        "pub(super) const FIRST_PREDICTED_MJD: f64 = {first_predicted:?};\n"
    )?;
    text.push_str(
        "/// The largest error, in seconds, of extrapolation from the last row for up\n\
         /// to n whole years: a backtest over the rows from 1900 up to 80 years, and\n\
         /// over the rows from 1802 after that.\n\
         #[rustfmt::skip]\n",
    );
    writeln!(
        text,
        "pub(super) const EXTRAPOLATION_ERROR: [f64; {}] = [",
        errors.len()
    )?;
    for chunk in errors.chunks(9) {
        let values: Vec<String> = chunk.iter().map(|error| format!("{error:?},")).collect();
        writeln!(text, "    {}", values.join(" "))?;
    }
    text.push_str(
        "];\n\n\
         /// Modified Julian date (TT), Delta T in seconds, and its uncertainty in seconds.\n\
         #[rustfmt::skip]\n\
         #[allow(clippy::approx_constant, reason = \"measured values; one is close to TAU\")]\n",
    );
    writeln!(
        text,
        "pub(super) static ROWS: [[f64; 3]; {}] = [",
        rows.len()
    )?;
    for chunk in rows.chunks(3) {
        let values: Vec<String> = chunk
            .iter()
            .map(|(day, value, error)| format!("[{day:?}, {value:?}, {error:?}],"))
            .collect();
        writeln!(text, "    {}", values.join(" "))?;
    }
    text.push_str("];\n");
    Ok(text)
}

//! Delta T (TT - UT1), and the TT and UT1 that a civil clock reading gives.
//!
//! The Earth rotation angle follows UT1. The ephemeris uses TT. Delta T
//! connects them, and a table gives it from 1849 to the last IERS prediction:
//! historical values to 1972, IERS measurements from 1973, then the IERS
//! predictions for about one year. `fixtures/delta_t_table.py` writes the
//! table from the USNO files. After the last prediction, Delta T continues
//! with the slope of the last year, and its uncertainty grows as a backtest of
//! that rule on the table from 1900 measured.
//!
//! A civil clock reading means UT1 before 1972, because civil time (GMT)
//! then followed the Earth's rotation. From 1972 it means UTC, which TAI
//! fixes: the leap seconds to 2017 give TAI - UTC = 37 s, and from
//! 20 May 2027 UTC has no leap seconds (CGPM 2026, Resolution 5). UT1 - UTC
//! is then free to grow, up to 3600 s, and Delta T gives it.

use crate::Epoch;
use hifitime::TimeScale;

mod table;

const DAYS_PER_YEAR: f64 = 365.25;
const SECONDS_PER_DAY: f64 = 86_400.0;

/// How a value of Delta T is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaTBasis {
    /// A historical value before 1973, or an IERS measurement from 1973.
    Observed,
    /// An IERS prediction, for about one year after the last measurement.
    Predicted,
    /// After the last IERS prediction: the last value continues with the
    /// slope of the last year.
    Extrapolated,
}

/// Delta T (TT - UT1) at one epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeltaT {
    /// TT - UT1, in seconds.
    pub seconds: f64,
    /// Uncertainty of `seconds`, in seconds. One second of Delta T turns
    /// the Earth-fixed directions by 15 arcseconds about the pole.
    pub uncertainty_seconds: f64,
    /// How the value is known.
    pub basis: DeltaTBasis,
}

/// The TT and UT1 that a civil clock reading gives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CivilTime {
    /// The same instant in TT, for the ephemeris.
    pub terrestrial: Epoch,
    /// UT1 - UTC, in seconds. It is zero before 1972, where the reading is UT1.
    pub ut1_minus_utc_seconds: f64,
    /// Delta T at the instant.
    pub delta_t: DeltaT,
}

/// Delta T at `epoch`, from the epoch's TT.
pub fn delta_t(epoch: Epoch) -> DeltaT {
    at_mjd(epoch.to_mjd_tt_days())
}

/// The TT and UT1 that the civil clock reading `epoch` gives.
///
/// Before 1972 the reading is UT1, and TT is UT1 + Delta T. The UTC that
/// hifitime keeps before 1972 is not used for TT, because it omits the rate
/// terms of the 1961 to 1971 UTC. From 1972 the reading is UTC, TT comes
/// from TAI, and UT1 is TT - Delta T.
pub fn civil_time(epoch: Epoch) -> CivilTime {
    if epoch < Epoch::from_gregorian_utc_at_midnight(1972, 1, 1) {
        let ut1_mjd = epoch.to_mjd_utc_days();
        let delta_t = at_mjd(ut1_mjd);
        CivilTime {
            terrestrial: Epoch::from_mjd_in_time_scale(
                ut1_mjd + delta_t.seconds / SECONDS_PER_DAY,
                TimeScale::TT,
            ),
            ut1_minus_utc_seconds: 0.0,
            delta_t,
        }
    } else {
        let delta_t = delta_t(epoch);
        let tt_minus_utc = (epoch.to_tt_duration() - epoch.to_utc_duration()).to_seconds();
        CivilTime {
            terrestrial: epoch,
            ut1_minus_utc_seconds: tt_minus_utc - delta_t.seconds,
            delta_t,
        }
    }
}

/// UT1 as a Julian date, for the TT epoch `epoch`.
pub(crate) fn ut1_julian_days(epoch: Epoch) -> f64 {
    epoch.to_jde_tt_days() - delta_t(epoch).seconds / SECONDS_PER_DAY
}

fn at_mjd(mjd: f64) -> DeltaT {
    let rows = &table::ROWS;
    match rows.last() {
        Some(&[last_mjd, _, last_error]) if mjd > last_mjd => DeltaT {
            seconds: extrapolate(last_mjd, mjd),
            uncertainty_seconds: last_error + extrapolation_error((mjd - last_mjd) / DAYS_PER_YEAR),
            basis: DeltaTBasis::Extrapolated,
        },
        _ => {
            let (seconds, uncertainty_seconds, upper_mjd) = interpolate(mjd);
            DeltaT {
                seconds,
                uncertainty_seconds,
                basis: if upper_mjd >= table::FIRST_PREDICTED_MJD {
                    DeltaTBasis::Predicted
                } else {
                    DeltaTBasis::Observed
                },
            }
        }
    }
}

/// Delta T at `mjd` from the value and the last year's slope at `base_mjd`.
fn extrapolate(base_mjd: f64, mjd: f64) -> f64 {
    let value = interpolate(base_mjd).0;
    let slope = value - interpolate(base_mjd - DAYS_PER_YEAR).0;
    value + slope * (mjd - base_mjd) / DAYS_PER_YEAR
}

/// The backtest error for `years` after the last row, rounded up to whole
/// years. The table reaches past 2100, the end of the ephemeris range; after
/// it the error grows at its last yearly rate.
fn extrapolation_error(years: f64) -> f64 {
    let whole = years.max(0.0).ceil() as usize;
    match table::EXTRAPOLATION_ERROR.get(whole) {
        Some(&error) => error,
        None => {
            let errors = &table::EXTRAPOLATION_ERROR;
            let last = errors.last().copied().unwrap_or(0.0);
            let rate = last - errors.iter().rev().nth(1).copied().unwrap_or(0.0);
            last + rate * (whole + 1 - errors.len()) as f64
        }
    }
}

/// Linear interpolation of Delta T, the larger uncertainty of the two rows,
/// and the date of the later row. Outside the table it gives the nearest row.
fn interpolate(mjd: f64) -> (f64, f64, f64) {
    let rows = &table::ROWS;
    let upper = rows.partition_point(|&[row_mjd, _, _]| row_mjd < mjd);
    match (
        upper.checked_sub(1).and_then(|i| rows.get(i)),
        rows.get(upper),
    ) {
        (Some(&[mjd0, value0, error0]), Some(&[mjd1, value1, error1])) => {
            let fraction = (mjd - mjd0) / (mjd1 - mjd0);
            (
                value0 + (value1 - value0) * fraction,
                error0.max(error1),
                mjd1,
            )
        }
        (None, Some(&[mjd1, value1, error1])) | (Some(&[mjd1, value1, error1]), None) => {
            (value1, error1, mjd1)
        }
        (None, None) => (0.0, 0.0, f64::INFINITY),
    }
}

#[cfg(test)]
mod tests;

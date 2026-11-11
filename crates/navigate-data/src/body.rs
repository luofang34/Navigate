//! The body, its ellipsoid, the horizontal datum, and dates.
//!
//! Each enumeration has an `Unsupported` value that decoding gives for a
//! name that this crate does not know. Validation refuses it with
//! [`DataError::Unsupported`], so an unknown value never reaches a reader.

use serde::{Deserialize, Serialize};

use crate::DataError;

/// The body that the data describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Body {
    /// The Earth.
    Earth,
    /// A body that this crate does not know.
    #[serde(other)]
    Unsupported,
}

/// The reference ellipsoid of the body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EllipsoidId {
    /// WGS84.
    Wgs84,
    /// An ellipsoid that this crate does not know.
    #[serde(other)]
    Unsupported,
}

/// The horizontal datum of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HorizontalDatum {
    /// WGS84, as the provider realises it.
    Wgs84,
    /// A datum that this crate does not know.
    #[serde(other)]
    Unsupported,
}

/// Whole days from 2000-01-01 (day 0), in the proleptic Gregorian calendar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DayNumber(pub i32);

impl DayNumber {
    /// The day of a calendar date.
    ///
    /// # Errors
    ///
    /// Refuses a month outside 1 to 12 and a day outside the month.
    pub fn from_ymd(year: i32, month: u32, day: u32) -> Result<Self, DataError> {
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let days_in = [
            31,
            if leap { 29 } else { 28 },
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ];
        let valid = month
            .checked_sub(1)
            .and_then(|m| days_in.get(m as usize))
            .is_some_and(|&n| (1..=n).contains(&day));
        if !valid {
            return Err(DataError::Invalid {
                field: "date",
                detail: alloc::format!("{year}-{month}-{day}"),
            });
        }
        // Days from 1970-01-01 (H. Hinnant, "days_from_civil").
        let y = i64::from(year) - i64::from(month <= 2);
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let m = i64::from(month);
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(day) - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        let unix_days = era * 146_097 + doe - 719_468;
        Ok(Self((unix_days - 10_957) as i32))
    }
}

#[cfg(test)]
mod tests;

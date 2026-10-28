//! Calendar dates as the catalogues give them.

#[cfg(test)]
mod tests;

/// A date in the calendar of the catalogues: Julian before 1582 October 15
/// and Gregorian from that day. Years are astronomical: year 0 is 1 BCE and
/// year -1999 is 2000 BCE.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CalendarDate {
    /// The astronomical year.
    pub year: i16,
    /// The month, 1 to 12.
    pub month: u8,
    /// The day of the month, from 1.
    pub day: u8,
}

impl CalendarDate {
    /// Whether the date is in the Gregorian calendar, that is, on or after
    /// 1582 October 15.
    #[must_use]
    pub const fn is_gregorian(self) -> bool {
        let (year, month, day) = (self.year, self.month, self.day);
        year > 1582 || (year == 1582 && (month > 10 || (month == 10 && day >= 15)))
    }

    /// The Julian date at 0h of the date.
    #[must_use]
    pub fn julian_date_at_midnight(self) -> f64 {
        let (mut year, mut month) = (i64::from(self.year), i64::from(self.month));
        if month <= 2 {
            year -= 1;
            month += 12;
        }
        let gregorian = if self.is_gregorian() {
            let century = year.div_euclid(100);
            2 - century + century.div_euclid(4)
        } else {
            0
        };
        // Integer forms of floor(365.25 (year + 4716)) and floor(30.6001 (month + 1)).
        let days = (1461 * (year + 4716)).div_euclid(4)
            + (306_001 * (month + 1)).div_euclid(10_000)
            + i64::from(self.day)
            + gregorian;
        days as f64 - 1524.5
    }
}

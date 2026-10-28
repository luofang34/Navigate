//! The solar and lunar eclipses of the Five Millennium Canon, -1999 to 3000,
//! as embedded data.
//!
//! The crate embeds the two catalogues of the Five Millennium Canon of Solar
//! Eclipses and the Five Millennium Canon of Lunar Eclipses (Espenak and
//! Meeus), 11,898 solar and 12,064 lunar eclipses, with every column of the
//! catalogue pages. It reads no file and no clock, it does not allocate, and
//! it has no dependency. It builds without the standard library.
//!
//! Eclipse Predictions by Fred Espenak (NASA's GSFC).
//!
//! Times are TD, which is TT. The catalogues give each eclipse the Delta T
//! that they used, in [`SolarEclipse::delta_t_s`] and
//! [`LunarEclipse::delta_t_s`]. To compare the geometry with another
//! ephemeris, use the TD of greatest eclipse. A different model of Delta T
//! changes only UT and the longitudes.
//!
//! ```
//! use eclipse_canon::{SolarKind, solar_eclipses};
//!
//! let total = solar_eclipses()
//!     .filter(|eclipse| eclipse.date.year == 2024 && eclipse.kind == SolarKind::Total)
//!     .next()
//!     .unwrap();
//! assert_eq!((total.date.month, total.date.day), (4, 8));
//! assert_eq!(total.type_code.as_str(), "T");
//! assert_eq!(total.central_duration_s, Some(268));
//! ```

#![no_std]

mod code;
mod date;
mod lunar;
mod solar;
mod table;

pub use code::Code;
pub use date::CalendarDate;
pub use lunar::{LunarEclipse, LunarKind, lunar_eclipses};
pub use solar::{SolarEclipse, SolarKind, solar_eclipses};

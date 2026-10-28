//! Celestial directions for a specified observer and reception event.
//!
//! Callers supply time, states, and ephemeris data. No function reads a clock
//! or loads a file. Earth surface coordinates belong in a caller adapter.
//! [`AnalyticEphemeris`] supplies Sun, Earth, and Moon states without kernel
//! files. [`sun_ecef`] and [`moon_ecef`] give their apparent directions in
//! Earth-fixed axes from a civil clock reading, [`sun_ecef_tt`] and
//! [`moon_ecef_tt`] from an instant such as TT, and [`stars_ecef`] and
//! [`stars_ecef_tt`] give the bright stars. [`civil_time`] and
//! [`delta_t`](fn@delta_t) give the TT and UT1 of a civil clock reading.
//!
//! The Sun, the Moon, and the stars cover 1800 to 2200. The `long-range`
//! feature extends them to -2000 to 3000 with larger series tables fitted to
//! JPL DE441, and it improves the Sun and the Moon to 0.3 arcsecond in the
//! modern era.

mod analytic;
#[cfg(feature = "anise")]
mod anise_adapter;
mod delta_t;
mod earth_fixed;
mod earth_rotation;
mod error;
mod observation;
mod precession;
mod stars;
mod state;

pub use analytic::{AnalyticEphemeris, AnalyticError};
#[cfg(feature = "anise")]
pub use anise_adapter::{AniseEphemeris, AniseError};
pub use delta_t::{CivilTime, DeltaT, DeltaTBasis, civil_time, delta_t};
pub use earth_fixed::{
    EarthFixedSight, SubsolarPoint, moon_ecef, moon_ecef_tt, subsolar_point, sun_ecef, sun_ecef_tt,
};
pub use error::ObservationError;
pub use hifitime::{Duration, Epoch, TimeScale};
pub use observation::{Correction, Sight, observe_body};
pub use stars::{StarSight, stars_ecef, stars_ecef_tt};
pub use state::{Ephemeris, FrameId, State};

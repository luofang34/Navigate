//! Celestial directions for a specified observer and reception event.
//!
//! Callers supply time, states, and ephemeris data. No function reads a clock
//! or loads a file. Earth surface coordinates belong in a caller adapter.
//! [`AnalyticEphemeris`] supplies Sun, Earth, and Moon states without kernel
//! files. [`sun_ecef`] and [`moon_ecef`] give their apparent directions in
//! Earth-fixed axes from a civil clock reading, [`sun_ecef_tt`] and
//! [`moon_ecef_tt`] from an instant such as TT, and [`stars_ecef`] and
//! [`stars_ecef_tt`] give the bright stars, and [`stars_ecef_tt_into`]
//! writes them into a caller buffer. [`civil_time`] and
//! [`delta_t`](fn@delta_t) give the TT and UT1 of a civil clock reading.
//!
//! [`FrameId`] is the frame name of `navigate-contract`. [`physical_from_epoch`]
//! and [`epoch_from_physical`] connect an [`Epoch`] to a
//! [`navigate_contract::PhysicalInstant`] on any of its time scales,
//! including the lunar scale TCL ([`tt_to_tcl`]).
//!
//! The Sun, the Moon, and the stars cover -2000 to 3000 with the default
//! `long-range` feature: series tables fitted to JPL DE431, with the Sun and
//! the Moon within 0.3 arcsecond in the modern era. Without default features,
//! smaller series cover 1800 to 2200, with the Moon within 18 arcseconds.

mod analytic;
#[cfg(feature = "anise")]
mod anise_adapter;
mod delta_t;
mod earth_fixed;
mod earth_rotation;
mod error;
mod observation;
mod precession;
mod spk;
mod stars;
mod state;
mod time_bridge;

pub use analytic::{AnalyticEphemeris, AnalyticError};
#[cfg(feature = "anise")]
pub use anise_adapter::{AniseEphemeris, AniseError};
pub use delta_t::{CivilTime, DeltaT, DeltaTBasis, civil_time, delta_t};
pub use earth_fixed::{
    EarthFixedSight, SubsolarPoint, moon_ecef, moon_ecef_tt, moon_ecef_tt_with, moon_ecef_with,
    subsolar_point, sun_ecef, sun_ecef_tt, sun_ecef_tt_with, sun_ecef_with,
};
pub use error::ObservationError;
pub use hifitime::{Duration, Epoch, TimeScale};
pub use navigate_contract::FrameId;
pub use observation::{Correction, Sight, observe_body};
pub use spk::{DafEndian, SpkDataType, SpkEphemeris, SpkError, SpkFile, SpkSegment};
pub use stars::{StarSight, stars_ecef, stars_ecef_tt, stars_ecef_tt_into};
pub use state::{Ephemeris, State};
pub use time_bridge::{
    TimeBridgeError, epoch_from_physical, physical_from_epoch, tcb_to_tdb, tcg_to_tt, tcl_to_tt,
    tdb_to_tcb, tt_to_tcg, tt_to_tcl, ut1_minus_tai_s,
};

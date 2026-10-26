//! Celestial directions for a specified observer and reception event.
//!
//! Callers supply time, states, and ephemeris data. No function reads a clock
//! or loads a file. Earth surface coordinates belong in a caller adapter.
//! [`AnalyticEphemeris`] supplies Sun, Earth, and Moon states without kernel
//! files. [`sun_ecef`] and [`moon_ecef`] give their apparent directions in
//! Earth-fixed axes.

mod analytic;
#[cfg(feature = "anise")]
mod anise_adapter;
mod earth_fixed;
mod earth_rotation;
mod error;
mod observation;
mod precession;
mod state;

pub use analytic::{AnalyticEphemeris, AnalyticError};
#[cfg(feature = "anise")]
pub use anise_adapter::{AniseEphemeris, AniseError};
pub use earth_fixed::{EarthFixedSight, SubsolarPoint, moon_ecef, subsolar_point, sun_ecef};
pub use error::ObservationError;
pub use hifitime::{Duration, Epoch, TimeScale};
pub use observation::{Correction, Sight, observe_body};
pub use state::{Ephemeris, FrameId, State};

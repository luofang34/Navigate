//! Celestial directions for a specified observer and reception event.
//!
//! Callers supply time, states, and ephemeris data. No function reads a clock
//! or loads a file. Earth surface coordinates belong in a caller adapter.

mod anise_adapter;
mod error;
mod observation;
mod state;

pub use anise::time::{Duration, Epoch, TimeScale};
pub use anise_adapter::{AniseEphemeris, AniseError};
pub use error::ObservationError;
pub use observation::{Correction, Sight, observe_body};
pub use state::{Ephemeris, FrameId, State};

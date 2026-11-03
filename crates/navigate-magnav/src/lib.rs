//! Magnetic anomaly navigation without heap allocation (ADR-0014,
//! ADR-0016).
//!
//! - [`OnlineTlCalibrator`] ([`calibration`]) learns the 18
//!   Tolles–Lawson coefficients of the magnetic interference of the
//!   aircraft with a recursive estimator.
//! - [`BilinearGridMap`] ([`grid_map`]) is a crustal anomaly map on a
//!   regular latitude and longitude grid. It borrows the grid values from
//!   the caller and implements [`navigate_fusion::ScalarFieldMap`], so the
//!   filters of `navigate-fusion` sample it.
//! - [`CrlbTracker`] ([`crlb`]) gives the posterior Cramér–Rao lower
//!   bound of the inertial error along a reference trajectory, so a
//!   planner can judge a map leg before a flight.
//!
//! The crate is `#![no_std]`. The `std` feature (the default) or the
//! `libm` feature supplies the float functions.

#![no_std]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("Enable feature `std` or `libm`.");

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod calibration;
pub mod crlb;
mod float_math;
pub mod grid_map;

pub use calibration::{
    OnlineTlCalibrator, TOLLES_LAWSON_TERMS, TlCalibratorConfig, TlError, TlUpdate,
    tolles_lawson_basis,
};
pub use crlb::{CrlbTracker, ReferencePoint};
pub use grid_map::{BilinearGridMap, GridMapError};

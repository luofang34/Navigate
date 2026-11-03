//! Pure surface guidance: a navigation solution plus the active leg
//! geometry gives one setpoint or one typed refusal (ADR-0015).
//!
//! [`guide`] maps `(solution, leg geometry, now, clock, config)` to a
//! deviation-tracking [`navigate_contract::GuidanceCommand`].
//! [`guide_velocity`] maps the same inputs to a local-level NED velocity
//! command. Neither function keeps internal state. `navigate-fpl` owns leg
//! sequencing, capture criteria, and procedure selection. The host
//! platform owns transports and actuator authority.
//!
//! Guidance checks integrity with fail-closed logic (ADR-0004). A solution
//! below the configured quality floor, older than the configured bound,
//! stamped after `now`, read on a mismatched clock domain, or defined on a
//! mismatched ellipsoid gives a [`GuidanceRefusal`]. Both derivations call
//! one shared admission helper so their bounds match. Caller code supplies
//! `now` ([`navigate_contract::MonotonicNanos`]); this crate reads no
//! clock and allocates no heap memory (ADR-0002, ADR-0015).

#![no_std]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!(
    "navigate-guidance needs the `std` feature or the `libm` feature for float functions"
);

#[cfg(any(feature = "std", test))]
extern crate std;

mod admission;
mod config;
mod derive;
mod float_math;
mod refusal;
mod velocity;
mod vertical;

#[cfg(test)]
mod scenario;

pub use config::{GuidanceConfig, VelocityGuidanceConfig};
pub use derive::guide;
pub use refusal::GuidanceRefusal;
pub use velocity::guide_velocity;

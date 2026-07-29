//! Pure guidance: a navigation solution plus the active leg's geometry
//! becomes one setpoint, or a typed refusal.
//!
//! This crate is a pair of functions, not a machine. [`guide`] maps
//! `(solution, leg geometry, now and its clock domain, config)` to a
//! deviation-tracking [`navigate_contract::GuidanceCommand`];
//! [`guide_velocity`] maps the same inputs to a local-level NED velocity
//! command. Neither holds state beyond the caller's config. Leg
//! sequencing, capture criteria, and procedure selection live in
//! `navigate-fpl`; transports and authority live in the host platform.
//!
//! Guidance consumes integrity fail-closed (ADR-0004): a solution below
//! the configured quality floor, older than the configured bound,
//! timestamped after the caller's `now`, or judged against a `now` read
//! on a different clock domain yields a [`GuidanceRefusal`], never a
//! best-effort setpoint. Both derivations admit through one helper, so
//! the floors they enforce cannot drift apart. `now` is always
//! caller-supplied ([`navigate_contract::MonotonicNanos`]); nothing here
//! reads a clock (ADR-0002).

mod admission;
mod config;
mod derive;
mod refusal;
mod velocity;
mod vertical;

#[cfg(test)]
mod scenario;

pub use config::{GuidanceConfig, VelocityGuidanceConfig};
pub use derive::guide;
pub use refusal::GuidanceRefusal;
pub use velocity::guide_velocity;

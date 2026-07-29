//! Pure guidance: a navigation solution plus the active leg's geometry
//! becomes one deviation-tracking setpoint, or a typed refusal.
//!
//! This crate is a function, not a machine: [`guide`] maps
//! `(solution, leg geometry, now and its clock domain, config)` to a
//! [`navigate_contract::GuidanceCommand`] and holds no state beyond the
//! caller's [`GuidanceConfig`]. Leg sequencing, capture criteria, and
//! procedure selection live in `navigate-fpl`; transports and authority
//! live in the host platform.
//!
//! Guidance consumes integrity fail-closed (ADR-0004): a solution below
//! the configured quality floor, older than the configured bound,
//! timestamped after the caller's `now`, or judged against a `now` read
//! on a different clock domain yields a [`GuidanceRefusal`], never a
//! best-effort setpoint. `now` is always caller-supplied
//! ([`navigate_contract::MonotonicNanos`]); nothing here reads a clock
//! (ADR-0002).

mod config;
mod derive;
mod refusal;
mod vertical;

pub use config::GuidanceConfig;
pub use derive::guide;
pub use refusal::GuidanceRefusal;

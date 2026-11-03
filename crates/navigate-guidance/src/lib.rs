//! Pure guidance on the surface and in space: a navigation solution plus
//! a target gives one setpoint or one typed refusal (ADR-0015,
//! ADR-0019).
//!
//! Surface: [`guide`] maps `(solution, leg geometry, now, clock, config)`
//! to a deviation-tracking [`navigate_contract::GuidanceCommand`].
//! [`guide_velocity`] maps the same inputs to a local-level NED velocity
//! command. `navigate-route` owns leg sequencing, capture criteria, and
//! procedure selection.
//!
//! Space: [`guide_burn`], [`guide_rendezvous`], [`guide_entry`], and
//! [`guide_powered_descent`] map a Cartesian solution and a target of
//! `navigate-trajectory` to a thrust-vector, Cartesian-velocity, or
//! entry-bank command. [`step_mission`] steps a [`MissionProfile`] from
//! an orbital coast through a burn and an entry to a route, or through a
//! powered descent to touchdown.
//!
//! Guidance checks integrity with fail-closed logic (ADR-0004). A surface
//! solution below the configured quality floor, older than the configured
//! bound, stamped after `now`, read on a mismatched clock domain, or
//! defined on a mismatched ellipsoid gives a [`GuidanceRefusal`]. A space
//! solution in the wrong frame or with too much position uncertainty
//! also gives a refusal ([`admit_space_solution`]). Caller code supplies
//! `now` ([`navigate_contract::MonotonicNanos`]); this crate reads no
//! clock and allocates no heap memory (ADR-0002, ADR-0015). The host
//! platform owns transports and actuator authority.

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
mod mission;
mod refusal;
mod space;
mod velocity;
mod vertical;

#[cfg(test)]
mod scenario;

pub use admission::admit_space_solution;
pub use config::{GuidanceConfig, SpaceGuidanceConfig, VelocityGuidanceConfig};
pub use derive::guide;
pub use mission::{
    MissionConfig, MissionInput, MissionPhase, MissionProfile, MissionState, MissionStep,
    step_mission,
};
pub use refusal::GuidanceRefusal;
pub use space::{
    EntryGuidance, PoweredDescentPlan, RendezvousPlan, guide_burn, guide_entry,
    guide_powered_descent, guide_rendezvous,
};
pub use velocity::guide_velocity;

//! Flight-plan execution: active-leg sequencing with capture criteria,
//! terminal behavior, and role-based procedure selection.
//!
//! The plan model — waypoints, roles, structural validation — lives in
//! [`navigate_contract::plan`]; this crate drives it. A validated plan
//! becomes a [`PlanExecution`] that sequences legs as caller-supplied
//! positions arrive, and a [`PlanSet`] holds the loaded procedures a
//! host selects among by [`navigate_contract::PlanRole`]. Everything is
//! sans-IO, `#![no_std]`, zero-heap, and deterministic (ADR-0002,
//! ADR-0015): positions come from the caller; nothing here reads a clock
//! or a sensor.

#![no_std]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("Enable feature `std` or `libm` on `navigate-fpl`.");

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod execution;
mod float_math;
pub mod plan_set;
pub mod turn;

pub use execution::{ExecutionConfig, Leg, PlanExecution, SequenceEvent, SequenceReason};
pub use plan_set::{PlanActivationError, PlanSet};
pub use turn::{EARTH_STANDARD_GRAVITY_MPS2, turn_anticipation_m, turn_radius_m};

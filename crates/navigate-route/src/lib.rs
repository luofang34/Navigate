//! Route execution on any body: active-leg sequencing with capture
//! criteria, terminal behavior, and role-based procedure selection
//! (ADR-0016).
//!
//! The plan model (waypoints, roles, and structural validation) is in
//! [`navigate_contract::plan`]. This crate drives it. A
//! [`PlanExecution`] sequences the legs of a validated plan as the
//! caller supplies positions. A [`PlanSet`] holds the loaded procedures,
//! and a host selects one by [`navigate_contract::PlanRole`]. The fly-by
//! turn model reads the surface gravity of the body from
//! [`ExecutionConfig`], so a route on Mars or Titan does not use the
//! gravity of the Earth.
//!
//! The crate is sans-IO, `#![no_std]`, and deterministic, and it does
//! not allocate (ADR-0002, ADR-0015). The caller supplies the positions.
//! The crate does not read a clock or a sensor.

#![no_std]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("Enable feature `std` or `libm` on `navigate-route`.");

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod execution;
mod float_math;
pub mod plan_set;
pub mod turn;

pub use execution::{
    Corner, ExecutionConfig, ExecutionState, HoldEntry, HoldStage, HoldStatus, Leg, PlanExecution,
    SequenceEvent, SequenceReason, hold_entry,
};
pub use plan_set::{PlanActivationError, PlanSet};
pub use turn::{
    EARTH_STANDARD_GRAVITY_MPS2, MARS_SURFACE_GRAVITY_MPS2, TITAN_SURFACE_GRAVITY_MPS2,
    turn_anticipation_m, turn_radius_m,
};

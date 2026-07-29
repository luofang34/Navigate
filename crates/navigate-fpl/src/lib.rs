//! Flight-plan execution: active-leg sequencing with capture criteria,
//! terminal behavior, and role-based procedure selection.
//!
//! The plan model — waypoints, roles, structural validation — lives in
//! [`navigate_contract::plan`]; this crate drives it. A validated plan
//! becomes a [`PlanExecution`] that sequences legs as caller-supplied
//! positions arrive, and a [`PlanSet`] holds the loaded procedures a
//! host selects among by [`navigate_contract::PlanRole`]. Everything is
//! sans-IO and deterministic (ADR-0002): positions come from the caller;
//! nothing here reads a clock or a sensor.

pub mod execution;
pub mod plan_set;

pub use execution::{ExecutionConfig, Leg, PlanExecution, SequenceEvent};
pub use plan_set::{PlanActivationError, PlanSet};

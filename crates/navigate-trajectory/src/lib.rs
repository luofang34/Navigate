//! Trajectory laws for space flight on any body (ADR-0019).
//!
//! Each law maps a state vector and a target to one command:
//!
//! - [`burn`]: closed-loop steering of an orbital burn on the velocity
//!   to go, with cross-product steering for a target that is a function
//!   of position.
//! - [`rendezvous`]: the local-vertical local-horizontal frame of a
//!   target, the Clohessy–Wiltshire transfer velocity, and the approach
//!   corridor check.
//! - [`entry`]: drag-versus-energy tracking of an atmospheric entry
//!   corridor, with bank reversals on the heading error.
//! - [`powered_descent`]: polynomial E-guidance to a touchdown target.
//!
//! The crate is `#![no_std]` and does not allocate. A corridor borrows
//! its knots from the caller. All values are `Copy`. The `std` feature
//! (the default) or the `libm` feature supplies the float functions. The
//! crate reads no clock: the caller supplies every time.

#![no_std]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("Enable feature `std` or `libm` on `navigate-trajectory`.");

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod burn;
pub mod entry;
mod error;
mod float_math;
pub mod powered_descent;
pub mod rendezvous;
mod state;

pub use burn::{BurnArc, BurnCommand, BurnTarget, RadialBranch, evaluate_burn};
pub use entry::{
    EntryCommand, EntryCorridor, EntryLaw, EntryNode, EntryState, evaluate_entry,
    specific_energy_j_per_kg,
};
pub use error::TrajectoryError;
pub use powered_descent::{
    DescentCommand, GlideSlope, PoweredDescentTarget, evaluate_powered_descent,
};
pub use rendezvous::{
    ApproachCorridor, LvlhFrame, RendezvousCommand, RendezvousTarget, cw_transfer_velocity,
    evaluate_rendezvous,
};
pub use state::StateVector;

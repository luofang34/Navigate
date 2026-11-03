//! Sans-IO navigation filters: typed observations, admission gates,
//! correlation discipline, and integrity-honest solutions (ADR-0003,
//! ADR-0004).
//!
//! The filters consume [`Observation`]s, never sensors. Every
//! admission-gate refusal increments a counter in [`RejectionCounters`].
//! Every published [`navigate_contract::NavigationSolution`] carries the
//! integrity that its sensor set supports. Identical observation scripts
//! give field-identical solutions, because the caller supplies all time
//! as [`navigate_contract::MonotonicNanos`].
//!
//! The crate has these filters:
//!
//! - [`NavigationFilter`] ([`kinematic`]): a six-state constant-velocity
//!   filter in a local NED plane on one reference ellipsoid of any body,
//!   with scalar map models and a bounded delayed-measurement history
//!   (ADR-0012, ADR-0013).
//! - [`InertialNavigationFilter`] ([`inertial`]): a strapdown error-state
//!   filter with seventeen states on a body surface or in a Cartesian
//!   space frame (ADR-0014).
//! - [`MarginalizedParticleFilter`] ([`particle`]): a scalar field map
//!   match without a linearization, on particles that the caller owns.
//!
//! The crate is `#![no_std]` and does not allocate (ADR-0016). Every
//! table and history has a fixed inline capacity. The `std` feature (the
//! default) or the `libm` feature supplies the float functions. The
//! gravity models come from `navigate-astrodynamics`, and this crate
//! re-exports them.

#![no_std]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("Enable feature `std` or `libm`.");

#[cfg(any(feature = "std", test))]
extern crate std;

mod admission;
pub mod config;
mod float_math;
mod history_ring;
pub mod inertial;
mod integrity;
pub mod kalman;
pub mod kinematic;
pub mod observation;
pub mod particle;
mod provenance;
pub mod rejection;

pub use config::FusionConfig;
pub use inertial::{ImuSample, InertialConfig, InertialNavigationFilter};
pub use kalman::FogmBias;
pub use kinematic::{InnovationStats, NavigationFilter};
pub use navigate_astrodynamics::{
    CentralBodyGravity, PlanetaryBody, PointMassBody, SpaceEnvironment, SpaceGravityModel,
    ThirdBodySet,
};
pub use observation::{MeasurementKind, Observation, ObservationValue};
pub use particle::{
    MarginalizedParticleFilter, MpfConfig, MpfError, MpfEstimate, MpfParticle, ParticlePool,
    ScalarFieldMap,
};
pub use provenance::{MAX_SOURCES, MAX_WINDOW_ADMISSIONS};
pub use rejection::{IngestOutcome, RejectionCounters, RejectionReason};

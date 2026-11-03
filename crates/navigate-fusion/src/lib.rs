//! Sans-IO navigation fusion: typed observations, admission gates,
//! correlation discipline, and integrity-honest solutions (ADR-0003,
//! ADR-0004).
//!
//! The filter consumes [`Observation`]s — never sensors — over a
//! local-level NED state anchored at the first admitted position fix.
//! Every admission-gate refusal increments a counter in
//! [`RejectionCounters`], every published
//! [`navigate_contract::NavigationSolution`] carries the integrity its
//! sensor set actually supports, and identical observation scripts yield
//! field-identical solutions: time is always supplied by the caller as
//! [`navigate_contract::MonotonicNanos`], never read from a clock.
//!
//! Scalar map models (altitude and magnetic anomaly) and a bounded
//! delayed-measurement history extend the core (ADR-0012). The state is
//! on one reference ellipsoid of any body, and the filter refuses an
//! observation that declares another frame (ADR-0013).
//!
//! [`InertialNavigationFilter`] is a strapdown error-state filter with
//! seventeen states on a body surface or in a Cartesian space frame.
//! [`MarginalizedParticleFilter`] matches a scalar field map without a
//! linearization. [`CrlbTracker`] gives the posterior Cramér–Rao bound,
//! and [`OnlineTlCalibrator`] estimates the Tolles–Lawson coefficients of
//! the aircraft (ADR-0014).

mod admission;
pub mod calibration;
pub mod config;
pub mod crlb;
pub mod environment;
mod filter;
pub mod inertial;
mod integrity;
pub mod mpf;
pub mod navigation_filter;
pub mod observation;
mod provenance;
pub mod rejection;

pub use calibration::{
    OnlineTlCalibrator, TOLLES_LAWSON_TERMS, TlCalibratorConfig, TlError, TlUpdate,
    tolles_lawson_basis,
};
pub use config::FusionConfig;
pub use crlb::{CrlbTracker, ReferencePoint};
pub use environment::{
    CentralBodyGravity, PlanetaryBody, PointMassBody, SpaceEnvironment, SpaceGravityModel,
};
pub use filter::FogmBias;
pub use inertial::{ImuSample, InertialConfig, InertialNavigationFilter};
pub use mpf::{
    BilinearGridMap, MarginalizedParticleFilter, MpfConfig, MpfError, MpfEstimate, ScalarFieldMap,
};
pub use navigation_filter::{InnovationStats, NavigationFilter};
pub use observation::{MeasurementKind, Observation, ObservationValue};
pub use rejection::{IngestOutcome, RejectionCounters, RejectionReason};

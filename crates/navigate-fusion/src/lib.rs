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
//! delayed-measurement history extend the core (ADR-0012).

pub mod config;
mod filter;
pub mod navigation_filter;
pub mod observation;
pub mod rejection;

pub use config::FusionConfig;
pub use filter::FogmBias;
pub use navigation_filter::{InnovationStats, NavigationFilter};
pub use observation::{MeasurementKind, Observation, ObservationValue};
pub use rejection::{IngestOutcome, RejectionCounters, RejectionReason};

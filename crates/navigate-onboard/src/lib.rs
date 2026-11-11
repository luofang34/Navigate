//! The onboard boundary (ADR-0022).
//!
//! A host links two parts:
//!
//! - a [`Navigator`] that takes sensor inputs and gives a
//!   `NavigationSolution` and [`AcceptanceRecord`]s;
//! - a [`FlightDirector`] that sequences the route and computes the
//!   guidance setpoint from a solution.
//!
//! [`Onboard`] owns one of each. A navigator declares its data
//! requirements before load, reports what it did with each product chunk,
//! and gives a snapshot of its complete state that it can restore. The host declares the [`SensorSuite`]
//! once, when it builds the navigator. Each [`SensorInput`] then carries the
//! sensor identifier, the times, and the data. A sensor is unavailable only
//! when its inputs do not come: no navigator has a mode that disables one.
//!
//! [`ReferenceNavigator`] is the reference composition: the 22-state
//! inertial filter with GNSS, barometric, star-camera, and terrain-camera
//! aiding. The terrain-camera means is in the Cargo feature
//! `terrain-camera`; without the feature, a suite with a terrain camera is
//! refused. The terrain camera needs a terrain map: the host opens a store
//! and gives it with a `TerrainMapRequest`, and the navigator reads the
//! package and never fetches.
//! [`RouteDirector`] executes a flight plan with Navigate's sequencer and
//! guidance laws.

mod acceptance;
mod alignment;
mod bundle;
mod director;
mod error;
mod navigator;
mod reference;
mod sensor;

pub use acceptance::{AcceptanceRecord, AcceptanceSink, Outcome};
pub use alignment::{
    AlignmentConfig, AlignmentPhase, AlignmentRecord, AlignmentSource, AlignmentStatus,
    HeadingSource, PositionSource,
};
pub use bundle::Onboard;
pub use director::{
    DirectorEvent, DirectorOutput, FlightDirector, GuidanceLaw, RouteDirector, SequenceCause,
};
pub use error::{MapRefusal, OnboardError};
pub use navigator::{Alignment, DataRequirement, Navigator, Need, ProductReport};
#[cfg(feature = "terrain-camera")]
pub use reference::{
    AccuracySource, AssumedAccuracy, BeliefSource, HostGeoid, LayerReport, MapReport,
    TerrainConfig, TerrainMapRequest, WorkingCamera, WorkingLimit,
};
pub use reference::{NavigatorFilter, ReferenceNavigator, StarCatalogueConfig, own_prior};
pub use sensor::{
    ErrorBudget, GnssCorrelatedError, SensorData, SensorInput, SensorKind, SensorSpec, SensorSuite,
};

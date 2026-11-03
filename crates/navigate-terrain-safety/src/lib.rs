//! Terrain clearance on any body: typed availability, alert vocabulary,
//! and vertical-datum adapters (ADR-0004, ADR-0016).
//!
//! The crate owns the assessment and the refusal vocabulary. The host
//! side binds the terrain data through [`TerrainElevation`]. In line
//! with ADR-0004, [`assess_clearance`] refuses with a typed
//! [`ClearanceUnavailable`] when an input that it needs is absent,
//! stale, on the wrong clock domain, or below the configured integrity
//! floor. It does not invent a clearance.
//!
//! # Vertical datum
//!
//! The altitude of a [`navigate_contract::NavigationSolution`] is a
//! height above the solution ellipsoid. [`TerrainElevation`] gives
//! terrain heights above the same ellipsoid, so the clearance is the
//! difference of two ellipsoidal heights. Terrain models usually give
//! orthometric heights `H` above an equipotential surface: the geoid of
//! the Earth, the selenoid of the Moon, or the areoid of Mars.
//! [`OrthometricTerrain`] adds the undulation `N` of a
//! [`navigate_geodesy::GeoidUndulation`] model, `h = H + N`, and it gives
//! no height where the model has no coverage.
//!
//! The horizontal reference is explicit (ADR-0013, ADR-0015). Each
//! terrain source names its ellipsoid, and [`assess_clearance`] refuses
//! a solution on another ellipsoid with
//! [`ClearanceUnavailable::EllipsoidMismatch`]. The crate is
//! `#![no_std]` and does not allocate.

#![no_std]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("Enable feature `std` or `libm`.");

#[cfg(test)]
extern crate std;

mod assessment;
mod config;
mod terrain;
mod vertical_datum;

pub use assessment::{AlertSeverity, ClearanceAssessment, ClearanceUnavailable, assess_clearance};
pub use config::ClearanceConfig;
pub use terrain::{OrthometricElevation, TerrainElevation};
pub use vertical_datum::OrthometricTerrain;

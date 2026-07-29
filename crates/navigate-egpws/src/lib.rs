//! Terrain-awareness seam: typed availability and alert vocabulary.
//!
//! This crate is a seam, not a full EGPWS. It owns the assessment and
//! refusal vocabulary; the terrain data itself is bound by the host side
//! through [`TerrainDatabase`], and no implementation exists in this
//! repository. In line with ADR-0004, [`assess`] refuses with a typed
//! [`EgpwsUnavailable`] whenever an input it needs is absent, stale,
//! read on the wrong clock domain, or below the configured integrity
//! floor — an honest refusal instead of an invented clearance.
//!
//! # Open integration question: vertical datum
//!
//! [`navigate_contract::NavigationSolution`] altitude is height above the
//! WGS84 ellipsoid, while production terrain databases publish
//! orthometric elevations (height above the geoid). Subtracting one from
//! the other without a geoid undulation model injects an error of tens of
//! meters in places. Resolving the datum — requiring [`TerrainDatabase`]
//! implementations to serve ellipsoidal elevations, or binding a geoid
//! model at the seam — is owed to the host-side terrain-package
//! integration; this crate states the gap rather than papering over it
//! with a constant offset.

mod assessment;
mod config;
mod terrain;

pub use assessment::{AlertSeverity, EgpwsAssessment, EgpwsUnavailable, assess};
pub use config::EgpwsConfig;
pub use terrain::TerrainDatabase;

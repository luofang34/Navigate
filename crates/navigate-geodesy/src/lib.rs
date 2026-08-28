//! WGS84 geodesy: geodetic/ECEF/local-NED conversions and great-circle
//! guidance geometry, hand-rolled on `f64` in canonical units (radians,
//! meters).
//!
//! Builds on the vocabulary of `navigate-contract`
//! ([`navigate_contract::GeodeticPosition`] is the geodetic type); this
//! crate adds the frame types the contract deliberately omits
//! ([`EcefPosition`], [`NedOffset`]) and the pure conversion math between
//! them. Everything is sans-IO and deterministic (ADR-0002): plain
//! functions over typed values, no clocks, no allocation on the hot path.
//!
//! Accuracy contracts:
//!
//! - Geodetic ↔ ECEF uses a fixed two-pass Bowring scheme
//!   ([`ecef_to_geodetic`]); round trips agree to better than 1e-6 m
//!   over the tested envelope of −400 m to 100 km above the ellipsoid.
//! - [`LocalTangentPlane`] conversions go through ECEF and an exact
//!   rotation, so they inherit the same bound — no small-angle
//!   approximations.
//! - Great-circle helpers use the spherical mean-radius approximation
//!   ([`wgs84::MEAN_RADIUS_M`]); they serve guidance geometry and are not
//!   survey-grade (see [`great_circle`] for the error budget). The
//!   track-deviation pair ([`cross_track_m`], [`along_track_m`]) returns
//!   `Result` and refuses tracks whose endpoints are closer than 1 mm
//!   ([`GeodesyError::DegenerateTrack`]).

pub mod ecef;
pub mod error;
pub mod great_circle;
pub mod local_plane;
pub mod wgs84;

pub use ecef::{EcefPosition, ecef_to_geodetic, geodetic_to_ecef};
pub use error::GeodesyError;
pub use great_circle::{
    along_track_m, cross_track_from_course_m, cross_track_m, distance_m, initial_bearing_rad,
};
pub use local_plane::{LocalTangentPlane, NedOffset};

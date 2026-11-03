//! Multi-body geodesy: geodetic/body-fixed/local-NED conversions and
//! great-circle guidance geometry on any [`BodyEllipsoid`], hand-rolled
//! on `f64` in canonical units (radians, meters) (ADR-0013).
//!
//! Builds on the vocabulary of `navigate-contract`
//! ([`navigate_contract::GeodeticPosition`] is the geodetic type and
//! [`navigate_contract::Ellipsoid`] is the reference ellipsoid); this
//! crate adds the frame types the contract deliberately omits
//! ([`BodyFixedPosition`], [`NedOffset`]) and the pure conversion math
//! between them. The functions without an ellipsoid argument use
//! [`BodyEllipsoid::WGS84`]. Everything is sans-IO and deterministic
//! (ADR-0002): plain functions over typed values, no clocks, no
//! allocation on the hot path.
//!
//! Accuracy contracts:
//!
//! - Geodetic ↔ body-fixed uses a fixed two-pass Bowring scheme
//!   ([`body_fixed_to_geodetic`]); round trips agree to better than
//!   1e-6 m over the tested envelopes, for example −400 m to 100 km
//!   above the WGS84 ellipsoid. Spheres and the polar axis are exact.
//! - [`LocalTangentPlane`] conversions go through the body-fixed frame
//!   and an exact rotation, so they inherit the same bound — no
//!   small-angle approximations.
//! - Great-circle helpers use the spherical mean-radius approximation
//!   ([`BodyEllipsoid::mean_radius_m`]); they serve guidance geometry
//!   and are not survey-grade (see [`great_circle`] for the error budget). The
//!   track-deviation pair ([`cross_track_m`], [`along_track_m`]) returns
//!   `Result` and refuses tracks whose endpoints are closer than 1 mm
//!   ([`GeodesyError::DegenerateTrack`]).

pub mod ecef;
pub mod ellipsoid;
pub mod error;
pub mod great_circle;
pub mod local_plane;
pub mod wgs84;

pub use ecef::{
    BodyFixedPosition, EcefPosition, body_fixed_to_geodetic, ecef_to_geodetic,
    geodetic_to_body_fixed, geodetic_to_ecef,
};
pub use ellipsoid::{BodyEllipsoid, west_to_east_lon_rad};
pub use error::GeodesyError;
pub use great_circle::{
    along_track_m, along_track_on_m, cross_track_from_course_m, cross_track_from_course_on_m,
    cross_track_m, cross_track_on_m, distance_m, distance_on_m, initial_bearing_rad,
};
pub use local_plane::{LocalTangentPlane, NedOffset};

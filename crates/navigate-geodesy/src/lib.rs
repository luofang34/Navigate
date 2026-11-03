//! Multi-body geodesy: geodetic/body-fixed/local-NED conversions,
//! great-circle guidance geometry, vertical datums, and surface ray
//! intersection on any [`BodyEllipsoid`], hand-rolled on `f64` in
//! canonical units (radians, meters) (ADR-0013, ADR-0015).
//!
//! Builds on the vocabulary of `navigate-contract`
//! ([`navigate_contract::GeodeticPosition`] is the geodetic type and
//! [`navigate_contract::Ellipsoid`] is the reference ellipsoid); this
//! crate adds the frame types the contract deliberately omits
//! ([`BodyFixedPosition`], [`NedOffset`]) and the pure conversion math
//! between them. Every function at the crate root takes its ellipsoid
//! explicitly. The Earth-only shorthand is in [`wgs84`]. Everything is
//! sans-IO and deterministic (ADR-0002): plain functions over typed
//! values, no clocks, no allocation.
//!
//! The crate is `#![no_std]`. The `std` feature (the default) or the
//! `libm` feature supplies the float functions ([`float_math`]).
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

#![no_std]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!(
    "navigate-geodesy needs the `std` feature or the `libm` feature for float functions"
);

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod datum;
pub mod ecef;
pub mod ellipsoid;
pub mod error;
pub mod float_math;
pub mod great_circle;
pub mod local_plane;
pub mod shape;
pub mod wgs84;

pub use datum::{
    GeoidUndulation, ZeroUndulation, ellipsoidal_to_orthometric_m, orthometric_to_ellipsoidal_m,
};
pub use ecef::{BodyFixedPosition, body_fixed_to_geodetic, geodetic_to_body_fixed};
pub use ellipsoid::{BodyEllipsoid, west_to_east_lon_rad};
pub use error::GeodesyError;
pub use great_circle::{
    along_track_m, cross_track_from_course_m, cross_track_m, distance_m, initial_bearing_rad,
};
pub use local_plane::{LocalTangentPlane, NedOffset};
pub use shape::{SurfaceHit, SurfaceShape};

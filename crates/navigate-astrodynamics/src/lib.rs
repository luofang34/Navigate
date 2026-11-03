//! Planetary bodies, gravity fields, coordinate time scales, and frame
//! rotations for navigation on any body (ADR-0016).
//!
//! [`PlanetaryBody`] gives the constants of a rotating body and its
//! apparent gravity at a body-fixed position. The [`gravity`] module
//! gives the gravity of a Cartesian frame on one body, with a fixed
//! number of third bodies in [`ThirdBodySet`] and an optional
//! [`SphericalHarmonicField`]. [`OrbitalForceModel`] adds the solar
//! radiation pressure of [`CannonballSrp`] and the drag of
//! [`ExponentialDrag`] for a coast propagation. The [`time_scales`]
//! module converts a [`navigate_contract::PhysicalInstant`] between the
//! coordinate time scales. The [`frame_rotation`] module rotates a
//! position and a velocity between a body-fixed frame and an inertial
//! frame.
//!
//! The crate is `#![no_std]` and does not allocate. All values are
//! `Copy`. The `std` feature (the default) or the `libm` feature
//! supplies the float functions.

#![no_std]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("Enable feature `std` or `libm`.");

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod body;
pub mod drag;
mod float_math;
pub mod force_model;
pub mod frame_rotation;
pub mod gravity;
pub mod srp;
pub mod time_scales;

pub use body::PlanetaryBody;
pub use drag::ExponentialDrag;
pub use force_model::{ForceEvaluation, OrbitalForceModel, OrbitalGravity};
pub use frame_rotation::{FrameTransform, era_rad, polar_motion_matrix};
pub use gravity::{
    CentralBodyGravity, GravityField, HarmonicFieldError, HarmonicNormalization,
    MAX_INLINE_HARMONIC_DEGREE, MAX_THIRD_BODIES, PointMassBody, SpaceEnvironment,
    SpaceGravityModel, SphericalHarmonicField, SurfaceFrame, ThirdBodyCapacityError, ThirdBodySet,
    centrifugal, centrifugal_gradient, harmonic_coefficient_count, harmonic_index,
};
pub use srp::{ASTRONOMICAL_UNIT_M, CannonballSrp, SOLAR_PRESSURE_AT_1_AU_N_PER_M2};
pub use time_scales::{EarthOrientation, TimeScaleError};

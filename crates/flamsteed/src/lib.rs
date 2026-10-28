//! Bright stars and their motion in space, from public-domain catalogue data.
//!
//! The crate embeds the Yale Bright Star Catalogue, 5th revised edition, and
//! moves each star on a straight line in space to any epoch. It reads no file
//! and no clock, and it does not allocate. It builds for
//! `wasm32-unknown-unknown` and, without the standard library, for
//! microcontrollers: use `default-features = false, features = ["libm"]`, so
//! that `libm` gives the float functions. [`Motion`] evaluates in `f64` or, for
//! a single-precision FPU, in `f32`. Directions are as seen from the solar
//! system barycentre, in the axes of the catalogue ([`Catalog::axes`]): for
//! the embedded catalogue, those of FK5, which differ from ICRS by less than
//! 0.1 arcsecond at J2000 and turn by up to about 0.9 milliarcsecond per year.
//! Aberration, the observer's parallax, and Earth axes belong to the caller.
//!
//! The straight line is the motion of a binary system. For 133 stars in
//! binaries with an ORB6 orbit, [`Star::component_direction_at`] adds the
//! orbital motion of the star itself.
//!
//! The core API is meant to stay compatible into 1.0: [`Catalog`] with
//! [`Catalog::bright_star_catalogue`], [`Catalog::brighter_than`],
//! [`Catalog::stars`], and [`Catalog::from_bytes`] for file format 1, the
//! fields of [`Star`], [`StarId`], [`FormatError`], and
//! [`Star::direction_at`]. [`Star`] can gain fields, and [`StarId`] and
//! [`FormatError`] can gain variants, without a breaking release.
//!
//! These parts can still change in a 0.x minor release: file format 2 with
//! [`Astrometry`] and [`Axes`], [`Motion`], [`Float`], and
//! [`Catalog::motions`], the binary-star orbits
//! ([`Star::component_direction_at`], [`Star::orbital_offset_rad_at`]), and
//! [`Star::barycentric_position_pc_at`].
//!
//! [`Catalog::bright_star_catalogue`] is the embedded Bright Star Catalogue.
//! [`Catalog::from_bytes`] reads other catalogue data from bytes that the
//! caller supplies, for example a file in flash: format version 1, the
//! embedded format, or format version 2, with `f64` astrometry in ICRS axes
//! and the provenance of each star.
//!
//! ```
//! use flamsteed::Catalog;
//!
//! // Julian years (TT) from J2000: -2330.0 is 331 BC.
//! let bright: Vec<[f64; 3]> = Catalog::bright_star_catalogue()
//!     .brighter_than(2.0)
//!     .stars()
//!     .map(|star| star.direction_at(-2330.0))
//!     .collect();
//! assert!(bright.len() > 40);
//! assert!(bright.iter().all(|v| (v.iter().map(|c| c * c).sum::<f64>() - 1.0).abs() < 1e-12));
//! ```

#![no_std]
#![cfg_attr(feature = "nightly", feature(core_float_math))]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("flamsteed needs the `std` feature or the `libm` feature for float functions");

#[cfg(any(feature = "std", test))]
extern crate std;

mod catalog;
mod float;
mod format;
mod motion;
mod orbit;

pub use catalog::{Astrometry, Axes, Catalog, Star, StarId};
pub use float::Float;
pub use format::FormatError;
pub use motion::Motion;

/// The README examples, compiled and run as documentation tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

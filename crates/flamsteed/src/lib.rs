//! Bright stars and their motion in space, from public-domain catalogue data.
//!
//! The crate embeds the Yale Bright Star Catalogue, 5th revised edition, and
//! moves each star on a straight line in space to any epoch. It reads no file
//! and no clock, and it does not allocate. It builds for
//! `wasm32-unknown-unknown` and, without the standard library, for
//! microcontrollers: use `default-features = false, features = ["libm"]`, so
//! that `libm` gives the float functions. [`Motion`] evaluates in `f64` or, for
//! a single-precision FPU, in `f32`. Directions are in J2000 equatorial axes as
//! seen from the solar system barycentre. The axes are those of FK5, which
//! differ from ICRS by less than 0.1 arcsecond at J2000 and turn by up to about
//! 0.9 milliarcsecond per year. Aberration, the observer's
//! parallax, and Earth axes belong to the caller.
//!
//! The straight line is the motion of a binary system. For 133 stars in
//! binaries with an ORB6 orbit, [`Star::component_direction_at`] adds the
//! orbital motion of the star itself.
//!
//! No ESA data is in this crate, in its tests, or in any step that makes its
//! tables. ESA catalogues serve only for offline comparison, and the crate
//! documentation states those comparisons as measured differences.
//!
//! The stable API in 0.1 is [`Catalog`], the fields of [`Star`], [`StarId`],
//! [`Motion`], and the `direction_at` functions. [`Star`] and [`StarId`] can gain fields and
//! variants without a major version. The binary file format is internal: the
//! crate does not read external files yet.
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
mod motion;
mod orbit;

pub use catalog::{Catalog, Star, StarId};
pub use float::Float;
pub use motion::Motion;

/// The README examples, compiled and run as documentation tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

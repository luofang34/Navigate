//! Bright stars and their motion in space, from public-domain catalogue data.
//!
//! The crate embeds the Yale Bright Star Catalogue, 5th revised edition, and
//! moves each star on a straight line in space to any epoch. It reads no
//! file and no clock, so it builds for `wasm32-unknown-unknown`. Directions
//! are in J2000 equatorial axes as seen from the solar system barycentre. The
//! axes are those of FK5, which differ from ICRS by less than 0.1 arcsecond. Aberration, the observer's parallax, and
//! Earth axes belong to the caller.
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
//!     .map(|star| star.direction_at(-2330.0))
//!     .collect();
//! assert!(bright.len() > 40);
//! assert!(bright.iter().all(|v| (v.iter().map(|c| c * c).sum::<f64>() - 1.0).abs() < 1e-12));
//! ```

mod catalog;
mod motion;

pub use catalog::{Catalog, Star, StarId};
pub use motion::Motion;

/// The README examples, compiled and run as documentation tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

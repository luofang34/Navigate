//! Bright stars and their motion in space, from public-domain catalogue data.
//!
//! The crate embeds the Yale Bright Star Catalogue, 5th revised edition, and
//! moves each star on a straight line in space to any epoch. It reads no
//! file and no clock, so it builds for `wasm32-unknown-unknown`. Directions
//! are in J2000 equatorial axes (FK5, within 0.03 arcsecond of ICRS) as seen
//! from the solar system barycentre. Aberration, the observer's parallax, and
//! Earth axes belong to the caller.
//!
//! No ESA data is in this crate, in its tests, or in any step that makes its
//! tables. ESA catalogues serve only for offline comparison, and the crate
//! documentation states those comparisons as measured differences.
//!
//! ```
//! use flamsteed::Catalog;
//!
//! // Julian years (TT) from J2000: -2330.0 is 331 BC.
//! let bright: Vec<[f64; 3]> = Catalog::bright_star_catalogue()
//!     .stars()
//!     .filter(|star| star.v_magnitude <= 2.0)
//!     .map(|star| star.direction_at(-2330.0))
//!     .collect();
//! assert!(bright.len() > 40);
//! assert!(bright.iter().all(|v| (v.iter().map(|c| c * c).sum::<f64>() - 1.0).abs() < 1e-12));
//! ```

mod catalog;
mod motion;

pub use catalog::{Catalog, Star};

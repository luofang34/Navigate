//! Portable coverage planning and immutable imagery packages.
//!
//! The default library has no network, filesystem, GPU, or Python dependency.
//! The `native` feature supplies the NAIP provider and a GDAL raster adapter.
mod coverage;
mod error;
mod package;
mod reading;
mod tiles;

#[cfg(feature = "native")]
pub mod native;

pub use coverage::{CoveragePlan, CoverageRequest, plan};
pub use error::ImageryError;
pub use package::{Asset, Chunk, Package, PackageBuilder, TileRecord, digest};
pub use reading::{is_digest, verify_asset};
pub use tiles::{Tile, tile_bounds, tile_position};

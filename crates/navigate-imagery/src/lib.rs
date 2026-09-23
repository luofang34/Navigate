//! Portable coverage planning and immutable imagery packages.
//!
//! The library has no network, filesystem, GPU, or Python dependency. A data
//! provider supplies its source terms to [`plan`] and writes the package chunks.
mod coverage;
mod error;
mod package;
mod reading;
mod source;
mod tiles;

pub use coverage::{CoveragePlan, CoverageRequest, SourceTerms, plan, tile_envelope};
pub use error::ImageryError;
pub use package::{Asset, Chunk, Package, PackageBuilder, TileRecord, digest};
pub use reading::{is_digest, verify_asset};
pub use source::{SourceAsset, SourceManifest, SourceTile};
pub use tiles::{Tile, tile_bounds, tile_position};

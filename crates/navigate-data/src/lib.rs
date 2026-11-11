//! The navigation data contract (ADR-0022).
//!
//! Navigate consumes navigation data and never fetches it. A host fetches
//! tiles through its own [`FetchPort`], installs blobs in a store, and gives
//! Navigate a [`ReadPort`]. This crate holds:
//!
//! - the package [`Manifest`]: layers, their sources, vertical datums,
//!   accuracy, tiles, and gaps, in one canonical encoding;
//! - the package identity: the SHA-256 of the canonical manifest bytes;
//! - [`PackageWriter`], which stores each blob by its digest and writes the
//!   manifest last;
//! - [`Package`], which checks the manifest on open and each blob on read;
//! - the fixed global [`grid`] of correlation cells and product tiles;
//! - Web Mercator [`tile`] arithmetic and the Terrarium height decoder;
//! - the [`resource`] model, which computes what each leg of a flight needs
//!   before a fetch, and [`prepare`](fn@prepare), which fetches the planned tiles
//!   through the host port and seals a package.
//!
//! The crate is `#![no_std]` with `alloc`. The `std` feature (the default)
//! adds [`DirectoryStore`]; the `libm` feature supplies the float functions
//! without the standard library.

#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("navigate-data needs the `std` or the `libm` feature for float functions");

pub mod accuracy;
pub mod corridor;
pub mod datum;
pub mod digest;
mod error;
pub mod fetch;
pub mod grid;
pub mod layer;
pub mod manifest;
pub mod package;
pub mod prepare;
pub mod resource;
pub mod store;
pub mod tile;

pub use accuracy::{Accuracy, Acquisition, Sigma, SurfaceModel};
pub use datum::{EllipsoidalHeight, UNKNOWN_UNDULATION_BOUND_M, VerticalDatum};
pub use digest::Sha256Digest;
pub use error::DataError;
pub use fetch::{FetchPort, Fetched, TileKey};
pub use grid::{CORRELATION_ZOOM, GridCell, GridSpec};
pub use layer::{Encoding, LayerKind, SourceRef};
pub use manifest::{Layer, LayerZoom, Manifest, PACKAGE_FORMAT, PACKAGE_FORMAT_VERSION, TileRef};
pub use package::{Package, PackageWriter, TileRead};
pub use prepare::{LayerTemplate, PlannedTiles, PrepareSummary, plan_tiles, prepare};
pub use resource::{
    GroundHeight, LayerLimits, LegResources, PlannedCamera, PlannedLeg, ResourcePlan,
    ResourceRequest, ZoomBand, plan_resources,
};
#[cfg(feature = "std")]
pub use store::DirectoryStore;
pub use store::{MemoryStore, ReadPort, WritePort, blob_path, manifest_path};
pub use tile::{MAX_TILE_LATITUDE_DEG, MAX_ZOOM, TileAddress};

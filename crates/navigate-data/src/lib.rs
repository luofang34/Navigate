//! The navigation data contract (ADR-0022).
//!
//! Navigate consumes navigation data and never fetches it. A host fetches
//! tiles through its own [`FetchPort`], installs blobs in a store, and gives
//! Navigate a [`ReadPort`]. This crate holds:
//!
//! - the package [`Manifest`]: body, ellipsoid, grid scheme, and layers with
//!   their source, datums, accuracy, correlation zoom, and tile index, in
//!   one canonical encoding with integers only;
//! - the package identity: the SHA-256 of the canonical manifest bytes;
//! - tile indexes: canonical binary blobs that bind each address to a blob
//!   digest or a gap;
//! - [`PackageWriter`], which stores each blob by its digest and writes the
//!   manifest last, and [`Package`], which checks the manifest and the
//!   indexes on open and each blob on read;
//! - the [`Descriptor`]: labels, attribution, and licence text outside the
//!   identity;
//! - [`CorrelationKey`]: correlated map error by source and grid cell;
//! - derived [`product`] chunks, reusable across missions, and per-mission
//!   product lists;
//! - Web Mercator tile arithmetic and the Terrarium height decoder;
//! - the [`resource`] model, which computes what each leg of a flight needs
//!   before a fetch, and [`prepare`](fn@prepare), which fetches the planned
//!   tiles through the host port and seals a package.
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

mod accuracy;
mod body;
mod correlation;
pub mod corridor;
mod datum;
mod descriptor;
mod digest;
mod error;
mod fetch;
mod grid;
pub mod index;
mod manifest;
mod package;
mod prepare;
pub mod product;
pub mod resource;
mod source;
mod store;
pub mod tile;

pub use accuracy::{Accuracy, SurfaceModel};
pub use body::{Body, DayNumber, EllipsoidId, HorizontalDatum};
pub use correlation::CorrelationKey;
pub use datum::{EllipsoidalHeight, GeoidModel, UNKNOWN_UNDULATION_BOUND_M, VerticalDatum};
pub use descriptor::{Descriptor, LayerLabel};
pub use digest::Sha256Digest;
pub use error::DataError;
pub use fetch::{FetchPort, Fetched, TileKey};
pub use grid::{DEFAULT_CORRELATION_ZOOM, GridCell, GridScheme};
pub use manifest::{IndexRef, Layer, LayerKey, Manifest, PACKAGE_FORMAT, PACKAGE_FORMAT_VERSION};
pub use package::{LayerSpec, Package, PackageWriter, TileRead, read_blob};
pub use prepare::{PlannedTiles, PrepareSummary, plan_tiles, prepare};
pub use resource::{
    GroundHeight, LayerLimits, LayerSize, LegResources, PlannedCamera, PlannedLeg, ResourcePlan,
    ResourceRequest, SizeEstimate, ZoomBand, estimate_size, plan_resources,
};
pub use source::{
    Encoding, LayerKind, Revision, SOURCE_FORMAT, SOURCE_FORMAT_VERSION, SourceDeclaration,
    SourceKey,
};
#[cfg(feature = "std")]
pub use store::{DirectoryFile, DirectoryStore};
pub use store::{
    MemoryFile, MemoryStore, RandomAccess, ReadPort, WritePort, blob_path, checked_path,
    descriptor_path, manifest_path, read_all,
};
pub use tile::{MAX_TILE_LATITUDE_DEG, MAX_ZOOM, TileAddress};

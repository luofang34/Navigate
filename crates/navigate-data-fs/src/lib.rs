//! Native byte-range access below a navigation data directory.
mod roots;
mod store;
pub use roots::{ClassStore, StorageRoots};
pub use store::FileStore;

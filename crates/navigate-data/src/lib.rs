//! Storage contracts for immutable navigation data on native and web hosts.
mod access;
mod class;
mod uri;
pub use access::{DataError, DataStore, OpenFuture, RandomAccess, ReadFuture, check_range};
pub use class::StorageClass;
pub use uri::DataUri;

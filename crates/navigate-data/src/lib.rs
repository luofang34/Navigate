//! Storage contracts for immutable navigation data on native and web hosts.
mod access;
mod uri;
pub use access::{DataError, DataStore, OpenFuture, RandomAccess, ReadFuture, check_range};
pub use uri::DataUri;

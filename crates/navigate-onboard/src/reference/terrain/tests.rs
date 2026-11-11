//! Tests of the terrain means through the navigator: a package written
//! with `PackageWriter` in a temporary directory, frames rendered over it
//! at true poses, and the reference navigator run on inertial increments
//! and those frames.

mod anchors;
mod content;
mod cost;
mod dem;
pub(crate) mod fixes;
pub(crate) mod flight;
mod inspection;
mod load;
mod precheck;
mod snapshot;
mod world;

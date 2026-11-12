//! Tests of the terrain means through the navigator: a package written
//! with `PackageWriter` in a temporary directory, frames rendered over it
//! at true poses, and the reference navigator run on inertial increments
//! and those frames.

mod appearance;
mod content;
mod cost;
mod dem;
mod fixes;
pub(crate) mod flight;
mod geometry;
mod inspection;
mod load;
mod precheck;
mod reacquire;
mod snapshot;
mod world;

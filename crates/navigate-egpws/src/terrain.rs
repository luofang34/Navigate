//! The terrain-elevation seam.

use navigate_contract::GeodeticPosition;

/// One queryable terrain-elevation source.
///
/// Implementations bind signed terrain packages supplied by the host
/// side; none exist in this repository. The trait is the seam that lets
/// [`crate::assess`] refuse honestly when no database is bound instead
/// of guessing a clearance.
pub trait TerrainDatabase {
    /// Terrain elevation in meters at `position`, or `None` where the
    /// database has no coverage.
    ///
    /// The vertical datum of the returned elevation is an open
    /// integration question (see the crate-level docs); an
    /// implementation documents which datum it serves.
    fn elevation_m(&self, position: &GeodeticPosition) -> Option<f64>;
}

//! The terrain-elevation seams.

use navigate_contract::{Ellipsoid, GeodeticPosition};

/// One queryable source of ellipsoidal terrain heights.
///
/// Implementations bind signed terrain packages that the host side
/// supplies. The trait is the seam that lets
/// [`crate::assess_clearance`] refuse when no source is bound, instead
/// of a guess of the clearance.
pub trait TerrainElevation {
    /// The terrain height in meters above [`Self::ellipsoid`] at
    /// `position`, or `None` where the source has no coverage.
    ///
    /// The height is ellipsoidal. A source of orthometric heights (above
    /// a geoid, a selenoid, or an areoid) goes through
    /// [`crate::OrthometricTerrain`].
    fn elevation_m(&self, position: &GeodeticPosition) -> Option<f64>;

    /// The reference ellipsoid of the positions and the heights
    /// (ADR-0013, ADR-0015). [`crate::assess_clearance`] refuses a
    /// solution on another ellipsoid. Each implementation names its
    /// ellipsoid explicitly.
    fn ellipsoid(&self) -> Ellipsoid;
}

/// One queryable source of orthometric terrain heights, such as a
/// digital elevation model above mean sea level, the selenoid, or the
/// areoid.
pub trait OrthometricElevation {
    /// The orthometric terrain height `H` in meters at `position`, or
    /// `None` where the source has no coverage.
    fn orthometric_height_m(&self, position: &GeodeticPosition) -> Option<f64>;

    /// The reference ellipsoid of the horizontal positions.
    fn ellipsoid(&self) -> Ellipsoid;
}

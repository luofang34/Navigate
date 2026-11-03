//! The vertical-datum adapter from orthometric to ellipsoidal terrain
//! heights (ADR-0016).

use navigate_contract::{Ellipsoid, GeodeticPosition};
use navigate_geodesy::{GeoidUndulation, orthometric_to_ellipsoidal_m};

use crate::terrain::{OrthometricElevation, TerrainElevation};

/// A source of ellipsoidal terrain heights from an orthometric source and
/// an undulation model of its vertical datum.
///
/// The ellipsoidal height is `h = H + N`, with `H` the orthometric height
/// of the terrain and `N` the undulation of the geoid, the selenoid, or
/// the areoid. Thus [`crate::assess_clearance`] compares two
/// ellipsoidal heights on any body. The adapter fails closed: it gives
/// no height where the undulation model has no coverage, where the model
/// is on a different ellipsoid, or where a value is not finite. It never
/// uses a zero undulation in place of a missing one.
#[derive(Debug, Clone, Copy)]
pub struct OrthometricTerrain<'a, T: ?Sized, D: ?Sized> {
    terrain: &'a T,
    datum: &'a D,
}

impl<'a, T, D> OrthometricTerrain<'a, T, D>
where
    T: OrthometricElevation + ?Sized,
    D: GeoidUndulation + ?Sized,
{
    /// The adapter of the orthometric source `terrain` and the undulation
    /// model `datum`.
    #[must_use]
    pub const fn new(terrain: &'a T, datum: &'a D) -> Self {
        Self { terrain, datum }
    }
}

impl<T, D> TerrainElevation for OrthometricTerrain<'_, T, D>
where
    T: OrthometricElevation + ?Sized,
    D: GeoidUndulation + ?Sized,
{
    fn elevation_m(&self, position: &GeodeticPosition) -> Option<f64> {
        let orthometric_m = self.terrain.orthometric_height_m(position)?;
        orthometric_to_ellipsoidal_m(
            self.datum,
            self.terrain.ellipsoid(),
            position.latitude_rad,
            position.longitude_rad,
            orthometric_m,
        )
    }

    fn ellipsoid(&self) -> Ellipsoid {
        self.terrain.ellipsoid()
    }
}

#[cfg(test)]
mod tests;

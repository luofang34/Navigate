//! Orthographic map pixels with surface heights in one local metric frame.
use crate::{CameraPose, EstimateQuality, MapRevision, VisualError};
use image::GrayImage;
use nalgebra::{Matrix2, SMatrix, Vector2, Vector3};

/// Affine horizontal coordinates of integer-centred raster pixels, in metres.
#[derive(Clone, Copy, Debug)]
pub struct RasterGeometry {
    /// Horizontal coordinates at column zero, row zero.
    pub origin: Vector2<f64>,
    /// Horizontal displacement for one column.
    pub column: Vector2<f64>,
    /// Horizontal displacement for one row.
    pub row: Vector2<f64>,
}
impl RasterGeometry {
    /// Check for finite coordinates and a non-singular horizontal grid.
    ///
    /// # Errors
    /// Rejects a singular or poorly conditioned pixel basis.
    pub fn validate(&self) -> Result<(), VisualError> {
        let basis = Matrix2::from_columns(&[self.column, self.row]);
        let determinant = basis.determinant().abs();
        let scale = self.column.norm() * self.row.norm();
        if !self
            .origin
            .iter()
            .chain(basis.iter())
            .all(|v| v.is_finite())
            || !scale.is_finite()
            || scale <= 0.0
            || determinant <= scale * 1e-8
        {
            return Err(VisualError::Invalid {
                field: "orthographic raster geometry",
            });
        }
        Ok(())
    }
    pub(crate) fn project(&self, world: Vector3<f64>) -> Option<Vector2<f64>> {
        let inverse = Matrix2::from_columns(&[self.column, self.row]).try_inverse()?;
        let pixel = inverse * (Vector2::new(world.x, world.y) - self.origin);
        pixel.iter().all(|v| v.is_finite()).then_some(pixel)
    }
}

/// Image and surface heights on an orthographic raster.
///
/// No query pose is required to prepare this reference. The host must verify the
/// map identity, horizontal registration, and height datum. This surface cannot
/// represent overhangs or separate surfaces at the same horizontal location.
pub struct OrthographicReference {
    /// Immutable source selection, including the pixel geometry and height datum.
    pub map: MapRevision,
    /// Reference pixels. Rows increase down the image.
    pub image: GrayImage,
    /// Row-major heights in metres. Non-finite cells have no surface evidence.
    pub heights_m: Vec<f32>,
    /// Horizontal coordinates in the same metric frame as the heights and prior.
    pub geometry: RasterGeometry,
}
impl OrthographicReference {
    /// Check raster dimensions, geometry, and source identity.
    ///
    /// # Errors
    /// Rejects dimensions outside 32 through 8192, inconsistent height counts,
    /// an invalid grid, or an empty source identity.
    pub fn validate(&self) -> Result<(), VisualError> {
        let (width, height) = self.image.dimensions();
        if !(32..=8192).contains(&width)
            || !(32..=8192).contains(&height)
            || self.heights_m.len() != self.image.as_raw().len()
            || self.map.release_id.is_empty()
            || self.map.manifest_sha256.len() != 64
            || !self
                .map
                .manifest_sha256
                .bytes()
                .all(|c| c.is_ascii_hexdigit())
        {
            return Err(VisualError::Invalid {
                field: "orthographic reference",
            });
        }
        self.geometry.validate()
    }
    /// Sample the nearest height and retain subpixel horizontal coordinates.
    ///
    /// Missing cells and pixels outside the raster interior return `None`.
    pub fn world_at(&self, pixel: Vector2<f64>) -> Option<Vector3<f64>> {
        let (width, height) = self.image.dimensions();
        if width < 2
            || height < 2
            || !pixel.iter().all(|v| v.is_finite())
            || pixel.x < 0.0
            || pixel.y < 0.0
            || pixel.x >= f64::from(width - 1)
            || pixel.y >= f64::from(height - 1)
        {
            return None;
        }
        let index = pixel.y.round() as usize * width as usize + pixel.x.round() as usize;
        let z = f64::from(*self.heights_m.get(index)?);
        let xy =
            self.geometry.origin + pixel.x * self.geometry.column + pixel.y * self.geometry.row;
        let world = Vector3::new(xy.x, xy.y, z);
        world.iter().all(|v| v.is_finite()).then_some(world)
    }
}

/// Pose relative to the supplied orthographic map coordinate system.
///
/// This result does not establish the map's geographic registration. The host
/// must bind the local frame before navigation fusion. Geometry covariance
/// excludes map, calibration, and correspondence systematic errors.
pub struct OrthographicEstimate {
    /// Camera pose in the reference's local metric frame.
    pub pose: CameraPose,
    /// Unchanged geometric acceptance diagnostics.
    pub quality: EstimateQuality,
    /// Pixel-noise-only local covariance, in metres and radians.
    pub geometry_covariance: SMatrix<f64, 6, 6>,
    /// Current observation identity.
    pub observation_sha256: String,
    /// Supplied map source identity.
    pub map: MapRevision,
    /// Image matching backend identity.
    pub backend: String,
}

#[cfg(test)]
mod tests;

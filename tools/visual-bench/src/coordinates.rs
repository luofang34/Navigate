//! Coordinates in the renderer's local Mercator plane.
//!
//! These conversions use MapLibre's spherical Earth scale at the map anchor.
//! They are not an ellipsoidal ENU conversion over long distances. Altitude
//! retains the package's declared datum. No geoid correction is applied.

use nalgebra::Vector3;
use navigate_visual::VisualError;

#[derive(Clone, Copy)]
pub(crate) struct MapFrame {
    anchor_x: f64,
    anchor_y: f64,
    scale_m: f64,
}

impl MapFrame {
    pub fn new([lat, lon]: [f64; 2]) -> Self {
        Self {
            anchor_x: (lon + 180.0) / 360.0,
            anchor_y: (1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0,
            scale_m: std::f64::consts::TAU * 6_371_008.8 * lat.to_radians().cos(),
        }
    }

    pub fn local(&self, [lat, lon, alt]: [f64; 3]) -> Result<Vector3<f64>, VisualError> {
        if ![lat, lon, alt].iter().all(|v| v.is_finite()) || lat.abs() > 85.0 || lon.abs() > 180.0 {
            return Err(VisualError::Invalid {
                field: "geographic camera position",
            });
        }
        let x = (lon + 180.0) / 360.0;
        let y = (1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0;
        Ok(Vector3::new(
            (x - self.anchor_x) * self.scale_m,
            (self.anchor_y - y) * self.scale_m,
            alt,
        ))
    }

    pub fn longitude_latitude_altitude(&self, position: Vector3<f64>) -> [f64; 3] {
        let x = self.anchor_x + position.x / self.scale_m;
        let y = self.anchor_y - position.y / self.scale_m;
        [
            (x * 360.0).rem_euclid(360.0) - 180.0,
            (std::f64::consts::PI * (1.0 - 2.0 * y))
                .sinh()
                .atan()
                .to_degrees(),
            position.z,
        ]
    }
}

#[cfg(test)]
mod tests;

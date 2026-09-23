//! The declared local frame shared by a reference render and the poses solved in it.

use nalgebra::Vector3;

use crate::VisualError;

/// Sphere radius of the Web Mercator scale that MapLibre renders with.
pub const MERCATOR_SPHERE_RADIUS_M: f64 = 6_371_008.8;

/// Latitude limit of the Web Mercator frame in degrees.
const MAX_LATITUDE_DEG: f64 = 85.0;

/// Local Cartesian frame of a [`crate::ReferenceView`] and of every pose derived from it.
///
/// Axes are x east, y north and z up, in metres. Altitude keeps the map
/// package's declared vertical datum. [`LocalFrame::local`] and
/// [`LocalFrame::geodetic`] are exact inverses, so a pose solved in the frame
/// converts back to the geodetic position that the renderer used.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum LocalFrame {
    /// A flat Web Mercator plane scaled at the anchor latitude.
    ///
    /// This is not an ellipsoidal east, north, up frame. The rendered world has
    /// no Earth curvature, and horizontal scale drifts away from the anchor
    /// latitude. [`LocalFrame::model_error_m`] bounds both effects for a point.
    AnchorMercator {
        /// Anchor latitude in degrees.
        anchor_lat_deg: f64,
        /// Anchor longitude in degrees.
        anchor_lon_deg: f64,
    },
}

impl LocalFrame {
    /// Declare an anchor Mercator frame.
    ///
    /// # Errors
    /// Rejects non-finite anchors and anchors outside the Mercator latitude limit.
    pub fn anchor_mercator(anchor_lat_deg: f64, anchor_lon_deg: f64) -> Result<Self, VisualError> {
        let frame = Self::AnchorMercator {
            anchor_lat_deg,
            anchor_lon_deg,
        };
        frame.validate()?;
        Ok(frame)
    }

    pub(crate) fn validate(&self) -> Result<(), VisualError> {
        let Self::AnchorMercator {
            anchor_lat_deg,
            anchor_lon_deg,
        } = *self;
        if !geodetic_in_range(anchor_lat_deg, anchor_lon_deg) {
            return Err(VisualError::Invalid {
                field: "local frame anchor",
            });
        }
        Ok(())
    }

    /// Convert latitude, longitude (degrees) and altitude (metres) into the frame.
    ///
    /// # Errors
    /// Rejects non-finite values and positions outside the Mercator latitude limit.
    pub fn local(&self, [lat, lon, alt]: [f64; 3]) -> Result<Vector3<f64>, VisualError> {
        if !geodetic_in_range(lat, lon) || !alt.is_finite() {
            return Err(VisualError::Invalid {
                field: "geographic position",
            });
        }
        let [ax, ay] = self.anchor_mercator_xy();
        let [x, y] = mercator_xy(lat, lon);
        let scale = self.mercator_scale_m();
        Ok(Vector3::new((x - ax) * scale, (ay - y) * scale, alt))
    }

    /// Convert a frame position into latitude, longitude (degrees) and altitude (metres).
    pub fn geodetic(&self, position: Vector3<f64>) -> [f64; 3] {
        let [x, y] = self.mercator_xy(position);
        [
            (std::f64::consts::PI * (1.0 - 2.0 * y))
                .sinh()
                .atan()
                .to_degrees(),
            (x * 360.0).rem_euclid(360.0) - 180.0,
            position.z,
        ]
    }

    /// Unit Web Mercator coordinates of a frame position, as MapLibre addresses tiles.
    pub fn mercator_xy(&self, position: Vector3<f64>) -> [f64; 2] {
        let [ax, ay] = self.anchor_mercator_xy();
        let scale = self.mercator_scale_m();
        [ax + position.x / scale, ay - position.y / scale]
    }

    /// Frame-model error in metres at a position, compared with a curved Earth.
    ///
    /// The sum of the curvature drop at the horizontal range from the origin and
    /// the horizontal scale drift from the anchor latitude. A fusion adapter adds
    /// this to its error budget. It is a model bound, not a measured error.
    pub fn model_error_m(&self, position: Vector3<f64>) -> f64 {
        let Self::AnchorMercator { anchor_lat_deg, .. } = *self;
        let range = position.xy().norm();
        let curvature = range * range / (2.0 * MERCATOR_SPHERE_RADIUS_M);
        let [lat, _, _] = self.geodetic(position);
        let drift = (lat.to_radians().cos() / anchor_lat_deg.to_radians().cos() - 1.0).abs();
        curvature + range * drift
    }

    fn anchor_mercator_xy(&self) -> [f64; 2] {
        let Self::AnchorMercator {
            anchor_lat_deg,
            anchor_lon_deg,
        } = *self;
        mercator_xy(anchor_lat_deg, anchor_lon_deg)
    }

    /// Metres per unit of Web Mercator coordinate at the anchor latitude.
    pub fn mercator_scale_m(&self) -> f64 {
        let Self::AnchorMercator { anchor_lat_deg, .. } = *self;
        std::f64::consts::TAU * MERCATOR_SPHERE_RADIUS_M * anchor_lat_deg.to_radians().cos()
    }
}

fn geodetic_in_range(lat: f64, lon: f64) -> bool {
    lat.is_finite() && lon.is_finite() && lat.abs() <= MAX_LATITUDE_DEG && lon.abs() <= 180.0
}

fn mercator_xy(lat: f64, lon: f64) -> [f64; 2] {
    [
        (lon + 180.0) / 360.0,
        (1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0,
    ]
}

#[cfg(test)]
mod tests;

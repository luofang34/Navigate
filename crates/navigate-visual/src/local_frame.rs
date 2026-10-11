//! The declared local frame shared by a reference render and the poses solved in it.

use nalgebra::{UnitQuaternion, Vector3};
mod spherical;

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
    /// East, north, up axes at an anchor on the mean-radius sphere.
    ///
    /// The origin is at zero map altitude. Position z is a Cartesian coordinate,
    /// not geodetic altitude away from the anchor. Terrain rays include curvature.
    SphericalEnu {
        /// Anchor latitude in degrees.
        anchor_lat_deg: f64,
        /// Anchor longitude in degrees.
        anchor_lon_deg: f64,
    },
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

    /// Declare a curved spherical frame with the mean Earth radius.
    ///
    /// # Errors
    /// Rejects non-finite anchors and anchors outside the tile latitude limit.
    pub fn spherical_enu(anchor_lat_deg: f64, anchor_lon_deg: f64) -> Result<Self, VisualError> {
        let frame = Self::SphericalEnu {
            anchor_lat_deg,
            anchor_lon_deg,
        };
        frame.validate()?;
        Ok(frame)
    }

    /// Geographic anchor in latitude, longitude order, in degrees.
    pub fn anchor(&self) -> [f64; 2] {
        match *self {
            Self::AnchorMercator {
                anchor_lat_deg,
                anchor_lon_deg,
            }
            | Self::SphericalEnu {
                anchor_lat_deg,
                anchor_lon_deg,
            } => [anchor_lat_deg, anchor_lon_deg],
        }
    }

    pub(crate) fn validate(&self) -> Result<(), VisualError> {
        let [lat, lon] = self.anchor();
        if !geodetic_in_range(lat, lon) {
            return Err(VisualError::Invalid {
                field: "local frame anchor",
            });
        }
        Ok(())
    }

    /// Rotate local east, north, up axes at a geographic point into this frame.
    ///
    /// Mercator axes keep their flat-map orientation. Spherical axes follow curvature.
    ///
    /// # Errors
    /// Rejects invalid geographic coordinates.
    pub fn local_rotation(&self, [lat, lon]: [f64; 2]) -> Result<UnitQuaternion<f64>, VisualError> {
        self.validate()?;
        if !geodetic_in_range(lat, lon) {
            return Err(VisualError::Invalid {
                field: "geographic orientation",
            });
        }
        Ok(match self {
            Self::AnchorMercator { .. } => UnitQuaternion::identity(),
            Self::SphericalEnu { .. } => UnitQuaternion::from_matrix(
                &(spherical::basis(self.anchor()).transpose() * spherical::basis([lat, lon])),
            ),
        })
    }

    /// Convert latitude, longitude (degrees) and altitude (metres) into the frame.
    ///
    /// # Errors
    /// Rejects non-finite values and positions outside the Mercator latitude limit.
    pub fn local(&self, [lat, lon, alt]: [f64; 3]) -> Result<Vector3<f64>, VisualError> {
        self.validate()?;
        if !geodetic_in_range(lat, lon) || !alt.is_finite() {
            return Err(VisualError::Invalid {
                field: "geographic position",
            });
        }
        if matches!(self, Self::SphericalEnu { .. }) {
            if alt <= -MERCATOR_SPHERE_RADIUS_M {
                return Err(VisualError::Invalid {
                    field: "spherical altitude",
                });
            }
            return Ok(spherical::local(self.anchor(), [lat, lon, alt]));
        }
        let [ax, ay] = self.anchor_mercator_xy();
        let [x, y] = mercator_xy(lat, lon);
        let scale = self.mercator_scale_m();
        Ok(Vector3::new((x - ax) * scale, (ay - y) * scale, alt))
    }

    /// Convert a frame position into latitude, longitude (degrees) and altitude (metres).
    pub fn geodetic(&self, position: Vector3<f64>) -> [f64; 3] {
        if matches!(self, Self::SphericalEnu { .. }) {
            return spherical::geodetic(self.anchor(), position);
        }
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
        if matches!(self, Self::SphericalEnu { .. }) {
            let [lat, lon, _] = self.geodetic(position);
            return mercator_xy(lat, lon);
        }
        let [ax, ay] = self.anchor_mercator_xy();
        let scale = self.mercator_scale_m();
        [ax + position.x / scale, ay - position.y / scale]
    }

    /// Frame-model error in metres at a position, compared with a curved Earth.
    ///
    /// Mercator frames bound curvature drop and horizontal scale drift. Spherical
    /// frames report the displacement from WGS84 for the same geographic position
    /// and numeric altitude. Datum and terrain errors require separate budgets.
    pub fn model_error_m(&self, position: Vector3<f64>) -> f64 {
        if matches!(self, Self::SphericalEnu { .. }) {
            return spherical::ellipsoid_displacement(self.anchor(), position);
        }
        let [anchor_lat_deg, _] = self.anchor();
        let range = position.xy().norm();
        let curvature = range * range / (2.0 * MERCATOR_SPHERE_RADIUS_M);
        let [lat, _, _] = self.geodetic(position);
        let drift = (lat.to_radians().cos() / anchor_lat_deg.to_radians().cos() - 1.0).abs();
        curvature + range * drift
    }

    fn anchor_mercator_xy(&self) -> [f64; 2] {
        let [lat, lon] = self.anchor();
        mercator_xy(lat, lon)
    }

    /// Metres per unit of Web Mercator coordinate at the anchor latitude.
    pub fn mercator_scale_m(&self) -> f64 {
        let [lat, _] = self.anchor();
        std::f64::consts::TAU * MERCATOR_SPHERE_RADIUS_M * lat.to_radians().cos()
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

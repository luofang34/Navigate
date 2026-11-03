//! Exact east, north, up coordinates at a geodetic origin on one body ellipsoid.
//!
//! A conversion goes through the body-fixed frame of the ellipsoid and one
//! rotation, so it has no latitude limit. At a pole the origin longitude
//! selects the east axis.
use super::spherical::basis;
use crate::VisualError;
use nalgebra::Vector3;
use navigate_contract::GeodeticPosition;
use navigate_geodesy::{
    BodyEllipsoid, BodyFixedPosition, body_fixed_to_geodetic, geodetic_to_body_fixed,
};

/// Largest round-trip error of an accepted conversion, in metres.
const ROUND_TRIP_TOLERANCE_M: f64 = 0.001;

/// An ellipsoid and a frame origin in latitude, longitude (degrees) and height (metres).
#[derive(Clone, Copy)]
pub(super) struct Tangent {
    pub ellipsoid: BodyEllipsoid,
    pub origin: [f64; 3],
}

impl Tangent {
    fn body_fixed(&self, [lat, lon, height]: [f64; 3]) -> Vector3<f64> {
        let geodetic = GeodeticPosition::new(lat.to_radians(), lon.to_radians(), height);
        let p = geodetic_to_body_fixed(&geodetic, self.ellipsoid);
        Vector3::new(p.x_m, p.y_m, p.z_m)
    }

    pub(super) fn local(&self, geographic: [f64; 3]) -> Result<Vector3<f64>, VisualError> {
        let (meridian_radius, _) = self
            .ellipsoid
            .radii_of_curvature_m(geographic[0].to_radians());
        // Below the centre of curvature, normal coordinates fold and do not define a unique height.
        if geographic[2] + meridian_radius <= 0.0 {
            return Err(VisualError::Invalid {
                field: "ellipsoid normal coordinate fold",
            });
        }
        let [lat, lon, _] = self.origin;
        let position = basis([lat, lon]).transpose()
            * (self.body_fixed(geographic) - self.body_fixed(self.origin));
        let recovered = self.geodetic(position);
        let a = self.ellipsoid.semi_major_axis_m();
        let longitude_error = (recovered[1] - geographic[1] + 180.0).rem_euclid(360.0) - 180.0;
        let errors = [
            (recovered[0] - geographic[0]).to_radians() * a,
            longitude_error.to_radians() * geographic[0].to_radians().cos() * a,
            recovered[2] - geographic[2],
        ];
        // A numerical check rejects non-invertible depths and precision loss without a flight-height cap.
        if !position.iter().all(|v| v.is_finite())
            || errors
                .iter()
                .any(|e| !e.is_finite() || e.abs() > ROUND_TRIP_TOLERANCE_M)
        {
            return Err(VisualError::Invalid {
                field: "ellipsoid coordinate roundtrip exceeds 1 mm",
            });
        }
        Ok(position)
    }

    pub(super) fn geodetic(&self, position: Vector3<f64>) -> [f64; 3] {
        let [lat, lon, _] = self.origin;
        let p = self.body_fixed(self.origin) + basis([lat, lon]) * position;
        let geo = body_fixed_to_geodetic(&BodyFixedPosition::new(p.x, p.y, p.z), self.ellipsoid);
        [
            geo.latitude_rad.to_degrees(),
            geo.longitude_rad.to_degrees(),
            geo.altitude_m,
        ]
    }
}

#[cfg(test)]
mod tests;

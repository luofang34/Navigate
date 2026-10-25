//! Ellipsoid coordinates use the shared geodesy implementation.
use super::spherical::basis;
use crate::VisualError;
use nalgebra::Vector3;
use navigate_contract::GeodeticPosition;
use navigate_geodesy::{EcefPosition, ecef_to_geodetic, geodetic_to_ecef};

fn ecef([lat, lon, height]: [f64; 3]) -> Vector3<f64> {
    let p = geodetic_to_ecef(&GeodeticPosition::new(
        lat.to_radians(),
        lon.to_radians(),
        height,
    ));
    Vector3::new(p.x_m, p.y_m, p.z_m)
}
pub(super) fn local(anchor: [f64; 2], geographic: [f64; 3]) -> Result<Vector3<f64>, VisualError> {
    let a = navigate_geodesy::wgs84::SEMI_MAJOR_AXIS_M;
    let e2 = navigate_geodesy::wgs84::FIRST_ECCENTRICITY_SQUARED;
    let denominator = 1.0 - e2 * geographic[0].to_radians().sin().powi(2);
    let meridian_radius = a * (1.0 - e2) / denominator.powf(1.5);
    // Below the centre of curvature, normal coordinates fold and do not define a unique height.
    if geographic[2] + meridian_radius <= 0.0 {
        return Err(VisualError::Invalid {
            field: "WGS84 normal coordinate fold",
        });
    }
    let position =
        basis(anchor).transpose() * (ecef(geographic) - ecef([anchor[0], anchor[1], 0.0]));
    let recovered = geodetic(anchor, position);
    let longitude_error = (recovered[1] - geographic[1] + 180.0).rem_euclid(360.0) - 180.0;
    let errors = [
        (recovered[0] - geographic[0]).to_radians() * navigate_geodesy::wgs84::SEMI_MAJOR_AXIS_M,
        longitude_error.to_radians()
            * geographic[0].to_radians().cos()
            * navigate_geodesy::wgs84::SEMI_MAJOR_AXIS_M,
        recovered[2] - geographic[2],
    ];
    // A numerical check rejects non-invertible depths and precision loss without a flight-height cap.
    if !position.iter().all(|v| v.is_finite())
        || errors.iter().any(|e| !e.is_finite() || e.abs() > 0.001)
    {
        return Err(VisualError::Invalid {
            field: "WGS84 coordinate roundtrip exceeds 1 mm",
        });
    }
    Ok(position)
}

pub(super) fn geodetic(anchor: [f64; 2], position: Vector3<f64>) -> [f64; 3] {
    let p = ecef([anchor[0], anchor[1], 0.0]) + basis(anchor) * position;
    let geo = ecef_to_geodetic(&EcefPosition::new(p.x, p.y, p.z));
    [
        geo.latitude_rad.to_degrees(),
        geo.longitude_rad.to_degrees(),
        geo.altitude_m,
    ]
}

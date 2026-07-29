//! Local tangent plane: North-East-Down offsets relative to a geodetic
//! origin, converted through ECEF with an exact rotation — no
//! small-angle approximations, so accuracy degrades only with the
//! (real) curvature divergence between the plane and the ellipsoid.

use navigate_contract::GeodeticPosition;

use crate::ecef::{EcefPosition, ecef_to_geodetic, geodetic_to_ecef};
use crate::error::GeodesyError;

/// An offset in a local North-East-Down frame. Down is positive toward
/// the ellipsoid, along the origin's ellipsoidal normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NedOffset {
    /// North component in meters.
    pub north_m: f64,
    /// East component in meters.
    pub east_m: f64,
    /// Down component in meters (positive descending).
    pub down_m: f64,
}

impl NedOffset {
    /// Builds an offset from its components.
    #[must_use]
    pub const fn new(north_m: f64, east_m: f64, down_m: f64) -> Self {
        Self {
            north_m,
            east_m,
            down_m,
        }
    }
}

/// A local tangent plane anchored at a geodetic origin. Precomputes the
/// origin's ECEF position and the ECEF→NED rotation so conversions are
/// pure arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalTangentPlane {
    origin: GeodeticPosition,
    origin_ecef: EcefPosition,
    sin_lat: f64,
    cos_lat: f64,
    sin_lon: f64,
    cos_lon: f64,
}

impl LocalTangentPlane {
    /// Anchors a plane at `origin`. Rejecting an implausible origin here
    /// keeps [`Self::to_ned`] and [`Self::from_ned`] total.
    pub fn new(origin: GeodeticPosition) -> Result<Self, GeodesyError> {
        if !origin.is_plausible() {
            return Err(GeodesyError::ImplausibleOrigin {
                latitude_rad: origin.latitude_rad,
                longitude_rad: origin.longitude_rad,
                altitude_m: origin.altitude_m,
            });
        }
        let (sin_lat, cos_lat) = origin.latitude_rad.sin_cos();
        let (sin_lon, cos_lon) = origin.longitude_rad.sin_cos();
        Ok(Self {
            origin,
            origin_ecef: geodetic_to_ecef(&origin),
            sin_lat,
            cos_lat,
            sin_lon,
            cos_lon,
        })
    }

    /// The anchoring origin.
    #[must_use]
    pub const fn origin(&self) -> GeodeticPosition {
        self.origin
    }

    /// Expresses `position` as a NED offset from the origin: ECEF delta
    /// rotated into the origin's tangent frame.
    #[must_use]
    pub fn to_ned(&self, position: &GeodeticPosition) -> NedOffset {
        let ecef = geodetic_to_ecef(position);
        let dx = ecef.x_m - self.origin_ecef.x_m;
        let dy = ecef.y_m - self.origin_ecef.y_m;
        let dz = ecef.z_m - self.origin_ecef.z_m;
        // Rows of the ECEF→NED rotation for the origin's latitude and
        // longitude.
        NedOffset {
            north_m: -self.sin_lat * self.cos_lon * dx - self.sin_lat * self.sin_lon * dy
                + self.cos_lat * dz,
            east_m: -self.sin_lon * dx + self.cos_lon * dy,
            down_m: -self.cos_lat * self.cos_lon * dx
                - self.cos_lat * self.sin_lon * dy
                - self.sin_lat * dz,
        }
    }

    /// Resolves a NED offset back to a geodetic position: the transpose
    /// rotation into ECEF, then [`ecef_to_geodetic`].
    #[must_use]
    pub fn from_ned(&self, offset: &NedOffset) -> GeodeticPosition {
        let n = offset.north_m;
        let e = offset.east_m;
        let d = offset.down_m;
        let ecef = EcefPosition {
            x_m: self.origin_ecef.x_m
                - self.sin_lat * self.cos_lon * n
                - self.sin_lon * e
                - self.cos_lat * self.cos_lon * d,
            y_m: self.origin_ecef.y_m - self.sin_lat * self.sin_lon * n + self.cos_lon * e
                - self.cos_lat * self.sin_lon * d,
            z_m: self.origin_ecef.z_m + self.cos_lat * n - self.sin_lat * d,
        };
        ecef_to_geodetic(&ecef)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;

    fn plane_at_47n_8e() -> LocalTangentPlane {
        let origin = GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 500.0);
        LocalTangentPlane::new(origin).expect("plausible origin")
    }

    #[test]
    fn implausible_origin_is_rejected() {
        let origin = GeodeticPosition::new(2.0, 0.0, 0.0);
        assert!(matches!(
            LocalTangentPlane::new(origin),
            Err(GeodesyError::ImplausibleOrigin { .. })
        ));
    }

    #[test]
    fn origin_maps_to_zero_offset() {
        let plane = plane_at_47n_8e();
        let ned = plane.to_ned(&plane.origin());
        assert!(ned.north_m.abs() < 1e-9);
        assert!(ned.east_m.abs() < 1e-9);
        assert!(ned.down_m.abs() < 1e-9);
    }

    #[test]
    fn point_above_origin_is_pure_negative_down() {
        let plane = plane_at_47n_8e();
        let above = GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 600.0);
        let ned = plane.to_ned(&above);
        assert!(ned.north_m.abs() < 1e-6);
        assert!(ned.east_m.abs() < 1e-6);
        assert!((ned.down_m + 100.0).abs() < 1e-6);
    }

    #[test]
    fn small_northward_step_matches_the_meridian_arc() {
        let plane = plane_at_47n_8e();
        let north = GeodeticPosition::new(47.01_f64.to_radians(), 8.0_f64.to_radians(), 500.0);
        let ned = plane.to_ned(&north);
        // Meridian arc for 0.01° at 47°N is ≈ 1111.7 m; the plane drops
        // below the ellipsoid ahead, so down is small and positive.
        assert!((ned.north_m - 1_111.7).abs() < 1.0);
        assert!(ned.east_m.abs() < 1e-6);
        assert!(ned.down_m > 0.0 && ned.down_m < 0.2);
    }

    #[test]
    fn geodetic_round_trip_within_100_km_stays_below_a_micrometer() {
        let plane = plane_at_47n_8e();
        let candidates = [
            GeodeticPosition::new(47.5_f64.to_radians(), 8.0_f64.to_radians(), 10_500.0),
            GeodeticPosition::new(46.4_f64.to_radians(), 9.0_f64.to_radians(), 0.0),
            GeodeticPosition::new(47.0_f64.to_radians(), 6.9_f64.to_radians(), 3_000.0),
        ];
        for original in candidates {
            let ned = plane.to_ned(&original);
            let round_tripped = plane.from_ned(&ned);
            let error_m = geodetic_to_ecef(&original).distance_m(&geodetic_to_ecef(&round_tripped));
            assert!(error_m < 1e-6, "round trip error {error_m} m");
        }
    }

    #[test]
    fn ned_round_trip_within_100_km_stays_below_a_micrometer() {
        let plane = plane_at_47n_8e();
        let offsets = [
            NedOffset::new(100_000.0, 0.0, 0.0),
            NedOffset::new(-30_000.0, 70_000.0, -9_500.0),
            NedOffset::new(0.0, -100_000.0, 2_000.0),
        ];
        for original in offsets {
            let geodetic = plane.from_ned(&original);
            let round_tripped = plane.to_ned(&geodetic);
            let dn = round_tripped.north_m - original.north_m;
            let de = round_tripped.east_m - original.east_m;
            let dd = round_tripped.down_m - original.down_m;
            let error_m = (dn * dn + de * de + dd * dd).sqrt();
            assert!(error_m < 1e-6, "round trip error {error_m} m");
        }
    }
}

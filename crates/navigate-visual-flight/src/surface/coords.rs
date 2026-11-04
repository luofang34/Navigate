//! Ellipsoidal tangent-frame coordinates, Web Mercator tile addressing, and terrain trait impls.

use super::Surface;
use nalgebra::{Matrix3, Vector3};
use navigate_contract::GeodeticPosition;
use navigate_geodesy::{
    BodyEllipsoid, BodyFixedPosition, body_fixed_to_geodetic, geodetic_to_body_fixed,
};
use navigate_locate::corridor::plausibility::CorridorTerrain;

impl Surface {
    /// Convert `(lat_deg, lon_deg, alt_m)` on `self.ellipsoid` to local `ENU` meters.
    #[must_use]
    pub fn local(&self, lat_deg: f64, lon_deg: f64, alt_m: f64) -> [f64; 3] {
        let p = geodetic_to_body_fixed(
            &GeodeticPosition::new(lat_deg.to_radians(), lon_deg.to_radians(), alt_m),
            self.ellipsoid,
        );
        let body_fixed = Vector3::new(p.x_m, p.y_m, p.z_m);
        let enu = self.enu_from_ecef * (body_fixed - self.origin_ecef);
        [enu.x, enu.y, enu.z]
    }

    /// Convert local `ENU` meters to `(lat_deg, lon_deg, alt_m)` on `self.ellipsoid`.
    #[must_use]
    pub fn geodetic(&self, local_enu: [f64; 3]) -> [f64; 3] {
        let enu = Vector3::new(local_enu[0], local_enu[1], local_enu[2]);
        let bf = self.origin_ecef + self.enu_from_ecef.transpose() * enu;
        let geo = body_fixed_to_geodetic(&BodyFixedPosition::new(bf.x, bf.y, bf.z), self.ellipsoid);
        let lon_deg = if geo.latitude_rad.abs() > 89.999_999_f64.to_radians() {
            self.origin_lon
        } else {
            geo.longitude_rad.to_degrees()
        };
        [geo.latitude_rad.to_degrees(), lon_deg, geo.altitude_m]
    }

    /// Build a world-to-camera rotation matrix `R` at `(lat_deg, lon_deg)` for `(heading, down, roll)`.
    #[must_use]
    pub fn camera_rotation(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        heading_deg: f64,
        down_deg: f64,
        roll_deg: f64,
    ) -> Matrix3<f64> {
        let (_, local_enu_from_bf) = enu_frame_on(self.ellipsoid, lat_deg, lon_deg);
        let basis = local_enu_from_bf * self.enu_from_ecef.transpose();
        let h = heading_deg.to_radians();
        let d = down_deg.to_radians();
        let r = roll_deg.to_radians();

        let fwd_local = Vector3::new(h.sin() * d.cos(), h.cos() * d.cos(), -d.sin());
        let right0 = Vector3::new(h.cos(), -h.sin(), 0.0);
        let down0 = fwd_local.cross(&right0).normalize();
        let right = right0 * r.cos() + down0 * r.sin();
        let down = -right0 * r.sin() + down0 * r.cos();

        let cam_in_local =
            Matrix3::from_rows(&[right.transpose(), down.transpose(), fwd_local.transpose()]);
        cam_in_local * basis
    }

    /// Compute `(tile_x, tile_y, frac_x, frac_y)` at `zoom` using Web Mercator on non-polar
    /// WGS84 surfaces and metric tangent-plane coordinates on polar or planetary surfaces.
    #[must_use]
    pub(super) fn tile_coords(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        zoom: u32,
    ) -> (u32, u32, f64, f64) {
        if self.ellipsoid == BodyEllipsoid::WGS84 && self.origin_lat.abs() < 85.0 {
            return mercator_tile_coords(lat_deg, lon_deg, zoom);
        }
        let enu = self.local(lat_deg, lon_deg, 0.0);
        tangent_tile_coords(self.ellipsoid, enu[0], enu[1], zoom)
    }
}

pub(super) fn tangent_tile_coords(
    ellipsoid: BodyEllipsoid,
    east_m: f64,
    north_m: f64,
    zoom: u32,
) -> (u32, u32, f64, f64) {
    let n_u = 1_u32 << zoom.min(24);
    let n = f64::from(n_u);
    let half = 0.5 * n;
    let span_m = (std::f64::consts::TAU * ellipsoid.semi_major_axis_m() / n).max(1.0);
    let x = half + east_m / span_m;
    let y = half - north_m / span_m;
    let tx = (x.floor() as i64).clamp(0, i64::from(n_u) - 1) as u32;
    let ty = (y.floor() as i64).clamp(0, i64::from(n_u) - 1) as u32;
    (
        tx,
        ty,
        (x - f64::from(tx)).clamp(0.0, 0.999_999),
        (y - f64::from(ty)).clamp(0.0, 0.999_999),
    )
}

impl CorridorTerrain for Surface {
    fn geodetic(&self, local_enu: [f64; 3]) -> [f64; 3] {
        Self::geodetic(self, local_enu)
    }

    fn local(&self, lat_deg: f64, lon_deg: f64, alt_m: f64) -> [f64; 3] {
        Self::local(self, lat_deg, lon_deg, alt_m)
    }

    fn sample_dem(&self, lat_deg: f64, lon_deg: f64) -> Option<f64> {
        Self::sample_dem(self, lat_deg, lon_deg)
    }

    fn camera_rotation(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        heading_deg: f64,
        down_deg: f64,
        roll_deg: f64,
    ) -> Matrix3<f64> {
        Self::camera_rotation(self, lat_deg, lon_deg, heading_deg, down_deg, roll_deg)
    }
}

/// Build the WGS84 `(origin_body_fixed, enu_from_body_fixed)` tangent basis at `(lat_deg, lon_deg)`.
#[must_use]
pub fn enu_frame(lat_deg: f64, lon_deg: f64) -> (Vector3<f64>, Matrix3<f64>) {
    enu_frame_on(BodyEllipsoid::WGS84, lat_deg, lon_deg)
}

/// Build `(origin_body_fixed, enu_from_body_fixed)` on `ellipsoid` at `(lat_deg, lon_deg)`.
#[must_use]
pub fn enu_frame_on(
    ellipsoid: BodyEllipsoid,
    lat_deg: f64,
    lon_deg: f64,
) -> (Vector3<f64>, Matrix3<f64>) {
    let p = geodetic_to_body_fixed(
        &GeodeticPosition::new(lat_deg.to_radians(), lon_deg.to_radians(), 0.0),
        ellipsoid,
    );
    let origin = Vector3::new(p.x_m, p.y_m, p.z_m);
    let (slat, clat) = lat_deg.to_radians().sin_cos();
    let (slon, clon) = lon_deg.to_radians().sin_cos();
    let east = Vector3::new(-slon, clon, 0.0);
    let north = Vector3::new(-slat * clon, -slat * slon, clat);
    let up = Vector3::new(clat * clon, clat * slon, slat);
    let rot = Matrix3::from_rows(&[east.transpose(), north.transpose(), up.transpose()]);
    (origin, rot)
}

/// Compute Web Mercator `(tile_x, tile_y, frac_x, frac_y)` at `zoom` for `(lat_deg, lon_deg)`.
#[must_use]
pub fn mercator_tile_coords(lat_deg: f64, lon_deg: f64, zoom: u32) -> (u32, u32, f64, f64) {
    let n = f64::from(1_u32 << zoom);
    let lat_clamped = lat_deg.clamp(-85.051_128_78, 85.051_128_78);
    let x = ((lon_deg + 180.0) / 360.0) * n;
    let y = (1.0 - lat_clamped.to_radians().tan().asinh() / std::f64::consts::PI) * 0.5 * n;
    let tx = (x.floor() as i64).clamp(0, (n as i64) - 1) as u32;
    let ty = (y.floor() as i64).clamp(0, (n as i64) - 1) as u32;
    (
        tx,
        ty,
        (x - f64::from(tx)).clamp(0.0, 0.999_999),
        (y - f64::from(ty)).clamp(0.0, 0.999_999),
    )
}

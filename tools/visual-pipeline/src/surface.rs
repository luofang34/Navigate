//! WGS84 terrain surface and ray-marching renderer (`photo_surface`).

mod catalog;
pub(crate) mod gpu;
pub mod horizon;
mod sparse;

pub use catalog::{ContextSurface, build_reference_catalog, build_reference_catalog_zoom};
pub use gpu::{GpuSurfaceView, SurfaceGpuRenderer};
pub use horizon::{RansacHorizonEstimator, horizon_attitude};
pub use sparse::{SurfaceAdaptiveStats, SurfaceRayHit};

use crate::{
    PipelineError,
    util::{file_sha256, read_json},
};
use image::{Rgb, RgbImage};
use nalgebra::{Matrix3, Vector3};
use navigate_contract::GeodeticPosition;
use navigate_geodesy::{EcefPosition, ecef_to_geodetic, geodetic_to_ecef};
use navigate_visual_render::RenderScene;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, path::Path};

/// Rendered view from [`Surface::render`] or [`ContextSurface::render`].
pub type SurfaceRender = (RgbImage, Vec<f32>, Vec<[f64; 3]>);

/// Tile record inside `catalog.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CatalogTile {
    /// Tile kind (`"imagery"` or `"dem"`).
    pub kind: String,
    /// Web Mercator `[z, x, y]` coordinates.
    pub xyz: [u32; 3],
    /// Relative tile path inside the catalog directory.
    pub path: String,
    /// Hex SHA-256 digest of the tile bytes.
    pub sha256: String,
    /// Byte length of the tile file.
    pub bytes: usize,
    /// Source URL or local origin.
    pub url: String,
}

/// Spherical terrain and orthoimagery surface loaded from a `photo-reference-catalog`.
#[derive(Clone, Debug)]
pub struct Surface {
    /// Origin latitude in degrees.
    pub origin_lat: f64,
    /// Origin longitude in degrees.
    pub origin_lon: f64,
    /// Geographic bounding box `[west, south, east, north]` in degrees.
    pub bbox: [f64; 4],
    /// Web Mercator zoom level for imagery tiles.
    pub imagery_zoom: u32,
    /// Web Mercator zoom level for DEM tiles.
    pub dem_zoom: u32,
    /// Catalog SHA-256 digest.
    pub catalog_sha256: String,
    /// Minimum DEM elevation in metres.
    pub min_elev_m: f64,
    /// Maximum DEM elevation in metres.
    pub max_elev_m: f64,
    pub(crate) origin_ecef: Vector3<f64>,
    pub(crate) enu_from_ecef: Matrix3<f64>,
    pub(super) imagery_tiles: HashMap<(u32, u32), RgbImage>,
    pub(super) dem_tiles: HashMap<(u32, u32), Vec<f32>>,
    pub(super) dem_tile_size: u32,
}

impl Surface {
    /// Load a [`Surface`] from `catalog.json` on disk.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when `catalog.json` or any present tile fails to decode.
    pub fn from_catalog(catalog_path: &Path) -> Result<Self, PipelineError> {
        let catalog_sha256 = file_sha256(catalog_path)?;
        let doc: Value = read_json(catalog_path)?;
        if doc.get("format").and_then(Value::as_str) != Some("photo-reference-catalog") {
            return Err(PipelineError::Invalid(format!(
                "{} is not a photo-reference-catalog",
                catalog_path.display()
            )));
        }
        let cfg = doc
            .get("configuration")
            .ok_or_else(|| PipelineError::Invalid("missing catalog configuration".into()))?;
        let bbox: [f64; 4] =
            serde_json::from_value(cfg["bbox"].clone()).map_err(|source| PipelineError::Json {
                context: "catalog.configuration.bbox".into(),
                source,
            })?;
        let imagery_zoom = cfg
            .get("imagery_zoom")
            .and_then(Value::as_u64)
            .unwrap_or(14) as u32;
        let dem_zoom = cfg.get("dem_zoom").and_then(Value::as_u64).unwrap_or(12) as u32;
        let origin_lon = 0.5 * (bbox[0] + bbox[2]);
        let origin_lat = 0.5 * (bbox[1] + bbox[3]);

        let root = catalog_path.parent().unwrap_or_else(|| Path::new("."));
        let tiles: Vec<CatalogTile> = doc
            .get("tiles")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|source| PipelineError::Json {
                context: "catalog.tiles".into(),
                source,
            })?
            .unwrap_or_default();

        let mut imagery_tiles = HashMap::new();
        let mut dem_tiles = HashMap::new();
        let mut min_elev = f64::INFINITY;
        let mut max_elev = f64::NEG_INFINITY;
        let mut dem_tile_size = 256u32;

        for tile in &tiles {
            let tile_path = root.join(&tile.path);
            if !tile_path.is_file() {
                continue;
            }
            let img = image::open(&tile_path)
                .map_err(|source| PipelineError::Image {
                    path: tile_path,
                    source,
                })?
                .to_rgb8();
            let xy = (tile.xyz[1], tile.xyz[2]);
            if tile.kind == "dem" {
                dem_tile_size = img.width().max(1);
                let mut elev = Vec::with_capacity((img.width() * img.height()) as usize);
                for p in img.pixels() {
                    let h = (f64::from(p[0]) * 256.0 + f64::from(p[1]) + f64::from(p[2]) / 256.0)
                        - 32_768.0;
                    min_elev = min_elev.min(h);
                    max_elev = max_elev.max(h);
                    elev.push(h as f32);
                }
                dem_tiles.insert(xy, elev);
            } else {
                imagery_tiles.insert(xy, img);
            }
        }
        if !min_elev.is_finite() {
            min_elev = 0.0;
            max_elev = 500.0;
        }
        let (origin_ecef, enu_from_ecef) = enu_frame(origin_lat, origin_lon);
        Ok(Self {
            origin_lat,
            origin_lon,
            bbox,
            imagery_zoom,
            dem_zoom,
            catalog_sha256,
            min_elev_m: min_elev,
            max_elev_m: max_elev,
            origin_ecef,
            enu_from_ecef,
            imagery_tiles,
            dem_tiles,
            dem_tile_size,
        })
    }

    /// Construct an in-memory synthetic [`Surface`] for deterministic unit tests.
    pub fn synthetic(origin_lat: f64, origin_lon: f64, elev_m: f32, image: RgbImage) -> Self {
        let (origin_ecef, enu_from_ecef) = enu_frame(origin_lat, origin_lon);
        let z = 12u32;
        let (tx, ty, _, _) = mercator_tile_coords(origin_lat, origin_lon, z);
        let mut imagery_tiles = HashMap::new();
        let mut dem_tiles = HashMap::new();
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let key = (
                    (tx as i32 + dx).max(0) as u32,
                    (ty as i32 + dy).max(0) as u32,
                );
                imagery_tiles.insert(key, image.clone());
                dem_tiles.insert(key, vec![elev_m; 256 * 256]);
            }
        }
        Self {
            origin_lat,
            origin_lon,
            bbox: [
                origin_lon - 0.2,
                origin_lat - 0.2,
                origin_lon + 0.2,
                origin_lat + 0.2,
            ],
            imagery_zoom: z,
            dem_zoom: z,
            catalog_sha256: "0".repeat(64),
            min_elev_m: f64::from(elev_m),
            max_elev_m: f64::from(elev_m),
            origin_ecef,
            enu_from_ecef,
            imagery_tiles,
            dem_tiles,
            dem_tile_size: 256,
        }
    }

    /// Build a WGS84 [`RenderScene`] for [`navigate_visual_render::GpuRenderer`] or [`navigate_visual_render::SparseSceneSampler`].
    #[must_use]
    pub fn to_render_scene(&self) -> Option<RenderScene> {
        gpu::SceneTemplate::from_surface(self).map(|t| t.to_render_scene())
    }

    /// Convert WGS84 `(lat_deg, lon_deg, alt_m)` to local `ENU` coordinates in meters.
    pub fn local(&self, lat_deg: f64, lon_deg: f64, alt_m: f64) -> [f64; 3] {
        let p = geodetic_to_ecef(&GeodeticPosition::new(
            lat_deg.to_radians(),
            lon_deg.to_radians(),
            alt_m,
        ));
        let ecef = Vector3::new(p.x_m, p.y_m, p.z_m);
        let enu = self.enu_from_ecef * (ecef - self.origin_ecef);
        [enu.x, enu.y, enu.z]
    }

    /// Convert local `ENU` coordinates in meters to WGS84 `(lat_deg, lon_deg, alt_m)`.
    pub fn geodetic(&self, local_enu: [f64; 3]) -> [f64; 3] {
        let enu = Vector3::new(local_enu[0], local_enu[1], local_enu[2]);
        let ecef = self.origin_ecef + self.enu_from_ecef.transpose() * enu;
        let geo = ecef_to_geodetic(&EcefPosition::new(ecef.x, ecef.y, ecef.z));
        [
            geo.latitude_rad.to_degrees(),
            geo.longitude_rad.to_degrees(),
            geo.altitude_m,
        ]
    }

    /// Build a world-to-camera rotation matrix `R` at `(lat_deg, lon_deg)` for `(heading, down, roll)`.
    pub fn camera_rotation(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        heading_deg: f64,
        down_deg: f64,
        roll_deg: f64,
    ) -> Matrix3<f64> {
        let (_, local_enu_from_ecef) = enu_frame(lat_deg, lon_deg);
        let basis = local_enu_from_ecef * self.enu_from_ecef.transpose();
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

    /// Sample bare-earth DEM elevation in metres at `(lat_deg, lon_deg)`.
    pub fn sample_dem(&self, lat_deg: f64, lon_deg: f64) -> Option<f64> {
        let (tx, ty, fx, fy) = mercator_tile_coords(lat_deg, lon_deg, self.dem_zoom);
        let tile = self.dem_tiles.get(&(tx, ty))?;
        let s = self.dem_tile_size;
        let px = ((fx * f64::from(s)).floor() as u32).min(s - 1);
        let py = ((fy * f64::from(s)).floor() as u32).min(s - 1);
        tile.get((py * s + px) as usize).copied().map(f64::from)
    }

    /// Sample RGB orthoimagery at `(lat_deg, lon_deg)`.
    pub fn sample_rgb(&self, lat_deg: f64, lon_deg: f64) -> Option<Rgb<u8>> {
        let (tx, ty, fx, fy) = mercator_tile_coords(lat_deg, lon_deg, self.imagery_zoom);
        let tile = self.imagery_tiles.get(&(tx, ty))?;
        let w = tile.width().max(1);
        let h = tile.height().max(1);
        let px = ((fx * f64::from(w)).floor() as u32).min(w - 1);
        let py = ((fy * f64::from(h)).floor() as u32).min(h - 1);
        Some(*tile.get_pixel(px, py))
    }

    /// Ray-cast a pinhole view `(eye, world_to_camera, k, size)` against the spherical DEM.
    ///
    /// Returns `(rgb_image, optical_axis_depth_m, world_points_enu)`.
    pub fn render(
        &self,
        eye: [f64; 3],
        world_to_cam: &Matrix3<f64>,
        k: &Matrix3<f64>,
        size: [u32; 2],
    ) -> SurfaceRender {
        let (w, h) = (size[0], size[1]);
        let mut rgb = RgbImage::from_pixel(w, h, Rgb([0, 0, 0]));
        let count = (w * h) as usize;
        let mut depth = vec![0.0f32; count];
        let mut points = vec![[0.0f64; 3]; count];
        let cam_to_world = world_to_cam.transpose();
        let fx = k[(0, 0)].max(1e-6);
        let fy = k[(1, 1)].max(1e-6);
        let cx = k[(0, 2)];
        let cy = k[(1, 2)];
        let eye_v = Vector3::new(eye[0], eye[1], eye[2]);

        for y in 0..h {
            for x in 0..w {
                let ray_cam = Vector3::new((f64::from(x) - cx) / fx, (f64::from(y) - cy) / fy, 1.0);
                let dir_world = cam_to_world * (ray_cam / ray_cam.norm());
                if let Some((hit, color)) = self.intersect_ray(eye_v, dir_world) {
                    let idx = (y * w + x) as usize;
                    let rel_cam = world_to_cam * (hit - eye_v);
                    if rel_cam.z > 0.5 {
                        rgb.put_pixel(x, y, color);
                        depth[idx] = rel_cam.z as f32;
                        points[idx] = [hit.x, hit.y, hit.z];
                    }
                }
            }
        }
        (rgb, depth, points)
    }

    pub(crate) fn intersect_ray(
        &self,
        eye: Vector3<f64>,
        dir: Vector3<f64>,
    ) -> Option<(Vector3<f64>, Rgb<u8>)> {
        let max_t = 65_000.0f64;
        let mut prev_t = 10.0f64;
        let mut t = 10.0f64;
        while t <= max_t {
            let p = eye + dir * t;
            let [lat, lon, alt] = self.geodetic([p.x, p.y, p.z]);
            if alt <= self.max_elev_m + 50.0
                && let Some(dem_h) = self.sample_dem(lat, lon)
                && alt <= dem_h
            {
                let mut lo = prev_t;
                let mut hi = t;
                for _ in 0..8 {
                    let mid = 0.5 * (lo + hi);
                    let pm = eye + dir * mid;
                    let [mlat, mlon, malt] = self.geodetic([pm.x, pm.y, pm.z]);
                    let mh = self.sample_dem(mlat, mlon).unwrap_or(dem_h);
                    if malt <= mh {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let hit = eye + dir * hi;
                let [hlat, hlon, _] = self.geodetic([hit.x, hit.y, hit.z]);
                let color = self.sample_rgb(hlat, hlon)?;
                return Some((hit, color));
            }
            prev_t = t;
            t += (t * 0.035).clamp(15.0, 600.0);
        }
        None
    }
}

pub(crate) fn enu_frame(lat_deg: f64, lon_deg: f64) -> (Vector3<f64>, Matrix3<f64>) {
    let p = geodetic_to_ecef(&GeodeticPosition::new(
        lat_deg.to_radians(),
        lon_deg.to_radians(),
        0.0,
    ));
    let origin = Vector3::new(p.x_m, p.y_m, p.z_m);
    let (slat, clat) = lat_deg.to_radians().sin_cos();
    let (slon, clon) = lon_deg.to_radians().sin_cos();
    let east = Vector3::new(-slon, clon, 0.0);
    let north = Vector3::new(-slat * clon, -slat * slon, clat);
    let up = Vector3::new(clat * clon, clat * slon, slat);
    let rot = Matrix3::from_rows(&[east.transpose(), north.transpose(), up.transpose()]);
    (origin, rot)
}

pub(crate) fn mercator_tile_coords(lat_deg: f64, lon_deg: f64, zoom: u32) -> (u32, u32, f64, f64) {
    let n = f64::from(1u32 << zoom);
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

#[cfg(test)]
mod tests;

//! Ellipsoidal terrain surface and ray-marching renderer.

mod coords;
pub mod gpu;
mod sparse;

pub use coords::{enu_frame, enu_frame_on, mercator_tile_coords};
pub use gpu::{GpuQueueRole, GpuSurfaceView, SharedSurfaceGpu, SurfaceGpuRenderer};
pub use sparse::{SurfaceAdaptiveStats, SurfaceRayHit};

use crate::error::FlightVisualError;
use image::{Rgb, RgbImage};
use nalgebra::{Matrix3, Vector3};
use navigate_contract::GeodeticPosition;
use navigate_geodesy::BodyEllipsoid;
use navigate_visual::LocalFrame;
use navigate_visual_render::RenderScene;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, path::Path};

/// Rendered view `(rgb_image, optical_axis_depth_m, world_points_enu)` from [`Surface::render`].
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

/// Terrain and orthoimagery surface on an explicit [`BodyEllipsoid`].
#[derive(Clone, Debug)]
pub struct Surface {
    /// Reference ellipsoid of the surface body (`WGS84` by default).
    pub ellipsoid: BodyEllipsoid,
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
    /// Minimum DEM elevation in meters.
    pub min_elev_m: f64,
    /// Maximum DEM elevation in meters.
    pub max_elev_m: f64,
    pub(crate) origin_ecef: Vector3<f64>,
    pub(crate) enu_from_ecef: Matrix3<f64>,
    pub(crate) imagery_tiles: HashMap<(u32, u32), RgbImage>,
    pub(crate) dem_tiles: HashMap<(u32, u32), Vec<f32>>,
    pub(crate) dem_tile_size: u32,
}

struct LoadedTileSet {
    imagery_tiles: HashMap<(u32, u32), RgbImage>,
    dem_tiles: HashMap<(u32, u32), Vec<f32>>,
    min_elev_m: f64,
    max_elev_m: f64,
    dem_tile_size: u32,
}

impl Surface {
    /// Load a [`Surface`] from `catalog.json` on disk.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when `catalog.json` or any present tile fails to decode.
    pub fn from_catalog(catalog_path: &Path) -> Result<Self, FlightVisualError> {
        let raw_bytes = fs::read(catalog_path).map_err(|source| FlightVisualError::Io {
            path: catalog_path.to_path_buf(),
            source,
        })?;
        let catalog_sha256 = hex_sha256(&raw_bytes);
        let doc: Value =
            serde_json::from_slice(&raw_bytes).map_err(|source| FlightVisualError::Json {
                context: catalog_path.display().to_string(),
                source,
            })?;
        if doc.get("format").and_then(Value::as_str) != Some("photo-reference-catalog") {
            return Err(FlightVisualError::Invalid(format!(
                "{} is not a photo-reference-catalog",
                catalog_path.display()
            )));
        }
        let cfg = doc
            .get("configuration")
            .ok_or_else(|| FlightVisualError::Invalid("missing catalog configuration".into()))?;
        let ellipsoid = parse_catalog_ellipsoid(cfg)?;
        let bbox: [f64; 4] = serde_json::from_value(cfg["bbox"].clone()).map_err(|source| {
            FlightVisualError::Json {
                context: "catalog.configuration.bbox".into(),
                source,
            }
        })?;
        let imagery_zoom = cfg
            .get("imagery_zoom")
            .and_then(Value::as_u64)
            .unwrap_or(14) as u32;
        let dem_zoom = cfg.get("dem_zoom").and_then(Value::as_u64).unwrap_or(12) as u32;
        let origin_lon = 0.5 * (bbox[0] + bbox[2]);
        let origin_lat = 0.5 * (bbox[1] + bbox[3]);
        let tiles: Vec<CatalogTile> = doc
            .get("tiles")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|source| FlightVisualError::Json {
                context: "catalog.tiles".into(),
                source,
            })?
            .unwrap_or_default();
        let root = catalog_path.parent().unwrap_or_else(|| Path::new("."));
        let loaded = load_catalog_tiles(catalog_path, root, &tiles)?;
        let (origin_ecef, enu_from_ecef) = enu_frame_on(ellipsoid, origin_lat, origin_lon);
        Ok(Self {
            ellipsoid,
            origin_lat,
            origin_lon,
            bbox,
            imagery_zoom,
            dem_zoom,
            catalog_sha256,
            min_elev_m: loaded.min_elev_m,
            max_elev_m: loaded.max_elev_m,
            origin_ecef,
            enu_from_ecef,
            imagery_tiles: loaded.imagery_tiles,
            dem_tiles: loaded.dem_tiles,
            dem_tile_size: loaded.dem_tile_size,
        })
    }

    /// Construct an in-memory synthetic WGS84 [`Surface`] for deterministic unit tests.
    #[must_use]
    pub fn synthetic(origin_lat: f64, origin_lon: f64, elev_m: f32, image: RgbImage) -> Self {
        Self::synthetic_on_body(BodyEllipsoid::WGS84, origin_lat, origin_lon, elev_m, image)
    }

    /// Construct an in-memory synthetic [`Surface`] on `ellipsoid` at any latitude (`-90..=90`).
    #[must_use]
    pub fn synthetic_on_body(
        ellipsoid: BodyEllipsoid,
        origin_lat: f64,
        origin_lon: f64,
        elev_m: f32,
        image: RgbImage,
    ) -> Self {
        let (origin_ecef, enu_from_ecef) = enu_frame_on(ellipsoid, origin_lat, origin_lon);
        let z = 12_u32;
        let (tx, ty, _, _) = if ellipsoid == BodyEllipsoid::WGS84 && origin_lat.abs() < 85.0 {
            mercator_tile_coords(origin_lat, origin_lon, z)
        } else {
            coords::tangent_tile_coords(ellipsoid, 0.0, 0.0, z)
        };
        let mut imagery_tiles = HashMap::new();
        let mut dem_tiles = HashMap::new();
        for dy in -1_i32..=1 {
            for dx in -1_i32..=1 {
                let key = (
                    (tx as i32 + dx).max(0) as u32,
                    (ty as i32 + dy).max(0) as u32,
                );
                imagery_tiles.insert(key, image.clone());
                dem_tiles.insert(key, vec![elev_m; 256 * 256]);
            }
        }
        Self {
            ellipsoid,
            origin_lat,
            origin_lon,
            bbox: [
                (origin_lon - 0.2).clamp(-180.0, 180.0),
                (origin_lat - 0.2).clamp(-90.0, 90.0),
                (origin_lon + 0.2).clamp(-180.0, 180.0),
                (origin_lat + 0.2).clamp(-90.0, 90.0),
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

    /// Return the validated [`LocalFrame`] of this surface.
    ///
    /// Uses [`LocalFrame::wgs84_enu`] for non-polar Earth catalogs so Web Mercator GPU
    /// atlases remain active, and [`LocalFrame::body_tangent_enu`] for polar latitudes
    /// (`|lat| >= 85` degrees) or non-Earth bodies (`MOON_PA`, `MARS_IAU`).
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError::MissingLocalFrame`] when `origin_lat` or `origin_lon` is invalid.
    pub fn local_frame(&self) -> Result<LocalFrame, FlightVisualError> {
        if self.ellipsoid == BodyEllipsoid::WGS84 && self.origin_lat.abs() < 85.0 {
            LocalFrame::wgs84_enu(self.origin_lat, self.origin_lon)
                .map_err(|_| FlightVisualError::MissingLocalFrame)
        } else {
            let origin = GeodeticPosition::new(
                self.origin_lat.to_radians(),
                self.origin_lon.to_radians(),
                0.0,
            );
            LocalFrame::body_tangent_enu(self.ellipsoid, origin)
                .map_err(|_| FlightVisualError::MissingLocalFrame)
        }
    }

    /// Borrow the body-fixed-to-`ENU` rotation matrix of this surface.
    #[must_use]
    pub const fn enu_from_ecef(&self) -> &Matrix3<f64> {
        &self.enu_from_ecef
    }

    /// Borrow the loaded imagery tiles map.
    #[must_use]
    pub const fn imagery_tiles(&self) -> &HashMap<(u32, u32), RgbImage> {
        &self.imagery_tiles
    }

    /// Mutably borrow the loaded imagery tiles map (for synthetic test construction).
    pub fn imagery_tiles_mut(&mut self) -> &mut HashMap<(u32, u32), RgbImage> {
        &mut self.imagery_tiles
    }

    /// Borrow the loaded DEM tiles map.
    #[must_use]
    pub const fn dem_tiles(&self) -> &HashMap<(u32, u32), Vec<f32>> {
        &self.dem_tiles
    }

    /// Mutably borrow the loaded DEM tiles map (for synthetic test construction).
    pub fn dem_tiles_mut(&mut self) -> &mut HashMap<(u32, u32), Vec<f32>> {
        &mut self.dem_tiles
    }

    /// Set the DEM tile edge size in pixels (for synthetic test construction).
    pub fn set_dem_tile_size(&mut self, size: u32) {
        self.dem_tile_size = size.max(1);
    }

    /// Build a WGS84 [`RenderScene`] for [`navigate_visual_render::GpuRenderer`].
    #[must_use]
    pub fn to_render_scene(&self) -> Option<RenderScene> {
        (self.ellipsoid == BodyEllipsoid::WGS84)
            .then(|| gpu::build_render_scene_for_surface(self))
            .flatten()
    }

    /// Sample bare-earth DEM elevation in meters at `(lat_deg, lon_deg)`.
    #[must_use]
    pub fn sample_dem(&self, lat_deg: f64, lon_deg: f64) -> Option<f64> {
        let (tx, ty, fx, fy) = self.tile_coords(lat_deg, lon_deg, self.dem_zoom);
        let tile = self.dem_tiles.get(&(tx, ty))?;
        let s = self.dem_tile_size;
        let px = ((fx * f64::from(s)).floor() as u32).min(s - 1);
        let py = ((fy * f64::from(s)).floor() as u32).min(s - 1);
        tile.get((py * s + px) as usize).copied().map(f64::from)
    }

    /// Sample `RGB` orthoimagery at `(lat_deg, lon_deg)`.
    #[must_use]
    pub fn sample_rgb(&self, lat_deg: f64, lon_deg: f64) -> Option<Rgb<u8>> {
        let (tx, ty, fx, fy) = self.tile_coords(lat_deg, lon_deg, self.imagery_zoom);
        let tile = self.imagery_tiles.get(&(tx, ty))?;
        let w = tile.width().max(1);
        let h = tile.height().max(1);
        let px = ((fx * f64::from(w)).floor() as u32).min(w - 1);
        let py = ((fy * f64::from(h)).floor() as u32).min(h - 1);
        Some(*tile.get_pixel(px, py))
    }

    /// Ray-cast a pinhole view `(eye, world_to_cam, k, size)` against the DEM.
    #[must_use]
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
        let mut depth = vec![0.0_f32; count];
        let mut points = vec![[0.0_f64; 3]; count];
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
        let [_, _, eye_alt] = self.geodetic([eye.x, eye.y, eye.z]);
        if eye_alt > self.max_elev_m + 50.0 && dir.z >= 0.0 {
            return None;
        }
        let max_t = 65_000.0_f64;
        let mut prev_t = 10.0_f64;
        let mut t = 10.0_f64;
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

fn parse_catalog_ellipsoid(cfg: &Value) -> Result<BodyEllipsoid, FlightVisualError> {
    let Some(raw) = cfg.get("ellipsoid").and_then(Value::as_str) else {
        return Ok(BodyEllipsoid::WGS84);
    };
    match raw {
        "wgs84" | "WGS84" => Ok(BodyEllipsoid::WGS84),
        "grs80" | "GRS80" => Ok(BodyEllipsoid::GRS80),
        "moon_pa" | "MOON_PA" => Ok(BodyEllipsoid::MOON_PA),
        "moon_me" | "MOON_ME" => Ok(BodyEllipsoid::MOON_ME),
        "mars_iau" | "MARS_IAU" => Ok(BodyEllipsoid::MARS_IAU),
        other => Err(FlightVisualError::Invalid(format!(
            "unsupported catalog ellipsoid `{other}`"
        ))),
    }
}

fn load_catalog_tiles(
    catalog_path: &Path,
    root: &Path,
    tiles: &[CatalogTile],
) -> Result<LoadedTileSet, FlightVisualError> {
    let mut imagery_tiles = HashMap::new();
    let mut dem_tiles = HashMap::new();
    let mut min_elev_m = f64::INFINITY;
    let mut max_elev_m = f64::NEG_INFINITY;
    let mut dem_tile_size = 256_u32;

    for tile in tiles {
        let tile_path = root.join(&tile.path);
        if !tile_path.is_file() {
            continue;
        }
        let img = image::open(&tile_path)
            .map_err(|source| FlightVisualError::Image {
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
                min_elev_m = min_elev_m.min(h);
                max_elev_m = max_elev_m.max(h);
                elev.push(h as f32);
            }
            dem_tiles.insert(xy, elev);
        } else {
            imagery_tiles.insert(xy, img);
        }
    }
    if !tiles.is_empty() && imagery_tiles.is_empty() {
        return Err(FlightVisualError::Invalid(format!(
            "catalog {} lists {} tiles, but zero imagery tiles exist on disk",
            catalog_path.display(),
            tiles.len()
        )));
    }
    if !min_elev_m.is_finite() {
        min_elev_m = 0.0;
        max_elev_m = 500.0;
    }
    Ok(LoadedTileSet {
        imagery_tiles,
        dem_tiles,
        min_elev_m,
        max_elev_m,
        dem_tile_size,
    })
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

//! Context surface fallback, horizon attitude estimation, and reference catalog builder.

use super::{CatalogTile, SPHERE_RADIUS_M, Surface, SurfaceRender, mercator_tile_coords};
use crate::{
    PipelineError,
    util::{bytes_sha256, read_json, run_tool, write_json},
};
use nalgebra::Matrix3;
use serde_json::{Value, json};
use std::{fs, path::Path};

/// Primary detail surface backed by a regional context surface for distant horizon rays.
#[derive(Clone, Debug)]
pub struct ContextSurface {
    /// High-resolution primary surface.
    pub primary: Surface,
    /// Optional wider-area context surface.
    pub context: Option<Surface>,
}

impl ContextSurface {
    /// Render against `primary`, filling uncovered pixels from `context` in `primary` ENU coordinates.
    pub fn render(
        &self,
        eye: [f64; 3],
        world_to_cam: &Matrix3<f64>,
        k: &Matrix3<f64>,
        size: [u32; 2],
    ) -> SurfaceRender {
        let (mut rgb, mut depth, mut points) = self.primary.render(eye, world_to_cam, k, size);
        let Some(ctx) = &self.context else {
            return (rgb, depth, points);
        };
        let eye_geo = self.primary.geodetic(eye);
        let ctx_eye = ctx.local(eye_geo[0], eye_geo[1], eye_geo[2]);
        let r_ctx_to_pri = self.primary.enu_from_ecef * ctx.enu_from_ecef.transpose();
        let w2c_ctx = world_to_cam * r_ctx_to_pri;
        let (crgb, cdepth, cpts) = ctx.render(ctx_eye, &w2c_ctx, k, size);
        let (w, h) = (size[0], size[1]);
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                if depth[idx] <= 0.0 && cdepth[idx] > 0.0 {
                    let g = ctx.geodetic(cpts[idx]);
                    rgb.put_pixel(x, y, *crgb.get_pixel(x, y));
                    depth[idx] = cdepth[idx];
                    points[idx] = self.primary.local(g[0], g[1], g[2]);
                }
            }
        }
        (rgb, depth, points)
    }
}

#[allow(unused_imports)]
pub use super::horizon::horizon_attitude;

/// Build a bounded `photo-reference-catalog` (`catalog.json`) at zoom `14` / `12`.
///
/// # Errors
///
/// Returns [`PipelineError`] when tile fetching or catalog writing fails.
pub fn build_reference_catalog(
    profile_path: &Path,
    output_dir: &Path,
    imagery_url_template: &str,
    dem_url_template: &str,
    cache_dir: Option<&Path>,
    radius_m: f64,
) -> Result<Value, PipelineError> {
    build_reference_catalog_zoom(
        profile_path,
        output_dir,
        imagery_url_template,
        dem_url_template,
        cache_dir,
        radius_m,
        14,
        12,
    )
}

/// Build a bounded `photo-reference-catalog` (`catalog.json`) with explicit tile zoom levels.
///
/// # Errors
///
/// Returns [`PipelineError`] when tile fetching or catalog writing fails.
#[expect(
    clippy::too_many_arguments,
    reason = "catalog builder requires profile, output, URLs, cache, radius, and zoom levels"
)]
pub fn build_reference_catalog_zoom(
    profile_path: &Path,
    output_dir: &Path,
    imagery_url_template: &str,
    dem_url_template: &str,
    cache_dir: Option<&Path>,
    radius_m: f64,
    imagery_zoom: u32,
    dem_zoom: u32,
) -> Result<Value, PipelineError> {
    let profile: Value = read_json(profile_path)?;
    let cases = profile
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| PipelineError::Invalid("missing cases in profile".into()))?;
    let mut west = f64::INFINITY;
    let mut south = f64::INFINITY;
    let mut east = f64::NEG_INFINITY;
    let mut north = f64::NEG_INFINITY;
    for c in cases {
        if let Some(pos) = c.get("prior").and_then(|p| p.get("position"))
            && let (Some(lat), Some(lon)) = (
                pos.get("latitude").and_then(Value::as_f64),
                pos.get("longitude").and_then(Value::as_f64),
            )
        {
            let dlat = (radius_m / SPHERE_RADIUS_M).to_degrees();
            let dlon = dlat / lat.to_radians().cos().abs().max(0.1);
            west = west.min(lon - dlon);
            east = east.max(lon + dlon);
            south = south.min(lat - dlat);
            north = north.max(lat + dlat);
        }
    }
    if !west.is_finite() {
        return Err(PipelineError::Invalid(
            "profile contains no position priors to bound the catalog".into(),
        ));
    }
    fs::create_dir_all(output_dir).map_err(|source| PipelineError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;

    let mut tiles = Vec::new();
    for (kind, zoom, tmpl, ext) in [
        ("imagery", imagery_zoom, imagery_url_template, "jpg"),
        ("dem", dem_zoom, dem_url_template, "png"),
    ] {
        let (x0, y1, _, _) = mercator_tile_coords(south, west, zoom);
        let (x1, y0, _, _) = mercator_tile_coords(north, east, zoom);
        for x in x0.min(x1)..=x0.max(x1) {
            for y in y0.min(y1)..=y0.max(y1) {
                let rel = format!("{kind}/{zoom}/{x}/{y}.{ext}");
                let dst = output_dir.join(&rel);
                let url = tmpl
                    .replace("{z}", &zoom.to_string())
                    .replace("{x}", &x.to_string())
                    .replace("{y}", &y.to_string());
                fetch_tile(&url, &dst, cache_dir, &rel)?;
                let bytes =
                    fs::read(&dst).map_err(|source| PipelineError::Io { path: dst, source })?;
                tiles.push(CatalogTile {
                    kind: kind.into(),
                    xyz: [zoom, x, y],
                    path: rel,
                    sha256: bytes_sha256(&bytes),
                    bytes: bytes.len(),
                    url,
                });
            }
        }
    }
    let catalog = json!({
        "format": "photo-reference-catalog",
        "version": 1,
        "complete": true,
        "scope": "Bounded regional catalog; no-prior tests are not global geolocation.",
        "configuration": {
            "bbox": [west, south, east, north],
            "imagery_zoom": imagery_zoom,
            "dem_zoom": dem_zoom,
        },
        "tiles": tiles,
    });
    write_json(&output_dir.join("catalog.json"), &catalog)?;
    Ok(catalog)
}

fn fetch_tile(
    url: &str,
    dst: &Path,
    cache_dir: Option<&Path>,
    rel: &str,
) -> Result<(), PipelineError> {
    if dst.is_file() {
        return Ok(());
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    if let Some(cache) = cache_dir {
        let cached = cache.join(rel);
        if cached.is_file() {
            fs::copy(&cached, dst).map_err(|source| PipelineError::Io {
                path: dst.to_path_buf(),
                source,
            })?;
            return Ok(());
        }
    }
    if let Some(local) = url.strip_prefix("file://") {
        fs::copy(local, dst).map_err(|source| PipelineError::Io {
            path: dst.to_path_buf(),
            source,
        })?;
    } else {
        let dst_s = dst.to_string_lossy().into_owned();
        run_tool("curl", &["-fsSL", url, "-o", &dst_s])?;
    }
    if let Some(cache) = cache_dir {
        let cached = cache.join(rel);
        if let Some(p) = cached.parent() {
            drop(fs::create_dir_all(p));
        }
        drop(fs::copy(dst, cached));
    }
    Ok(())
}

//! A deterministic terrain package for renderer verification.

use crate::{
    BenchError, directory_blocking,
    package::{Artifact, Manifest, Tile, digest},
    read_blocking, write_blocking,
};
use image::{Rgba, RgbaImage};
use std::path::Path;

const ZOOM: u32 = 13;
const CENTER: [u32; 2] = [4357, 2870];

pub(crate) fn prepare_blocking(root: &Path) -> Result<(), BenchError> {
    directory_blocking(root)?;
    let mut tiles = Vec::new();
    for y in CENTER[1] - 2..=CENTER[1] + 2 {
        for x in CENTER[0] - 2..=CENTER[0] + 2 {
            let imagery = RgbaImage::from_fn(512, 512, |px, py| {
                texture(
                    i64::from(x) * 512 + i64::from(px),
                    i64::from(y) * 512 + i64::from(py),
                )
            });
            tiles.push(Tile {
                xyz: [ZOOM, x, y],
                imagery: Some(save_blocking(
                    root,
                    &format!("{ZOOM}-{x}-{y}.png"),
                    &imagery,
                )?),
                elevation: None,
            });
        }
    }
    let dem_zoom = ZOOM - 2;
    for y in (CENTER[1] - 2) / 4..=(CENTER[1] + 2) / 4 {
        for x in (CENTER[0] - 2) / 4..=(CENTER[0] + 2) / 4 {
            let elevation = elevation_image(dem_zoom, x, y);
            tiles.push(Tile {
                xyz: [dem_zoom, x, y],
                imagery: None,
                elevation: Some(save_blocking(
                    root,
                    &format!("{dem_zoom}-{x}-{y}.dem.png"),
                    &elevation,
                )?),
            });
        }
    }
    let n = f64::from(1 << ZOOM);
    let lon = (f64::from(CENTER[0]) + 0.5) / n * 360.0 - 180.0;
    let mercator_y = std::f64::consts::PI * (1.0 - 2.0 * (f64::from(CENTER[1]) + 0.5) / n);
    let lat = mercator_y.sinh().atan().to_degrees();
    let manifest = Manifest {
        schema_version: 2,
        release_id: "synthetic-terrain-v1".into(),
        anchor_lat_lon: [lat, lon],
        elevation_datum: "synthetic-msl".into(),
        attribution: "Procedural test data. No real terrain or imagery accuracy claim.".into(),
        tiles,
    };
    let path = root.join("map.json");
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|source| BenchError::Json {
        path: path.clone(),
        source,
    })?;
    write_blocking(&path, &bytes)?;
    tracing::info!(path = %path.display(), lat, lon, "prepared shared map package");
    Ok(())
}

fn elevation_image(zoom: u32, x: u32, y: u32) -> RgbaImage {
    let factor = f64::from(1 << (ZOOM - zoom));
    RgbaImage::from_fn(256, 256, |px, py| {
        let wx = (f64::from(x) * factor - f64::from(CENTER[0])) * 256.0
            + (f64::from(px) + 0.5) * factor
            - 0.5;
        let wy = (f64::from(y) * factor - f64::from(CENTER[1])) * 256.0
            + (f64::from(py) + 0.5) * factor
            - 0.5;
        let height = 450.0
            + 180.0 * (wx / 95.0).sin() * (wy / 120.0).cos()
            + 300.0 * (-((wx - 180.0).powi(2) + (wy + 260.0).powi(2)) / 40000.0).exp();
        let encoded = ((height + 32768.0) * 256.0).round() as u32;
        Rgba([
            (encoded >> 16) as u8,
            (encoded >> 8) as u8,
            encoded as u8,
            255,
        ])
    })
}

fn hash(x: i64, y: i64) -> u64 {
    let mut v =
        (x as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ (y as u64).wrapping_mul(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
    v ^ (v >> 31)
}

fn texture(x: i64, y: i64) -> Rgba<u8> {
    let field = hash(x.div_euclid(47), y.div_euclid(39));
    let detail = hash(x.div_euclid(5), y.div_euclid(5));
    let noise = (hash(x, y) % 17) as u8;
    let road = (x + (60.0 * (y as f64 / 130.0).sin()) as i64).rem_euclid(193) < 3
        || (y + x / 7).rem_euclid(271) < 3;
    if road {
        return Rgba([215, 203, 183, 255]);
    }
    let v = 35 + (field % 105) as u8 + (detail % 55) as u8 + noise;
    Rgba([
        v,
        v.saturating_add((field % 25) as u8),
        v.saturating_sub(12),
        255,
    ])
}

fn save_blocking(root: &Path, name: &str, image: &RgbaImage) -> Result<Artifact, BenchError> {
    let path = root.join(name);
    image.save(&path).map_err(|source| BenchError::Image {
        path: path.clone(),
        source,
    })?;
    Ok(Artifact {
        path: name.into(),
        sha256: digest(&read_blocking(&path)?),
    })
}

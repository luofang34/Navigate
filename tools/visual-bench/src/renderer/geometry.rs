//! Projection and coverage in MapLibre's local Mercator frame.

use crate::package::Manifest;
use cgmath::Matrix4;
use maplibre::render::camera::EyeFrustum;
use nalgebra::Vector2;
use navigate_visual::{CameraModel, CameraPose};

pub(super) fn eye_transform(pose: CameraPose) -> Matrix4<f64> {
    let mut m = pose.orientation.to_homogeneous();
    m[(0, 3)] = pose.position.x;
    m[(1, 3)] = pose.position.y;
    m[(2, 3)] = pose.position.z;
    let column = |i| cgmath::Vector4::new(m[(0, i)], m[(1, i)], m[(2, i)], m[(3, i)]);
    Matrix4::from_cols(column(0), column(1), column(2), column(3))
}

pub(super) fn frustum(camera: CameraModel) -> EyeFrustum {
    EyeFrustum {
        left: (camera.cx + 0.5) / camera.fx,
        right: (f64::from(camera.width) - camera.cx - 0.5) / camera.fx,
        top: (camera.cy + 0.5) / camera.fy,
        bottom: (f64::from(camera.height) - camera.cy - 0.5) / camera.fy,
        near: 10.0,
        far: 100000.0,
    }
}

pub(super) struct Coverage(Vec<[f64; 4]>);

impl Coverage {
    pub fn new(manifest: &Manifest) -> Self {
        let [lat, lon] = manifest.anchor_lat_lon;
        let anchor_x = (lon + 180.0) / 360.0;
        let anchor_y = (1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0;
        let scale = std::f64::consts::TAU * 6_371_008.8 * lat.to_radians().cos();
        Self(
            manifest
                .tiles
                .iter()
                .map(|tile| {
                    let [z, x, y] = tile.xyz;
                    let n = f64::from(1 << z);
                    [
                        (f64::from(x) / n - anchor_x) * scale,
                        (f64::from(x + 1) / n - anchor_x) * scale,
                        (anchor_y - f64::from(y + 1) / n) * scale,
                        (anchor_y - f64::from(y) / n) * scale,
                    ]
                })
                .collect(),
        )
    }

    pub fn mask(&self, depths: &mut [f32], camera: CameraModel, pose: CameraPose) {
        for (index, depth) in depths.iter_mut().enumerate() {
            if !depth.is_finite() || *depth <= 0.0 {
                *depth = 0.0;
                continue;
            }
            let pixel = Vector2::new(
                (index % camera.width as usize) as f64,
                (index / camera.width as usize) as f64,
            );
            let world = camera.unproject(&pose, pixel, f64::from(*depth));
            if !self.0.iter().any(|[west, east, south, north]| {
                world.x >= *west && world.x < *east && world.y >= *south && world.y < *north
            }) {
                *depth = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests;

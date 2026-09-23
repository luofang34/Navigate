//! Projection and coverage in MapLibre's local Mercator frame.

use crate::package::Manifest;
use cgmath::Matrix4;
use maplibre::render::camera::EyeFrustum;
use nalgebra::{Vector2, Vector3};
use navigate_visual::{CameraModel, CameraPose, LocalFrame};

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

pub(super) struct Coverage {
    imagery: Vec<[f64; 4]>,
    elevation: Vec<[f64; 4]>,
}

impl Coverage {
    pub fn new(manifest: &Manifest, frame: LocalFrame) -> Self {
        let [anchor_x, anchor_y] = frame.mercator_xy(Vector3::zeros());
        let scale = frame.mercator_scale_m();
        let bounds = |tile: &crate::package::Tile| {
            let [z, x, y] = tile.xyz;
            let n = f64::from(1 << z);
            [
                (f64::from(x) / n - anchor_x) * scale,
                (f64::from(x + 1) / n - anchor_x) * scale,
                (anchor_y - f64::from(y + 1) / n) * scale,
                (anchor_y - f64::from(y) / n) * scale,
            ]
        };
        Self {
            imagery: manifest
                .tiles
                .iter()
                .filter(|tile| tile.imagery.is_some())
                .map(bounds)
                .collect(),
            elevation: manifest
                .tiles
                .iter()
                .filter(|tile| tile.elevation.is_some())
                .map(bounds)
                .collect(),
        }
    }

    pub fn mask(
        &self,
        depths: &mut [f32],
        camera: CameraModel,
        pose: CameraPose,
        surface_valid: impl Fn(Vector3<f64>) -> bool,
    ) {
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
            let contains = |bounds: &[f64; 4]| {
                world.x >= bounds[0]
                    && world.x < bounds[1]
                    && world.y >= bounds[2]
                    && world.y < bounds[3]
            };
            if !self.imagery.iter().any(contains)
                || !self.elevation.iter().any(contains)
                || !surface_valid(world)
            {
                *depth = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests;

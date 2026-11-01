//! Camera intrinsics, SO(3) pose conversion, and render limit helpers for [`super::SurfaceGpuRenderer`].

use super::{Surface, atlas::MAX_SCENE_BYTES};
use crate::PipelineError;
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use navigate_visual::{CameraModel, CameraPose, LocalFrame};
use navigate_visual_render::RenderLimits;
use sha2::{Digest, Sha256};

const MAX_VIEW_PIXELS: usize = 4_194_304;

pub(crate) fn surface_pose_to_gpu(
    surface: &Surface,
    frame: LocalFrame,
    eye_enu: [f64; 3],
    world_to_cam: &Matrix3<f64>,
) -> Result<CameraPose, PipelineError> {
    let [lat, lon, alt] = surface.geodetic(eye_enu);
    let position = frame.local([lat, lon, alt])?;
    let gl_to_cv = Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0));
    let raw_eye_to_world = world_to_cam.transpose() * gl_to_cv;
    let ortho = orthonormalize_so3(&raw_eye_to_world);
    let orientation =
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(ortho));
    let pose = CameraPose {
        position,
        orientation,
    };
    pose.validate()?;
    Ok(pose)
}

fn orthonormalize_so3(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    if let (Some(u), Some(v_t)) = (svd.u, svd.v_t) {
        let mut r = u * v_t;
        if r.determinant() < 0.0 {
            let mut u_fix = u;
            u_fix.set_column(2, &(-u.column(2)));
            r = u_fix * v_t;
        }
        r
    } else {
        Matrix3::identity()
    }
}

pub(super) fn camera_model(k: &Matrix3<f64>, size: [u32; 2]) -> Option<CameraModel> {
    let model = CameraModel {
        width: size[0],
        height: size[1],
        fx: k[(0, 0)],
        fy: k[(1, 1)],
        cx: k[(0, 2)],
        cy: k[(1, 2)],
    };
    model.validate().ok()?;
    Some(model)
}

pub(super) fn default_limits(camera: &CameraModel) -> RenderLimits {
    let corner_norm = [
        (0.0 - camera.cx) / camera.fx,
        (f64::from(camera.width - 1) - camera.cx) / camera.fx,
    ]
    .into_iter()
    .fold(0.0_f64, |acc, dx| {
        let dy0 = (camera.cy - 0.0) / camera.fy;
        let dy1 = (camera.cy - f64::from(camera.height - 1)) / camera.fy;
        acc.max((dx * dx + dy0 * dy0 + 1.0).sqrt())
            .max((dx * dx + dy1 * dy1 + 1.0).sqrt())
    })
    .max(1.0);
    let max_far = (195_000.0 / corner_norm).clamp(1_000.0, 65_000.0) as f32;
    RenderLimits {
        scene_bytes: MAX_SCENE_BYTES,
        pixels: MAX_VIEW_PIXELS,
        far_m: max_far,
        steps: 144,
        refinements: 12,
    }
}

pub(super) fn manifest_hex64(raw: &str) -> String {
    if raw.len() == 64 && raw.bytes().all(|b| b.is_ascii_hexdigit()) {
        raw.to_ascii_lowercase()
    } else {
        let digest = Sha256::digest(raw.as_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

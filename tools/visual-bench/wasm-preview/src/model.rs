use crate::error::PreviewError;
use cgmath::Matrix4;
use maplibre::render::camera::EyeFrustum;
use nalgebra::{Quaternion, UnitQuaternion};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Deserialize, Serialize)]
pub(crate) struct Camera {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}
impl Camera {
    pub fn validate(&self) -> Result<(), PreviewError> {
        if !(64..=1920).contains(&self.width)
            || !(64..=1920).contains(&self.height)
            || ![self.fx, self.fy, self.cx, self.cy]
                .iter()
                .all(|v| v.is_finite())
            || self.fx <= 0.0
            || self.fy <= 0.0
            || self.cx < 0.0
            || self.cy < 0.0
            || self.cx >= f64::from(self.width)
            || self.cy >= f64::from(self.height)
        {
            return Err(PreviewError::Input {
                reason: "invalid camera intrinsics".into(),
            });
        }
        Ok(())
    }
    pub fn model(&self) -> navigate_visual::CameraModel {
        navigate_visual::CameraModel {
            width: self.width,
            height: self.height,
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
        }
    }
    pub fn frustum(&self) -> EyeFrustum {
        EyeFrustum {
            left: (self.cx + 0.5) / self.fx,
            right: (f64::from(self.width) - self.cx - 0.5) / self.fx,
            top: (self.cy + 0.5) / self.fy,
            bottom: (f64::from(self.height) - self.cy - 0.5) / self.fy,
            near: 10.0,
            far: 50_000_000.0,
        }
    }
}
#[derive(Clone, Copy, Deserialize, Serialize)]
pub(crate) struct Pose {
    pub position_enu_m: [f64; 3],
    pub eye_to_enu_xyzw: [f64; 4],
}
impl Pose {
    pub fn model(&self) -> Result<navigate_visual::CameraPose, PreviewError> {
        self.transform()?;
        let [x, y, z, w] = self.eye_to_enu_xyzw;
        Ok(navigate_visual::CameraPose {
            position: nalgebra::Vector3::from(self.position_enu_m),
            orientation: UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z)),
        })
    }
    pub fn transform(&self) -> Result<Matrix4<f64>, PreviewError> {
        let [x, y, z, w] = self.eye_to_enu_xyzw;
        if !self
            .position_enu_m
            .iter()
            .chain(self.eye_to_enu_xyzw.iter())
            .all(|v| v.is_finite())
            || (x * x + y * y + z * z + w * w - 1.0).abs() > 1e-5
        {
            return Err(PreviewError::Input {
                reason: "invalid camera pose or non-unit quaternion".into(),
            });
        }
        let mut m = UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z)).to_homogeneous();
        for i in 0..3 {
            m[(i, 3)] = self.position_enu_m[i];
        }
        let col = |i| cgmath::Vector4::new(m[(0, i)], m[(1, i)], m[(2, i)], m[(3, i)]);
        Ok(Matrix4::from_cols(col(0), col(1), col(2), col(3)))
    }
}

pub(crate) use navigate_imagery::{Asset, Package as Manifest, is_digest as hash};

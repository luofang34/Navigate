//! Internal camera coordinates for local scene optimization.
use super::LocalScenePose;
use crate::CameraModel;
use nalgebra::{Matrix3, SVector, UnitQuaternion, Vector2, Vector3};
#[derive(Clone, Copy)]
pub(crate) struct Pose {
    pub r: Matrix3<f64>,
    pub t: Vector3<f64>,
}
fn axes() -> Matrix3<f64> {
    Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0))
}
impl Pose {
    pub fn from_scene(pose: LocalScenePose) -> Self {
        let r = axes() * pose.orientation.inverse().to_rotation_matrix().into_inner();
        Self {
            r,
            t: -r * pose.position,
        }
    }
    pub fn to_scene(self) -> LocalScenePose {
        LocalScenePose {
            position: self.center(),
            orientation: UnitQuaternion::from_matrix(&(self.r.transpose() * axes())),
        }
    }
    pub fn center(self) -> Vector3<f64> {
        -self.r.transpose() * self.t
    }
    pub fn project(self, camera: &CameraModel, world: Vector3<f64>) -> Option<Vector2<f64>> {
        let p = self.r * world + self.t;
        (p.z > 1e-6).then(|| {
            Vector2::new(
                camera.fx * p.x / p.z + camera.cx,
                camera.fy * p.y / p.z + camera.cy,
            )
        })
    }
    pub fn increment(self, delta: SVector<f64, 6>) -> Self {
        let r =
            UnitQuaternion::from_scaled_axis(Vector3::new(delta[3], delta[4], delta[5]) * 0.001)
                .to_rotation_matrix()
                .into_inner();
        Self {
            r: r * self.r,
            t: r * self.t + delta.fixed_rows::<3>(0),
        }
    }
}

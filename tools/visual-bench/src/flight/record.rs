//! Sequence records preserve candidate identities and conditional geometry.
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use navigate_visual::{CameraPose, EstimateQuality, LocalFrame, VisualError};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PoseRecord {
    pub position_enu_m: [f64; 3],
    pub eye_to_enu_xyzw: [f64; 4],
}
impl PoseRecord {
    pub fn pose(self) -> Result<CameraPose, VisualError> {
        let [x, y, z, w] = self.eye_to_enu_xyzw;
        let q = Quaternion::new(w, x, y, z);
        if !q.coords.iter().all(|v| v.is_finite()) || (q.norm_squared() - 1.0).abs() > 1e-6 {
            return Err(VisualError::Invalid {
                field: "flight candidate quaternion",
            });
        }
        let pose = CameraPose {
            position: Vector3::from(self.position_enu_m),
            orientation: UnitQuaternion::new_normalize(q),
        };
        pose.validate()?;
        Ok(pose)
    }
}
impl From<CameraPose> for PoseRecord {
    fn from(pose: CameraPose) -> Self {
        Self {
            position_enu_m: pose.position.into(),
            eye_to_enu_xyzw: pose.orientation.coords.into(),
        }
    }
}
pub(super) fn pose_report(pose: CameraPose, frame: LocalFrame, q: EstimateQuality) -> Value {
    let [lat, lon, alt] = frame.geodetic(pose.position);
    json!({"position_enu_m":pose.position.as_slice(),"eye_to_enu_xyzw":pose.orientation.coords.as_slice(),
        "latitude_deg":lat,"longitude_deg":lon,"altitude_m":alt,"inliers":q.inliers,
        "spatial_support":q.spatial_support,"occupied_cells":q.occupied_cells,"reprojection_rms_px":q.reprojection_rms_px,
        "condition_number":q.condition_number})
}

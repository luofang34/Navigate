//! Candidate camera poses carry no navigation admission bounds.
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use navigate_visual::{CameraPose, VisualError};
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CandidateRecord {
    pub position_enu_m: [f64; 3],
    pub eye_to_enu_xyzw: [f64; 4],
}
impl CandidateRecord {
    pub fn pose(&self) -> Result<CameraPose, VisualError> {
        let [x, y, z, w] = self.eye_to_enu_xyzw;
        let q = Quaternion::new(w, x, y, z);
        if !q.coords.iter().all(|v| v.is_finite()) || (q.norm_squared() - 1.0).abs() > 1e-6 {
            return Err(VisualError::Invalid {
                field: "unit candidate quaternion",
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

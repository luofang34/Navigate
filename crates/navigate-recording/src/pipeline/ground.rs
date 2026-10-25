//! Ground normals constrain tilt independently of the pose used to render tracking depth.

use super::{Processor, VisualObservation, key};
use crate::Error;
use navigate_visual::{PlaneMotionConfig, plane_motion};
use navigate_visual_session::{GroundPlaneObservation, TerrainNormal};

impl Processor {
    pub(super) fn ground(
        &mut self,
        input: &VisualObservation,
        diagnostics: &mut Vec<String>,
    ) -> Result<(), Error> {
        let (Some(previous), Some(reference)) = (&self.previous, &input.tracking_reference) else {
            return Ok(());
        };
        let frame = key(previous.stamp.sequence);
        if self.session.pose_at(&frame)?.odometry.is_none() {
            return Ok(());
        }
        let plane = match plane_motion(
            &input.frame.camera,
            &input.tracking_matches,
            &PlaneMotionConfig::default(),
        ) {
            Ok(plane) => plane,
            Err(error) => {
                diagnostics.push(format!("Ground plane unavailable: {error}"));
                return Ok(());
            }
        };
        let Some(terrain) = TerrainNormal::from_reference(&input.frame.camera, reference) else {
            diagnostics.push("Ground plane unavailable: terrain normal unsupported".into());
            return Ok(());
        };
        let distance = (reference.pose.position.z - input.terrain_m).max(10.0);
        let Some(observation) = GroundPlaneObservation::from_plane_motion(
            frame,
            &plane,
            terrain,
            self.config.object_height_m / distance,
        ) else {
            diagnostics.push("Ground plane ambiguous; no attitude constraint submitted".into());
            return Ok(());
        };
        let decision = self.session.submit_ground_plane(observation)?;
        diagnostics.push(format!(
            "Ground plane: {decision:?}; {} inliers",
            plane.inliers
        ));
        Ok(())
    }
}

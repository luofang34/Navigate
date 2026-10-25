//! Queries apply all accepted map corrections without changing relative odometry.

use super::{Processor, key};
use crate::{Error, ResultFrame};
use navigate_visual_session::{
    Confirmation, Height, HeightInputs, MapPose, TerrainSample, TerrainSource, VerticalReference,
    to_camera,
};

impl Processor {
    /// Supply map terrain at a corrected camera location without adding visual evidence.
    ///
    /// # Errors
    /// Rejects unknown frames and non-finite terrain elevations.
    pub fn update_terrain(&mut self, sequence: u64, elevation_m: f64) -> Result<(), Error> {
        if !elevation_m.is_finite() {
            return Err(Error::Invalid("non-finite terrain".into()));
        }
        let record = self
            .records
            .get_mut(&sequence)
            .ok_or_else(|| Error::Invalid(format!("unknown frame {sequence}")))?;
        record.terrain_m = elevation_m;
        Ok(())
    }

    /// All retained frames at the current revision, in source order.
    ///
    /// # Errors
    /// Returns an error if a required frame was evicted from the session.
    pub fn results(&self) -> Result<Vec<ResultFrame>, Error> {
        self.records
            .keys()
            .map(|index| self.result(*index))
            .collect()
    }

    pub(super) fn result(&self, sequence: u64) -> Result<ResultFrame, Error> {
        let record = self
            .records
            .get(&sequence)
            .ok_or_else(|| Error::Invalid(format!("unknown frame {sequence}")))?;
        let pose = self.session.pose_at(&key(sequence))?;
        let height = self.session.height_at(
            &key(sequence),
            &HeightInputs {
                terrain: Some(TerrainSample {
                    elevation_m: record.terrain_m,
                    sigma_m: self.config.map_sigma_m[1],
                    vertical: VerticalReference::Unknown,
                    source: TerrainSource::MapElevation { map: self.map() },
                }),
                ..Default::default()
            },
        )?;
        let mut result = ResultFrame {
            sequence,
            media_ns: record.media_ns,
            observation_sha256: record.digest.clone(),
            pose: None,
            geodetic: None,
            camera_agl: None,
            position_bound_m: None,
            confirmed: false,
            map_observation_accepted: record.map_observation_accepted,
            support: None,
            revision: pose.revision,
            diagnostics: record.diagnostics.clone(),
        };
        match pose.map {
            MapPose::Located {
                pose,
                position_bound_m,
                confirmation,
                local_frame,
                usability,
                tilt_bound_rad,
                nearest_anchor_ns,
                ..
            } => {
                result.support = Some(crate::OutputSupport {
                    recent_map_support: usability.recent_map_support,
                    navigation: usability.navigation,
                    ground_projection: usability.ground_projection,
                    tilt_bound_deg: tilt_bound_rad.map(f64::to_degrees),
                    nearest_anchor_s: nearest_anchor_ns.map(|ns| ns as f64 / 1e9),
                });
                result.pose = Some(to_camera(&pose).into());
                result.geodetic = Some(local_frame.geodetic(pose.translation.vector));
                result.position_bound_m = Some(position_bound_m);
                result.confirmed = matches!(confirmation, Confirmation::Confirmed { .. });
                if let Height::Estimated {
                    value_m, sigma_m, ..
                } = height.camera_agl
                {
                    result.camera_agl = Some([value_m, sigma_m]);
                }
            }
            MapPose::Unlocated(reason) => result.diagnostics.push(format!("Unlocated: {reason:?}")),
        }
        Ok(result)
    }
}

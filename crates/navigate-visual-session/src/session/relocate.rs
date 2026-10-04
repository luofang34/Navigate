//! Map relocation proposals that do not inherit an unsupported tilt.
//!
//! A proposal is a pose at which the host renders a map reference and runs a
//! map match. It is not an observation and never enters the trajectory.

use super::VisualSession;
use crate::{FrameKey, SessionError, TerrainNormal, pose::Pose};
use nalgebra::{Unit, UnitQuaternion, Vector3};
use navigate_visual::CameraPose;

/// Where the position of a proposal comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionBasis {
    /// The predicted map position of the frame.
    Predicted,
    /// The newest anchor, for a frame that the session cannot locate.
    LastAnchor,
}

/// Where the tilt of a proposal comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TiltBasis {
    /// The ground normal observed in this frame.
    GroundPlane,
    /// The predicted pose, with a tilt bound inside the projection limit.
    Supported,
    /// The attitude of the newest anchor.
    LastAnchor,
}

/// One map relocation proposal.
#[derive(Clone, Copy, Debug)]
pub struct RelocationCandidate {
    /// Render pose in the session local frame.
    pub pose: CameraPose,
    /// Position source.
    pub position: PositionBasis,
    /// Tilt source.
    pub tilt: TiltBasis,
}

/// A ground normal measured in the frame camera, with the terrain normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundView {
    /// Unit ground normal in the camera eye frame, from the ground to the camera.
    pub normal_camera: Unit<Vector3<f64>>,
    /// Terrain normal under the camera.
    pub terrain: TerrainNormal,
}

impl VisualSession {
    /// Render poses for a map relocation of `frame`, nearest first.
    ///
    /// The search radius grows with the position bound. Returns no proposal
    /// when the session has neither a prediction nor an anchor; the host then
    /// needs a wide-area retrieval.
    ///
    /// # Errors
    /// Rejects an unknown frame.
    pub fn relocation_candidates(
        &self,
        frame: &FrameKey,
        ground: Option<&GroundView>,
    ) -> Result<Vec<RelocationCandidate>, SessionError> {
        let state = self
            .frames
            .get(frame)
            .ok_or(SessionError::UnknownFrame(*frame))?;
        let policy = self.config.relocation;
        let predicted = state.odometry.as_ref().and_then(|o| {
            let (correction, bounds, _) = self.locate(frame, o)?;
            Some((crate::pose::compose(&correction, &o.pose), bounds))
        });
        let anchor = self.newest_anchor();
        let (base, position, radius, supported) = match (&predicted, anchor) {
            (Some((pose, bounds)), _) if bounds.position_m.is_some() => {
                let tilt_ok = bounds
                    .tilt_rad
                    .is_some_and(|t| t <= self.config.output.projection_tilt_bound_rad);
                let bound = bounds.position_m.unwrap_or(policy.max_radius_m);
                (
                    *pose,
                    PositionBasis::Predicted,
                    self.config.anchors.gate_sigma * bound,
                    tilt_ok,
                )
            }
            (_, Some(pose)) => (pose, PositionBasis::LastAnchor, policy.max_radius_m, false),
            _ => return Ok(Vec::new()),
        };
        let (rotation, tilt) = match (ground, supported, anchor) {
            (Some(view), _, _) => (
                level(&base, &view.normal_camera, &view.terrain.normal),
                TiltBasis::GroundPlane,
            ),
            (None, true, _) => (base.rotation, TiltBasis::Supported),
            (None, false, Some(anchored)) => {
                let up = Unit::new_normalize(anchored.rotation.inverse() * Vector3::z());
                (level(&base, &up, &Vector3::z_axis()), TiltBasis::LastAnchor)
            }
            (None, false, None) => return Ok(Vec::new()),
        };
        let radius = if radius.is_finite() {
            radius.clamp(policy.spacing_m, policy.max_radius_m)
        } else {
            policy.max_radius_m
        };
        Ok(offsets(policy.spacing_m, radius, policy.max_candidates)
            .into_iter()
            .map(|offset| RelocationCandidate {
                pose: CameraPose {
                    position: base.translation.vector + offset,
                    orientation: rotation,
                },
                position,
                tilt,
            })
            .collect())
    }

    /// Map pose of the newest anchor by capture time.
    fn newest_anchor(&self) -> Option<Pose> {
        let (keyframe, _) = self.anchors.iter().max_by_key(|(_, a)| a.capture_ns)?;
        self.keyframes.get(keyframe).map(|k| k.estimate)
    }
}

/// Rotate `pose` by the smallest rotation that maps the camera direction to the world direction.
fn level(
    pose: &Pose,
    camera: &Unit<Vector3<f64>>,
    world: &Unit<Vector3<f64>>,
) -> UnitQuaternion<f64> {
    let current = pose.rotation * camera.into_inner();
    let correction = UnitQuaternion::rotation_between(&current, &world.into_inner())
        .unwrap_or_else(UnitQuaternion::identity);
    crate::pose::unit(&(correction * pose.rotation))
}

/// Horizontal offsets on a square grid that spans `radius`, nearest first.
///
/// The grid uses the smallest spacing that both respects `spacing` and lets
/// `count` points reach the radius, so a larger bound searches farther.
fn offsets(spacing: f64, radius: f64, count: usize) -> Vec<Vector3<f64>> {
    let side = ((count as f64).sqrt().floor() as i32).max(1);
    let steps = (side - 1) / 2;
    let spacing = if steps > 0 {
        spacing.max(radius / f64::from(steps))
    } else {
        spacing
    };
    let mut all: Vec<Vector3<f64>> = (-steps..=steps)
        .flat_map(|i| {
            (-steps..=steps)
                .map(move |j| Vector3::new(f64::from(i) * spacing, f64::from(j) * spacing, 0.0))
        })
        .collect();
    all.sort_by(|a, b| {
        a.norm()
            .total_cmp(&b.norm())
            .then_with(|| a.x.total_cmp(&b.x))
            .then_with(|| a.y.total_cmp(&b.y))
    });
    all.truncate(count);
    all
}

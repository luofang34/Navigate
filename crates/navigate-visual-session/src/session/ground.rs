//! Ground-plane attitude observations: camera tilt relative to the terrain.
//!
//! Two-view plane geometry (`navigate_visual::plane_motion`) measures the
//! ground normal in the camera frame from image matches alone. With the
//! terrain normal from the elevation model, it constrains camera tilt. It
//! does not constrain heading or position. It does not depend on earlier
//! poses, so it stops the attitude feedback of map-depth tracking.

use super::{Ground, VisualSession};
use crate::{FrameKey, RevisionCause, SessionError, SessionEvent};
use nalgebra::{Matrix3, Unit, Vector2, Vector3};
use navigate_visual::{CameraModel, PlaneMotion, ReferenceView};

/// The measured plane is the dominant visible surface. Fields, slopes, roofs,
/// and the elevation-model resolution separate it from the terrain normal by
/// about this angle, whatever the image residuals are.
const MIN_GROUND_SIGMA_RAD: f64 = 0.026;

/// Terrain normal under the camera, in the world frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainNormal {
    /// Unit normal that points up from the terrain.
    pub normal: Unit<Vector3<f64>>,
    /// One-sigma direction error, including terrain curvature in the view.
    pub sigma_rad: f64,
}

impl TerrainNormal {
    /// Fit a plane to the surface depth of a reference view rendered with `camera`.
    ///
    /// The error combines the fit residual with the surface extent, so curved
    /// terrain gives a larger error. Returns `None` with fewer than 50 valid
    /// depth samples or a degenerate fit.
    pub fn from_reference(camera: &CameraModel, view: &ReferenceView) -> Option<Self> {
        let mut points = Vec::new();
        for v in (0..camera.height).step_by(8) {
            for u in (0..camera.width).step_by(8) {
                let index = usize::try_from(v * camera.width + u).ok()?;
                let depth = f64::from(*view.depth_m.get(index)?);
                if depth.is_finite() && depth > 0.0 {
                    let pixel = Vector2::new(f64::from(u), f64::from(v));
                    points.push(camera.unproject(&view.pose, pixel, depth));
                }
            }
        }
        fit_surface(&points)
    }
}

/// One accepted observation of the ground normal in a camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundPlaneObservation {
    /// Earlier frame of the matched pair. The normal is in its camera frame.
    pub frame: FrameKey,
    /// Unit ground normal in the camera eye frame, from the ground to the camera.
    pub normal_camera: Unit<Vector3<f64>>,
    /// Image-geometry error of the normal, in radians.
    pub geometry_sigma_rad: f64,
    /// Terrain normal under the camera.
    pub terrain: TerrainNormal,
    /// Declared object height divided by the distance to the ground plane.
    pub relief_ratio: f64,
}

impl GroundPlaneObservation {
    /// An observation from a plane motion. Returns `None` when the matches
    /// do not select one solution.
    pub fn from_plane_motion(
        frame: FrameKey,
        motion: &PlaneMotion,
        terrain: TerrainNormal,
        relief_ratio: f64,
    ) -> Option<Self> {
        let solution = motion.unique()?;
        Some(Self {
            frame,
            normal_camera: solution.normal,
            geometry_sigma_rad: motion.normal_sigma_rad,
            terrain,
            relief_ratio,
        })
    }

    fn valid(&self) -> bool {
        let values = [
            self.geometry_sigma_rad,
            self.terrain.sigma_rad,
            self.relief_ratio,
        ];
        values.iter().all(|v| v.is_finite() && *v >= 0.0)
            && self.normal_camera.iter().all(|v| v.is_finite())
            && self.terrain.normal.iter().all(|v| v.is_finite())
    }
}

/// The result of a ground-plane submission.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GroundDecision {
    /// The observation constrains the keyframe tilt.
    Accepted,
    /// An observation with a smaller error already constrains this keyframe.
    SameKeyframe,
    /// The trajectory has no map attitude yet, so the terrain normal cannot
    /// be compared with it. Nothing changed.
    Unanchored,
    /// The observation entered the graph, but it remained an outlier after
    /// optimization and was retracted.
    Retracted,
    /// The observation disagrees with the estimate by more than its error and
    /// the tilt bound from map evidence allow.
    Inconsistent {
        /// Angle between the predicted and the observed normal, in radians.
        angle_rad: f64,
        /// Gate, in radians.
        gate_rad: f64,
    },
}

impl VisualSession {
    /// Submit a ground normal observed in a camera.
    ///
    /// The gate width uses the tilt bound from map evidence, not the bound
    /// of tracked attitude, because tracked attitude can drift without a
    /// visible residual. A trajectory without map attitude takes no
    /// observation. An observation that stays an outlier after optimization
    /// is retracted.
    ///
    /// # Errors
    /// Rejects invalid values and frames without odometry.
    pub fn submit_ground_plane(
        &mut self,
        observation: GroundPlaneObservation,
    ) -> Result<GroundDecision, SessionError> {
        if !observation.valid() {
            return Err(SessionError::Invalid {
                field: "ground plane observation",
            });
        }
        if self
            .frames
            .get(&observation.frame)
            .and_then(|s| s.odometry.as_ref())
            .is_none()
        {
            return Err(SessionError::NoOdometry(observation.frame));
        }
        let relief = self.config.ground.relief_fraction * observation.relief_ratio;
        let measured = [
            observation.geometry_sigma_rad,
            relief,
            observation.terrain.sigma_rad,
        ];
        let sigma = measured
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(MIN_GROUND_SIGMA_RAD);
        let world = observation.terrain.normal.into_inner();
        let camera = observation.normal_camera.into_inner();
        // Every refusal happens before the session changes.
        if let Some(decision) = self.ground_gate(&observation.frame, &world, &camera, sigma) {
            if matches!(decision, GroundDecision::Inconsistent { .. }) {
                self.event(SessionEvent::GroundPlaneRejected {
                    frame: observation.frame,
                });
            }
            return Ok(decision);
        }
        let attachment = self.attach(observation.frame, &[])?;
        let keyframe = attachment.keyframe;
        let camera = attachment.frame_to_keyframe.rotation.inverse() * camera;
        let sigma_rad = sigma + attachment.drift_rad;
        if let Some(existing) = self.grounds.get(&keyframe)
            && existing.sigma_rad <= sigma_rad
        {
            return Ok(GroundDecision::SameKeyframe);
        }
        let before = self.snapshot();
        let frame = observation.frame;
        self.grounds.insert(
            keyframe,
            Ground {
                world,
                camera,
                sigma_rad,
                frame,
            },
        );
        self.event(SessionEvent::GroundPlaneAccepted { frame });
        self.optimize(RevisionCause::GroundPlane, &before);
        if !self.grounds.contains_key(&keyframe) {
            return Ok(GroundDecision::Retracted);
        }
        Ok(GroundDecision::Accepted)
    }

    /// Refuse an observation for a frame without map attitude, or one that
    /// disagrees with the current estimate by more than the observation error
    /// and the tilt bound from map evidence allow.
    fn ground_gate(
        &self,
        frame: &FrameKey,
        world: &Vector3<f64>,
        camera: &Vector3<f64>,
        sigma_rad: f64,
    ) -> Option<GroundDecision> {
        let odometry = self.frames.get(frame)?.odometry.as_ref()?;
        let Some((correction, bounds, _)) = self.locate(frame, odometry) else {
            return Some(GroundDecision::Unanchored);
        };
        // Before a map anchor, the session frame is the odometry frame: the
        // terrain normal has no meaning in it.
        let Some(bound) = bounds.map_tilt_rad else {
            return Some(GroundDecision::Unanchored);
        };
        let pose = crate::pose::compose(&correction, &odometry.pose);
        let angle_rad = (pose.rotation.inverse() * world).angle(camera);
        let gate_rad = self.config.anchors.gate_sigma * (sigma_rad.powi(2) + bound.powi(2)).sqrt();
        (angle_rad > gate_rad).then_some(GroundDecision::Inconsistent {
            angle_rad,
            gate_rad,
        })
    }
}

/// Least-squares plane `z = a x + b y + c` through surface points.
fn fit_surface(points: &[Vector3<f64>]) -> Option<TerrainNormal> {
    if points.len() < 50 {
        return None;
    }
    let mean = points.iter().sum::<Vector3<f64>>() / points.len() as f64;
    let mut normal_matrix = Matrix3::zeros();
    let mut rhs = Vector3::zeros();
    for p in points {
        let row = Vector3::new(p.x - mean.x, p.y - mean.y, 1.0);
        normal_matrix += row * row.transpose();
        rhs += row * (p.z - mean.z);
    }
    let [a, b, _] = normal_matrix.lu().solve(&rhs)?.into();
    let residual = points
        .iter()
        .map(|p| (p.z - mean.z - a * (p.x - mean.x) - b * (p.y - mean.y)).powi(2))
        .sum::<f64>()
        / points.len() as f64;
    let extent = points
        .iter()
        .map(|p| (p.xy() - mean.xy()).norm_squared())
        .sum::<f64>()
        / points.len() as f64;
    let half_width = extent.sqrt();
    if !(half_width > 1.0 && residual.is_finite()) {
        return None;
    }
    Some(TerrainNormal {
        normal: Unit::new_normalize(Vector3::new(-a, -b, 1.0)),
        sigma_rad: residual.sqrt().atan2(half_width),
    })
}

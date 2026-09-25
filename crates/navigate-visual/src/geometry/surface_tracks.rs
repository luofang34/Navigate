//! Persistent feature identities with immutable, conditional surface positions.
use super::{PoseVerifier, TrackingProposal};
use crate::{CameraPose, Frame, LocalFrame, MapRevision, ReferenceView, VisualError, pose_solver};
use nalgebra::{Vector2, Vector3};
use std::collections::BTreeSet;

struct SurfacePoint {
    world: Vector3<f64>,
    pixel: Vector2<f64>,
    depth_observation: String,
}

/// A bounded set of tracked pixels seeded from rendered surface depth.
///
/// A feature keeps its conditional world position while its image location moves.
/// Re-rendering at each estimated pose would redefine that feature's world point
/// and can conceal a growing pose error. These positions are not verified scene
/// geometry. This type does not refine the map, create absolute covariance, or
/// claim that different features or camera observations are independent.
pub struct SurfaceTracks {
    observation: Frame,
    pose: CameraPose,
    map: MapRevision,
    frame: LocalFrame,
    points: Vec<SurfacePoint>,
}

/// A conditional pose update and the observations used to seed its depth.
pub struct SurfaceTrackUpdate {
    /// Pose and image support. This is not an independent geographic fix.
    pub proposal: TrackingProposal,
    /// Observation identities at which the surviving surface points were seeded.
    pub depth_observations: Vec<String>,
}
impl SurfaceTracks {
    /// Seed track identities from a camera observation and a surface at its estimated pose.
    ///
    /// The adapter supplies feature locations. Missing surface depth is excluded.
    ///
    /// # Errors
    /// Rejects invalid reference data or fewer than six supported features.
    pub fn new(
        observation: &Frame,
        surface: &ReferenceView,
        pixels: &[Vector2<f64>],
    ) -> Result<Self, VisualError> {
        surface.validate(observation)?;
        let mut tracks = Self {
            observation: copy_frame(observation),
            pose: surface.pose,
            map: surface.map.clone(),
            frame: surface.frame,
            points: Vec::new(),
        };
        tracks.replenish(surface, pixels)?;
        if tracks.points.len() < 6 {
            return Err(VisualError::InsufficientMatches {
                found: tracks.points.len(),
                required: 6,
            });
        }
        Ok(tracks)
    }

    /// Seed fixed surface positions from verified map-to-camera correspondences.
    ///
    /// Each world position comes from the matched reference pixel and its depth.
    /// The tracked pixel comes from the camera observation. A fitted camera pose
    /// does not redefine the world position to remove its reprojection residual.
    /// The host must retain the reference identity and the returned map estimate.
    /// Rendered depth and map registration remain uncertain and correlated.
    ///
    /// # Errors
    /// Rejects invalid reference data or matches that fail the map acceptance policy.
    pub fn from_map_matches(
        observation: &Frame,
        reference: &ReferenceView,
        prior: &crate::PosePrior,
        matches: &[crate::PixelMatch],
        verifier: &PoseVerifier,
        backend: &str,
    ) -> Result<(Self, crate::Estimate), VisualError> {
        let estimate = verifier.verify(observation, reference, prior, matches, backend)?;
        let origin = observation.evidence_sha256();
        let points = super::depth_correspondences(observation, reference, matches)
            .into_iter()
            .filter(|point| {
                pose_solver::residual(&observation.camera, &estimate.pose, point)
                    <= verifier.config.inlier_threshold_px
            })
            .take(4096)
            .map(|point| SurfacePoint {
                world: point.world,
                pixel: point.pixel,
                depth_observation: origin.clone(),
            })
            .collect();
        Ok((
            Self {
                observation: copy_frame(observation),
                pose: estimate.pose,
                map: reference.map.clone(),
                frame: reference.frame,
                points,
            },
            estimate,
        ))
    }

    /// The exact previous camera pixels to send to the tracking adapter.
    pub fn observation(&self) -> &Frame {
        &self.observation
    }

    /// Feature locations in the previous observation, in adapter input order.
    pub fn pixels(&self) -> Vec<Vector2<f64>> {
        self.points.iter().map(|p| p.pixel).collect()
    }

    /// Number of active feature identities.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether no feature identity remains.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Add features using a new surface at the current conditional pose.
    ///
    /// Surviving world points are not moved. The host must record the surface
    /// pose and depth identity for this observation. Data versions cannot change.
    ///
    /// # Errors
    /// Rejects mismatched camera, map, local frame, pose, or feature locations.
    pub fn replenish(
        &mut self,
        surface: &ReferenceView,
        pixels: &[Vector2<f64>],
    ) -> Result<(), VisualError> {
        surface.validate(&self.observation)?;
        crate::matching::validate_points(&self.observation.image, &surface.image, pixels)?;
        if surface.map != self.map
            || surface.frame != self.frame
            || (surface.pose.position - self.pose.position).norm() > 1e-6
            || surface.pose.orientation.angle_to(&self.pose.orientation) > 1e-8
        {
            return Err(VisualError::Invalid {
                field: "tracked surface identity or pose",
            });
        }
        let camera = self.observation.camera;
        let origin = self.observation.evidence_sha256();
        for &pixel in pixels {
            if self.points.len() >= 4096 {
                break;
            }
            if self
                .points
                .iter()
                .any(|p| (p.pixel - pixel).norm_squared() < 64.0)
            {
                continue;
            }
            let x = pixel.x.round() as usize;
            let y = pixel.y.round() as usize;
            if x >= camera.width as usize || y >= camera.height as usize {
                continue;
            }
            let depth = f64::from(surface.depth_m[y * camera.width as usize + x]);
            if !depth.is_finite() || depth <= 0.0 {
                continue;
            }
            self.points.push(SurfacePoint {
                pixel,
                world: camera.unproject(&surface.pose, pixel, depth),
                depth_observation: origin.clone(),
            });
        }
        Ok(())
    }

    /// Fit all six camera pose components and retain only geometrically supported tracks.
    ///
    /// `locations` must follow the order of [`Self::pixels`]. A failed update
    /// leaves the previous observation and all feature identities unchanged.
    ///
    /// # Errors
    /// Rejects reused or reordered observations, calibration changes, weak geometry,
    /// malformed adapter output, and violations of the supplied navigation prior.
    pub fn update(
        &mut self,
        verifier: &PoseVerifier,
        frame: &Frame,
        prior: &crate::PosePrior,
        locations: &[Option<Vector2<f64>>],
        backend: &str,
    ) -> Result<SurfaceTrackUpdate, VisualError> {
        self.validate_update(frame, locations, backend)?;
        let (points, indices) = self.correspondences(frame, locations);
        let (pose, inliers, count) =
            verifier.fit_points(frame, self.pose, prior, points, pose_solver::Motion::Free)?;
        let (quality, _) = verifier.assess(frame, &pose, &inliers, count)?;
        let supported: BTreeSet<_> = inliers.iter().map(|p| pixel_key(p.pixel)).collect();
        let retained: BTreeSet<_> = indices
            .into_iter()
            .filter(|&i| locations[i].is_some_and(|p| supported.contains(&pixel_key(p))))
            .collect();
        let reference_observation_sha256 = self.observation.evidence_sha256();
        let mut index = 0_usize;
        self.points.retain_mut(|point| {
            let i = index;
            index = index.wrapping_add(1);
            if !retained.contains(&i) {
                return false;
            }
            if let Some(pixel) = locations[i] {
                point.pixel = pixel;
            }
            true
        });
        let depth_observations = self
            .points
            .iter()
            .map(|p| p.depth_observation.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        self.pose = pose;
        self.observation = copy_frame(frame);
        Ok(SurfaceTrackUpdate {
            proposal: TrackingProposal {
                pose,
                quality,
                observation_sha256: frame.evidence_sha256(),
                reference_observation_sha256,
                map: self.map.clone(),
                frame: self.frame,
                backend: backend.to_owned(),
                motion: super::TrackingMotion::Free,
            },
            depth_observations,
        })
    }

    fn validate_update(
        &self,
        frame: &Frame,
        locations: &[Option<Vector2<f64>>],
        backend: &str,
    ) -> Result<(), VisualError> {
        let advance = frame
            .stamp
            .sequence
            .wrapping_sub(self.observation.stamp.sequence);
        if frame.stamp.capture_time_ns <= self.observation.stamp.capture_time_ns
            || advance == 0
            || advance >= 1_u64 << 63
        {
            return Err(VisualError::Invalid {
                field: "tracked observation order",
            });
        }
        if frame.camera != self.observation.camera
            || frame.image.dimensions() != self.observation.image.dimensions()
        {
            return Err(VisualError::Invalid {
                field: "tracking camera calibration",
            });
        }
        if locations.len() != self.points.len() || backend.is_empty() {
            return Err(VisualError::Invalid {
                field: "tracked feature output or backend identity",
            });
        }
        Ok(())
    }

    fn correspondences(
        &self,
        frame: &Frame,
        locations: &[Option<Vector2<f64>>],
    ) -> (Vec<pose_solver::Correspondence>, Vec<usize>) {
        let mut occupied = BTreeSet::new();
        let mut indices = Vec::new();
        let points = self
            .points
            .iter()
            .zip(locations)
            .enumerate()
            .filter_map(|(i, (point, pixel))| {
                let pixel = (*pixel)?;
                if !pixel.iter().all(|v| v.is_finite())
                    || pixel.x < 0.0
                    || pixel.y < 0.0
                    || pixel.x >= f64::from(frame.camera.width - 1)
                    || pixel.y >= f64::from(frame.camera.height - 1)
                    || !occupied.insert(pixel_key(pixel))
                {
                    return None;
                }
                indices.push(i);
                Some(pose_solver::Correspondence {
                    world: point.world,
                    pixel,
                })
            })
            .collect();
        (points, indices)
    }
}
fn pixel_key(pixel: Vector2<f64>) -> (u32, u32) {
    (pixel.x.round() as u32, pixel.y.round() as u32)
}
fn copy_frame(frame: &Frame) -> Frame {
    Frame {
        camera: frame.camera,
        stamp: frame.stamp,
        image: frame.image.clone(),
    }
}

#[cfg(test)]
mod tests;

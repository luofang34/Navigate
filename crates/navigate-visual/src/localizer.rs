//! Frame admission and geometric pose acceptance.

use crate::{
    CameraPose, Frame, FrameStamp, ImageMatcher, MapRevision, PosePrior, ReferenceView,
    VisualError,
    pose_solver::{self, Correspondence},
};
use nalgebra::{SMatrix, Vector2};

/// Acceptance thresholds for one visual observation.
#[derive(Clone, Copy, Debug)]
pub struct LocalizerConfig {
    /// Minimum count of geometric inliers.
    pub min_inliers: usize,
    /// Maximum reprojection error for an inlier, in pixels.
    pub inlier_threshold_px: f64,
    /// Minimum occupied cells in a four-column, three-row image grid.
    pub min_occupied_cells: usize,
    /// Assumed pixel noise floor for the local geometry covariance.
    pub pixel_noise_floor: f64,
}

impl Default for LocalizerConfig {
    fn default() -> Self {
        Self {
            min_inliers: 20,
            inlier_threshold_px: 2.5,
            min_occupied_cells: 4,
            pixel_noise_floor: 1.0,
        }
    }
}

/// Measured evidence for an accepted pose. These values are not an integrity guarantee.
#[derive(Clone, Copy, Debug)]
pub struct EstimateQuality {
    /// Number of matches with usable reference depth.
    pub depth_matches: usize,
    /// Number of final geometric inliers.
    pub inliers: usize,
    /// Root mean square reprojection error, in pixels.
    pub reprojection_rms_px: f64,
    /// Number of occupied cells in the query image.
    pub occupied_cells: usize,
    /// Condition number of the scaled pose normal matrix.
    pub condition_number: f64,
}

/// A map-relative camera pose from one frame.
pub struct Estimate {
    /// Acquisition stamp of the query image.
    pub stamp: FrameStamp,
    /// Selected map release.
    pub map: MapRevision,
    /// Camera pose in the map anchor's local east, north, up frame.
    pub pose: CameraPose,
    /// Acceptance evidence.
    pub quality: EstimateQuality,
    /// Local linearized covariance from image residuals only.
    ///
    /// First three axes are ENU position in metres. Last three are local camera
    /// rotation in radians. Map, calibration, and association errors are excluded.
    /// A fusion adapter must account for these errors before admission.
    pub geometry_covariance: SMatrix<f64, 6, 6>,
    /// Matcher and model identity.
    pub backend: String,
}

/// A stream of independent visual map observations.
///
/// The caller supplies each frame's prior and reference. This component does not
/// feed its own result back as an independent measurement or fuse IMU data.
pub struct Localizer<M> {
    matcher: M,
    config: LocalizerConfig,
    last_stamp: Option<FrameStamp>,
}

impl<M: ImageMatcher> Localizer<M> {
    /// Build a localizer with explicit acceptance thresholds.
    ///
    /// # Errors
    ///
    /// Returns [`VisualError::Invalid`] for invalid acceptance thresholds.
    pub fn new(matcher: M, config: LocalizerConfig) -> Result<Self, VisualError> {
        if config.min_inliers < 6
            || !(1..=12).contains(&config.min_occupied_cells)
            || !config.inlier_threshold_px.is_finite()
            || config.inlier_threshold_px <= 0.0
            || !config.pixel_noise_floor.is_finite()
            || config.pixel_noise_floor <= 0.0
        {
            return Err(VisualError::Invalid {
                field: "localizer thresholds",
            });
        }
        Ok(Self {
            matcher,
            config,
            last_stamp: None,
        })
    }

    /// Estimate the pose from image evidence and validate it against the prior bounds.
    ///
    /// A valid input consumes its stamp before matching starts. A visual rejection
    /// therefore consumes the stamp. Use a new localizer for a new capture stream.
    /// This call waits for the matcher, including GPU readback when enabled.
    /// Call it on a worker that may block.
    ///
    /// # Errors
    ///
    /// Rejects invalid inputs, repeated or out-of-order stamps, backend failures,
    /// insufficient unique matches, weak geometry, and corrections outside the
    /// prior bounds. A rejection contains no substitute pose.
    pub fn estimate_blocking(
        &mut self,
        frame: &Frame,
        reference: &ReferenceView,
        prior: &PosePrior,
    ) -> Result<Estimate, VisualError> {
        reference.validate(frame)?;
        prior.validate()?;
        self.admit_stamp(frame.stamp)?;
        let matches = self
            .matcher
            .match_images_blocking(&reference.image, &frame.image)?;
        let points = depth_correspondences(frame, reference, &matches);
        let depth_matches = points.len();
        self.require_inliers(depth_matches)?;
        let first = pose_solver::optimize(&frame.camera, &points, prior.pose)?;
        let inliers: Vec<_> = points
            .into_iter()
            .filter(|point| {
                pose_solver::residual(&frame.camera, &first, point)
                    <= self.config.inlier_threshold_px
            })
            .collect();
        self.require_inliers(inliers.len())?;
        let pose = pose_solver::optimize(&frame.camera, &inliers, first)?;
        let inliers: Vec<_> = inliers
            .into_iter()
            .filter(|point| {
                pose_solver::residual(&frame.camera, &pose, point)
                    <= self.config.inlier_threshold_px
            })
            .collect();
        self.require_inliers(inliers.len())?;
        check_bounds(&pose, prior)?;
        let (quality, covariance) = self.assess(frame, &pose, &inliers, depth_matches)?;
        Ok(Estimate {
            stamp: frame.stamp,
            map: reference.map.clone(),
            pose,
            quality,
            geometry_covariance: covariance,
            backend: self.matcher.identity().to_owned(),
        })
    }

    fn admit_stamp(&mut self, stamp: FrameStamp) -> Result<(), VisualError> {
        if let Some(previous) = self.last_stamp {
            if stamp.capture_time_ns <= previous.capture_time_ns {
                return Err(VisualError::FrameOrder {
                    previous_ns: previous.capture_time_ns,
                    received_ns: stamp.capture_time_ns,
                });
            }
            let advance = stamp.sequence.wrapping_sub(previous.sequence);
            if advance == 0 || advance >= (1_u64 << 63) {
                return Err(VisualError::Invalid {
                    field: "frame sequence",
                });
            }
        }
        self.last_stamp = Some(stamp);
        Ok(())
    }

    fn require_inliers(&self, found: usize) -> Result<(), VisualError> {
        if found < self.config.min_inliers {
            return Err(VisualError::InsufficientMatches {
                found,
                required: self.config.min_inliers,
            });
        }
        Ok(())
    }

    fn assess(
        &self,
        frame: &Frame,
        pose: &CameraPose,
        points: &[Correspondence],
        depth_matches: usize,
    ) -> Result<(EstimateQuality, SMatrix<f64, 6, 6>), VisualError> {
        let mut cells = [false; 12];
        let mut squared_error = 0.0;
        for point in points {
            let x = (point.pixel.x * 4.0 / f64::from(frame.camera.width)).clamp(0.0, 3.0) as usize;
            let y = (point.pixel.y * 3.0 / f64::from(frame.camera.height)).clamp(0.0, 2.0) as usize;
            cells[y * 4 + x] = true;
            squared_error += pose_solver::residual(&frame.camera, pose, point).powi(2);
        }
        let occupied_cells = cells.into_iter().filter(|occupied| *occupied).count();
        let (h, _) = pose_solver::normal_equations(&frame.camera, points, pose);
        let eigenvalues = h.symmetric_eigen().eigenvalues;
        let condition_number = eigenvalues.max() / eigenvalues.min();
        if occupied_cells < self.config.min_occupied_cells
            || eigenvalues.min() <= 1e-10
            || !condition_number.is_finite()
            || condition_number > 1e10
        {
            return Err(VisualError::DegenerateGeometry);
        }
        let variance = (squared_error / (2 * points.len() - 6) as f64)
            .max(self.config.pixel_noise_floor.powi(2));
        let scale = SMatrix::<f64, 6, 6>::from_diagonal(&nalgebra::SVector::from_row_slice(&[
            1.0, 1.0, 1.0, 0.001, 0.001, 0.001,
        ]));
        let covariance =
            scale * h.try_inverse().ok_or(VisualError::DegenerateGeometry)? * scale * variance;
        Ok((
            EstimateQuality {
                depth_matches,
                inliers: points.len(),
                occupied_cells,
                condition_number,
                reprojection_rms_px: (squared_error / points.len() as f64).sqrt(),
            },
            covariance,
        ))
    }
}

fn check_bounds(pose: &CameraPose, prior: &PosePrior) -> Result<(), VisualError> {
    let meters = (pose.position - prior.pose.position).norm();
    let radians = pose.orientation.angle_to(&prior.pose.orientation);
    if meters > prior.position_radius_m || radians > prior.attitude_radius_rad {
        return Err(VisualError::OutsidePrior { meters, radians });
    }
    Ok(())
}

fn depth_correspondences(
    frame: &Frame,
    reference: &ReferenceView,
    matches: &[crate::PixelMatch],
) -> Vec<Correspondence> {
    let mut reference_pixels = std::collections::BTreeSet::new();
    let mut query_pixels = std::collections::BTreeSet::new();
    matches
        .iter()
        .filter_map(|pair| {
            let p = pair.reference;
            let q = pair.query;
            if ![p.x, p.y, q.x, q.y].iter().all(|v| v.is_finite()) {
                return None;
            }
            let inside = |p: Vector2<f64>| {
                p.x >= 0.0
                    && p.y >= 0.0
                    && p.x < f64::from(frame.camera.width - 1)
                    && p.y < f64::from(frame.camera.height - 1)
            };
            if !inside(p) || !inside(q) {
                return None;
            }
            let x = p.x.round() as usize;
            let y = p.y.round() as usize;
            let depth = f64::from(reference.depth_m[y * frame.camera.width as usize + x]);
            if !depth.is_finite() || depth <= 0.0 {
                return None;
            }
            let query_key = (q.x.round() as u32, q.y.round() as u32);
            if reference_pixels.contains(&(x, y)) || query_pixels.contains(&query_key) {
                return None;
            }
            reference_pixels.insert((x, y));
            query_pixels.insert(query_key);
            Some(Correspondence {
                world: frame.camera.unproject(&reference.pose, p, depth),
                pixel: q,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests;

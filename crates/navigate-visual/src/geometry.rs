//! Shared surface geometry checks for any image correspondence backend.
use crate::{
    CameraPose, Estimate, EstimateQuality, Frame, LocalizerConfig, PixelMatch, PosePrior,
    ReferenceView, VisualError,
    pose_solver::{self, Correspondence},
};
use nalgebra::{SMatrix, Vector2};
mod surface_tracks;
mod tracking;
pub use surface_tracks::{SurfaceTrackUpdate, SurfaceTracks};
pub use tracking::{TrackingMotion, TrackingProposal, TrackingReference};

/// Full acceptance result and a possible seed for further reference rendering.
pub struct CandidateEvaluation {
    /// The unchanged geometric and navigation-prior acceptance decision.
    pub acceptance: Result<Estimate, VisualError>,
    /// A fitted pose inside the prior, with enough unique inliers to refine.
    /// The inlier count or spatial support can still fail acceptance. This is not a measurement.
    pub refinement: Option<CameraPose>,
}

/// Stateless geometric validation for a candidate reference.
///
/// Repeated calls evaluate alternatives from the same evidence. They do not
/// consume a stream stamp, fuse estimates, or reduce uncertainty across calls.
pub struct PoseVerifier {
    config: LocalizerConfig,
}
impl PoseVerifier {
    /// Construct the common visual acceptance policy.
    ///
    /// # Errors
    /// Rejects invalid acceptance thresholds.
    pub fn new(config: LocalizerConfig) -> Result<Self, VisualError> {
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
        Ok(Self { config })
    }
    /// Fit and check one candidate using backend-independent pixel matches.
    ///
    /// The reference pose initializes fitting. The navigation prior only sets
    /// admission bounds. This method imposes no planar homography model.
    ///
    /// # Errors
    /// Rejects invalid inputs, missing depth, weak geometry, and prior violations.
    pub fn verify(
        &self,
        frame: &Frame,
        reference: &ReferenceView,
        prior: &PosePrior,
        matches: &[PixelMatch],
        matcher_identity: &str,
    ) -> Result<Estimate, VisualError> {
        self.evaluate(frame, reference, prior, matches, matcher_identity)
            .acceptance
    }

    /// Evaluate a candidate and retain a bounded pose for another reference render.
    ///
    /// A refinement pose is not an accepted measurement. It can have weak spatial
    /// support. A new render and new correspondences must pass the full policy.
    /// Repeated calls retain the same observation identity and add no confidence.
    pub fn evaluate(
        &self,
        frame: &Frame,
        reference: &ReferenceView,
        prior: &PosePrior,
        matches: &[PixelMatch],
        matcher_identity: &str,
    ) -> CandidateEvaluation {
        if matcher_identity.is_empty() {
            return CandidateEvaluation {
                acceptance: Err(VisualError::Invalid {
                    field: "matcher identity",
                }),
                refinement: None,
            };
        }
        let (pose, inliers, depth_matches) =
            match self.fit_candidate(frame, reference, prior, matches, pose_solver::Motion::Free) {
                Ok(fitted) => fitted,
                Err(error) => {
                    return CandidateEvaluation {
                        acceptance: Err(error),
                        refinement: None,
                    };
                }
            };
        let acceptance =
            self.assess(frame, &pose, &inliers, depth_matches)
                .map(|(quality, covariance)| Estimate {
                    stamp: frame.stamp,
                    observation_sha256: frame.evidence_sha256(),
                    map: reference.map.clone(),
                    frame: reference.frame,
                    pose,
                    quality,
                    geometry_covariance: covariance,
                    backend: matcher_identity.to_owned(),
                });
        CandidateEvaluation {
            acceptance,
            refinement: Some(pose),
        }
    }

    fn fit_candidate(
        &self,
        frame: &Frame,
        reference: &ReferenceView,
        prior: &PosePrior,
        matches: &[PixelMatch],
        motion: pose_solver::Motion,
    ) -> Result<(CameraPose, Vec<Correspondence>, usize), VisualError> {
        reference.validate(frame)?;
        prior.validate()?;
        let points = depth_correspondences(frame, reference, matches);
        self.fit_points(frame, reference.pose, prior, points, motion)
    }

    fn fit_points(
        &self,
        frame: &Frame,
        initial: CameraPose,
        prior: &PosePrior,
        points: Vec<Correspondence>,
        motion: pose_solver::Motion,
    ) -> Result<(CameraPose, Vec<Correspondence>, usize), VisualError> {
        prior.validate()?;
        initial.validate()?;
        let depth_matches = points.len();
        Self::require_fit_points(depth_matches)?;
        let first = pose_solver::initialize(
            &frame.camera,
            &points,
            initial,
            self.config.inlier_threshold_px,
            motion,
        );
        let inliers: Vec<_> = points
            .into_iter()
            .filter(|point| {
                pose_solver::residual(&frame.camera, &first, point)
                    <= self.config.inlier_threshold_px
            })
            .collect();
        Self::require_fit_points(inliers.len())?;
        let pose = pose_solver::optimize_motion(&frame.camera, &inliers, first, motion)?;
        let inliers: Vec<_> = inliers
            .into_iter()
            .filter(|point| {
                pose_solver::residual(&frame.camera, &pose, point)
                    <= self.config.inlier_threshold_px
            })
            .collect();
        Self::require_fit_points(inliers.len())?;
        check_bounds(&pose, prior)?;
        Ok((pose, inliers, depth_matches))
    }

    fn require_fit_points(found: usize) -> Result<(), VisualError> {
        if found < 6 {
            return Err(VisualError::InsufficientMatches { found, required: 6 });
        }
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
        self.require_inliers(points.len())?;
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

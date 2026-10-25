//! Require raw sensor support before admitting a rectified map pose.
use super::{FitSupport, PoseVerifier, orthographic::correspondences};
use crate::{
    Frame, OrthographicEstimate, OrthographicReference, PixelMatch, PosePrior, RawPixelQuality,
    RectificationMap, VisualError,
    pose_solver::{self, Correspondence},
};
use nalgebra::Vector2;

struct Evidence {
    point: Correspondence,
    raw: Vector2<f64>,
    reference: Vector2<f64>,
    squared_error: f64,
}

impl PoseVerifier {
    /// Check a fitted map pose in both rectified and raw sensor pixels.
    ///
    /// The pose is fixed. Both pixel spaces use this verifier's thresholds.
    /// Map heights supply the world points. Successful checks replace the
    /// estimate's quality and pinhole covariance with the common support subset.
    /// The covariance remains in the rectified pixel noise model; it does not
    /// describe calibration error or a raw sensor noise model.
    ///
    /// # Errors
    /// Rejects source identity or camera mismatches, invalid inputs, prior
    /// violations, insufficient raw support, and poorly conditioned geometry.
    /// A rejection does not change the supplied estimate.
    pub fn check_rectified_orthographic(
        &self,
        frame: &Frame,
        reference: &OrthographicReference,
        prior: &PosePrior,
        matches: &[PixelMatch],
        map: &RectificationMap,
        estimate: &mut OrthographicEstimate,
    ) -> Result<RawPixelQuality, VisualError> {
        frame.camera.validate()?;
        reference.validate()?;
        prior.validate()?;
        estimate.pose.validate()?;
        if frame.camera != map.camera()
            || frame.image.dimensions() != (frame.camera.width, frame.camera.height)
            || matches.len() > 16384
            || estimate.map != reference.map
            || estimate.observation_sha256 != frame.evidence_sha256()
            || estimate.backend.is_empty()
        {
            return Err(VisualError::Invalid {
                field: "rectified map evidence binding",
            });
        }
        super::check_bounds(&estimate.pose, prior)?;
        let points = correspondences(frame, reference, matches);
        let count = points.len();
        let mut evidence = raw_evidence(
            frame,
            reference,
            &estimate.pose,
            points,
            map,
            self.config.inlier_threshold_px,
        );
        let inliers = evidence.len();
        self.require_inliers(inliers)?;
        let selected = separated(&mut evidence, self.config.support_separation_px);
        let raw_camera = map.source_camera();
        let query_cells = cells(
            selected.iter().map(|p| p.raw),
            [raw_camera.width, raw_camera.height],
        );
        let reference_cells = cells(
            selected.iter().map(|p| p.reference),
            [reference.image.width(), reference.image.height()],
        );
        if query_cells < self.config.min_occupied_cells {
            return Err(VisualError::DegenerateGeometry);
        }
        let support = FitSupport::from_separated(
            &frame.camera,
            selected.iter().map(|p| p.point.clone()).collect(),
            inliers,
            reference_cells,
        );
        let (quality, covariance) = self.assess_support(frame, &estimate.pose, &support, count)?;
        let raw = RawPixelQuality {
            inliers,
            spatial_support: selected.len(),
            query_cells,
            reference_cells,
            reprojection_rms_px: (selected.iter().map(|p| p.squared_error).sum::<f64>()
                / selected.len() as f64)
                .sqrt(),
        };
        estimate.quality = quality;
        estimate.geometry_covariance = covariance;
        Ok(raw)
    }
}
fn raw_evidence(
    frame: &Frame,
    reference: &OrthographicReference,
    pose: &crate::CameraPose,
    points: Vec<Correspondence>,
    map: &RectificationMap,
    threshold: f64,
) -> Vec<Evidence> {
    points
        .into_iter()
        .filter_map(|point| {
            if pose_solver::residual(&frame.camera, pose, &point) > threshold {
                return None;
            }
            let predicted = frame.camera.project(pose, point.world)?;
            let raw = map.raw_pixel(point.pixel)?;
            let error = (raw - map.raw_pixel(predicted)?).norm_squared();
            if error > threshold * threshold {
                return None;
            }
            Some(Evidence {
                reference: reference.geometry.project(point.world)?,
                point,
                raw,
                squared_error: error,
            })
        })
        .collect()
}
fn separated(points: &mut [Evidence], distance: f64) -> Vec<&Evidence> {
    points.sort_by(|a, b| {
        a.raw
            .x
            .total_cmp(&b.raw.x)
            .then(a.raw.y.total_cmp(&b.raw.y))
            .then(a.point.world.x.total_cmp(&b.point.world.x))
            .then(a.point.world.y.total_cmp(&b.point.world.y))
            .then(a.point.world.z.total_cmp(&b.point.world.z))
    });
    let mut selected: Vec<&Evidence> = Vec::new();
    let squared = distance * distance;
    for point in points.iter() {
        if selected.iter().any(|p| {
            (p.raw - point.raw).norm_squared() < squared
                || (p.point.pixel - point.point.pixel).norm_squared() < squared
                || (p.reference - point.reference).norm_squared() < squared
        }) {
            continue;
        }
        selected.push(point);
    }
    selected
}
fn cells(pixels: impl Iterator<Item = Vector2<f64>>, size: [u32; 2]) -> usize {
    let mut occupied = [false; 12];
    for p in pixels {
        let x = (p.x * 4.0 / f64::from(size[0])).clamp(0.0, 3.0) as usize;
        let y = (p.y * 3.0 / f64::from(size[1])).clamp(0.0, 2.0) as usize;
        occupied[y * 4 + x] = true;
    }
    occupied.into_iter().filter(|v| *v).count()
}
#[cfg(test)]
mod tests;

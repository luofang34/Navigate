//! Two-view geometry of a dominant ground plane, from image matches alone.
//!
//! The homography between two views of a plane fixes the relative rotation,
//! the translation divided by the plane distance, and the plane normal in the
//! camera frame. No map depth or earlier pose enters this estimate, so an
//! attitude error in the host state cannot feed back into it. The plane
//! normal gives the camera tilt relative to the ground. It does not give the
//! heading or the metric scale.
//!
//! The decomposition has two physical solutions in general. A solution must
//! put the matched points in front of both cameras. Often only one passes;
//! in general both can. The result then keeps both solutions, and only
//! independent attitude evidence can select one.

use crate::{
    CameraModel, PixelMatch, VisualError,
    retrieval::{fit, planar_consensus},
};
use nalgebra::{Matrix3, Unit, UnitQuaternion, Vector2, Vector3};

/// Acceptance limits for a two-view plane estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneMotionConfig {
    /// Transfer error limit of a homography inlier, in pixels.
    pub inlier_threshold_px: f64,
    /// Smallest number of inliers.
    pub min_inliers: usize,
    /// Smallest share of matches that are inliers. A lower share means the
    /// scene is not dominated by one plane.
    pub min_inlier_fraction: f64,
    /// Smallest occupied cells of a 4 by 3 grid in the earlier image.
    pub min_occupied_cells: usize,
    /// Smallest translation divided by plane distance. Below this the plane
    /// normal is not observable.
    pub min_parallax: f64,
    /// Share of inliers that must be in front of the camera for a solution.
    pub min_visible_fraction: f64,
}

impl Default for PlaneMotionConfig {
    fn default() -> Self {
        Self {
            inlier_threshold_px: 1.5,
            min_inliers: 30,
            min_inlier_fraction: 0.4,
            min_occupied_cells: 6,
            min_parallax: 0.005,
            min_visible_fraction: 0.97,
        }
    }
}

/// One physical decomposition of a plane homography.
///
/// Vectors use the eye axes of `CameraPose`: right, up, and back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneSolution {
    /// Orientation of the later camera in the earlier camera frame.
    pub rotation: UnitQuaternion<f64>,
    /// Position of the later camera in the earlier camera frame, divided by
    /// the distance from the earlier camera to the plane.
    pub translation_per_distance: Vector3<f64>,
    /// Unit plane normal in the earlier camera frame, from the plane to the camera.
    pub normal: Unit<Vector3<f64>>,
}

/// Relative camera motion and ground-plane normal from two views.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneMotion {
    /// One solution, or two when the matches cannot separate them.
    pub solutions: Vec<PlaneSolution>,
    /// Homography inliers.
    pub inliers: usize,
    /// Occupied cells of a 4 by 3 grid in the earlier image.
    pub occupied_cells: usize,
    /// Root mean square transfer error of the inliers, in pixels.
    pub residual_rms_px: f64,
    /// First-order normal error from pixel noise, propagated through the fit
    /// and the decomposition, in radians. The pixel noise comes from the
    /// transfer residual. Scene relief and terrain slope are not included.
    pub normal_sigma_rad: f64,
}

impl PlaneMotion {
    /// The solution, when the matches select exactly one.
    pub fn unique(&self) -> Option<&PlaneSolution> {
        match self.solutions.as_slice() {
            [only] => Some(only),
            _ => None,
        }
    }
}

/// Estimate the plane motion between an earlier image (match reference) and
/// a later image (match query).
///
/// # Errors
/// Rejects invalid calibration or matches, weak or nonplanar support, and
/// too little parallax.
pub fn plane_motion(
    camera: &CameraModel,
    matches: &[PixelMatch],
    config: &PlaneMotionConfig,
) -> Result<PlaneMotion, VisualError> {
    camera.validate()?;
    if matches.len() > 16_384
        || matches.iter().any(|m| {
            !(m.reference
                .iter()
                .chain(m.query.iter())
                .all(|v| v.is_finite()))
        })
    {
        return Err(VisualError::Invalid {
            field: "plane matches",
        });
    }
    let rays: Vec<_> = matches
        .iter()
        .map(|m| (normalized(camera, m.reference), normalized(camera, m.query)))
        .collect();
    if rays.len() < config.min_inliers.max(8) {
        return Err(VisualError::InsufficientMatches {
            found: rays.len(),
            required: config.min_inliers.max(8),
        });
    }
    let inliers = planar_consensus(*camera, &rays, config.inlier_threshold_px);
    let needed = config
        .min_inliers
        .max((config.min_inlier_fraction * rays.len() as f64).ceil() as usize);
    if inliers.len() < needed {
        return Err(VisualError::InsufficientMatches {
            found: inliers.len(),
            required: needed,
        });
    }
    let cells = occupied_cells(camera, matches, &inliers);
    if cells < config.min_occupied_cells {
        return Err(VisualError::InsufficientSpatialSupport {
            found: cells,
            required: config.min_occupied_cells,
        });
    }
    let h = fit(&rays, &inliers).ok_or(VisualError::DegenerateGeometry)?;
    let residual_rms_px = transfer_rms(camera, &h, &rays, &inliers);
    let points: Vec<_> = inliers
        .iter()
        .filter_map(|&i| rays.get(i).copied())
        .collect();
    let visible = decompose::visible_solutions(&h, &points, config)?;
    let normals: Vec<_> = visible.iter().map(|(_, n)| *n).collect();
    // The transfer residual adds the noise of both images.
    let sigma_px = (residual_rms_px / std::f64::consts::SQRT_2).max(MIN_PIXEL_SIGMA);
    let sigmas = uncertainty::normal_sigmas(&points, &normals, [camera.fx, camera.fy], sigma_px);
    Ok(PlaneMotion {
        solutions: visible.into_iter().map(|(s, _)| s).collect(),
        inliers: inliers.len(),
        occupied_cells: cells,
        residual_rms_px,
        normal_sigma_rad: sigmas.into_iter().fold(0.0, f64::max),
    })
}

/// Normalized image coordinates with the optical axis forward and rows down.
fn normalized(camera: &CameraModel, pixel: Vector2<f64>) -> Vector2<f64> {
    Vector2::new(
        (pixel.x - camera.cx) / camera.fx,
        (pixel.y - camera.cy) / camera.fy,
    )
}

fn transfer_rms(
    camera: &CameraModel,
    h: &Matrix3<f64>,
    rays: &[(Vector2<f64>, Vector2<f64>)],
    inliers: &[usize],
) -> f64 {
    let sum: f64 = inliers
        .iter()
        .filter_map(|&i| rays.get(i))
        .map(|(p, q)| {
            let r = h * Vector3::new(p.x, p.y, 1.0);
            ((Vector2::new(r.x / r.z, r.y / r.z) - q)
                .component_mul(&Vector2::new(camera.fx, camera.fy)))
            .norm_squared()
        })
        .sum();
    (sum / inliers.len().max(1) as f64).sqrt()
}

fn occupied_cells(camera: &CameraModel, matches: &[PixelMatch], inliers: &[usize]) -> usize {
    let mut cells = [false; 12];
    for m in inliers.iter().filter_map(|&i| matches.get(i)) {
        let col = ((m.reference.x / f64::from(camera.width)) * 4.0).clamp(0.0, 3.0) as usize;
        let row = ((m.reference.y / f64::from(camera.height)) * 3.0).clamp(0.0, 2.0) as usize;
        if let Some(cell) = cells.get_mut(row * 4 + col) {
            *cell = true;
        }
    }
    cells.iter().filter(|c| **c).count()
}

/// Matching noise below this value is not credible, in pixels.
const MIN_PIXEL_SIGMA: f64 = 0.2;

mod decompose;
mod uncertainty;

#[cfg(test)]
mod tests;

//! Geometric evidence for prioritizing another render of the same observation.
use super::support;
use crate::{CameraModel, CameraPose, pose_solver::Correspondence};

/// A bounded render seed. It is not an accepted pose or a new measurement.
///
/// Counts describe the current fit. They can order work within a resource budget.
/// They are not probabilities and do not establish independent evidence.
#[derive(Clone, Copy, Debug)]
pub struct RefinementSeed {
    /// Fitted pose inside the supplied navigation bounds.
    pub pose: CameraPose,
    /// Geometric inliers before spatial separation.
    pub inliers: usize,
    /// Inliers separated in both query and reference pixels.
    pub spatial_support: usize,
    /// Occupied cells in the four-column, three-row query grid.
    pub query_cells: usize,
    /// Occupied cells in the four-column, three-row reference grid.
    pub reference_cells: usize,
}

pub(super) struct FitSupport {
    pub points: Vec<Correspondence>,
    pub inliers: usize,
    pub query_cells: usize,
    pub reference_cells: usize,
}
impl FitSupport {
    pub fn new(
        camera: &CameraModel,
        reference: &CameraPose,
        points: &[Correspondence],
        separation: f64,
    ) -> Self {
        let inliers = points.len();
        let (points, reference_cells) = support::separated(camera, reference, points, separation);
        let mut cells = [false; 12];
        for point in &points {
            let x = (point.pixel.x * 4.0 / f64::from(camera.width)).clamp(0.0, 3.0) as usize;
            let y = (point.pixel.y * 3.0 / f64::from(camera.height)).clamp(0.0, 2.0) as usize;
            cells[y * 4 + x] = true;
        }
        Self {
            points,
            inliers,
            query_cells: cells.into_iter().filter(|c| *c).count(),
            reference_cells,
        }
    }
    pub fn seed(&self, pose: CameraPose) -> RefinementSeed {
        RefinementSeed {
            pose,
            inliers: self.inliers,
            spatial_support: self.points.len(),
            query_cells: self.query_cells,
            reference_cells: self.reference_cells,
        }
    }
}

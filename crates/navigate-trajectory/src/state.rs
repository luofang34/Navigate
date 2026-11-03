//! The position and the velocity that every trajectory law reads.

use nalgebra::Vector3;
use navigate_contract::CartesianSolution;

use crate::error::TrajectoryError;

/// Below this length in meters or meters per second, a vector has no
/// direction that a law can use.
pub(crate) const MIN_DIRECTION_NORM: f64 = 1e-9;

/// A position in meters and a velocity in m/s in the axes of one
/// Cartesian frame. The caller names the frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateVector {
    /// The position in meters.
    pub position_m: Vector3<f64>,
    /// The velocity in m/s.
    pub velocity_mps: Vector3<f64>,
}

impl StateVector {
    /// Builds a state from a position in meters and a velocity in m/s.
    #[must_use]
    pub const fn new(position_m: Vector3<f64>, velocity_mps: Vector3<f64>) -> Self {
        Self {
            position_m,
            velocity_mps,
        }
    }

    /// The position and the velocity of `solution`, in the axes of
    /// `solution.frame`.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::NonFiniteState`] when a component is not
    /// finite.
    pub fn from_solution(solution: &CartesianSolution) -> Result<Self, TrajectoryError> {
        let p = solution.position;
        let v = solution.velocity;
        Self::new(
            Vector3::new(p.x_m, p.y_m, p.z_m),
            Vector3::new(v.vx_mps, v.vy_mps, v.vz_mps),
        )
        .checked()
    }

    /// This state when every component is finite.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::NonFiniteState`] when a component is not
    /// finite.
    pub fn checked(self) -> Result<Self, TrajectoryError> {
        let finite = self.position_m.iter().all(|c| c.is_finite())
            && self.velocity_mps.iter().all(|c| c.is_finite());
        if finite {
            Ok(self)
        } else {
            Err(TrajectoryError::NonFiniteState)
        }
    }
}

/// The unit vector of `v`, or `None` when `v` is too short to give a
/// direction.
pub(crate) fn unit(v: &Vector3<f64>) -> Option<Vector3<f64>> {
    let norm = v.norm();
    (norm > MIN_DIRECTION_NORM && norm.is_finite()).then(|| v / norm)
}

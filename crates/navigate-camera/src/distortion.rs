//! Lens distortion in normalized image coordinates.

use nalgebra::Vector2;

use crate::CameraError;

/// Fixed-point iterations that remove distortion.
const UNDISTORT_ITERATIONS: usize = 20;
/// Residual in normalized coordinates at which the inversion stops.
const UNDISTORT_TOLERANCE: f64 = 1e-12;

/// The lens distortion of a camera.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum Distortion {
    /// An ideal pinhole.
    #[default]
    None,
    /// The Brown–Conrady model with three radial and two tangential terms,
    /// with the coefficient names of OpenCV.
    BrownConrady {
        /// First radial coefficient.
        k1: f64,
        /// Second radial coefficient.
        k2: f64,
        /// First tangential coefficient.
        p1: f64,
        /// Second tangential coefficient.
        p2: f64,
        /// Third radial coefficient.
        k3: f64,
    },
}

impl Distortion {
    /// The distorted coordinates of ideal normalized coordinates.
    #[must_use]
    pub fn distort(&self, ideal: &Vector2<f64>) -> Vector2<f64> {
        match *self {
            Self::None => *ideal,
            Self::BrownConrady { k1, k2, p1, p2, k3 } => {
                let (x, y) = (ideal.x, ideal.y);
                let r2 = x * x + y * y;
                let radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
                Vector2::new(
                    x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
                    y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
                )
            }
        }
    }

    /// The ideal coordinates of distorted normalized coordinates.
    ///
    /// # Errors
    /// Returns [`CameraError::NotConverged`] when the fixed-point iteration
    /// does not reach its tolerance, for example outside the valid area of
    /// a strong lens model.
    pub fn undistort(&self, distorted: &Vector2<f64>) -> Result<Vector2<f64>, CameraError> {
        if matches!(self, Self::None) {
            return Ok(*distorted);
        }
        let mut ideal = *distorted;
        for _ in 0..UNDISTORT_ITERATIONS {
            let residual = self.distort(&ideal) - distorted;
            if !residual.x.is_finite() || !residual.y.is_finite() {
                return Err(CameraError::NotConverged);
            }
            if residual.norm_squared() < UNDISTORT_TOLERANCE * UNDISTORT_TOLERANCE {
                return Ok(ideal);
            }
            ideal -= residual;
        }
        Err(CameraError::NotConverged)
    }

    /// Checks that every coefficient is finite.
    ///
    /// # Errors
    /// Refuses a coefficient that is not finite.
    pub fn validate(&self) -> Result<(), CameraError> {
        match *self {
            Self::None => Ok(()),
            Self::BrownConrady { k1, k2, p1, p2, k3 } => {
                if [k1, k2, p1, p2, k3].iter().all(|c| c.is_finite()) {
                    Ok(())
                } else {
                    Err(CameraError::Invalid {
                        field: "distortion coefficient",
                    })
                }
            }
        }
    }
}

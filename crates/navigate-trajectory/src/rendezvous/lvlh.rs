//! The local-vertical local-horizontal frame of a target orbit.

use nalgebra::{Matrix3, Vector3};

use crate::error::TrajectoryError;
use crate::state::{StateVector, unit};

/// The local-vertical local-horizontal frame of a target.
///
/// With the target position `r` and velocity `v` in an inertial frame,
/// the axes are the radial axis `R = r/|r|`, the cross-track axis
/// `N = (r × v)/|r × v|`, and the along-track axis `T = N × R`. LVLH
/// coordinates are `(R, T, N)` in this order. The frame turns about `N`
/// at the rate `|r × v|/|r|²`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LvlhFrame {
    origin: StateVector,
    rotation: Matrix3<f64>,
    angular_rate_rad_per_s: f64,
}

impl LvlhFrame {
    /// The LVLH frame of the target state `target`, in inertial axes.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::NonFiniteState`] when a component is not
    /// finite, and [`TrajectoryError::DegenerateOrbit`] when `|r|` or
    /// `|r × v|` is too small to define the axes.
    pub fn from_target(target: &StateVector) -> Result<Self, TrajectoryError> {
        let target = target.checked()?;
        let r = target.position_m;
        let h = r.cross(&target.velocity_mps);
        let radial = unit(&r).ok_or(TrajectoryError::DegenerateOrbit)?;
        let normal = unit(&h).ok_or(TrajectoryError::DegenerateOrbit)?;
        let along = normal.cross(&radial);
        Ok(Self {
            origin: target,
            rotation: Matrix3::from_rows(&[
                radial.transpose(),
                along.transpose(),
                normal.transpose(),
            ]),
            angular_rate_rad_per_s: h.norm() / r.norm_squared(),
        })
    }

    /// The rate in rad/s at which the frame turns about its cross-track
    /// axis. For a circular target orbit it is the mean motion.
    #[must_use]
    pub const fn angular_rate_rad_per_s(&self) -> f64 {
        self.angular_rate_rad_per_s
    }

    /// The radial, along-track, and cross-track unit vectors in the
    /// inertial axes.
    #[must_use]
    pub fn axes(&self) -> [Vector3<f64>; 3] {
        let t = self.rotation.transpose();
        [
            t.column(0).into_owned(),
            t.column(1).into_owned(),
            t.column(2).into_owned(),
        ]
    }

    /// The position and the velocity of `chaser` relative to the target,
    /// in LVLH coordinates. The velocity is the rate of change seen in
    /// the turning frame.
    #[must_use]
    pub fn relative_state(&self, chaser: &StateVector) -> StateVector {
        let rho = self.rotation * (chaser.position_m - self.origin.position_m);
        let rho_dot = self.rotation * (chaser.velocity_mps - self.origin.velocity_mps)
            - self.rate_vector().cross(&rho);
        StateVector::new(rho, rho_dot)
    }

    /// The inertial velocity of a chaser at the LVLH position `rho` that
    /// has the LVLH velocity `rho_dot`.
    #[must_use]
    pub fn inertial_velocity(&self, rho: &Vector3<f64>, rho_dot: &Vector3<f64>) -> Vector3<f64> {
        self.origin.velocity_mps
            + self.rotation.transpose() * (rho_dot + self.rate_vector().cross(rho))
    }

    const fn rate_vector(&self) -> Vector3<f64> {
        Vector3::new(0.0, 0.0, self.angular_rate_rad_per_s)
    }
}

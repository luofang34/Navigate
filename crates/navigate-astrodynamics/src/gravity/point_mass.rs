//! Third bodies as point masses, in a set of fixed capacity.

use core::fmt;

use nalgebra::{Matrix3, Vector3};

use super::MIN_CENTRAL_DISTANCE_M;
use crate::float_math::powi;

/// Largest number of third bodies in one [`ThirdBodySet`].
pub const MAX_THIRD_BODIES: usize = 8;

/// A third body as a point mass at a known position in the frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointMassBody {
    /// NAIF code of the body.
    pub naif_id: i32,
    /// Gravitational parameter `μ` in m³/s².
    pub gm_m3_per_s2: f64,
    /// Position of the body in the frame in meters.
    pub position_in_frame_m: [f64; 3],
}

impl PointMassBody {
    /// The value of the unused slots of a [`ThirdBodySet`].
    const VACANT: Self = Self {
        naif_id: 0,
        gm_m3_per_s2: 0.0,
        position_in_frame_m: [0.0; 3],
    };

    fn position(&self) -> Vector3<f64> {
        Vector3::from(self.position_in_frame_m)
    }

    /// Whether `μ` is finite and positive and the position is finite and
    /// not at the frame center.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        let r = self.position();
        self.gm_m3_per_s2.is_finite()
            && self.gm_m3_per_s2 > 0.0
            && r.iter().all(|c| c.is_finite())
            && r.norm() >= MIN_CENTRAL_DISTANCE_M
    }

    /// The tidal acceleration `μ·(d/|d|³ − r_k/|r_k|³)` in m/s² at `r`,
    /// with `d = r_k − r`. The second term is the acceleration of the
    /// frame center, so the acceleration is zero at the center.
    #[must_use]
    pub fn tidal_acceleration(&self, r: &Vector3<f64>) -> Vector3<f64> {
        let body = self.position();
        let d = body - r;
        let d_norm = d.norm();
        let body_norm = body.norm();
        if d_norm < MIN_CENTRAL_DISTANCE_M || body_norm < MIN_CENTRAL_DISTANCE_M {
            return Vector3::zeros();
        }
        self.gm_m3_per_s2 * (d / powi(d_norm, 3) - body / powi(body_norm, 3))
    }

    /// The tidal gradient `μ/|d|³·(3·d·dᵀ/|d|² − I)` in 1/s² at `r`.
    #[must_use]
    pub fn tidal_gradient(&self, r: &Vector3<f64>) -> Matrix3<f64> {
        let d = self.position() - r;
        let d_norm = d.norm();
        if d_norm < MIN_CENTRAL_DISTANCE_M {
            return Matrix3::zeros();
        }
        let unit = d / d_norm;
        self.gm_m3_per_s2 / powi(d_norm, 3) * (3.0 * unit * unit.transpose() - Matrix3::identity())
    }
}

/// The refusal of a third body when a [`ThirdBodySet`] holds
/// [`MAX_THIRD_BODIES`] bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a third-body set holds at most {capacity} bodies")]
pub struct ThirdBodyCapacityError {
    /// The capacity of the set.
    pub capacity: usize,
}

/// Up to [`MAX_THIRD_BODIES`] third bodies, stored inline.
///
/// The set is `Copy` and does not allocate. Two sets are equal when
/// they hold the same bodies in the same sequence.
#[derive(Clone, Copy)]
pub struct ThirdBodySet {
    bodies: [PointMassBody; MAX_THIRD_BODIES],
    len: usize,
}

impl ThirdBodySet {
    /// A set with no body.
    pub const EMPTY: Self = Self {
        bodies: [PointMassBody::VACANT; MAX_THIRD_BODIES],
        len: 0,
    };

    /// A set with the bodies of `bodies` in the same sequence.
    ///
    /// # Errors
    ///
    /// [`ThirdBodyCapacityError`] when `bodies` holds more than
    /// [`MAX_THIRD_BODIES`] bodies.
    pub fn from_slice(bodies: &[PointMassBody]) -> Result<Self, ThirdBodyCapacityError> {
        let mut set = Self::EMPTY;
        for body in bodies {
            set.push(*body)?;
        }
        Ok(set)
    }

    /// Adds `body` at the end of the set.
    ///
    /// # Errors
    ///
    /// [`ThirdBodyCapacityError`] when the set is full. The set does not
    /// change.
    pub fn push(&mut self, body: PointMassBody) -> Result<(), ThirdBodyCapacityError> {
        let slot = self
            .bodies
            .get_mut(self.len)
            .ok_or(ThirdBodyCapacityError {
                capacity: MAX_THIRD_BODIES,
            })?;
        *slot = body;
        self.len = self.len.wrapping_add(1);
        Ok(())
    }

    /// The bodies in the set.
    #[must_use]
    pub fn as_slice(&self) -> &[PointMassBody] {
        self.bodies.get(..self.len).unwrap_or(&[])
    }

    /// The number of bodies in the set.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the set holds no body.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether each body in the set is plausible.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.as_slice().iter().all(PointMassBody::is_plausible)
    }

    /// `central` plus the tidal acceleration and the tidal gradient of
    /// each body at `r`.
    #[must_use]
    pub fn add_tidal(
        &self,
        central: (Vector3<f64>, Matrix3<f64>),
        r: &Vector3<f64>,
    ) -> (Vector3<f64>, Matrix3<f64>) {
        self.as_slice().iter().fold(central, |(g, gradient), body| {
            (
                g + body.tidal_acceleration(r),
                gradient + body.tidal_gradient(r),
            )
        })
    }
}

impl Default for ThirdBodySet {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl PartialEq for ThirdBodySet {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl fmt::Debug for ThirdBodySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

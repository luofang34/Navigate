//! Straight-line space motion from the catalogue epoch.

use crate::{Float, Star};

/// The straight-line motion of one star, prepared for repeated evaluation.
///
/// The position is the unit direction at the reference epoch. The velocity is
/// in the same unit (the distance at the reference epoch) per Julian year.
/// Evaluation needs no trigonometry, so a renderer can prepare the motions
/// once and evaluate them for each frame. [`Motion::cast`] gives the motion
/// in `f32` for a single-precision FPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motion<T: Float = f64> {
    /// Unit direction at the reference epoch, in the axes of the catalogue.
    pub position: [T; 3],
    /// Velocity in reference distances per Julian year.
    pub velocity: [T; 3],
    /// The reference epoch, in Julian years (TT) from J2000.
    pub reference_epoch_years: T,
}

impl<T: Float> Motion<T> {
    /// The direction `years` Julian years (TT) after J2000, not normalized.
    ///
    /// This takes only multiplications and additions. Normalize the result,
    /// or use [`Motion::direction_at`], where the length matters.
    #[inline]
    pub fn position_at(&self, years: T) -> [T; 3] {
        let elapsed = years - self.reference_epoch_years;
        core::array::from_fn(|i| self.position[i] + self.velocity[i] * elapsed)
    }

    /// The unit direction `years` Julian years (TT) after J2000.
    #[inline]
    pub fn direction_at(&self, years: T) -> [T; 3] {
        let [x, y, z] = self.position_at(years);
        let norm = (x * x + y * y + z * z).sqrt();
        [x / norm, y / norm, z / norm]
    }

    /// The same motion in another float precision.
    #[inline]
    pub fn cast<U: Float>(&self) -> Motion<U> {
        Motion {
            position: self.position.map(|c| U::from_f64(c.to_f64())),
            velocity: self.velocity.map(|c| U::from_f64(c.to_f64())),
            reference_epoch_years: U::from_f64(self.reference_epoch_years.to_f64()),
        }
    }
}

impl Star {
    /// The motion of the star in space, ready for [`Motion::direction_at`].
    ///
    /// The star moves on a straight line in space: its proper motion, and its
    /// radial velocity where the catalogue gives both a trigonometric
    /// parallax and a radial velocity. The light-time factor
    /// 1 / (1 - v_r / c) turns the observed rates into space motion (Stumpff
    /// 1985; SOFA `eraStarpv`). The table stores the result, so this does no
    /// arithmetic.
    #[inline]
    pub fn motion(&self) -> Motion {
        Motion {
            position: self.position,
            velocity: self.velocity,
            reference_epoch_years: self.reference_epoch_years,
        }
    }

    /// The unit direction `years` Julian years (TT) after J2000, from the solar
    /// system barycentre, in the axes of the catalogue.
    ///
    /// From -2000 to 3000 this matches Skyfield's star propagation to 0.01
    /// arcsecond. The catalogue proper motions limit the accuracy far from
    /// J2000. The direction follows the system motion of a binary;
    /// [`Star::component_direction_at`](crate::Star::component_direction_at)
    /// adds the orbit where the crate has one.
    #[inline]
    pub fn direction_at(&self, years: f64) -> [f64; 3] {
        self.motion().direction_at(years)
    }
}

#[cfg(test)]
mod tests;

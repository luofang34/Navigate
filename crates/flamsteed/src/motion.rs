//! Straight-line space motion from the catalogue epoch.

use crate::Star;

/// Parsecs per Julian year at 1 km/s.
const PARSEC_PER_YEAR_PER_KM_S: f64 = 1.022_712_165_053_707_7e-6;
const LIGHT_SPEED_KM_S: f64 = 299_792.458;

/// The straight-line motion of one star, prepared for repeated evaluation.
///
/// The position is the unit direction at the reference epoch. The velocity is
/// in the same unit (the distance at the reference epoch) per Julian year.
/// [`Motion::direction_at`] needs no trigonometry, so a renderer can prepare
/// the motions once and evaluate them for each frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motion {
    /// Unit direction at the reference epoch, in J2000 equatorial axes.
    pub position: [f64; 3],
    /// Velocity in reference distances per Julian year.
    pub velocity: [f64; 3],
    /// The reference epoch, in Julian years (TT) from J2000.
    pub reference_epoch_years: f64,
}

impl Motion {
    /// The unit direction `years` Julian years (TT) after J2000.
    #[inline]
    pub fn direction_at(&self, years: f64) -> [f64; 3] {
        let elapsed = years - self.reference_epoch_years;
        let moved: [f64; 3] =
            std::array::from_fn(|i| self.position[i] + self.velocity[i] * elapsed);
        let norm = moved.iter().map(|c| c * c).sum::<f64>().sqrt();
        moved.map(|c| c / norm)
    }
}

impl Star {
    /// The motion of the star in space, ready for [`Motion::direction_at`].
    ///
    /// The star moves on a straight line in space: its proper motion, and its
    /// radial velocity where the catalogue gives both a trigonometric
    /// parallax and a radial velocity. The light-time factor
    /// 1 / (1 - v_r / c) turns the observed rates into space motion (Stumpff
    /// 1985; SOFA `eraStarpv`). Otherwise the radial term is unknown and
    /// omitted.
    #[inline]
    pub fn motion(&self) -> Motion {
        let (sin_ra, cos_ra) = self.right_ascension_rad.sin_cos();
        let (sin_dec, cos_dec) = self.declination_rad.sin_cos();
        let position = [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec];
        let east = [-sin_ra, cos_ra, 0.0];
        let north = [-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec];
        // An absent value contributes nothing.
        let parallax = self.parallax_mas.unwrap_or(0.0);
        let velocity = if parallax > 0.0 {
            self.radial_velocity_km_s.unwrap_or(0.0)
        } else {
            0.0
        };
        let light_time = 1.0 / (1.0 - velocity / LIGHT_SPEED_KM_S);
        // Radial velocity over distance, per year.
        let radial_rate = light_time * velocity * parallax / 1000.0 * PARSEC_PER_YEAR_PER_KM_S;
        let tangential: [f64; 3] = std::array::from_fn(|i| {
            light_time
                * (east[i] * self.proper_motion_ra_rad_per_year
                    + north[i] * self.proper_motion_dec_rad_per_year)
        });
        Motion {
            position,
            velocity: std::array::from_fn(|i| position[i] * radial_rate + tangential[i]),
            reference_epoch_years: self.reference_epoch_years,
        }
    }

    /// The unit direction `years` Julian years (TT) after J2000, from the solar
    /// system barycentre, in J2000 equatorial axes. This is
    /// [`Star::motion`] and then [`Motion::direction_at`].
    ///
    /// From -2000 to 3000 this matches Skyfield's star propagation to 0.01
    /// arcsecond. The catalogue proper motions limit the accuracy far from
    /// J2000. Compared with the Hipparcos catalogue propagated the same way,
    /// stars to magnitude 6.5 differ by a median of 1.4 arcseconds in 1700, 5
    /// in 1000, 13 in -330, and 23 in -2000; 99 percent are within 9, 34, 81,
    /// and 141 arcseconds. Hipparcos is not truth for long spans either,
    /// because its proper motions come from three years of observations.
    /// Close binaries, such as Alpha Centauri, can differ by several
    /// arcminutes in the past, because their orbits are not modelled.
    #[inline]
    pub fn direction_at(&self, years: f64) -> [f64; 3] {
        self.motion().direction_at(years)
    }
}

#[cfg(test)]
mod tests;

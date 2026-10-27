//! Straight-line space motion from the catalogue epoch.

use crate::Star;

/// Parsecs per Julian year at 1 km/s.
const PARSEC_PER_YEAR_PER_KM_S: f64 = 1.022_712_165_053_707_7e-6;
const LIGHT_SPEED_KM_S: f64 = 299_792.458;

impl Star {
    /// The unit direction `years` Julian years (TT) after J2000, from the solar
    /// system barycentre, in J2000 equatorial axes.
    ///
    /// The star moves on a straight line in space: its proper motion, and its
    /// radial velocity where the catalogue gives a trigonometric parallax.
    /// The light-time factor 1 / (1 - v_r / c) turns the observed rates into
    /// space motion (Stumpff 1985; SOFA `eraStarpv`). Without a parallax the
    /// radial term is unknown and omitted. From -2000 to 3000 this matches
    /// Skyfield's star propagation to 0.01 arcsecond.
    ///
    /// The catalogue proper motions limit the accuracy far from J2000.
    /// Compared with the Hipparcos catalogue propagated the same way, stars
    /// to magnitude 6.5 differ by a median of 1.4 arcseconds in 1700, 5 in
    /// 1000, 13 in -330, and 23 in -2000; 99 percent are within 9, 34, 81, and
    /// 141 arcseconds. Hipparcos is not truth for long spans either, because
    /// its proper motions come from three years of observations. Close
    /// binaries, such as Alpha Centauri, can differ by several arcminutes in
    /// the past, because their orbits are not modelled.
    pub fn direction_at(&self, years: f64) -> [f64; 3] {
        let (sin_ra, cos_ra) = self.right_ascension_rad.sin_cos();
        let (sin_dec, cos_dec) = self.declination_rad.sin_cos();
        let position = [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec];
        let east = [-sin_ra, cos_ra, 0.0];
        let north = [-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec];
        let (light_time, radial_rate) = if self.parallax_mas > 0.0 {
            (
                1.0 / (1.0 - self.radial_velocity_km_s / LIGHT_SPEED_KM_S),
                // Radial velocity over distance, per year.
                self.radial_velocity_km_s * self.parallax_mas / 1000.0 * PARSEC_PER_YEAR_PER_KM_S,
            )
        } else {
            (1.0, 0.0)
        };
        let radial = 1.0 + light_time * radial_rate * years;
        let tangential = light_time * years;
        let moved: [f64; 3] = std::array::from_fn(|i| {
            position[i] * radial
                + (east[i] * self.proper_motion_ra_rad_per_year
                    + north[i] * self.proper_motion_dec_rad_per_year)
                    * tangential
        });
        let norm = moved.iter().map(|c| c * c).sum::<f64>().sqrt();
        moved.map(|c| c / norm)
    }
}

#[cfg(test)]
mod tests;

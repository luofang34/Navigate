//! The truth of a synthetic terrain anchor: the registration of the
//! imagery cell, and the tilt and the mean of a random elevation error
//! under the footprint, which the real matcher follows
//! (`super::super::tilt`).

use nalgebra::Vector3;

use crate::reference::tests::truth::{Draws, mix};

/// Count of cosine modes of a field.
const MODES: usize = 64;
/// Samples on each side of the footprint for the plane fit.
const SIDE: i32 = 6;

/// A stationary Gaussian random field of elevation error with the
/// covariance `σ²·exp(−r²/L²)`: a sum of cosines with random wave vectors
/// drawn from its spectrum and random phases.
pub(super) struct Field {
    sigma_m: f64,
    modes: Vec<([f64; 2], f64)>,
}

impl Field {
    /// The field of seed `seed`, one-sigma `sigma_m`, length `length_m`.
    pub(super) fn new(seed: u64, sigma_m: f64, length_m: f64) -> Self {
        let mut draws = Draws {
            seed: mix(seed ^ 0xF1E1_D000),
            index: 0,
        };
        // exp(−r²/L²) is a Gaussian of one-sigma L/√2, whose spectrum is a
        // Gaussian of one-sigma √2/L on each axis.
        let k = core::f64::consts::SQRT_2 / length_m;
        let modes = (0..MODES)
            .map(|_| {
                let wave = [draws.normal() * k, draws.normal() * k];
                let phase = (draws.normal() * 1e3).rem_euclid(core::f64::consts::TAU);
                (wave, phase)
            })
            .collect();
        Self { sigma_m, modes }
    }

    /// The error at `(north, east)` metres.
    pub(super) fn at(&self, [north, east]: [f64; 2]) -> f64 {
        let sum: f64 = self
            .modes
            .iter()
            .map(|([kn, ke], phase)| (kn * north + ke * east + phase).cos())
            .sum();
        self.sigma_m * (2.0 / MODES as f64).sqrt() * sum
    }

    /// The error that a nadir fix `agl_m` above the ground at `centre`,
    /// flying along `heading`, takes from the field: the height times the
    /// fitted tilt over the footprint, against the slope, and the mean
    /// over the footprint on the vertical, north, east and down. The
    /// footprint is 0.8 of the height along the track and 1.07 across, as
    /// the test camera has it.
    pub(super) fn fix_error(&self, centre: [f64; 2], heading: f64, agl_m: f64) -> Vector3<f64> {
        let (along_half, across_half) = (0.4 * agl_m, 0.533 * agl_m);
        let (sin, cos) = heading.sin_cos();
        let (mut su, mut sv, mut suu, mut svv, mut sz, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for i in -SIDE..=SIDE {
            for j in -SIDE..=SIDE {
                let u = along_half * f64::from(i) / f64::from(SIDE);
                let v = across_half * f64::from(j) / f64::from(SIDE);
                let point = [centre[0] + u * cos - v * sin, centre[1] + u * sin + v * cos];
                let z = self.at(point);
                su += u * z;
                sv += v * z;
                suu += u * u;
                svv += v * v;
                sz += z;
                n += 1.0;
            }
        }
        let (along, across) = (su / suu, sv / svv);
        let slope = [along * cos - across * sin, along * sin + across * cos];
        // A higher package moves the fix up: the down error is minus the
        // mean.
        Vector3::new(-agl_m * slope[0], -agl_m * slope[1], -sz / n)
    }
}

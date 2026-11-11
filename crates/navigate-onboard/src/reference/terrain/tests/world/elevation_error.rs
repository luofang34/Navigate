//! Errors of the elevation model of a test package against the ground of
//! its frames.

use super::{CENTRE, ELEVATION_ZOOM, noise};

/// An error of the elevation model of the package: the frames show the
/// true ground, and the package holds the true height plus this error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DemError {
    /// No error.
    None,
    /// A constant offset, metres.
    Offset(f64),
    /// A plane through [`CENTRE`] that rises by `north` and `east` metres
    /// for each metre to the north and to the east.
    Slope { north: f64, east: f64 },
    /// A smooth random field: features of `length_m`, scaled so that its
    /// one-sigma is near `sigma_m`.
    Field {
        sigma_m: f64,
        length_m: f64,
        seed: u64,
    },
}

/// Ground size of one elevation pixel at [`CENTRE`], metres.
fn elevation_pixel_m() -> f64 {
    156_543.033_928 * CENTRE[0].to_radians().cos() / f64::from(1_u32 << ELEVATION_ZOOM)
}

/// The global elevation pixel of [`CENTRE`].
fn centre_pixel() -> (f64, f64) {
    let n = 256.0 * f64::from(1_u32 << ELEVATION_ZOOM);
    let lat = CENTRE[0].to_radians();
    let x = (CENTRE[1] + 180.0) / 360.0 * n;
    let y = (1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / core::f64::consts::PI) / 2.0 * n;
    (x, y)
}

/// One-sigma of `noise - 0.5` for smooth value noise, measured.
const NOISE_SIGMA: f64 = 0.17;

impl DemError {
    /// The error at the global elevation pixel `(gx, gy)`, metres.
    pub(crate) fn at(self, gx: f64, gy: f64) -> f64 {
        let px = elevation_pixel_m();
        let (x0, y0) = centre_pixel();
        let (east, north) = ((gx - x0) * px, (y0 - gy) * px);
        match self {
            Self::None => 0.0,
            Self::Offset(m) => m,
            Self::Slope {
                north: sn,
                east: se,
            } => sn * north + se * east,
            Self::Field {
                sigma_m,
                length_m,
                seed,
            } => sigma_m * (noise(gx, gy, length_m / px, seed) - 0.5) / NOISE_SIGMA,
        }
    }
}

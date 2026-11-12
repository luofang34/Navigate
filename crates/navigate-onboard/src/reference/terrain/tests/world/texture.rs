//! The ground texture of the imagery of a test package.

use super::{lattice, noise};

/// What the ground of a test package looks like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Texture {
    /// Fields of features 50 m to 300 m with hard edges.
    Farmland,
    /// Blocks of about 20 m with their own grey and streets between them.
    Town,
    /// Smooth ground with a contrast of about ten counts.
    Low,
    /// Blocks of 10 m to 40 m in an irregular grid, streets between them.
    TownIrregular,
    /// Smooth ground with a contrast of about twenty-five counts.
    LowModerate,
    /// The fields of [`Self::Farmland`] in another season: about half of
    /// them change from dark to light or back, and the contrast differs.
    FarmlandSeason,
}

impl Texture {
    /// The grey of the global imagery pixel `(gx, gy)`.
    pub(crate) fn luma(self, gx: f64, gy: f64) -> u8 {
        match self {
            Self::Farmland => {
                let v = 0.5 * noise(gx, gy, 14.0, 1)
                    + 0.3 * noise(gx, gy, 37.0, 2)
                    + 0.2 * noise(gx, gy, 90.0, 3);
                // Harder edges, as fields have: stretch the middle of the
                // range.
                ((v - 0.5) * 3.2 + 0.5).clamp(0.0, 1.0).mul_add(235.0, 10.0) as u8
            }
            Self::Town => {
                let (bx, by) = ((gx / 6.0).floor(), (gy / 6.0).floor());
                let street = gx.rem_euclid(30.0) < 2.0 || gy.rem_euclid(30.0) < 2.0;
                if street {
                    40
                } else {
                    (lattice(bx as i64, by as i64, 21) * 180.0 + 60.0) as u8
                }
            }
            Self::FarmlandSeason => {
                let base = f64::from(Self::Farmland.luma(gx, gy));
                // The fields of one crop: cells of a coarse noise.
                let field = (noise(gx, gy, 37.0, 2) * 6.0).floor() as i64;
                let changed = lattice(field, 0, 41) < 0.5;
                let v = if changed { 255.0 - base } else { base };
                (128.0 + (v - 128.0) * 0.8).clamp(0.0, 255.0) as u8
            }
            Self::TownIrregular => {
                // Block edges where a smooth noise crosses levels: an
                // irregular grid.
                let a = noise(gx, gy, 9.0, 31);
                let b = noise(gx, gy, 5.0, 32);
                let block = (a * 7.0).floor() * 13.0 + (b * 5.0).floor();
                let edge = (a * 7.0).fract() < 0.08 || (b * 5.0).fract() < 0.1;
                if edge {
                    40
                } else {
                    (lattice(block as i64, 0, 33) * 180.0 + 60.0) as u8
                }
            }
            Self::LowModerate => {
                let v = 0.6 * noise(gx, gy, 20.0, 4) + 0.4 * noise(gx, gy, 60.0, 5);
                (128.0 + (v - 0.5) * 150.0).clamp(0.0, 255.0) as u8
            }
            Self::Low => {
                let v = 0.6 * noise(gx, gy, 40.0, 4) + 0.4 * noise(gx, gy, 120.0, 5);
                (128.0 + (v - 0.5) * 60.0).clamp(0.0, 255.0) as u8
            }
        }
    }
}

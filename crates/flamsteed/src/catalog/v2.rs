//! Format version 2: 72-byte records in ICRS axes with the position and the
//! velocity in `f64`, for astrometry more precise than the fixed point of
//! version 1 can hold.
//!
//! A record is, little-endian: the HR number (u16); the provenance (u16: 1
//! for the Bright Star Catalogue, another value for the catalogue that the
//! file's writer names); that catalogue's identifier of the star (u64, 0 for
//! provenance 1); the unit direction at the reference epoch (3 f64); the
//! space velocity in reference distances per Julian year, with the
//! light-time factor and, where the record gives a parallax and a radial
//! velocity, the radial term (3 f64); the parallax in milliarcseconds and the
//! radial velocity in km/s (f32, NaN when absent); and the V magnitude and
//! B-V colour in hundredths (i16, `i16::MIN` for an absent colour).

use super::{Astrometry, Star, StarId};
use crate::{Float, Motion};

pub(crate) const RECORD_BYTES: usize = 72;
/// A unit direction stays within this distance of length 1.
const UNIT_TOLERANCE: f64 = 1e-9;
/// Provenance 1 is the Bright Star Catalogue.
const PROVENANCE_BSC5: u16 = 1;

pub(super) type Record = [u8; RECORD_BYTES];

fn bytes<const N: usize>(record: &Record, offset: usize) -> [u8; N] {
    record
        .get(offset..offset + N)
        .and_then(|slice| slice.try_into().ok())
        .unwrap_or([0; N])
}

fn vector(record: &Record, offset: usize) -> [f64; 3] {
    [0, 1, 2].map(|i| f64::from_le_bytes(bytes(record, offset + 8 * i)))
}

fn optional(value: f32) -> Option<f64> {
    (!value.is_nan()).then(|| f64::from(value))
}

#[inline]
pub(super) fn magnitude(record: &Record) -> f64 {
    f64::from(i16::from_le_bytes(bytes(record, 68))) / 100.0
}

/// Whether the position is a unit vector and the velocity is finite.
pub(super) fn is_valid(record: &Record) -> bool {
    let [x, y, z] = vector(record, 12);
    let finite = vector(record, 36).iter().all(|c| c.is_finite());
    finite && ((x * x + y * y + z * z) - 1.0).abs() <= UNIT_TOLERANCE
}

#[inline]
pub(super) fn star(record: &Record, reference_epoch_years: f64) -> Star {
    let provenance = u16::from_le_bytes(bytes(record, 2));
    let colour = i16::from_le_bytes(bytes(record, 70));
    Star {
        id: StarId::Hr(u16::from_le_bytes(bytes(record, 0))),
        astrometry: if provenance == PROVENANCE_BSC5 {
            Astrometry::BrightStarCatalogue
        } else {
            Astrometry::Other {
                provenance,
                source_id: u64::from_le_bytes(bytes(record, 4)),
            }
        },
        reference_epoch_years,
        position: vector(record, 12),
        velocity: vector(record, 36),
        parallax_mas: optional(f32::from_le_bytes(bytes(record, 60))),
        radial_velocity_km_s: optional(f32::from_le_bytes(bytes(record, 64))),
        v_magnitude: magnitude(record),
        b_minus_v: (colour != i16::MIN).then(|| f64::from(colour) / 100.0),
    }
}

/// The motion in `T`, rounded from the stored `f64`.
#[inline]
pub(super) fn motion<T: Float>(record: &Record, reference_epoch_years: T) -> Motion<T> {
    Motion {
        position: vector(record, 12).map(T::from_f64),
        velocity: vector(record, 36).map(T::from_f64),
        reference_epoch_years,
    }
}

#[cfg(test)]
mod tests;

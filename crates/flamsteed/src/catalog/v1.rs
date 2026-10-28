//! Format version 1: the embedded layout, 34-byte fixed-point records in
//! FK5 J2000 axes. The `bsc5_table` example documents the layout.

use super::{Astrometry, Star, StarId};
use crate::{Float, Motion};

pub(crate) const RECORD_BYTES: usize = 34;
/// The fixed-point scale of a unit-vector component.
const POSITION_SCALE: f64 = 1.0 / i32::MAX as f64;
/// The fixed-point scale of a velocity component, per Julian year.
const VELOCITY_SCALE: f64 = 1e-12;

pub(super) type Record = [u8; RECORD_BYTES];

#[inline]
pub(super) fn magnitude(record: &Record) -> f64 {
    let &[.., v0, v1, _, _, _, _, _, _] = record;
    f64::from(i16::from_le_bytes([v0, v1])) / 100.0
}

#[inline]
fn vectors(record: &Record) -> ([i32; 3], [i32; 3]) {
    let &[
        _,
        _,
        p0,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        q0,
        q1,
        q2,
        q3,
        q4,
        q5,
        q6,
        q7,
        q8,
        q9,
        q10,
        q11,
        ..,
    ] = record;
    (
        [
            i32::from_le_bytes([p0, p1, p2, p3]),
            i32::from_le_bytes([p4, p5, p6, p7]),
            i32::from_le_bytes([p8, p9, p10, p11]),
        ],
        [
            i32::from_le_bytes([q0, q1, q2, q3]),
            i32::from_le_bytes([q4, q5, q6, q7]),
            i32::from_le_bytes([q8, q9, q10, q11]),
        ],
    )
}

#[inline]
pub(super) fn star(record: &Record, reference_epoch_years: f64) -> Star {
    let &[h0, h1, .., v0, v1, c0, c1, x0, x1, w0, w1] = record;
    let (position, velocity) = vectors(record);
    let colour = i16::from_le_bytes([c0, c1]);
    let parallax = u16::from_le_bytes([x0, x1]);
    let radial_velocity = i16::from_le_bytes([w0, w1]);
    Star {
        id: StarId::Hr(u16::from_le_bytes([h0, h1])),
        astrometry: Astrometry::BrightStarCatalogue,
        reference_epoch_years,
        position: position.map(|c| f64::from(c) * POSITION_SCALE),
        velocity: velocity.map(|c| f64::from(c) * VELOCITY_SCALE),
        parallax_mas: (parallax != u16::MAX).then(|| f64::from(parallax)),
        radial_velocity_km_s: (radial_velocity != i16::MIN).then(|| f64::from(radial_velocity)),
        v_magnitude: f64::from(i16::from_le_bytes([v0, v1])) / 100.0,
        b_minus_v: (colour != i16::MIN).then(|| f64::from(colour) / 100.0),
    }
}

/// The motion in `T`. The decode converts integers to `T` and does no other
/// arithmetic, so an `f32` motion needs no double-precision operation.
#[inline]
pub(super) fn motion<T: Float>(record: &Record, reference_epoch_years: T) -> Motion<T> {
    let (position, velocity) = vectors(record);
    let (position_scale, velocity_scale) =
        (T::from_f64(POSITION_SCALE), T::from_f64(VELOCITY_SCALE));
    Motion {
        position: position.map(|c| T::from_i32(c) * position_scale),
        velocity: velocity.map(|c| T::from_i32(c) * velocity_scale),
        reference_epoch_years,
    }
}

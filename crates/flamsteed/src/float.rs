//! The two float precisions of [`Motion`](crate::Motion), and the float
//! functions from the standard library or from `libm`.
//!
//! `core` has no square root or arctangent on stable Rust. With the `std`
//! feature these come from the standard library, which uses the hardware
//! instruction where the target has one. Without it they come from `libm`,
//! which uses a software square root on Cortex-M. The `nightly` feature takes
//! the square root from `core` (`core_float_math`), which uses the hardware
//! instruction, for example `vsqrt.f32` on the Cortex-M33.
//! Only [`Motion::direction_at`](crate::Motion::direction_at) needs a square
//! root, and only the right ascension and declination accessors need an
//! arctangent; [`Motion::position_at`](crate::Motion::position_at) needs
//! neither.

use core::ops::{Add, Div, Mul, Sub};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A float type for [`Motion`](crate::Motion): `f32` or `f64`.
///
/// `f64` keeps the full precision of the catalogue. `f32` holds a direction
/// to about 0.03 arcsecond and runs natively on single-precision FPUs, for
/// example the Arm Cortex-M33 of the RP2350.
pub trait Float:
    sealed::Sealed
    + 'static
    + Copy
    + PartialEq
    + PartialOrd
    + core::fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    /// The value nearest to `value`.
    fn from_f64(value: f64) -> Self;
    /// The value nearest to the integer `value`.
    fn from_i32(value: i32) -> Self;
    /// The value as an `f64`.
    fn to_f64(self) -> f64;
    /// The square root.
    fn sqrt(self) -> Self;
}

impl Float for f64 {
    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        f64::from(value)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        {
            f64::sqrt(self)
        }
        #[cfg(all(not(feature = "std"), feature = "nightly"))]
        {
            core::f64::math::sqrt(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "nightly")))]
        {
            libm::sqrt(self)
        }
    }
}

impl Float for f32 {
    #[inline]
    fn from_f64(value: f64) -> Self {
        value as f32
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        {
            f32::sqrt(self)
        }
        #[cfg(all(not(feature = "std"), feature = "nightly"))]
        {
            core::f32::math::sqrt(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "nightly")))]
        {
            libm::sqrtf(self)
        }
    }
}

/// The angle of the point (x, y) from the x axis, in radians from -pi to pi.
#[inline]
pub(crate) fn atan2(y: f64, x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        y.atan2(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::atan2(y, x)
    }
}

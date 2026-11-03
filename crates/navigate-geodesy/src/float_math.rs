//! The `f64` functions that `core` does not supply on stable Rust.
//!
//! With the `std` feature each function calls the standard library,
//! which uses the hardware instruction where the target has one. Without
//! it each function calls `libm`. The crates of this workspace that are
//! `#![no_std]` call these functions, so one feature list selects the
//! implementation for all of them.

#[cfg(not(feature = "std"))]
use libm as imp;

/// Generates one wrapper per function: the `f64` method under `std`, and
/// the `libm` function of the same name otherwise.
macro_rules! float_functions {
    ($($(#[$doc:meta])* $name:ident($($arg:ident),+);)+) => {
        $(
            $(#[$doc])*
            #[must_use]
            #[inline]
            pub fn $name($($arg: f64),+) -> f64 {
                #[cfg(feature = "std")]
                {
                    float_functions!(@std $name $($arg),+)
                }
                #[cfg(not(feature = "std"))]
                {
                    imp::$name($($arg),+)
                }
            }
        )+
    };
    (@std $name:ident $x:ident) => { $x.$name() };
    (@std $name:ident $x:ident, $y:ident) => { $x.$name($y) };
}

float_functions! {
    /// The sine of `x` radians.
    sin(x);
    /// The cosine of `x` radians.
    cos(x);
    /// The tangent of `x` radians.
    tan(x);
    /// The arcsine of `x`, in `[-π/2, π/2]` radians.
    asin(x);
    /// The arccosine of `x`, in `[0, π]` radians.
    acos(x);
    /// The arctangent of `x`, in `[-π/2, π/2]` radians.
    atan(x);
    /// The four-quadrant arctangent of `y / x`, in `[-π, π]` radians.
    atan2(y, x);
    /// The non-negative square root of `x`.
    sqrt(x);
    /// `sqrt(x² + y²)` without intermediate overflow.
    hypot(x, y);
    /// The exponential `e^x`.
    exp(x);
    /// The largest integer value that is not greater than `x`.
    floor(x);
    /// The smallest integer value that is not less than `x`.
    ceil(x);
    /// The integer value nearest to `x`. A half value rounds away from
    /// zero.
    round(x);
}

/// The natural logarithm of `x`.
#[must_use]
#[inline]
pub fn ln(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.ln()
    }
    #[cfg(not(feature = "std"))]
    {
        imp::log(x)
    }
}

/// `x` to the integer power `n`.
#[must_use]
#[inline]
pub fn powi(x: f64, n: i32) -> f64 {
    #[cfg(feature = "std")]
    {
        x.powi(n)
    }
    #[cfg(not(feature = "std"))]
    {
        imp::pow(x, f64::from(n))
    }
}

/// The sine and the cosine of `x` radians.
#[must_use]
#[inline]
pub fn sin_cos(x: f64) -> (f64, f64) {
    (sin(x), cos(x))
}

/// The least non-negative remainder of `x / y`, as `f64::rem_euclid`
/// gives it.
#[must_use]
#[inline]
pub fn rem_euclid(x: f64, y: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.rem_euclid(y)
    }
    #[cfg(not(feature = "std"))]
    {
        let remainder = imp::fmod(x, y);
        if remainder < 0.0 {
            remainder + y.abs()
        } else {
            remainder
        }
    }
}

#[cfg(test)]
mod tests;

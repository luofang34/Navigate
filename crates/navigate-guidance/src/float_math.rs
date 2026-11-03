//! Floating-point operations over `std` or `libm`.

#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub(crate) fn sin(x: f64) -> f64 {
    libm::sin(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn sin(x: f64) -> f64 {
    x.sin()
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub(crate) fn cos(x: f64) -> f64 {
    libm::cos(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn cos(x: f64) -> f64 {
    x.cos()
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub(crate) fn hypot(x: f64, y: f64) -> f64 {
    libm::hypot(x, y)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

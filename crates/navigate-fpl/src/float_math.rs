//! Floating-point operations over `std` or `libm`.

#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub(crate) fn tan(x: f64) -> f64 {
    libm::tan(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn tan(x: f64) -> f64 {
    x.tan()
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub(crate) fn rem_euclid(x: f64, y: f64) -> f64 {
    let r = libm::fmod(x, y);
    if r < 0.0 { r + libm::fabs(y) } else { r }
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn rem_euclid(x: f64, y: f64) -> f64 {
    x.rem_euclid(y)
}

//! One imaging sensor model for visible, long-wave infrared, and star
//! cameras (ADR-0021).
//!
//! - [`Intrinsics`] is a pinhole camera. Pixel centres have integer
//!   coordinates and rows increase down the image.
//! - [`Distortion`] maps normalized image coordinates between the ideal
//!   pinhole and the real lens.
//! - [`OpticalPose`] holds the optical centre and the rotation from the
//!   world axes into the camera axes: `+X` right, `+Y` down, `+Z` forward.
//! - [`CameraMount`] places a camera on a body with forward, right, and
//!   down (FRD) axes.
//! - [`ImagerModel`] adds the [`SpectralBand`] and the [`Shutter`].
//! - [`ImageView`] borrows row-strided [`Pixels`].
//!
//! The crate is `#![no_std]` and does not allocate. The `std` feature (the
//! default) or the `libm` feature supplies the float functions.

#![no_std]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("Enable feature `std` or `libm`.");

#[cfg(any(feature = "std", test))]
extern crate std;

mod distortion;
mod error;
mod image;
mod imager;
mod intrinsics;
mod mount;
mod pose;

pub use distortion::Distortion;
pub use error::CameraError;
pub use image::{ImageView, Pixels};
pub use imager::{ImageFrame, ImagerModel, Shutter, SpectralBand};
pub use intrinsics::Intrinsics;
pub use mount::CameraMount;
pub use pose::{OPTICAL_FROM_GL, OpticalPose};

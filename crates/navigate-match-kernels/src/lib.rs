//! CubeCL kernels of the Navigate matchers.
//!
//! `navigate-match` launches these kernels on burn's CubeCL tensors. They need no
//! matrix accelerator, so they compile to WGSL for browsers and to MSL, SPIR-V and
//! CUDA for native GPUs.
mod kernels;

pub use kernels::{TILE, UNITS, convolution, matmul};

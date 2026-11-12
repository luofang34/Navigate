//! What a real camera does to a frame that the map does not show: the
//! exposure and the response of the sensor, a change of the sun on the
//! ground, the blur of the optics, the noise of the sensor, and the loss
//! of compression.

use std::io::Cursor;

use image::codecs::jpeg::JpegEncoder;
use image::{GrayImage, ImageFormat};

use super::super::world::lattice;

/// How a camera frame differs from a render of the map imagery.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Mismatch {
    /// Gain of the exposure.
    pub gain: f64,
    /// Exponent of the response: the frame is `(v/255)^gamma`.
    pub gamma: f64,
    /// Contrast about the mean.
    pub contrast: f64,
    /// Amplitude of a smooth change of the light over the frame, as a
    /// change of sun angle on gentle ground gives it.
    pub shading: f64,
    /// One-sigma of a Gaussian blur, pixels.
    pub blur_px: f64,
    /// One-sigma of the sensor noise, counts.
    pub noise: f64,
    /// JPEG quality, or none.
    pub jpeg: Option<u8>,
}

impl Mismatch {
    /// A frame as the map renders it.
    pub(super) const EXACT: Self = Self {
        gain: 1.0,
        gamma: 1.0,
        contrast: 1.0,
        shading: 0.0,
        blur_px: 0.0,
        noise: 0.0,
        jpeg: None,
    };
    /// Another exposure, response, and light.
    pub(super) const PHOTOMETRIC: Self = Self {
        gain: 0.8,
        gamma: 1.3,
        contrast: 0.7,
        shading: 0.2,
        ..Self::EXACT
    };
    /// The optics and the sensor: a blur of a pixel, noise, JPEG at 75.
    pub(super) const OPTICS: Self = Self {
        blur_px: 1.0,
        noise: 3.0,
        jpeg: Some(75),
        ..Self::EXACT
    };
    /// Both.
    pub(super) const ALL: Self = Self {
        blur_px: 1.0,
        noise: 3.0,
        jpeg: Some(75),
        ..Self::PHOTOMETRIC
    };

    /// `luma`, `width` by `height`, as this camera gives it; `seed` draws
    /// the noise.
    pub(super) fn apply(&self, luma: &[u8], width: u32, height: u32, seed: u64) -> Vec<u8> {
        let (w, h) = (width as usize, height as usize);
        let mean = luma.iter().map(|v| f64::from(*v)).sum::<f64>() / luma.len() as f64;
        let mut v: Vec<f64> = luma
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let (px, py) = ((i % w) as f64 / w as f64, (i / w) as f64 / h as f64);
                let light =
                    1.0 + self.shading * (core::f64::consts::TAU * (0.7 * px + 0.4 * py)).cos();
                let c = mean + (f64::from(*x) - mean) * self.contrast;
                255.0
                    * (self.gain * light * c / 255.0)
                        .clamp(0.0, 1.0)
                        .powf(self.gamma)
            })
            .collect();
        if self.blur_px > 0.0 {
            v = blurred(&v, w, h, self.blur_px);
        }
        let bytes: Vec<u8> = v
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let noise = self.noise * normal(i as i64, seed);
                (x + noise).round().clamp(0.0, 255.0) as u8
            })
            .collect();
        match self.jpeg {
            Some(quality) => compressed(bytes, width, height, quality),
            None => bytes,
        }
    }
}

/// A standard normal draw for the pixel `i` of the frame `seed`.
fn normal(i: i64, seed: u64) -> f64 {
    let u = lattice(i, 1, seed).max(1e-12);
    let v = lattice(i, 2, seed);
    (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
}

/// A separable Gaussian blur of one-sigma `sigma` pixels.
fn blurred(v: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil() as i64;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|k| (-(k * k) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f64 = kernel.iter().sum();
    let pass = |src: &[f64], horizontal: bool| -> Vec<f64> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                kernel
                    .iter()
                    .enumerate()
                    .map(|(j, k)| {
                        let d = j as i64 - radius;
                        let (sx, sy) = if horizontal {
                            ((x + d).clamp(0, w as i64 - 1), y)
                        } else {
                            (x, (y + d).clamp(0, h as i64 - 1))
                        };
                        k * src[(sy as usize) * w + sx as usize]
                    })
                    .sum::<f64>()
                    / total
            })
            .collect()
    };
    pass(&pass(v, true), false)
}

/// `bytes` through a JPEG of `quality` and back.
fn compressed(bytes: Vec<u8>, width: u32, height: u32, quality: u8) -> Vec<u8> {
    let image = GrayImage::from_raw(width, height, bytes).unwrap();
    let mut out = Cursor::new(Vec::new());
    image
        .write_with_encoder(JpegEncoder::new_with_quality(&mut out, quality))
        .unwrap();
    image::load_from_memory_with_format(out.get_ref(), ImageFormat::Jpeg)
        .unwrap()
        .to_luma8()
        .into_raw()
}

//! The imaging sensor and one frame from it.

use navigate_contract::ObservationStamp;

use crate::{CameraError, Distortion, ImageView, Intrinsics};

/// The spectral band of an imaging sensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SpectralBand {
    /// Visible light, about 0.4 to 0.7 micrometres.
    #[default]
    Visible,
    /// Near infrared, about 0.7 to 1.0 micrometres.
    NearInfrared,
    /// Long-wave infrared (thermal), about 8 to 14 micrometres.
    LongWaveInfrared,
}

/// The exposure timing of the pixel rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Shutter {
    /// All rows have the same exposure time.
    #[default]
    Global,
    /// Each row starts `line_period_ns` after the row above it. The frame
    /// stamp is the exposure of row zero.
    Rolling {
        /// Time between the exposures of adjacent rows, in nanoseconds.
        line_period_ns: u64,
    },
}

/// The full description of an imaging sensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagerModel {
    /// Pinhole intrinsics of the undistorted image.
    pub intrinsics: Intrinsics,
    /// Lens distortion of the raw image.
    pub distortion: Distortion,
    /// Spectral band.
    pub band: SpectralBand,
    /// Row exposure timing.
    pub shutter: Shutter,
}

impl ImagerModel {
    /// A visible camera with a global shutter and no distortion.
    #[must_use]
    pub fn pinhole(intrinsics: Intrinsics) -> Self {
        Self {
            intrinsics,
            distortion: Distortion::None,
            band: SpectralBand::Visible,
            shutter: Shutter::Global,
        }
    }

    /// The same model in another band.
    #[must_use]
    pub fn with_band(mut self, band: SpectralBand) -> Self {
        self.band = band;
        self
    }

    /// Checks the intrinsics and the distortion.
    ///
    /// # Errors
    /// Returns the first invalid part.
    pub fn validate(&self) -> Result<(), CameraError> {
        self.intrinsics.validate()?;
        self.distortion.validate()
    }
}

/// One frame of an imaging sensor.
#[derive(Debug, Clone, Copy)]
pub struct ImageFrame<'a> {
    /// Source identity, sequence, and acquisition time.
    pub stamp: ObservationStamp,
    /// The sensor that made the frame.
    pub imager: ImagerModel,
    /// The pixels.
    pub image: ImageView<'a>,
}

impl ImageFrame<'_> {
    /// Checks that the sensor model is valid and that the image size is the
    /// size of the intrinsics.
    ///
    /// # Errors
    /// Returns the first invalid part.
    pub fn validate(&self) -> Result<(), CameraError> {
        self.imager.validate()?;
        self.image.validate()?;
        let k = &self.imager.intrinsics;
        if self.image.width() != k.width || self.image.height() != k.height {
            return Err(CameraError::Invalid {
                field: "image size differs from intrinsics",
            });
        }
        Ok(())
    }
}

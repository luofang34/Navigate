//! Cache misses retain source validation whether features come from disk or inference.
use super::{Cache, ENTRY_RESERVE, Engine};
use crate::{
    Error,
    config::Config,
    surface::{acquire::store::reader::Loaded, matching::rotate},
};
use image::GrayImage;
use navigate_visual_onnx::XFeatFeatures;
use navigate_visual_workflow::FeatureBackend;
use serde::Serialize;
use std::time::Instant;
#[derive(Default, Serialize)]
pub(in crate::surface::acquire) struct ImportTimes {
    pub loaded: bool,
    pub file_bytes: usize,
    pub read_ms: f64,
    pub decode_ms: f64,
    pub pixel_validation_ms: f64,
    pub rotation_ms: f64,
}
pub(super) struct Reference {
    pub features: XFeatFeatures,
    pub bytes: usize,
    pub extraction_ms: f64,
    pub import: ImportTimes,
}
impl Engine {
    pub(super) fn reference_blocking(
        &mut self,
        image: &GrayImage,
        rotation: u8,
        key: Option<[u8; 32]>,
        raster_bytes: usize,
        config: &Config,
    ) -> Result<Reference, Error> {
        let overlap = raster_bytes.saturating_add(image.as_raw().len());
        self.reserve_reference(overlap, config)?;
        let started = Instant::now();
        let rotated = rotate(image, rotation);
        self.counts.rotations = self.counts.rotations.wrapping_add(1);
        let rotation_ms = started.elapsed().as_secs_f64() * 1000.;
        let (features, buffer_bytes, path, mut import) =
            self.fetch_blocking(&rotated, key, config)?;
        import.rotation_ms = rotation_ms;
        let validate = Instant::now();
        let info = self
            .backend
            .validate_features(&features, &rotated)
            .map_err(|source| match path {
                Some(path) => Error::CatalogImage { path, source },
                None => Error::Visual(source),
            })?;
        if import.loaded {
            import.pixel_validation_ms = validate.elapsed().as_secs_f64() * 1000.;
        }
        let bytes = if self.cache.limit > 0 {
            Cache::<XFeatFeatures>::entry_bytes(info.resident_bytes)
        } else {
            info.resident_bytes
        };
        let transient = bytes.saturating_add(buffer_bytes);
        let peak = overlap
            .saturating_add(self.cache.bytes)
            .saturating_add(transient);
        if peak > config.limits.reference_bytes
            || info.count > config.limits.features_per_image
            || ((self.cache.limit > 0 || self.catalog.is_some()) && transient > ENTRY_RESERVE)
        {
            return Err(Error::Input("acquisition reference feature budget".into()));
        }
        self.peak_reference_bytes = self.peak_reference_bytes.max(peak);
        let extraction_ms = if import.loaded {
            0.
        } else {
            started.elapsed().as_secs_f64() * 1000.
        };
        Ok(Reference {
            features,
            bytes,
            extraction_ms,
            import,
        })
    }
    fn fetch_blocking(
        &mut self,
        rotated: &GrayImage,
        key: Option<[u8; 32]>,
        config: &Config,
    ) -> Result<
        (
            XFeatFeatures,
            usize,
            Option<std::path::PathBuf>,
            ImportTimes,
        ),
        Error,
    > {
        let loaded = if let Some(reader) = &self.catalog {
            reader.load_blocking(
                key.ok_or_else(|| Error::Input("catalog requires an input fingerprint".into()))?,
                config.keypoints,
            )?
        } else {
            None
        };
        let mut import = ImportTimes {
            rotation_ms: 0.,
            ..ImportTimes::default()
        };
        let (features, buffer_bytes, path) = if let Some(Loaded {
            features,
            buffer_bytes,
            file_bytes,
            read_ms,
            decode_ms,
            path,
        }) = loaded
        {
            self.counts.packet_loads = self.counts.packet_loads.wrapping_add(1);
            self.counts.packet_bytes = self.counts.packet_bytes.wrapping_add(file_bytes);
            import.loaded = true;
            import.file_bytes = file_bytes;
            import.read_ms = read_ms;
            import.decode_ms = decode_ms;
            (features, buffer_bytes, Some(path))
        } else {
            if self.catalog.is_some() {
                self.counts.packet_misses = self.counts.packet_misses.wrapping_add(1);
            }
            self.reference_encodings = self.reference_encodings.wrapping_add(1);
            (self.backend.extract_features_blocking(rotated)?, 0, None)
        };
        Ok((features, buffer_bytes, path, import))
    }
    fn reserve_reference(&mut self, overlap: usize, config: &Config) -> Result<(), Error> {
        if overlap > config.limits.reference_bytes {
            return Err(Error::Input("acquisition rotated raster budget".into()));
        }
        if self.cache.limit > 0 || self.catalog.is_some() {
            if overlap.saturating_add(ENTRY_RESERVE) > config.limits.reference_bytes {
                return Err(Error::Input(
                    "acquisition feature reservation budget".into(),
                ));
            }
            self.cache
                .trim(config.limits.reference_bytes - overlap - ENTRY_RESERVE);
        }
        Ok(())
    }
}

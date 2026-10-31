//! In-process matcher client wrapping `burn` neural matchers and pyramidal patch matching.

use crate::{PipelineError, util::file_sha256};
use burn::tensor::Device;
use image::{DynamicImage, GrayImage, RgbImage};
use navigate_match::{EngineKind, NativeMlEngine, Weights};
use navigate_visual::{ImageMatcher, PixelMatch, PyramidalMatcher};
use std::path::Path;

/// Image correspondence matcher used by offline pipeline stages.
pub enum PipelineMatcher {
    /// In-process pure-Rust `burn` matcher (`LoFTR-DS` or `XFeat + LighterGlue`).
    Native(Box<NativeMlEngine>),
    /// Classical pyramidal optical-flow and patch matcher (`navigate_visual::PyramidalMatcher`).
    Pyramidal(PyramidalMatcher),
}

impl PipelineMatcher {
    /// Open an in-process LoFTR-DS `.safetensors` model on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when reading or loading weights fails.
    pub fn load_loftr(
        safetensors_path: &Path,
        device: &Device,
        engine: EngineKind,
    ) -> Result<Self, PipelineError> {
        let sha = file_sha256(safetensors_path)?;
        let bytes = std::fs::read(safetensors_path).map_err(|source| PipelineError::Io {
            path: safetensors_path.to_path_buf(),
            source,
        })?;
        let weights = Weights::from_safetensors(&bytes)?;
        let ml = NativeMlEngine::load_loftr(weights, &sha, device, engine)?;
        Ok(Self::Native(Box::new(ml)))
    }

    /// Open an in-process XFeat (and optional LighterGlue) `.safetensors` matcher on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when reading or loading weights fails.
    pub fn load_xfeat(
        xfeat_path: &Path,
        lighterglue_path: Option<&Path>,
        limit: usize,
        device: &Device,
        engine: EngineKind,
    ) -> Result<Self, PipelineError> {
        let det_sha = file_sha256(xfeat_path)?;
        let det_bytes = std::fs::read(xfeat_path).map_err(|source| PipelineError::Io {
            path: xfeat_path.to_path_buf(),
            source,
        })?;
        let det_w = Weights::from_safetensors(&det_bytes)?;
        let lg_pair = if let Some(lg_p) = lighterglue_path {
            let lg_sha = file_sha256(lg_p)?;
            let lg_bytes = std::fs::read(lg_p).map_err(|source| PipelineError::Io {
                path: lg_p.to_path_buf(),
                source,
            })?;
            Some((Weights::from_safetensors(&lg_bytes)?, lg_sha))
        } else {
            None
        };
        let ml = NativeMlEngine::load_xfeat(
            det_w,
            &det_sha,
            lg_pair.as_ref().map(|(w, s)| (w.clone(), s.as_str())),
            limit,
            device,
            engine,
        )?;
        Ok(Self::Native(Box::new(ml)))
    }

    /// Construct a deterministic classical pyramidal matcher.
    #[must_use]
    pub fn pyramidal() -> Self {
        Self::Pyramidal(PyramidalMatcher)
    }

    /// Open a matcher from CLI options (`backend` + `model_path` + optional `lighterglue_path`).
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when configuration or model loading fails.
    pub fn from_options(
        backend: &str,
        model_path: Option<&Path>,
        lighterglue_path: Option<&Path>,
    ) -> Result<Self, PipelineError> {
        if backend == "pyramidal" {
            return Ok(Self::pyramidal());
        }
        let path = model_path.ok_or_else(|| {
            PipelineError::Invalid(format!("backend '{backend}' requires --model"))
        })?;
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if ext == "onnx" {
            return Err(PipelineError::Invalid(
                "offline stages require .safetensors weights; use bank-match or corridor-track --xfeat-runtime for ONNX Runtime".into(),
            ));
        }
        let (device, engine) = default_burn_device();
        match backend {
            "loftr" => Self::load_loftr(path, &device, engine),
            "xfeat" => Self::load_xfeat(path, lighterglue_path, 2048, &device, engine),
            other => Err(PipelineError::Invalid(format!(
                "unsupported matcher backend '{other}'; use loftr, xfeat, or pyramidal"
            ))),
        }
    }

    /// Return the stable matcher backend identity string.
    #[must_use]
    pub fn identity(&self) -> &str {
        match self {
            Self::Native(m) => m.identity(),
            Self::Pyramidal(m) => m.identity(),
        }
    }

    /// Match RGB `reference` and `query` images and return pixel-centre pairs.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when matching fails.
    pub fn match_rgb(
        &mut self,
        reference: &RgbImage,
        query: &RgbImage,
    ) -> Result<Vec<PixelMatch>, PipelineError> {
        let ref_gray = DynamicImage::ImageRgb8(reference.clone()).into_luma8();
        let qry_gray = DynamicImage::ImageRgb8(query.clone()).into_luma8();
        self.match_gray(&ref_gray, &qry_gray)
    }

    /// Match grayscale `reference` and `query` images and return pixel-centre pairs.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when matching fails.
    pub fn match_gray(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, PipelineError> {
        match self {
            Self::Native(m) => Ok(m.match_images_blocking(reference, query)?),
            Self::Pyramidal(m) => Ok(m.match_images_blocking(reference, query)?),
        }
    }
}

pub(crate) fn default_burn_device() -> (Device, EngineKind) {
    #[cfg(any(feature = "wgpu", feature = "metal", feature = "cuda"))]
    {
        (Device::default(), EngineKind::BurnGpu)
    }
    #[cfg(not(any(feature = "wgpu", feature = "metal", feature = "cuda")))]
    {
        (Device::flex(), EngineKind::BurnCpu)
    }
}

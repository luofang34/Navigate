//! Native and embedded static-dispatch inference engine (`not(wasm32)`).
//!
//! [`NativeMlEngine`] binds a burn [`Device`] (`wgpu`, `metal`, `vulkan`, `cuda`,
//! or `flex` CPU) with an explicit [`EngineKind`] and exposes both [`ModelInference`]
//! tensor sessions and [`navigate_visual::ImageMatcher`] image matchers.
use crate::{
    LighterGlue, Loftr, LoftrMatcher, MatchError, Weights, XFeat, XFeatLighterGlueMatcher,
    traits::{AsyncModelInference, EngineKind, HostTensor, ModelInference, NamedTensor},
};
use burn::tensor::Device;
use image::GrayImage;
use navigate_visual::{AsyncImageMatcher, ImageMatcher, PixelMatch, VisualError};

/// Dense XFeat session implementing [`ModelInference`] and [`AsyncModelInference`].
#[derive(Clone, Debug)]
pub struct XFeatDenseSession {
    model: XFeat,
    engine: EngineKind,
}

impl XFeatDenseSession {
    /// Wrap a loaded [`XFeat`] model with its active [`EngineKind`].
    pub fn new(model: XFeat, engine: EngineKind) -> Self {
        Self { model, engine }
    }

    async fn run_async(
        &self,
        inputs: &[NamedTensor<HostTensor>],
    ) -> Result<Vec<NamedTensor<HostTensor>>, MatchError> {
        let [input] = inputs else {
            return Err(MatchError::Output("XFeat session expects one input tensor"));
        };
        if input.name != "image" && input.name != "images" {
            return Err(MatchError::Output(
                "XFeat input tensor must be named image or images",
            ));
        }
        if input.tensor.shape() != [1, 1, 576, 800] {
            return Err(MatchError::Output(
                "XFeat dense session requires shape [1, 1, 576, 800]",
            ));
        }
        let maps = self
            .model
            .forward_dense(input.tensor.as_slice().to_vec())
            .await?;
        let host = |shape: Vec<usize>, values: Vec<f32>| {
            HostTensor::new(shape, values)
                .map_err(|_| MatchError::Output("invalid XFeat host tensor"))
        };
        Ok(vec![
            NamedTensor::new("descriptors", host(vec![1, 64, 72, 100], maps.descriptors)?),
            NamedTensor::new(
                "keypoint_logits",
                host(vec![1, 65, 72, 100], maps.keypoint_logits)?,
            ),
            NamedTensor::new("reliability", host(vec![1, 1, 72, 100], maps.reliability)?),
        ])
    }
}

impl ModelInference for XFeatDenseSession {
    type Tensor = HostTensor;
    type Error = MatchError;

    fn engine_kind(&self) -> EngineKind {
        self.engine
    }

    fn forward(
        &mut self,
        inputs: &[NamedTensor<HostTensor>],
    ) -> Result<Vec<NamedTensor<HostTensor>>, MatchError> {
        pollster::block_on(self.run_async(inputs))
    }
}

impl AsyncModelInference for XFeatDenseSession {
    type Tensor = HostTensor;
    type Error = MatchError;

    fn engine_kind(&self) -> EngineKind {
        self.engine
    }

    async fn forward_async(
        &mut self,
        inputs: &[NamedTensor<HostTensor>],
    ) -> Result<Vec<NamedTensor<HostTensor>>, MatchError> {
        self.run_async(inputs).await
    }
}

/// Native matcher engine wrapping either LoFTR-DS or XFeat + LighterGlue on burn.
#[derive(Clone)]
pub enum NativeMlEngine {
    /// Detector-free LoFTR-DS matcher.
    Loftr {
        /// Wrapped LoFTR matcher.
        matcher: Box<LoftrMatcher>,
        /// Active runtime track.
        engine: EngineKind,
    },
    /// Staged XFeat detector with optional LighterGlue matcher.
    XFeat(Box<XFeatLighterGlueMatcher>),
}

impl NativeMlEngine {
    /// Load LoFTR-DS onto `device` with explicit `engine` track.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when weight tensors do not match [`Loftr`].
    pub fn load_loftr(
        weights: Weights,
        weights_sha256: &str,
        device: &Device,
        engine: EngineKind,
    ) -> Result<Self, MatchError> {
        let model = Loftr::load(weights, device)?;
        let identity = format!(
            "loftr-dual-softmax-640x480/{}/{weights_sha256}",
            engine.as_str()
        );
        Ok(Self::Loftr {
            matcher: Box::new(LoftrMatcher::with_identity(model, identity)),
            engine,
        })
    }

    /// Load XFeat and optional LighterGlue onto `device` with explicit `engine` track.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when weight tensors or `limit` are invalid.
    pub fn load_xfeat(
        detector_weights: Weights,
        detector_sha256: &str,
        matcher_weights: Option<(Weights, &str)>,
        limit: usize,
        device: &Device,
        engine: EngineKind,
    ) -> Result<Self, MatchError> {
        let detector = XFeat::load(detector_weights, device)?;
        let lg = match matcher_weights {
            Some((w, sha)) => Some((LighterGlue::load(w, device)?, sha)),
            None => None,
        };
        Ok(Self::XFeat(Box::new(XFeatLighterGlueMatcher::new(
            detector,
            detector_sha256,
            lg,
            limit,
            engine,
        )?)))
    }

    /// Return the active [`EngineKind`].
    pub fn engine_kind(&self) -> EngineKind {
        match self {
            Self::Loftr { engine, .. } => *engine,
            Self::XFeat(staged) => staged.engine_kind(),
        }
    }

    /// Return the stable matcher identity string.
    pub fn identity(&self) -> &str {
        match self {
            Self::Loftr { matcher, .. } => matcher.identity(),
            Self::XFeat(staged) => staged.identity(),
        }
    }
}

impl ImageMatcher for NativeMlEngine {
    fn identity(&self) -> &str {
        NativeMlEngine::identity(self)
    }

    fn match_images_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        match self {
            Self::Loftr { matcher, .. } => matcher.match_images_blocking(reference, query),
            Self::XFeat(staged) => staged.match_images_blocking(reference, query),
        }
    }
}

impl AsyncImageMatcher for NativeMlEngine {
    fn identity(&self) -> &str {
        NativeMlEngine::identity(self)
    }

    async fn match_images(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        match self {
            Self::Loftr { matcher, .. } => {
                AsyncImageMatcher::match_images(&mut **matcher, reference, query).await
            }
            Self::XFeat(staged) => {
                AsyncImageMatcher::match_images(&mut **staged, reference, query).await
            }
        }
    }
}

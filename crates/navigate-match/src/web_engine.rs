//! Browser dual-track inference engine (`wasm32`).
//!
//! [`WebMlEngine`] probes browser WebNN (`navigator.ml`) first and degrades to
//! Burn WebGPU (`burn/webgpu`) or Burn CPU (`burn/flex`) when WebNN is unavailable
//! or rejects the model operators. The active [`EngineKind`] and WebNN status are
//! recorded without silent spoofing.
use crate::{
    LighterGlue, Loftr, LoftrMatcher, MatchError, Weights, XFeat, XFeatLighterGlueMatcher,
    traits::EngineKind,
};
use burn::tensor::Device;
use image::GrayImage;
use js_sys::{Function, Object, Promise, Reflect};
use navigate_visual::{AsyncImageMatcher, PixelMatch, VisualError};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

/// Result of probing `navigator.ml` in the browser global scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WebNnAvailability {
    /// `navigator.ml.createContext` succeeded, but direct safetensors matcher dispatch
    /// uses Burn WebGPU until a WebNN graph builder is registered.
    ContextCreated,
    /// `navigator.ml` is not exposed by this browser (for example Safari or Firefox).
    MissingNavigatorMl,
    /// `navigator.ml.createContext` rejected or threw an error.
    ContextRejected(String),
}

impl WebNnAvailability {
    /// Return a short status string for diagnostics and telemetry.
    pub fn as_str(&self) -> &str {
        match self {
            Self::ContextCreated => "webnn-context-ready-burn-fallback-for-custom-ops",
            Self::MissingNavigatorMl => "navigator-ml-unavailable",
            Self::ContextRejected(reason) => reason.as_str(),
        }
    }
}

/// Probe `navigator.ml` in either a `Window` or `WorkerGlobalScope`.
pub async fn probe_webnn() -> WebNnAvailability {
    let global = js_sys::global();
    let Ok(navigator) = Reflect::get(&global, &JsValue::from_str("navigator")) else {
        return WebNnAvailability::MissingNavigatorMl;
    };
    if navigator.is_undefined() || navigator.is_null() {
        return WebNnAvailability::MissingNavigatorMl;
    }
    let Ok(ml) = Reflect::get(&navigator, &JsValue::from_str("ml")) else {
        return WebNnAvailability::MissingNavigatorMl;
    };
    if ml.is_undefined() || ml.is_null() {
        return WebNnAvailability::MissingNavigatorMl;
    }
    let Ok(create_fn) = Reflect::get(&ml, &JsValue::from_str("createContext")) else {
        return WebNnAvailability::MissingNavigatorMl;
    };
    let Ok(create_fn) = create_fn.dyn_into::<Function>() else {
        return WebNnAvailability::MissingNavigatorMl;
    };
    let options = Object::new();
    if let Err(err) = Reflect::set(
        &options,
        &JsValue::from_str("deviceType"),
        &JsValue::from_str("gpu"),
    ) {
        return WebNnAvailability::ContextRejected(format!("deviceType option rejected: {err:?}"));
    }
    let call_result = create_fn.call1(&ml, &options);
    let Ok(promise_val) = call_result else {
        return WebNnAvailability::ContextRejected("createContext threw synchronously".into());
    };
    let Ok(promise) = promise_val.dyn_into::<Promise>() else {
        return WebNnAvailability::ContextRejected("createContext did not return a Promise".into());
    };
    match JsFuture::from(promise).await {
        Ok(_) => WebNnAvailability::ContextCreated,
        Err(err) => WebNnAvailability::ContextRejected(format!("createContext rejected: {err:?}")),
    }
}

#[derive(Clone)]
enum RuntimeMatcher {
    Loftr { model: Box<Loftr>, identity: String },
    XFeat(Box<XFeatLighterGlueMatcher>),
}

/// Dual-track browser matcher engine with automatic WebNN probe and Burn WebGPU fallback.
#[derive(Clone)]
pub struct WebMlEngine {
    matcher: RuntimeMatcher,
    engine: EngineKind,
    webnn: WebNnAvailability,
}

impl WebMlEngine {
    /// Open a browser engine for LoFTR-DS weights.
    ///
    /// Probes WebNN first, then initializes Burn WebGPU (or Burn CPU if `flex` is
    /// compiled and WebGPU is absent).
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when no browser compute backend can be initialized or
    /// when `weights` is invalid.
    pub async fn open_loftr(weights: Weights, weights_sha256: &str) -> Result<Self, MatchError> {
        let webnn = probe_webnn().await;
        let (device, engine) = open_burn_device().await?;
        let model = Loftr::load(weights, &device)?;
        let identity = format!(
            "loftr-dual-softmax-640x480/{}/{weights_sha256}",
            engine.as_str()
        );
        Ok(Self {
            matcher: RuntimeMatcher::Loftr {
                model: Box::new(model),
                identity,
            },
            engine,
            webnn,
        })
    }

    /// Open a browser engine for XFeat and optional LighterGlue weights.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when no browser compute backend can be initialized or
    /// when weight shapes do not match.
    pub async fn open_xfeat(
        detector_weights: Weights,
        detector_sha256: &str,
        matcher_weights: Option<(Weights, &str)>,
        limit: usize,
    ) -> Result<Self, MatchError> {
        let webnn = probe_webnn().await;
        let (device, engine) = open_burn_device().await?;
        let detector = XFeat::load(detector_weights, &device)?;
        let lg = match matcher_weights {
            Some((w, sha)) => Some((LighterGlue::load(w, &device)?, sha)),
            None => None,
        };
        let staged = XFeatLighterGlueMatcher::new(detector, detector_sha256, lg, limit, engine)?;
        Ok(Self {
            matcher: RuntimeMatcher::XFeat(Box::new(staged)),
            engine,
            webnn,
        })
    }

    /// Return the active [`EngineKind`].
    pub fn engine_kind(&self) -> EngineKind {
        self.engine
    }

    /// Return the probed [`WebNnAvailability`].
    pub fn webnn_availability(&self) -> &WebNnAvailability {
        &self.webnn
    }

    /// Return the stable matcher identity string.
    pub fn identity(&self) -> &str {
        match &self.matcher {
            RuntimeMatcher::Loftr { identity, .. } => identity,
            RuntimeMatcher::XFeat(staged) => staged.identity(),
        }
    }

    /// Match `reference` and `query` on the active browser backend.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when preprocessing or device execution fails.
    pub async fn match_pair(
        &self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, MatchError> {
        match &self.matcher {
            RuntimeMatcher::Loftr { model, identity } => {
                let adapter = LoftrMatcher::with_identity((**model).clone(), identity.clone());
                adapter.match_images(reference, query).await
            }
            RuntimeMatcher::XFeat(staged) => staged.match_pair(reference, query).await,
        }
    }
}

impl AsyncImageMatcher for WebMlEngine {
    fn identity(&self) -> &str {
        WebMlEngine::identity(self)
    }

    async fn match_images(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        self.match_pair(reference, query)
            .await
            .map_err(|source| VisualError::Backend {
                backend: self.identity().to_owned(),
                source: Box::new(source),
            })
    }
}

/// Open the browser burn device: WebGPU first, then the Burn CPU device when the
/// `flex` feature is compiled in and the browser has no WebGPU adapter.
///
/// # Errors
///
/// Returns [`MatchError::Device`] when no compiled browser runtime can start.
pub async fn open_burn_device() -> Result<(Device, EngineKind), MatchError> {
    #[cfg(feature = "webgpu")]
    {
        match Device::wgpu_options().init_async().await {
            Ok(device) => return Ok((device, EngineKind::BurnWebGpu)),
            Err(error) => {
                #[cfg(not(feature = "flex"))]
                return Err(MatchError::Device {
                    stage: "webgpu init",
                    message: format!("{error:?}"),
                });
                #[cfg(feature = "flex")]
                tracing::warn!("WebGPU unavailable ({error:?}); falling back to Burn Flex CPU");
            }
        }
    }
    #[cfg(feature = "flex")]
    {
        Ok((Device::flex(), EngineKind::BurnCpu))
    }
    #[cfg(not(any(feature = "webgpu", feature = "flex")))]
    Err(MatchError::Device {
        stage: "browser device init",
        message: "neither webgpu nor flex feature is enabled".into(),
    })
}

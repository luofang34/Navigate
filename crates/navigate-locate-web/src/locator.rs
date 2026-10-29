//! Exported browser locator running on [`navigate_match::WebMlEngine`].
use crate::{photo_info, request::prepare_photo, response_of, tiles::FetchSource};
use navigate_locate::{
    LocateRequest as Request, LookAt, Progress, SceneRequest, locate,
    messages::{LocateProgress, LocateRequest, LocateResponse, LocateStage},
};
use navigate_match::{WebMlEngine, Weights};
use wasm_bindgen::prelude::*;

/// Return the size and camera hints of `photo` as JSON `PhotoInfo` without a GPU.
///
/// # Errors
///
/// Rejects with a reason string when `photo` cannot be decoded.
#[wasm_bindgen]
pub fn inspect(photo: Vec<u8>) -> Result<String, JsValue> {
    let info = photo_info(&photo).map_err(|e| JsValue::from_str(&e))?;
    serde_json::to_string(&info).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Dual-track browser locator probing WebNN and falling back to Burn WebGPU or CPU.
#[wasm_bindgen]
pub struct Locator {
    engine: WebMlEngine,
}

#[wasm_bindgen]
impl Locator {
    /// Open a browser engine with LoFTR-DS weights (`navigate-model --recipe loftr-burn-f16`).
    ///
    /// # Errors
    ///
    /// Rejects with a reason string when no browser compute runtime is available or
    /// when `weights` is invalid.
    pub async fn create(weights: Vec<u8>, weights_sha256: String) -> Result<Locator, JsValue> {
        let weights =
            Weights::from_safetensors(&weights).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let engine = WebMlEngine::open_loftr(weights, &weights_sha256)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Locator { engine })
    }

    /// Open a browser engine with XFeat and optional LighterGlue weights.
    ///
    /// # Errors
    ///
    /// Rejects with a reason string when no browser compute runtime is available or
    /// when weight files are invalid.
    pub async fn create_xfeat(
        detector_weights: Vec<u8>,
        detector_sha256: String,
        matcher_weights: Option<Vec<u8>>,
        matcher_sha256: Option<String>,
        limit: usize,
    ) -> Result<Locator, JsValue> {
        let detector = Weights::from_safetensors(&detector_weights)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let matcher = match (matcher_weights, matcher_sha256) {
            (Some(bytes), Some(sha)) => {
                let w = Weights::from_safetensors(&bytes)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                Some((w, sha))
            }
            (None, None) => None,
            _ => {
                return Err(JsValue::from_str(
                    "matcher_weights and matcher_sha256 must both be present or both be absent",
                ));
            }
        };
        let engine = WebMlEngine::open_xfeat(
            detector,
            &detector_sha256,
            matcher.as_ref().map(|(w, s)| (w.clone(), s.as_str())),
            limit,
        )
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Locator { engine })
    }

    /// Return the active runtime track identifier (`"burn-webgpu"` or `"burn-cpu"`).
    pub fn engine(&self) -> String {
        self.engine.engine_kind().as_str().to_owned()
    }

    /// Return the probed browser WebNN status string.
    pub fn webnn_status(&self) -> String {
        self.engine.webnn_availability().as_str().to_owned()
    }

    /// Return the active matcher identity string.
    pub fn identity(&self) -> String {
        self.engine.identity().to_owned()
    }

    /// Locate one photo and return a serialized JSON `LocateResponse`.
    pub async fn locate(
        &self,
        request: String,
        photo: Vec<u8>,
        progress: js_sys::Function,
    ) -> String {
        let response = match self.run(&request, &photo, &progress).await {
            Ok(response) => response,
            Err(reason) => LocateResponse::Failed { reason },
        };
        serde_json::to_string(&response)
            .unwrap_or_else(|e| format!(r#"{{"outcome":"failed","reason":"{e}"}}"#))
    }
}

impl Locator {
    async fn run(
        &self,
        request: &str,
        photo: &[u8],
        progress: &js_sys::Function,
    ) -> Result<LocateResponse, String> {
        let request: LocateRequest =
            serde_json::from_str(request).map_err(|e| format!("the request is not valid: {e}"))?;
        let photo = prepare_photo(&request, photo)?;
        let target = [
            request.target_latitude_rad.to_degrees(),
            request.target_longitude_rad.to_degrees(),
        ];
        let pipeline = Request {
            image: photo.image,
            camera: photo.camera,
            target,
            look_at: LookAt::standard(request.heading_rad.to_degrees()),
            scene: SceneRequest {
                anchor: target,
                radius_m: request.area_radius_m.clamp(100.0, 2000.0),
                imagery_zoom: 18,
                terrain_zoom: 14,
                imagery_tile_pixels: 256,
                max_bytes: 512 << 20,
                concurrency: 16,
            },
            position_radius_m: 400.0,
            attitude_radius_rad: 0.9,
        };
        let matcher = self.engine.clone();
        let report = |p: Progress| {
            let message = match p {
                Progress::Tiles { done, total } => (LocateStage::Tiles, done, total),
                Progress::Rendering { done, total } => (LocateStage::Rendering, done, total),
                Progress::Matching { total } => (LocateStage::Matching, 0, total),
                Progress::Refining => (LocateStage::Refining, 0, 0),
            };
            let progress_message = LocateProgress {
                stage: message.0,
                done: u32::try_from(message.1).unwrap_or(u32::MAX),
                total: u32::try_from(message.2).unwrap_or(u32::MAX),
            };
            if let Ok(json) = serde_json::to_string(&progress_message) {
                progress
                    .call1(&JsValue::NULL, &JsValue::from_str(&json))
                    .ok();
            }
        };
        let outcome = locate(&FetchSource::public(), matcher, pipeline, report)
            .await
            .map_err(|e| format!("{e}: {}", source_chain(&e)))?;
        Ok(response_of(outcome, &photo.camera))
    }
}

fn source_chain(error: &dyn std::error::Error) -> String {
    let mut out = Vec::new();
    let mut next = error.source();
    while let Some(e) = next {
        out.push(e.to_string());
        next = e.source();
    }
    out.join(": ")
}

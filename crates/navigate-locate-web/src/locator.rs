//! The exported locator: weights on the browser's WebGPU device and one call per photo.
use crate::{photo_info, request::prepare_photo, response_of, tiles::FetchSource};
use burn::tensor::Device;
use navigate_contract::locate::{LocateProgress, LocateRequest, LocateResponse, LocateStage};
use navigate_locate::{LocateRequest as Request, LookAt, Progress, SceneRequest, locate};
use navigate_match::{Loftr, LoftrMatcher, Weights};
use wasm_bindgen::prelude::*;

/// The size and camera hints of a photo as JSON `PhotoInfo`, without weights or a GPU.
///
/// # Errors
///
/// Rejects with a reason when the photo cannot be decoded.
#[wasm_bindgen]
pub fn inspect(photo: Vec<u8>) -> Result<String, JsValue> {
    let info = photo_info(&photo).map_err(|e| JsValue::from_str(&e))?;
    serde_json::to_string(&info).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// LoFTR-DS on the browser's WebGPU device.
#[wasm_bindgen]
pub struct Locator {
    model: Loftr,
    weights_sha256: String,
}

#[wasm_bindgen]
impl Locator {
    /// Open the WebGPU device and load a weight file of `navigate-model --recipe loftr-burn-f16`.
    ///
    /// # Errors
    ///
    /// Rejects with a reason when WebGPU is not available or the file is not valid.
    pub async fn create(weights: Vec<u8>, weights_sha256: String) -> Result<Locator, JsValue> {
        let device = Device::wgpu_options()
            .init_async()
            .await
            .map_err(|e| JsValue::from_str(&format!("WebGPU is not available: {e:?}")))?;
        let weights =
            Weights::from_safetensors(&weights).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let model = Loftr::load(weights, &device).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Locator {
            model,
            weights_sha256,
        })
    }

    /// Locate one photo. `request` is a JSON `LocateRequest`; `progress` receives
    /// JSON `LocateProgress` values. Resolves to a JSON `LocateResponse`.
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
        let matcher = LoftrMatcher::new(self.model.clone(), &self.weights_sha256);
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

//! The exported searcher: DINOv2 and LoFTR on the browser burn device, a place
//! index over `fetch`, and one call per photo.
use crate::search_request::{prepare_search, search_response_of};
use crate::tiles::{FetchSource, fetch_bytes};
use navigate_locate::messages::search::{
    SearchProgress, SearchRequest, SearchResponse, SearchStage,
};
use navigate_locate::{
    IndexSource, PlaceEncoder, Progress, SearchProgress as Stage, SearchSources, search,
};
use navigate_match::{
    DINOV2_WIDTH, Dinov2, EngineKind, Loftr, LoftrMatcher, Weights, open_burn_device,
};
use navigate_place::{Canvas, GRID, IndexManifest, Tokens};
use wasm_bindgen::prelude::*;

/// DINOv2 input side, pixels.
const ENCODE_PX: usize = 224;

/// DINOv2 behind the place encoder port.
struct Encoder<'a>(&'a Dinov2);

impl PlaceEncoder for Encoder<'_> {
    async fn tokens(&mut self, canvases: &[Canvas]) -> Result<Vec<Tokens>, String> {
        let input: Vec<f32> = canvases
            .iter()
            .flat_map(|c| c.model_input(ENCODE_PX))
            .collect();
        let values = self
            .0
            .tokens(&input, canvases.len())
            .await
            .map_err(|e| e.to_string())?;
        let each = (1 + GRID * GRID) * DINOV2_WIDTH;
        Ok(values
            .chunks(each)
            .map(|v| Tokens {
                width: DINOV2_WIDTH,
                values: v.to_vec(),
            })
            .collect())
    }
}

/// Index files relative to a base URL.
struct Remote(String);

impl IndexSource for Remote {
    async fn read(&self, path: &str) -> Result<Vec<u8>, String> {
        let url = format!("{}{path}", self.0);
        fetch_bytes(&url)
            .await?
            .ok_or_else(|| format!("{url}: not found"))
    }
}

/// DINOv2 and LoFTR-DS on the browser burn device (WebGPU, or the CPU when WebGPU is
/// not available).
#[wasm_bindgen]
pub struct Searcher {
    loftr: Loftr,
    dinov2: Dinov2,
    weights_sha256: String,
    engine: EngineKind,
}

#[wasm_bindgen]
impl Searcher {
    /// Open the browser burn device and load the LoFTR and DINOv2 weight files.
    ///
    /// The device is WebGPU when the browser has an adapter, and the Burn CPU device
    /// when it does not.
    ///
    /// # Errors
    ///
    /// Rejects with a reason when no browser compute runtime can start or a file is
    /// not valid.
    pub async fn create(
        loftr: Vec<u8>,
        dinov2: Vec<u8>,
        weights_sha256: String,
    ) -> Result<Searcher, JsValue> {
        let reason = |e: navigate_match::MatchError| JsValue::from_str(&e.to_string());
        let (device, engine) = open_burn_device().await.map_err(reason)?;
        let loftr = Loftr::load(Weights::from_safetensors(&loftr).map_err(reason)?, &device)
            .map_err(reason)?;
        let dinov2 = Dinov2::load(
            Weights::from_safetensors(&dinov2).map_err(reason)?,
            ENCODE_PX,
            &device,
        )
        .map_err(reason)?;
        Ok(Searcher {
            loftr,
            dinov2,
            weights_sha256,
            engine,
        })
    }

    /// Return the active compute engine name for telemetry.
    pub fn engine(&self) -> String {
        self.engine.as_str().to_owned()
    }

    /// Search for the camera of one photo. `request` is a JSON `SearchRequest`;
    /// `progress` receives JSON `SearchProgress` values. Resolves to a JSON
    /// `SearchResponse`.
    pub async fn search(
        &self,
        request: String,
        photo: Vec<u8>,
        progress: js_sys::Function,
    ) -> String {
        let response = match self.run(&request, &photo, &progress).await {
            Ok(response) => response,
            Err(reason) => SearchResponse::Failed { reason },
        };
        serde_json::to_string(&response)
            .unwrap_or_else(|e| format!(r#"{{"outcome":"failed","reason":"{e}"}}"#))
    }
}

impl Searcher {
    async fn run(
        &self,
        request: &str,
        photo: &[u8],
        progress: &js_sys::Function,
    ) -> Result<SearchResponse, String> {
        let request: SearchRequest =
            serde_json::from_str(request).map_err(|e| format!("the request is not valid: {e}"))?;
        let (image, plan) = prepare_search(&request, photo)?;
        let index = Remote(request.index_url.clone());
        let manifest = index.read("manifest.json").await?;
        let manifest = IndexManifest::from_json(&String::from_utf8_lossy(&manifest))
            .map_err(|e| e.to_string())?;
        let tiles = FetchSource::public();
        let sources = SearchSources {
            tiles: &tiles,
            index: &index,
            manifest: &manifest,
        };
        let matcher = LoftrMatcher::new(self.loftr.clone(), &self.weights_sha256);
        let report = |stage: Stage| send(progress, stage);
        let outcome = search(
            &sources,
            &mut Encoder(&self.dinov2),
            matcher,
            &image,
            &plan,
            report,
        )
        .await
        .map_err(|e| e.to_string())?;
        Ok(search_response_of(
            outcome,
            image.width(),
            image.height(),
            plan.focal_px,
        ))
    }
}

fn send(progress: &js_sys::Function, stage: Stage) {
    let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
    let (stage, done, total) = match stage {
        Stage::Describing { done, total } => (SearchStage::Describing, done, total),
        Stage::Index { done, total } => (SearchStage::Index, done, total),
        Stage::Checking { done, total } => (SearchStage::Checking, done, total),
        Stage::Locating(Progress::Tiles { done, total }) => (SearchStage::Tiles, done, total),
        Stage::Locating(Progress::Rendering { done, total }) => {
            (SearchStage::Rendering, done, total)
        }
        Stage::Locating(Progress::Matching { total }) => (SearchStage::Matching, 0, total),
        Stage::Locating(Progress::Refining) => (SearchStage::Refining, 0, 0),
    };
    let message = SearchProgress {
        stage,
        done: count(done),
        total: count(total),
    };
    if let Ok(json) = serde_json::to_string(&message) {
        progress
            .call1(&JsValue::NULL, &JsValue::from_str(&json))
            .ok();
    }
}

//! Map tiles with the Worker's `fetch`. The browser HTTP cache shares them with
//! the page, so a page that already shows the area does not download them again.
use navigate_locate::{TileKind, TileSource};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Response, WorkerGlobalScope};

/// Tile URL templates with `{z}`, `{x}` and `{y}`.
pub(crate) struct FetchSource {
    pub imagery: String,
    pub terrain: String,
}

impl FetchSource {
    /// Esri World Imagery and the AWS Terrarium elevation tiles, as the viewer uses.
    pub(crate) fn public() -> Self {
        Self {
            imagery: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}?blankTile=false".into(),
            terrain: "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png".into(),
        }
    }
}

impl TileSource for FetchSource {
    async fn fetch(
        &self,
        kind: TileKind,
        z: u32,
        x: u32,
        y: u32,
    ) -> Result<Option<Vec<u8>>, String> {
        let template = match kind {
            TileKind::Imagery => &self.imagery,
            TileKind::Terrain => &self.terrain,
        };
        let url = template
            .replace("{z}", &z.to_string())
            .replace("{x}", &x.to_string())
            .replace("{y}", &y.to_string());
        fetch_bytes(&url).await
    }

    fn identity(&self) -> String {
        format!("{} + {}", self.imagery, self.terrain)
    }
}

/// The body of `url` through the Worker's `fetch`, or `None` for HTTP 404.
pub(crate) async fn fetch_bytes(url: &str) -> Result<Option<Vec<u8>>, String> {
    let scope: WorkerGlobalScope = js_sys::global()
        .dyn_into()
        .map_err(|_| "the module must run in a Worker".to_owned())?;
    let response: Response = JsFuture::from(scope.fetch_with_str(url))
        .await
        .map_err(|e| format!("{url}: {e:?}"))?
        .dyn_into()
        .map_err(|_| format!("{url}: not a response"))?;
    if response.status() == 404 {
        return Ok(None);
    }
    if !response.ok() {
        return Err(format!("{url}: HTTP {}", response.status()));
    }
    let buffer = JsFuture::from(
        response
            .array_buffer()
            .map_err(|e| format!("{url}: {e:?}"))?,
    )
    .await
    .map_err(|e| format!("{url}: {e:?}"))?;
    Ok(Some(js_sys::Uint8Array::new(&buffer).to_vec()))
}

use crate::{error::PreviewError, model::Manifest, preview::Preview, storage::BrowserStore};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl Preview {
    /// Add a verified display package without changing the observation reference.
    pub async fn add_display_package(
        &mut self,
        manifest_json: String,
        read: js_sys::Function,
    ) -> Result<(), JsValue> {
        if !self.globe {
            return Err(PreviewError::Input {
                reason: "dynamic display data cannot modify an observation reference".into(),
            }
            .into());
        }
        let manifest: Manifest =
            serde_json::from_str(&manifest_json).map_err(PreviewError::from)?;
        manifest.validate()?;
        crate::preview::load(
            &mut self.map,
            &manifest,
            BrowserStore::new(&manifest, read),
            false,
        )
        .await?;
        Ok(())
    }
}

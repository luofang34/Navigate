use crate::{
    error::PreviewError,
    model::{Asset, Manifest},
};
use js_sys::{Function, Promise, Uint8Array};
use navigate_data::{
    DataError, DataStore, DataUri, OpenFuture, RandomAccess, ReadFuture, check_range,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;
pub(crate) struct BrowserStore {
    read: Function,
    sizes: BTreeMap<String, u64>,
}
struct BrowserFile {
    read: Function,
    uri: DataUri,
    size: u64,
}
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct BrowserError(String);
fn backend(uri: &DataUri, value: JsValue) -> DataError {
    DataError::Backend {
        uri: uri.as_str().into(),
        operation: "OPFS range read",
        source: Box::new(BrowserError(format!("{value:?}"))),
    }
}
impl BrowserStore {
    pub fn new(manifest: &Manifest, read: Function) -> Self {
        Self {
            read,
            sizes: manifest
                .files
                .iter()
                .map(|c| (format!("pilotage://chunks/{}.bin", c.sha256), c.size))
                .collect(),
        }
    }
    pub async fn image(&self, asset: &Asset, size: u32) -> Result<image::RgbaImage, PreviewError> {
        if !crate::model::hash(&asset.chunk)
            || !crate::model::hash(&asset.sha256)
            || asset.length > 4 * 1024 * 1024
        {
            return Err(PreviewError::Input {
                reason: "invalid tile asset".into(),
            });
        }
        let uri = DataUri::parse(format!("pilotage://chunks/{}.bin", asset.chunk))?;
        let file = self.open(&uri).await?;
        let bytes = file.read_at(asset.offset, asset.length).await?;
        if format!("{:x}", Sha256::digest(&bytes)) != asset.sha256 {
            return Err(PreviewError::Input {
                reason: format!("tile checksum failed: {}", uri.as_str()),
            });
        }
        let image = image::load_from_memory(&bytes)
            .map_err(|source| PreviewError::Image {
                uri: uri.as_str().into(),
                source,
            })?
            .to_rgba8();
        if image.dimensions() != (size, size)
            || (size == 256 && image.pixels().any(|p| p[3] != 255))
        {
            return Err(PreviewError::Input {
                reason: "tile dimensions or DEM alpha are invalid".into(),
            });
        }
        Ok(image)
    }
}
impl DataStore for BrowserStore {
    fn open<'a>(&'a self, uri: &'a DataUri) -> OpenFuture<'a> {
        Box::pin(async move {
            let size = *self
                .sizes
                .get(uri.as_str())
                .ok_or_else(|| backend(uri, JsValue::from_str("chunk is absent from manifest")))?;
            Ok(Box::new(BrowserFile {
                read: self.read.clone(),
                uri: uri.clone(),
                size,
            }) as Box<dyn RandomAccess>)
        })
    }
}
impl RandomAccess for BrowserFile {
    fn len(&self) -> u64 {
        self.size
    }
    fn read_at(&self, offset: u64, length: usize) -> ReadFuture<'_> {
        Box::pin(async move {
            check_range(&self.uri, self.size, offset, length)?;
            let promise = self
                .read
                .call3(
                    &JsValue::NULL,
                    &JsValue::from_str(self.uri.as_str()),
                    &JsValue::from_f64(offset as f64),
                    &JsValue::from_f64(length as f64),
                )
                .map_err(|e| backend(&self.uri, e))?;
            let value = JsFuture::from(Promise::resolve(&promise))
                .await
                .map_err(|e| backend(&self.uri, e))?;
            let bytes = Uint8Array::new(&value).to_vec();
            if bytes.len() != length {
                return Err(backend(&self.uri, JsValue::from_str("short read")));
            }
            Ok(bytes)
        })
    }
}

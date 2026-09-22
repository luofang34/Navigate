use super::{
    GdalRaster, RasterSource, Region,
    files::{Overview, chunk_blocking, png, publish_blocking, read_blocking, write_blocking},
    raster_error,
};
use crate::{
    CoveragePlan, ImageryError, Package, PackageBuilder, Tile, digest, error::invalid, tile_bounds,
};
use image::RgbaImage;
use reqwest::blocking::Client;
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path, time::Duration};

const API: &str = "https://planetarycomputer.microsoft.com/api/stac/v1/search";
const ATTRIBUTION: &str = "Imagery: USDA NAIP (public domain), hosted by Microsoft Planetary Computer. Terrain: Mapzen / AWS Open Data; source vertical datum is unverified.";

/// NAIP imagery and Mapzen terrain provider. It has no model or GPU dependency.
pub struct NaipProvider {
    client: Client,
}

fn http(operation: &'static str) -> impl FnOnce(reqwest::Error) -> ImageryError {
    move |source| ImageryError::Provider {
        operation,
        source: source.without_url(),
    }
}

impl NaipProvider {
    /// Create an HTTP client with bounded request time.
    ///
    /// # Errors
    /// Returns a TLS or client configuration error.
    pub fn new_blocking() -> Result<Self, ImageryError> {
        Ok(Self {
            client: Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .map_err(http("create client"))?,
        })
    }

    fn search_blocking(&self, bounds: [f64; 4]) -> Result<Vec<Value>, ImageryError> {
        let result: Value = self
            .client
            .post(API)
            .json(&json!({"collections":["naip"],"bbox":bounds,"limit":100}))
            .send()
            .and_then(|r| r.error_for_status())
            .and_then(|r| r.json())
            .map_err(http("search NAIP"))?;
        if result["links"]
            .as_array()
            .is_some_and(|links| links.iter().any(|l| l["rel"] == "next"))
        {
            return Err(invalid(
                "NAIP selection requires more source records; reduce the area",
            ));
        }
        let items = result["features"]
            .as_array()
            .ok_or_else(|| invalid("NAIP search has no features"))?;
        let year = items
            .iter()
            .filter_map(source_year)
            .max()
            .ok_or_else(|| invalid("NAIP has no imagery for this area"))?;
        Ok(items
            .iter()
            .filter(|i| source_year(i) == Some(year))
            .cloned()
            .collect())
    }

    fn token_blocking(&self) -> Result<String, ImageryError> {
        let value: Value = self
            .client
            .get("https://planetarycomputer.microsoft.com/api/sas/v1/token/naip")
            .send()
            .and_then(|r| r.error_for_status())
            .and_then(|r| r.json())
            .map_err(http("authorize public NAIP"))?;
        Ok(value["token"]
            .as_str()
            .ok_or_else(|| invalid("NAIP token is missing"))?
            .into())
    }

    /// Download a planned area and publish an immutable reference package.
    ///
    /// # Errors
    /// Rejects unsupported provider data, missing coverage, or invalid raster data.
    /// Provider operations disclose the planned area. The precise prior is not an input.
    pub fn build_blocking(
        &self,
        plan: &CoveragePlan,
        state: &Path,
        mut progress: impl FnMut(String),
    ) -> Result<Region, ImageryError> {
        let checked = crate::plan(plan.requested.clone())?;
        if checked.imagery_tiles != plan.imagery_tiles {
            return Err(invalid("coverage plan differs from its request"));
        }
        progress("Searching public NAIP imagery".into());
        let items = self.search_blocking(plan.bounds)?;
        let sources:Vec<_>=items.iter().map(|i|json!({"id":i["id"],"datetime":i["properties"]["datetime"],"gsd":i["properties"]["gsd"],"url":i["assets"]["image"]["href"]})).collect();
        let identity = json!({"provider":"planetary-computer-naip","sources":sources,"selection":plan.requested,
            "imagery_tiles":plan.imagery_tiles,"processing":"rust-gdal-rgb-bilinear-strict-mask-512/v1","terrain":"mapzen-terrarium-z14"});
        let key = digest(&serde_json::to_vec(&identity)?);
        let [w, s, e, n] = plan.bounds;
        let manifest = Package {
            schema_version: 1,
            region_id: format!("naip-{}", &key[..16]),
            release_id: format!("naip-{key}"),
            anchor_lat_lon: [(s + n) / 2.0, (w + e) / 2.0],
            elevation_datum: "source_vertical_datum_unverified".into(),
            attribution: ATTRIBUTION.into(),
            files: vec![],
            tiles: vec![],
            provenance: Some(identity),
            pack_id: String::new(),
        };
        let mut builder =
            PackageBuilder::new(manifest, |sha, bytes| chunk_blocking(state, sha, bytes))?;
        let mut overview = Overview::new(plan.imagery_tiles.clone())?;
        let mut sources = self.open_sources_blocking(&items)?;
        let terrain = self.imagery_blocking(
            plan,
            &mut sources,
            &mut builder,
            &mut overview,
            &mut progress,
        )?;
        for (index, tile) in terrain.iter().enumerate() {
            let bytes = self.terrain_blocking(state, *tile)?;
            builder.add(*tile, true, &bytes, &digest(&bytes))?;
            progress(format!("Terrain {}/{}", index + 1, terrain.len()));
        }
        publish_blocking(
            state,
            builder.finish()?,
            "NAIP · downloaded coverage".into(),
            overview,
        )
    }

    fn open_sources_blocking(
        &self,
        items: &[Value],
    ) -> Result<Vec<([f64; 4], GdalRaster)>, ImageryError> {
        for (key, value) in [
            ("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"),
            ("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif"),
            ("GDAL_HTTP_MAX_RETRY", "2"),
            ("GDAL_HTTP_TIMEOUT", "120"),
        ] {
            gdal::config::set_thread_local_config_option(key, value)
                .map_err(raster_error("configure range reader"))?;
        }
        let token = self.token_blocking()?;
        items
            .iter()
            .map(|item| {
                let href = item["assets"]["image"]["href"]
                    .as_str()
                    .ok_or_else(|| invalid("NAIP image URL missing"))?;
                let url =
                    reqwest::Url::parse(href).map_err(|_| invalid("invalid NAIP asset URL"))?;
                if url.scheme() != "https"
                    || !url
                        .host_str()
                        .is_some_and(|h| h.ends_with(".blob.core.windows.net"))
                    || url.query().is_some()
                {
                    return Err(invalid("unsupported NAIP asset origin"));
                }
                let bounds = serde_json::from_value(item["bbox"].clone())?;
                Ok((
                    bounds,
                    GdalRaster::open_blocking(&format!("/vsicurl/{href}?{token}"))?,
                ))
            })
            .collect()
    }

    fn imagery_blocking<F: FnMut(&str, &[u8]) -> Result<(), ImageryError>>(
        &self,
        plan: &CoveragePlan,
        sources: &mut [([f64; 4], GdalRaster)],
        builder: &mut PackageBuilder<F>,
        overview: &mut Overview,
        progress: &mut impl FnMut(String),
    ) -> Result<BTreeSet<Tile>, ImageryError> {
        let mut terrain = BTreeSet::new();
        for (index, tile) in plan.imagery_tiles.iter().enumerate() {
            let bounds = tile_bounds(*tile);
            let mut image = RgbaImage::new(512, 512);
            for (source_bounds, source) in sources.iter_mut() {
                if !intersects(*source_bounds, bounds) {
                    continue;
                }
                let rendered = source.tile_blocking(*tile)?;
                for (out, pixel) in image.pixels_mut().zip(rendered.pixels()) {
                    if pixel[3] == 255 {
                        *out = *pixel;
                    }
                }
            }
            if image.pixels().any(|p| p[3] == 255) {
                overview.add(*tile, &image);
                let bytes = png(image)?;
                builder.add(*tile, false, &bytes, &digest(&bytes))?;
                terrain.insert(Tile(14, tile.1 >> (tile.0 - 14), tile.2 >> (tile.0 - 14)));
            }
            progress(format!(
                "NAIP imagery {}/{}",
                index + 1,
                plan.imagery_tiles.len()
            ));
        }
        Ok(terrain)
    }

    fn terrain_blocking(&self, state: &Path, Tile(z, x, y): Tile) -> Result<Vec<u8>, ImageryError> {
        let path = state
            .join("provider-cache")
            .join(format!("terrain-{z}-{x}-{y}.png"));
        let bytes = if path.exists() {
            read_blocking(&path)?
        } else {
            self.client
                .get(format!(
                    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
                ))
                .send()
                .and_then(|r| r.error_for_status())
                .and_then(|r| r.bytes())
                .map_err(http("download terrain"))?
                .to_vec()
        };
        let image = image::load_from_memory(&bytes)?.to_rgba8();
        if image.dimensions() != (256, 256) {
            return Err(invalid("unexpected terrain tile dimensions"));
        }
        let encoded = png(image)?;
        if !path.exists() {
            write_blocking(&path, &encoded)?;
        }
        Ok(encoded)
    }
}

fn intersects(a: [f64; 4], b: [f64; 4]) -> bool {
    a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
}

fn source_year(item: &Value) -> Option<u64> {
    let year = &item["properties"]["naip:year"];
    year.as_u64().or_else(|| year.as_str()?.parse().ok())
}

#[cfg(test)]
mod tests;

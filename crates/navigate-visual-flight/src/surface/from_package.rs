//! A [`Surface`] from a navigation data package (ADR-0022).

use super::{Surface, enu_frame_on};
use crate::error::FlightVisualError;
use image::RgbImage;
use navigate_data::index::IndexEntry;
use navigate_data::product::inputs_digest;
use navigate_data::tile::{decode_terrarium_m, tile_lat_lon};
use navigate_data::{LayerKey, Package, ReadPort, Sha256Digest, TileAddress, read_blob};
use navigate_geodesy::BodyEllipsoid;
use std::collections::BTreeSet;
use std::collections::HashMap;

/// The layers and zoom levels of a package that a [`Surface`] reads.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceLayers {
    /// Imagery layer.
    pub imagery_layer: LayerKey,
    /// Imagery zoom.
    pub imagery_zoom: u32,
    /// Elevation layer.
    pub elevation_layer: LayerKey,
    /// Elevation zoom.
    pub elevation_zoom: u32,
}

impl Surface {
    /// Loads the tiles of two layers of a package at one zoom each. The
    /// heights stay as the provider supplies them, as in
    /// [`Surface::from_catalog`]. `catalog_sha256` is the inputs digest of
    /// the tile blobs that the surface holds
    /// (`navigate_data::product::inputs_digest`), so a product made from
    /// the surface binds its inputs, not one mission package. Returns the
    /// surface and the input blob digests.
    ///
    /// # Errors
    ///
    /// Refuses a layer or zoom that the package does not hold, a changed or
    /// missing blob, and a tile that does not decode.
    pub async fn from_package<S: ReadPort + ?Sized>(
        package: &Package<S::File>,
        store: &S,
        layers: &SurfaceLayers,
    ) -> Result<(Self, BTreeSet<Sha256Digest>), FlightVisualError> {
        let imagery = read_layer(package, store, layers.imagery_layer, layers.imagery_zoom).await?;
        let dem = read_layer(
            package,
            store,
            layers.elevation_layer,
            layers.elevation_zoom,
        )
        .await?;
        if imagery.is_empty() {
            return Err(FlightVisualError::Invalid(format!(
                "package {} has no imagery at zoom {}",
                package.identity(),
                layers.imagery_zoom
            )));
        }
        let mut imagery_tiles = HashMap::new();
        let mut bbox = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for (address, _, bytes) in &imagery {
            let image = decode(bytes)?;
            let [north, west] =
                tile_lat_lon(f64::from(address.x), f64::from(address.y), address.zoom);
            let [south, east] = tile_lat_lon(
                f64::from(address.x + 1),
                f64::from(address.y + 1),
                address.zoom,
            );
            bbox = [
                bbox[0].min(west),
                bbox[1].min(south),
                bbox[2].max(east),
                bbox[3].max(north),
            ];
            imagery_tiles.insert((address.x, address.y), image);
        }
        let mut dem_tiles = HashMap::new();
        let (mut low, mut high, mut side) = (f64::INFINITY, f64::NEG_INFINITY, 256_u32);
        for (address, _, bytes) in &dem {
            let image = decode(bytes)?;
            side = image.width().max(1);
            let heights: Vec<f32> = image
                .pixels()
                .map(|p| decode_terrarium_m(p.0))
                .inspect(|h| {
                    low = low.min(*h);
                    high = high.max(*h);
                })
                .map(|h| h as f32)
                .collect();
            dem_tiles.insert((address.x, address.y), heights);
        }
        if !low.is_finite() {
            (low, high) = (0.0, 500.0);
        }
        let origin_lat = 0.5 * (bbox[1] + bbox[3]);
        let origin_lon = 0.5 * (bbox[0] + bbox[2]);
        let (origin_ecef, enu_from_ecef) =
            enu_frame_on(BodyEllipsoid::WGS84, origin_lat, origin_lon);
        let inputs: BTreeSet<Sha256Digest> = imagery
            .iter()
            .chain(&dem)
            .map(|(_, blob, _)| *blob)
            .collect();
        let surface = Self {
            ellipsoid: BodyEllipsoid::WGS84,
            origin_lat,
            origin_lon,
            bbox,
            imagery_zoom: layers.imagery_zoom,
            dem_zoom: layers.elevation_zoom,
            catalog_sha256: inputs_digest(&inputs.iter().copied().collect::<Vec<_>>()).to_string(),
            min_elev_m: low,
            max_elev_m: high,
            origin_ecef,
            enu_from_ecef,
            imagery_tiles,
            dem_tiles,
            dem_tile_size: side,
        };
        Ok((surface, inputs))
    }
}

fn decode(bytes: &[u8]) -> Result<RgbImage, FlightVisualError> {
    image::load_from_memory(bytes)
        .map(|image| image.to_rgb8())
        .map_err(|source| FlightVisualError::Image {
            path: "package tile".into(),
            source,
        })
}

async fn read_layer<S: ReadPort + ?Sized>(
    package: &Package<S::File>,
    store: &S,
    layer: LayerKey,
    zoom: u32,
) -> Result<Vec<(TileAddress, Sha256Digest, Vec<u8>)>, FlightVisualError> {
    let mut out = Vec::new();
    for (address, entry) in package.layer_entries(layer).await? {
        if let (true, IndexEntry::Tile { blob, bytes }) = (address.zoom == zoom, entry) {
            out.push((address, blob, read_blob(store, blob, bytes).await?));
        }
    }
    Ok(out)
}

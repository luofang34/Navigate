//! Render scenes from a navigation data package (ADR-0022, representation
//! R).

use std::collections::BTreeSet;

use navigate_data::index::IndexEntry;
use navigate_data::tile::tile_lat_lon;
use navigate_data::{
    EllipsoidalHeight, GeoidModel, LayerKey, Package, ReadPort, Sha256Digest, TileAddress,
    VerticalDatum, read_blob,
};
use navigate_geodesy::GeoidUndulation;
use navigate_visual::{LocalFrame, MapRevision};

use crate::{AtlasLayout, ImageAtlas, RenderError, RenderScene, TerrainAtlas};

/// The host geoid model and the model that it implements, for elevation
/// layers with an orthometric datum.
pub type Geoid<'a> = Option<(GeoidModel, &'a dyn GeoidUndulation)>;

/// What [`scene_around`] reads from a package.
#[derive(Clone, Copy, Debug)]
pub struct SceneRequest {
    /// Imagery layer.
    pub imagery_layer: LayerKey,
    /// Imagery zoom.
    pub imagery_zoom: u32,
    /// Elevation layer.
    pub elevation_layer: LayerKey,
    /// Elevation zoom.
    pub elevation_zoom: u32,
    /// `[lat, lon]` of the window centre and of the frame anchor, degrees.
    pub centre: [f64; 2],
    /// Half side of the window, metres.
    pub half_side_m: f64,
}

/// A render scene and what the package could supply for it.
pub struct SceneWindow {
    /// The scene, in a WGS84 east-north-up frame at the centre.
    pub scene: RenderScene,
    /// Share of the imagery pixels that the package covers, in `[0, 1]`.
    pub imagery_coverage: f64,
    /// Bound in metres on the height error from an unknown geoid
    /// undulation; zero when the heights are ellipsoidal.
    pub undulation_bound_m: f64,
    /// Digests of the tile blobs that the scene holds.
    pub inputs: BTreeSet<Sha256Digest>,
}

/// Assembles the scene of a square window from the tiles of a package.
/// Tiles in a gap or outside the coverage leave their pixels uncovered.
///
/// # Errors
///
/// Refuses an unknown layer, an elevation layer without a datum, a window
/// across the antimeridian, a changed or missing blob, and a tile that does
/// not decode at the declared size.
pub async fn scene_around<S: ReadPort + ?Sized>(
    package: &Package<S::File>,
    store: &S,
    request: &SceneRequest,
    geoid: Geoid<'_>,
) -> Result<SceneWindow, RenderError> {
    let [lat, lon] = request.centre;
    let frame = LocalFrame::wgs84_enu(lat, lon)?;
    let datum = package
        .manifest()
        .layer(request.elevation_layer)
        .and_then(|l| l.vertical_datum)
        .ok_or(RenderError::Invalid(
            "elevation layer has no vertical datum",
        ))?;
    let (image_layout, image_tiles) =
        window_tiles(request.centre, request.half_side_m, request.imagery_zoom)?;
    let (terrain_layout, terrain_tiles) =
        window_tiles(request.centre, request.half_side_m, request.elevation_zoom)?;
    let mut inputs = BTreeSet::new();
    let mut imagery = Grid::new(image_layout, 0_u8);
    for address in image_tiles {
        if let Some(bytes) =
            read(package, store, request.imagery_layer, address, &mut inputs).await?
        {
            let luma = decode(&bytes)?.to_luma8();
            imagery.put(address, luma.as_raw())?;
        }
    }
    let mut terrain = Grid::new(terrain_layout, 0.0_f32);
    let mut bound = 0.0_f64;
    for address in terrain_tiles {
        if let Some(bytes) = read(
            package,
            store,
            request.elevation_layer,
            address,
            &mut inputs,
        )
        .await?
        {
            let (heights, tile_bound) = terrain_heights(&bytes, address, &datum, geoid)?;
            bound = bound.max(tile_bound);
            terrain.put(address, &heights)?;
        }
    }
    let covered = imagery.coverage.iter().filter(|c| **c == 255).count();
    let imagery_coverage = covered as f64 / imagery.coverage.len().max(1) as f64;
    Ok(SceneWindow {
        scene: RenderScene {
            map: MapRevision {
                // The release label is in the descriptor, outside the identity;
                // the identity names the release here.
                release_id: package.identity().to_string(),
                manifest_sha256: package.identity().to_string(),
            },
            frame,
            imagery: ImageAtlas {
                layout: imagery.layout,
                pixels: imagery.values,
                coverage: imagery.coverage,
            },
            terrain: TerrainAtlas {
                layout: terrain.layout,
                heights_m: terrain.values,
                coverage: terrain.coverage,
            },
        },
        imagery_coverage,
        undulation_bound_m: bound,
        inputs,
    })
}

/// The bytes of one tile, or `None` for a gap or an address outside the
/// coverage. The blob digest goes into `inputs`.
async fn read<S: ReadPort + ?Sized>(
    package: &Package<S::File>,
    store: &S,
    layer: LayerKey,
    address: TileAddress,
    inputs: &mut BTreeSet<Sha256Digest>,
) -> Result<Option<Vec<u8>>, RenderError> {
    match package.entry(layer, address).await? {
        Some(IndexEntry::Tile { blob, bytes }) => {
            let data = read_blob(store, blob, bytes).await?;
            inputs.insert(blob);
            Ok(Some(data))
        }
        Some(IndexEntry::Gap) | None => Ok(None),
    }
}

fn decode(bytes: &[u8]) -> Result<image::DynamicImage, RenderError> {
    let image = image::load_from_memory(bytes)?;
    if image.width() != TILE_PIXELS || image.height() != TILE_PIXELS {
        return Err(RenderError::Invalid(
            "a package tile is not 256 pixels square",
        ));
    }
    Ok(image)
}

/// Pixels on a tile side in a package layer that this module reads.
const TILE_PIXELS: u32 = 256;

/// Ellipsoidal heights of one Terrarium tile, and the undulation bound.
fn terrain_heights(
    bytes: &[u8],
    address: TileAddress,
    datum: &VerticalDatum,
    geoid: Geoid<'_>,
) -> Result<(Vec<f32>, f64), RenderError> {
    let rgb = decode(bytes)?.to_rgb8();
    let [lat, lon] = tile_lat_lon(
        f64::from(address.x) + 0.5,
        f64::from(address.y) + 0.5,
        address.zoom,
    );
    // The geoid changes by centimetres across one elevation tile, so one
    // undulation serves the tile.
    let (offset, bound) = match datum.to_ellipsoidal(0.0, lat.to_radians(), lon.to_radians(), geoid)
    {
        EllipsoidalHeight::Known(n) => (n, 0.0),
        EllipsoidalHeight::UnknownUndulation { bound_m, .. } => (0.0, bound_m),
    };
    let heights = rgb
        .pixels()
        .map(|p| (navigate_data::tile::decode_terrarium_m(p.0) + offset) as f32)
        .collect();
    Ok((heights, bound))
}

/// The tile rectangle of a square window at `zoom`.
fn window_tiles(
    [lat, lon]: [f64; 2],
    half_side_m: f64,
    zoom: u32,
) -> Result<(AtlasLayout, Vec<TileAddress>), RenderError> {
    let dlat = (half_side_m / 6_356_752.314_245).to_degrees();
    let dlon = dlat / (lat.abs() + dlat).to_radians().cos().max(1e-6);
    let corner = |p: [f64; 2]| TileAddress::containing(p, zoom);
    let north_west = corner([lat + dlat, lon - dlon])?;
    let south_east = corner([lat - dlat, lon + dlon])?;
    if south_east.x < north_west.x {
        return Err(RenderError::Invalid(
            "a scene window across the antimeridian",
        ));
    }
    let mut tiles = Vec::new();
    for y in north_west.y..=south_east.y {
        for x in north_west.x..=south_east.x {
            tiles.push(TileAddress { zoom, x, y });
        }
    }
    let columns = south_east.x - north_west.x + 1;
    let rows = south_east.y - north_west.y + 1;
    let layout = AtlasLayout {
        zoom,
        origin_tile: [north_west.x, north_west.y],
        tile_pixels: TILE_PIXELS,
        size: [columns * TILE_PIXELS, rows * TILE_PIXELS],
    };
    Ok((layout, tiles))
}

/// A row-major grid of values with coverage.
struct Grid<T> {
    layout: AtlasLayout,
    values: Vec<T>,
    coverage: Vec<u8>,
}

impl<T: Copy> Grid<T> {
    fn new(layout: AtlasLayout, blank: T) -> Self {
        let size = layout.size[0] as usize * layout.size[1] as usize;
        Self {
            layout,
            values: vec![blank; size],
            coverage: vec![0; size],
        }
    }

    fn put(&mut self, address: TileAddress, tile: &[T]) -> Result<(), RenderError> {
        let side = TILE_PIXELS as usize;
        if tile.len() != side * side {
            return Err(RenderError::Invalid(
                "a package tile has the wrong sample count",
            ));
        }
        let width = self.layout.size[0] as usize;
        let column = (address.x - self.layout.origin_tile[0]) as usize * side;
        let row = (address.y - self.layout.origin_tile[1]) as usize * side;
        for (line, chunk) in tile.chunks_exact(side).enumerate() {
            let start = (row + line) * width + column;
            if let (Some(target), Some(mask)) = (
                self.values.get_mut(start..start + side),
                self.coverage.get_mut(start..start + side),
            ) {
                target.copy_from_slice(chunk);
                mask.fill(255);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

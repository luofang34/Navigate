use super::*;
use navigate_visual_render::{AtlasLayout, ImageAtlas, RenderScene, TerrainAtlas};
use std::collections::BTreeSet;

#[derive(Deserialize)]
pub(super) struct Catalog {
    format: String,
    version: u32,
    complete: bool,
    configuration: Configuration,
    tiles: Vec<Tile>,
}
#[derive(Deserialize)]
struct Configuration {
    bbox: [f64; 4],
    dem_zoom: u32,
    imagery_zoom: u32,
}
#[derive(Deserialize)]
pub(super) struct Tile {
    pub kind: String,
    pub xyz: [u32; 3],
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: usize,
}
impl Catalog {
    pub(super) fn load_blocking(
        &self,
        root: &Path,
        config: &SceneConfig,
        frame: LocalFrame,
        map: MapRevision,
    ) -> Result<RenderScene, Error> {
        let [west, south, east, north] = self.configuration.bbox;
        let anchor = [(south + north) * 0.5, (west + east) * 0.5];
        if self.format != "photo-reference-catalog"
            || self.version != 1
            || !self.complete
            || self.tiles.is_empty()
            || self.tiles.len() > 100_000
            || !self.configuration.bbox.iter().all(|v| v.is_finite())
            || west >= east
            || south >= north
            || west < -180.0
            || east > 180.0
            || south < -85.0
            || north > 85.0
            || frame
                .anchor()
                .iter()
                .zip(anchor)
                .any(|(a, b)| (*a - b).abs() > 1e-8)
        {
            return Err(Error::Input(
                "catalog format, completion, or geographic anchor".into(),
            ));
        }
        let image_layout = self.layout(
            "imagery",
            self.configuration.imagery_zoom,
            config.imagery_tile_pixels,
        )?;
        let dem_layout = self.layout("dem", self.configuration.dem_zoom, 256)?;
        let image_count = count(image_layout)?;
        let dem_count = count(dem_layout)?;
        if image_count
            .saturating_mul(2)
            .saturating_add(dem_count.saturating_mul(8))
            > config.scene_bytes
        {
            return Err(Error::Input("catalog exceeds scene memory limit; reduce imagery tile pixels or use a smaller catalog".into()));
        }
        let mut imagery = ImageAtlas {
            layout: image_layout,
            pixels: vec![0; image_count],
            coverage: vec![0; image_count],
        };
        let mut terrain = TerrainAtlas {
            layout: dem_layout,
            heights_m: vec![0.0; dem_count],
            coverage: vec![0; dem_count],
        };
        for tile in &self.tiles {
            let image = super::tiles::decode_blocking(root, tile)?;
            match tile.kind.as_str() {
                "imagery" => super::tiles::imagery(&mut imagery, tile, image),
                "dem" => super::tiles::terrain(&mut terrain, tile, image),
                _ => return Err(Error::Input("unknown catalog tile kind".into())),
            }
        }
        Ok(RenderScene {
            map,
            frame,
            imagery,
            terrain,
        })
    }

    fn layout(&self, kind: &str, zoom: u32, tile_pixels: u32) -> Result<AtlasLayout, Error> {
        if zoom > 22 {
            return Err(Error::Input("catalog zoom exceeds 22".into()));
        }
        let mut seen = BTreeSet::new();
        let mut start = [u32::MAX; 2];
        let mut end = [0; 2];
        for tile in self.tiles.iter().filter(|t| t.kind == kind) {
            let [z, x, y] = tile.xyz;
            if z != zoom || x >= 1 << zoom || y >= 1 << zoom || !seen.insert((x, y)) {
                return Err(Error::Input(format!(
                    "invalid or duplicate {kind} tile {z}/{x}/{y}"
                )));
            }
            for i in 0..2 {
                start[i] = start[i].min(tile.xyz[i + 1]);
                end[i] = end[i].max(tile.xyz[i + 1]);
            }
        }
        if seen.is_empty() {
            return Err(Error::Input(format!("catalog has no {kind} tiles")));
        }
        Ok(AtlasLayout {
            zoom,
            origin_tile: start,
            tile_pixels,
            size: [
                (end[0] - start[0] + 1) * tile_pixels,
                (end[1] - start[1] + 1) * tile_pixels,
            ],
        })
    }
}
fn count(layout: AtlasLayout) -> Result<usize, Error> {
    if layout.size.iter().any(|n| *n > 8192) {
        return Err(Error::Input(
            "atlas exceeds 8192 pixels; reduce imagery tile pixels or use a smaller catalog".into(),
        ));
    }
    Ok(layout.size[0] as usize * layout.size[1] as usize)
}

#[cfg(test)]
mod tests;

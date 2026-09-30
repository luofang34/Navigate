//! MapLibre color and depth rendering from the shared map package.

mod geometry;
#[cfg(test)]
mod tests;

use crate::{BenchError, package::MapPackage};
use image::{GrayImage, Luma};
use maplibre::{
    coords::{LatLon, WorldTileCoords},
    headless::{
        HeadlessPlugin, create_headless_renderer_with_settings,
        map::{
            HeadlessMap, ProcessedLayers,
            reference::{PinholeIntrinsics, ReferenceTarget},
        },
    },
    plugin::Plugin,
    raster::{AvailableRasterLayerData, DefaultRasterTransferables, RasterPlugin},
    render::{
        RenderPlugin,
        settings::{BufferPoolSizes, Msaa, RendererSettings},
        view_state::ExternalAnchor,
    },
    style::Style,
    terrain::{DefaultDemTransferables, TerrainPlugin},
};
use navigate_visual::{CameraModel, CameraPose, LocalFrame, MapRevision, ReferenceView};

pub(crate) struct ReferenceRenderer {
    map: HeadlessMap,
    target: ReferenceTarget,
    camera: CameraModel,
    anchor: ExternalAnchor,
    revision: MapRevision,
    frame: LocalFrame,
    coverage: geometry::Coverage,
    style_sha256: String,
}

fn render_error<E: std::error::Error + Send + Sync + 'static>(source: E) -> BenchError {
    BenchError::Render {
        source: Box::new(source),
    }
}

impl ReferenceRenderer {
    pub async fn new(package: MapPackage, camera: CameraModel) -> Result<Self, BenchError> {
        camera.validate()?;
        let settings = RendererSettings {
            msaa: Msaa { samples: 1 },
            buffer_pools: BufferPoolSizes {
                vertices: 10000,
                indices: 10000,
                feature_metadata: 10000,
                layer_metadata: 1024,
            },
            symbol_pools: BufferPoolSizes {
                vertices: 10000,
                indices: 10000,
                feature_metadata: 10000,
                layer_metadata: 1024,
            },
            texture_format: Some(wgpu::TextureFormat::Rgba8Unorm),
            ..Default::default()
        };
        let (kernel, renderer) =
            create_headless_renderer_with_settings(camera.width, camera.height, None, settings)
                .await
                .map_err(render_error)?;
        let anchor = ExternalAnchor {
            position: LatLon::new(
                package.manifest.anchor_lat_lon[0],
                package.manifest.anchor_lat_lon[1],
            ),
            altitude_meters: 0.0,
        };
        let (style, style_sha256) = style(&package)?;
        let coverage = geometry::Coverage::new(&package.manifest, package.frame);
        let plugins: Vec<Box<dyn Plugin<_>>> = vec![
            Box::new(RenderPlugin),
            Box::new(RasterPlugin::<DefaultRasterTransferables>::default()),
            Box::new(TerrainPlugin::<DefaultDemTransferables>::default()),
            Box::new(
                HeadlessPlugin::new(false)
                    .preserve_tile_sources()
                    .retain_supplied_tiles(),
            ),
        ];
        let mut map = HeadlessMap::new(style, renderer, kernel, plugins).map_err(render_error)?;
        load_sources_blocking(&mut map, package.tiles)?;
        let target = ReferenceTarget::new(&map, intrinsics(camera)).map_err(render_error)?;
        Ok(Self {
            map,
            target,
            camera,
            anchor,
            revision: package.revision,
            frame: package.frame,
            coverage,
            style_sha256,
        })
    }

    pub fn render_blocking(&mut self, pose: CameraPose) -> Result<ReferenceView, BenchError> {
        pose.validate()?;
        let camera = self.camera;
        let render = self
            .target
            .render_blocking(&mut self.map, self.anchor, geometry::eye_transform(pose))
            .map_err(render_error)?;
        let rgba = render.rgba;
        let mut depth_m = render.depth_m;
        let image = GrayImage::from_fn(camera.width, camera.height, |x, y| {
            let i = ((y * camera.width + x) * 4) as usize;
            Luma([((u32::from(rgba[i]) * 77
                + u32::from(rgba[i + 1]) * 150
                + u32::from(rgba[i + 2]) * 29)
                >> 8) as u8])
        });
        self.coverage.mask(&mut depth_m, camera, pose, |world| {
            self.map
                .rendered_terrain_sample_cached(self.frame.mercator_xy(world))
                .is_some_and(|sample| sample.covered && sample.dem_loaded)
        });
        Ok(ReferenceView {
            map: self.revision.clone(),
            frame: self.frame,
            pose,
            image,
            depth_m,
        })
    }
}

fn intrinsics(camera: CameraModel) -> PinholeIntrinsics {
    PinholeIntrinsics {
        width: camera.width,
        height: camera.height,
        fx: camera.fx,
        fy: camera.fy,
        cx: camera.cx,
        cy: camera.cy,
    }
}

fn load_sources_blocking(
    map: &mut HeadlessMap,
    tiles: Vec<crate::package::DecodedTile>,
) -> Result<(), BenchError> {
    let mut imagery = Vec::new();
    let mut elevation = Vec::new();
    for tile in tiles {
        let [z, x, y] = tile.xyz;
        let coords = WorldTileCoords::from((x as i32, y as i32, (z as u8).into()));
        if let Some(image) = tile.imagery {
            imagery.push(AvailableRasterLayerData {
                coords,
                source_layer: "imagery".into(),
                image,
            });
        }
        if let Some(image) = tile.elevation {
            elevation.push((coords, image));
        }
    }
    map.render_frames_with_terrain(ProcessedLayers::default(), imagery, elevation, 3)
        .map_err(render_error)
}

/// The reference style and the SHA-256 of its canonical JSON.
fn style(package: &MapPackage) -> Result<(Style, String), BenchError> {
    let imagery_maxzoom = package
        .manifest
        .tiles
        .iter()
        .filter(|tile| tile.imagery.is_some())
        .map(|t| t.xyz.0)
        .max()
        .unwrap_or(0);
    let dem_maxzoom = package
        .manifest
        .tiles
        .iter()
        .filter(|tile| tile.elevation.is_some())
        .map(|tile| tile.xyz.0)
        .max()
        .unwrap_or(0);
    let value = serde_json::json!({
        "version":8, "center":[package.manifest.anchor_lat_lon[1],package.manifest.anchor_lat_lon[0]],
        "zoom":12, "projection":{"type":"mercator"}, "terrain":{"source":"dem","exaggeration":1},
        "sources":{
            "imagery":{"type":"raster","tiles":["local://imagery/{z}/{x}/{y}"],"tileSize":512,"maxzoom":imagery_maxzoom},
            "dem":{"type":"raster-dem","tiles":["local://dem/{z}/{x}/{y}"],"tileSize":256,"maxzoom":dem_maxzoom,"encoding":"terrarium"}
        },
        "layers":[{"id":"background","type":"background","paint":{"background-color":"rgba(0,0,0,0)"}},
            {"id":"imagery","type":"raster","source":"imagery","paint":{"raster-fade-duration":0}}]
    });
    let json_error = |source| BenchError::Json {
        path: "generated-reference-style".into(),
        source,
    };
    let digest = crate::package::digest(&serde_json::to_vec(&value).map_err(json_error)?);
    Ok((serde_json::from_value(value).map_err(json_error)?, digest))
}

impl navigate_visual::ReferenceRenderer for ReferenceRenderer {
    type Error = BenchError;

    fn identity(&self) -> navigate_visual::RendererIdentity {
        navigate_visual::RendererIdentity {
            revision: crate::RENDERER_REVISION.trim().to_owned(),
            style_sha256: self.style_sha256.clone(),
        }
    }

    fn render_blocking(&mut self, pose: CameraPose) -> Result<ReferenceView, BenchError> {
        ReferenceRenderer::render_blocking(self, pose)
    }
}

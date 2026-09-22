//! MapLibre color and depth rendering from the shared map package.

mod geometry;
mod readback;

use crate::{BenchError, package::MapPackage};
use cgmath::{Matrix4, SquareMatrix};
use image::{GrayImage, Luma};
use maplibre::{
    coords::{LatLon, WorldTileCoords},
    headless::{
        HeadlessPlugin, create_headless_renderer_with_settings,
        map::{HeadlessMap, ProcessedLayers},
    },
    plugin::Plugin,
    raster::{AvailableRasterLayerData, DefaultRasterTransferables, RasterPlugin},
    render::{
        RenderPlugin,
        settings::{BufferPoolSizes, Msaa, RendererSettings},
        view_state::ExternalAnchor,
        xr::{EyeTarget, ScenePlacement, XrEye, XrFrame},
    },
    style::Style,
    terrain::{DefaultDemTransferables, TerrainPlugin},
};
use navigate_visual::{CameraModel, CameraPose, MapRevision, ReferenceView};
use std::time::Duration;

pub(crate) struct ReferenceRenderer {
    map: HeadlessMap,
    depth: wgpu::Texture,
    camera: CameraModel,
    anchor: ExternalAnchor,
    revision: MapRevision,
    tick: u64,
    coverage: geometry::Coverage,
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
        let style = style(&package)?;
        let coverage = geometry::Coverage::new(&package.manifest);
        let plugins: Vec<Box<dyn Plugin<_>>> = vec![
            Box::new(RenderPlugin),
            Box::new(RasterPlugin::<DefaultRasterTransferables>::default()),
            Box::new(TerrainPlugin::<DefaultDemTransferables>::default()),
            Box::new(HeadlessPlugin::new(false).preserve_tile_sources()),
        ];
        let mut map = HeadlessMap::new(style, renderer, kernel, plugins).map_err(render_error)?;
        let mut imagery = Vec::new();
        let mut elevation = Vec::new();
        for tile in package.tiles {
            let [z, x, y] = tile.xyz;
            let coords = WorldTileCoords::from((x as i32, y as i32, (z as u8).into()));
            imagery.push(AvailableRasterLayerData {
                coords,
                source_layer: "imagery".into(),
                image: tile.imagery,
            });
            elevation.push((coords, tile.elevation));
        }
        map.render_frames_with_terrain(ProcessedLayers::default(), imagery, elevation, 3)
            .map_err(render_error)?;
        let depth = map.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("visual reference depth"),
            size: wgpu::Extent3d {
                width: camera.width,
                height: camera.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Ok(Self {
            map,
            depth,
            camera,
            anchor,
            revision: package.revision,
            tick: 0,
            coverage,
        })
    }

    pub fn render_blocking(&mut self, pose: CameraPose) -> Result<ReferenceView, BenchError> {
        pose.validate()?;
        let camera = self.camera;
        let world_from_eye = geometry::eye_transform(pose);
        let frustum = geometry::frustum(camera);
        for _ in 0..4 {
            self.tick = self.tick.wrapping_add(1);
            self.map
                .run_xr_frame(XrFrame {
                    timestamp: Duration::from_millis(self.tick.wrapping_mul(16)),
                    opaque_environment: true,
                    placement: ScenePlacement {
                        anchor: self.anchor,
                        world_from_scene: Matrix4::identity(),
                    },
                    eyes: vec![XrEye {
                        world_from_eye,
                        frustum,
                        target: EyeTarget {
                            color: None,
                            depth: Some(self.depth.create_view(&Default::default())),
                        },
                    }],
                    request_overscan: 1.0,
                    prefetch: None,
                })
                .map_err(render_error)?;
        }
        let texture = self.map.head_texture().ok_or(BenchError::MissingTexture)?;
        let rgba = readback::read_blocking(&self.map, texture, wgpu::TextureAspect::All)?;
        let depths =
            readback::read_blocking(&self.map, &self.depth, wgpu::TextureAspect::DepthOnly)?;
        let image = GrayImage::from_fn(camera.width, camera.height, |x, y| {
            let i = ((y * camera.width + x) * 4) as usize;
            Luma([((u32::from(rgba[i]) * 77
                + u32::from(rgba[i + 1]) * 150
                + u32::from(rgba[i + 2]) * 29)
                >> 8) as u8])
        });
        let mut depth_m: Vec<f32> = depths
            .chunks_exact(4)
            .map(|b| {
                let d = f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                if d <= 0.0 {
                    return 0.0;
                }
                (frustum.near * frustum.far / (frustum.near + d * (frustum.far - frustum.near)))
                    as f32
            })
            .collect();
        self.coverage.mask(&mut depth_m, camera, pose);
        Ok(ReferenceView {
            map: self.revision.clone(),
            pose,
            image,
            depth_m,
        })
    }
}

fn style(package: &MapPackage) -> Result<Style, BenchError> {
    let maxzoom = package
        .manifest
        .tiles
        .iter()
        .map(|t| t.xyz[0])
        .max()
        .unwrap_or(0);
    serde_json::from_value(serde_json::json!({
        "version":8, "center":[package.manifest.anchor_lat_lon[1],package.manifest.anchor_lat_lon[0]],
        "zoom":12, "projection":{"type":"mercator"}, "terrain":{"source":"dem","exaggeration":1},
        "sources":{
            "imagery":{"type":"raster","tiles":["local://imagery/{z}/{x}/{y}"],"tileSize":512,"maxzoom":maxzoom},
            "dem":{"type":"raster-dem","tiles":["local://dem/{z}/{x}/{y}"],"tileSize":256,"maxzoom":maxzoom,"encoding":"terrarium"}
        },
        "layers":[{"id":"background","type":"background","paint":{"background-color":"#1b2430"}},
            {"id":"imagery","type":"raster","source":"imagery","paint":{"raster-fade-duration":0}}]
    })).map_err(|source| BenchError::Json { path: "generated-reference-style".into(), source })
}

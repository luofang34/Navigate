use crate::{
    error::{PreviewError, render_error},
    model::{Camera, Manifest, Pose},
    storage::BrowserStore,
};
use cgmath::{Matrix4, SquareMatrix};
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
use wasm_bindgen::prelude::*;
/// MapLibre terrain renderer backed by a verified offline package.
#[wasm_bindgen]
pub struct Preview {
    pub(crate) map: HeadlessMap,
    pub(crate) camera: Camera,
    anchor: ExternalAnchor,
    tick: u64,
    pub(crate) depth: wgpu::Texture,
    pub(crate) camera_session: Option<crate::session::Session>,
    pub(crate) reference: Option<navigate_visual::ReferenceView>,
    pub(crate) manifest: Manifest,
    pub(crate) coverage: crate::coverage::Coverage,
    pub(crate) globe: bool,
    pub(crate) surface: Option<wgpu::Surface<'static>>,
}
#[wasm_bindgen]
impl Preview {
    /// Read package ranges through the host and create the WebGPU renderer.
    pub async fn create(
        manifest_json: String,
        camera_json: String,
        read: js_sys::Function,
        globe: bool,
    ) -> Result<Preview, JsValue> {
        Self::initialize(manifest_json, camera_json, read, globe, None)
            .await
            .map_err(Into::into)
    }
    /// Add display context. Reference renderers reject this lower-resolution surface.
    pub fn set_globe_context(&mut self, bytes: Vec<u8>) -> Result<(), JsValue> {
        if !self.globe || bytes.len() > 1024 * 1024 {
            return Err(PreviewError::Input {
                reason: "invalid globe context".into(),
            }
            .into());
        }
        let image = image::load_from_memory(&bytes)
            .map_err(|source| PreviewError::Image {
                uri: "display context".into(),
                source,
            })?
            .to_rgba8();
        if image.dimensions() != (512, 512) {
            return Err(PreviewError::Input {
                reason: "context tile must be 512 by 512".into(),
            }
            .into());
        }
        self.map
            .render_frames_with_terrain(
                ProcessedLayers::default(),
                vec![AvailableRasterLayerData {
                    coords: WorldTileCoords::from((0, 0, 0_u8.into())),
                    source_layer: "context".into(),
                    image,
                }],
                vec![],
                3,
            )
            .map_err(|e| render_error("globe context upload", e))?;
        Ok(())
    }
    /// Render an eye-to-ENU pose and return tightly packed RGBA pixels.
    pub async fn render(&mut self, pose_json: String) -> Result<Vec<u8>, JsValue> {
        let pose: Pose = serde_json::from_str(&pose_json).map_err(PreviewError::from)?;
        let transform = pose.transform()?;
        self.draw(transform)?;
        crate::readback::read(&self.map).await.map_err(Into::into)
    }
}
impl Preview {
    pub(crate) fn draw(&mut self, transform: Matrix4<f64>) -> Result<(), PreviewError> {
        self.draw_to(transform, None, 4)
    }
    pub(crate) fn draw_to(
        &mut self,
        transform: Matrix4<f64>,
        color: Option<&wgpu::Texture>,
        frames: usize,
    ) -> Result<(), PreviewError> {
        for _ in 0..frames {
            self.tick = self.tick.wrapping_add(1);
            self.map
                .run_xr_frame(XrFrame {
                    timestamp: std::time::Duration::from_millis(self.tick.wrapping_mul(16)),
                    opaque_environment: true,
                    placement: ScenePlacement {
                        anchor: self.anchor,
                        world_from_scene: Matrix4::identity(),
                    },
                    eyes: vec![XrEye {
                        world_from_eye: transform,
                        frustum: self.camera.frustum(),
                        target: EyeTarget {
                            color: color.map(|t| t.create_view(&Default::default())),
                            depth: if color.is_none() {
                                Some(self.depth.create_view(&Default::default()))
                            } else {
                                None
                            },
                        },
                    }],
                    request_overscan: 1.0,
                    prefetch: None,
                })
                .map_err(|e| render_error("camera frame", e))?;
        }
        Ok(())
    }
}
impl Preview {
    pub(crate) async fn initialize(
        manifest_json: String,
        camera_json: String,
        read: js_sys::Function,
        globe: bool,
        canvas: Option<(web_sys::HtmlCanvasElement, wgpu::TextureFormat)>,
    ) -> Result<Self, PreviewError> {
        let manifest: Manifest = serde_json::from_str(&manifest_json)?;
        manifest.validate()?;
        let camera: Camera = serde_json::from_str(&camera_json)?;
        camera.validate()?;
        let (kernel, renderer) = create_headless_renderer_with_settings(
            camera.width,
            camera.height,
            None,
            renderer_settings(canvas.as_ref().map(|(_, format)| *format)),
        )
        .await
        .map_err(|e| render_error("WebGPU initialization", e))?;
        let surface = crate::display::surface(&renderer, canvas)?;
        let plugins: Vec<Box<dyn Plugin<_>>> = vec![
            Box::new(RenderPlugin),
            Box::new(maplibre::background::BackgroundPlugin),
            Box::new(RasterPlugin::<DefaultRasterTransferables>::default()),
            Box::new(TerrainPlugin::<DefaultDemTransferables>::default()),
            Box::new(
                HeadlessPlugin::new(false)
                    .preserve_tile_sources()
                    .retain_supplied_tiles(),
            ),
        ];
        let mut map = HeadlessMap::new(style(&manifest, globe)?, renderer, kernel, plugins)
            .map_err(|e| render_error("map initialization", e))?;
        load(
            &mut map,
            &manifest,
            BrowserStore::new(&manifest, read),
            globe,
        )
        .await?;
        let anchor = ExternalAnchor {
            position: LatLon::new(manifest.anchor_lat_lon[0], manifest.anchor_lat_lon[1]),
            altitude_meters: 0.0,
        };
        let depth = reference_depth(&map, camera);
        let mut preview = Self {
            map,
            camera,
            anchor,
            tick: 0,
            depth,
            camera_session: None,
            reference: None,
            coverage: crate::coverage::Coverage::new(&manifest),
            manifest,
            globe,
            surface,
        };
        preview.configure_surface();
        Ok(preview)
    }
}
pub(crate) async fn load(
    map: &mut HeadlessMap,
    manifest: &Manifest,
    store: BrowserStore,
    display: bool,
) -> Result<(), PreviewError> {
    let mut imagery = Vec::new();
    let mut elevation = Vec::new();
    for tile in &manifest.tiles {
        let [z, x, y] = tile.xyz;
        let coords = WorldTileCoords::from((x as i32, y as i32, (z as u8).into()));
        if let Some(asset) = &tile.imagery {
            imagery.push(AvailableRasterLayerData {
                coords,
                source_layer: "imagery".into(),
                image: store.image(asset, 512).await?,
            });
        }
        if let Some(asset) = &tile.elevation {
            elevation.push((coords, store.image(asset, 256).await?));
        }
    }
    if display {
        imagery.extend(crate::display_tiles::parents(&imagery));
    }
    map.render_frames_with_terrain(ProcessedLayers::default(), imagery, elevation, 3)
        .map_err(|e| render_error("tile upload", e))
}
fn style(manifest: &Manifest, globe: bool) -> Result<Style, PreviewError> {
    let imax = manifest
        .tiles
        .iter()
        .filter(|t| t.imagery.is_some())
        .map(|t| t.xyz[0])
        .max()
        .unwrap_or(0);
    let imax = if globe { 18 } else { imax };
    let dmax = manifest
        .tiles
        .iter()
        .filter(|t| t.elevation.is_some())
        .map(|t| t.xyz[0])
        .max()
        .unwrap_or(0);
    Ok(serde_json::from_value(
        serde_json::json!({"version":8,"center":[manifest.anchor_lat_lon[1],manifest.anchor_lat_lon[0]],"zoom":12,
        "projection":{"type":if globe {"vertical-perspective"} else {"mercator"}},"terrain":{"source":"dem","exaggeration":1},
        "sources":{"context":{"type":"raster","tiles":["pilotage://context/{z}/{x}/{y}"],"tileSize":512,"maxzoom":0},"imagery":{"type":"raster","tiles":["pilotage://imagery/{z}/{x}/{y}"],"tileSize":512,"maxzoom":imax},
        "dem":{"type":"raster-dem","tiles":["pilotage://dem/{z}/{x}/{y}"],"tileSize":256,"maxzoom":dmax,"encoding":"terrarium"}},
        "layers":[{"id":"background","type":"background","paint":{"background-color":if globe {"#254551"} else {"rgba(0,0,0,0)"}}},{"id":"context","type":"raster","source":"context","paint":{"raster-fade-duration":0}},{"id":"imagery","type":"raster","source":"imagery","paint":{"raster-fade-duration":0}}]}),
    )?)
}

fn renderer_settings(format: Option<wgpu::TextureFormat>) -> RendererSettings {
    let pools = || BufferPoolSizes {
        vertices: 10000,
        indices: 10000,
        feature_metadata: 10000,
        layer_metadata: 1024,
    };
    RendererSettings {
        msaa: Msaa { samples: 1 },
        buffer_pools: pools(),
        symbol_pools: pools(),
        texture_format: Some(format.unwrap_or(wgpu::TextureFormat::Rgba8Unorm)),
        ..Default::default()
    }
}

fn reference_depth(map: &HeadlessMap, camera: Camera) -> wgpu::Texture {
    map.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("browser reference depth"),
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
    })
}

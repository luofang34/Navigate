//! A localization job binds images, depth, calibration, and priors.
use crate::{Error, assets};
use navigate_recording::{Camera, Pose};
use navigate_visual::{Frame, FrameStamp, LocalFrame, MapRevision, PosePrior, ReferenceView};
use navigate_visual::{ReferenceRenderer, RendererIdentity};
use navigate_visual_render::GpuRenderer;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub query: Image,
    pub camera: Camera,
    pub prior: Pose,
    pub position_radius_m: f64,
    pub attitude_radius_rad: f64,
    pub anchor_lat_lon: [f64; 2],
    #[serde(default)]
    pub coordinate_frame: CoordinateFrame,
    pub render_scene: Option<crate::terrain::SceneConfig>,
    pub map_release: String,
    pub map_sha256: String,
    pub references: Vec<Reference>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Image {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub id: String,
    pub image: Option<Image>,
    pub depth: Option<PathBuf>,
    pub depth_sha256: Option<String>,
    pub pose: Pose,
    pub features: Option<PathBuf>,
}
#[derive(Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateFrame {
    #[default]
    AnchorMercator,
    SphericalEnu,
}
#[derive(Default)]
pub struct References {
    renderer: Option<GpuRenderer>,
    pub render_ms: Vec<f64>,
}
impl References {
    pub fn identity(&self) -> Option<RendererIdentity> {
        self.renderer.as_ref().map(ReferenceRenderer::identity)
    }
    pub fn texture_bytes(&self) -> usize {
        self.renderer
            .as_ref()
            .map_or(0, GpuRenderer::resident_texture_bytes)
    }
    pub fn readback_bytes(&self) -> u64 {
        self.renderer
            .as_ref()
            .map_or(0, GpuRenderer::readback_bytes_per_view)
    }
}
impl Job {
    pub fn frame(&self) -> Result<LocalFrame, Error> {
        let [lat, lon] = self.anchor_lat_lon;
        Ok(match self.coordinate_frame {
            CoordinateFrame::AnchorMercator => LocalFrame::anchor_mercator(lat, lon)?,
            CoordinateFrame::SphericalEnu => LocalFrame::spherical_enu(lat, lon)?,
        })
    }
    pub fn references_blocking(&self, root: &Path, max_pixels: usize) -> Result<References, Error> {
        let needed = self.references.iter().any(|r| r.image.is_none());
        let renderer = if needed {
            let config = self
                .render_scene
                .as_ref()
                .ok_or_else(|| Error::Input("generated references require render_scene".into()))?;
            Some(config.load_blocking(
                root,
                self.frame()?,
                MapRevision {
                    release_id: self.map_release.clone(),
                    manifest_sha256: self.map_sha256.clone(),
                },
                self.camera.model(),
                max_pixels,
            )?)
        } else {
            None
        };
        Ok(References {
            renderer,
            render_ms: Vec::new(),
        })
    }
    pub fn query_blocking(&self, root: &Path, max_pixels: usize) -> Result<Frame, Error> {
        let image = assets::image_verified_blocking(
            &root.join(&self.query.path),
            max_pixels,
            Some(&self.query.sha256),
        )?;
        Ok(Frame {
            stamp: FrameStamp {
                sequence: 0,
                capture_time_ns: 0,
            },
            camera: self.camera.model(),
            image,
        })
    }
    pub fn prior(&self) -> Result<PosePrior, Error> {
        Ok(PosePrior {
            pose: self.prior.camera()?,
            position_radius_m: self.position_radius_m,
            attitude_radius_rad: self.attitude_radius_rad,
        })
    }
    pub fn reference_blocking(
        &self,
        entry: &Reference,
        root: &Path,
        max_pixels: usize,
        source: &mut References,
    ) -> Result<ReferenceView, Error> {
        let Some(input) = &entry.image else {
            if entry.depth.is_some() || entry.depth_sha256.is_some() || entry.features.is_some() {
                return Err(Error::Input(
                    "generated references cannot bind file depth or cached image features".into(),
                ));
            }
            let renderer = source
                .renderer
                .as_mut()
                .ok_or_else(|| Error::Input("renderer missing".into()))?;
            let started = std::time::Instant::now();
            let view = renderer.render_blocking(entry.pose.camera()?)?;
            source
                .render_ms
                .push(started.elapsed().as_secs_f64() * 1000.0);
            return Ok(view);
        };
        let depth = entry
            .depth
            .as_ref()
            .ok_or_else(|| Error::Input("reference depth missing".into()))?;
        let depth_sha256 = entry
            .depth_sha256
            .as_ref()
            .ok_or_else(|| Error::Input("reference depth digest missing".into()))?;
        let image = assets::image_verified_blocking(
            &root.join(&input.path),
            max_pixels,
            Some(&input.sha256),
        )?;
        let bytes = assets::read_blocking(&root.join(depth), max_pixels.saturating_mul(4))?;
        assets::verify(&bytes, depth_sha256)?;
        if bytes.len() != image.as_raw().len() * 4 {
            return Err(Error::Input(format!("depth dimensions for {}", entry.id)));
        }
        let depth_m = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Ok(ReferenceView {
            map: MapRevision {
                release_id: self.map_release.clone(),
                manifest_sha256: self.map_sha256.clone(),
            },
            frame: self.frame()?,
            pose: entry.pose.camera()?,
            image,
            depth_m,
        })
    }
}

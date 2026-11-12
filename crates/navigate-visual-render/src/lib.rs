//! Resident imagery and terrain with asynchronous GPU reference rendering,
//! render scenes from navigation data packages, and route keyframes
//! (ADR-0022).
mod atlas;
mod error;
pub mod keyframes;
mod map_raster;
pub mod package_scene;
mod renderer;
mod sparse;
#[cfg(test)]
mod test_package;

pub use atlas::{AtlasLayout, ImageAtlas, TerrainAtlas};
pub use error::RenderError;
pub use keyframes::{KeyframeRendering, KeyframeSpec, build_keyframes};
pub use map_raster::{ground_height_m, map_raster};
pub use package_scene::{SceneRequest, SceneWindow, scene_around};
pub use renderer::{GpuRenderer, GpuScene, GpuViewport, RenderLimits, RenderScene};
pub use sparse::{AdaptiveRenderStats, SparseRayHit, SparseSceneSampler};
pub use wgpu;

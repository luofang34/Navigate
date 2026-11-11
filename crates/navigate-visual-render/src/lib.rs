//! Resident imagery and terrain with asynchronous GPU reference rendering,
//! render scenes from navigation data packages, and route keyframes
//! (ADR-0022).
mod atlas;
mod error;
pub mod keyframes;
pub mod package_scene;
mod renderer;
mod sparse;
#[cfg(test)]
mod test_package;

pub use atlas::{AtlasLayout, ImageAtlas, TerrainAtlas};
pub use error::RenderError;
pub use keyframes::{KeyframeSpec, build_keyframes};
pub use package_scene::{SceneRequest, SceneWindow, scene_around};
pub use renderer::{GpuRenderer, GpuScene, GpuViewport, RenderLimits, RenderScene};
pub use sparse::{AdaptiveRenderStats, SparseRayHit, SparseSceneSampler};
pub use wgpu;

//! Resident imagery and terrain with asynchronous GPU reference rendering.
mod atlas;
mod error;
mod renderer;
mod sparse;

pub use atlas::{AtlasLayout, ImageAtlas, TerrainAtlas};
pub use error::RenderError;
pub use renderer::{GpuRenderer, RenderLimits, RenderScene};
pub use sparse::{AdaptiveRenderStats, SparseRayHit, SparseSceneSampler};

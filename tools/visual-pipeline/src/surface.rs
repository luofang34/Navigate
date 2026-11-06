//! Terrain surface re-exports, reference catalog builder, and horizon attitude estimator.

mod catalog;
pub mod horizon;

pub use catalog::{ContextSurface, build_reference_catalog, build_reference_catalog_zoom};
pub use horizon::{RansacHorizonEstimator, horizon_attitude};
pub use navigate_visual_flight::surface::{
    CatalogTile, GpuQueueRole, GpuSurfaceView, SharedSurfaceGpu, Surface, SurfaceAdaptiveStats,
    SurfaceGpuRenderer, SurfaceRayHit, SurfaceRender, enu_frame, enu_frame_on, gpu,
    mercator_tile_coords,
};

#[cfg(test)]
mod tests;

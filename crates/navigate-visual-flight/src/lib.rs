//! Real-time visual localization and `30 Hz` corridor engine (`navigate-visual-flight`).
//!
//! [`FlightVisualEngine`] processes borrowed [`LiveFrameInput`] buffers in memory without disk I/O.
//! It uses bounded worker queues, shared GPU terrain textures, and multi-hypothesis basin checks.
//! It also propagates delayed-measurement covariance updates in monotonic order.

pub mod anchor;
mod anchor_arrival;
mod anchor_worker;
pub mod canonical;
pub mod corridor_build;
pub mod diagnostics;
pub mod engine;
pub mod error;
pub mod masking;
pub mod nadir;
pub mod pixels;
pub mod place_loader;
pub mod runtime;
pub mod surface;
mod tile_assemble;
pub mod tile_builder;
pub mod tracks;
pub mod types;
pub mod worker;

pub use corridor_build::{CorridorBuildSpec, CorridorBuildStats, build_corridor_bank_with_priors};
pub use diagnostics::{AnchorArrivalRecord, AttemptRecord, FrameDiagnostics};
pub use engine::{
    FlightEngineConfig, FlightEngineResourcePaths, FlightEngineResources, FlightVisualEngine,
};
pub use error::FlightVisualError;
pub use masking::{AmortizedMasker, MaskFrameTelemetry, NIGHT_BYPASS_LUMA_P50};
/// The spectral band of a live frame (ADR-0021).
pub use navigate_camera::SpectralBand;
pub use navigate_visual::fusion_bridge::VisualAttitudeBudget;
pub use pixels::{median_luma_u8, prepare_tracking_buffers, tracking_render_size};
pub use place_loader::{load_place_records, load_search_tier_matcher};
pub use runtime::{
    LoadedMatcher, LoadedXFeat, MatcherRuntimeConfig, XFeatRuntime, XFeatRuntimeConfig,
    default_burn_device,
};
pub use surface::{
    CatalogTile, GpuQueueRole, GpuSurfaceView, SharedSurfaceGpu, Surface, SurfaceAdaptiveStats,
    SurfaceGpuRenderer, SurfaceRayHit, SurfaceRender, enu_frame, enu_frame_on,
    mercator_tile_coords,
};
pub use tile_builder::{TileSynthParams, synthesize_corridor_tiles};
pub use types::{
    AnchorConfidence, AttitudePrior, CalibrationMode, CameraIntrinsics, CameraMountProfile,
    DeadReckonStep, ExecutionPolicy, FlightFrameOutcome, FramePixels, FrameTelemetry,
    FusionEligibility, LiveFrameInput, PlatformKind, SearchPrior, TrackingMode, WorkerHealth,
};

#[cfg(test)]
mod tests;

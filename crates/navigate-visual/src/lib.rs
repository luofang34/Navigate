//! Visual camera positioning from a pose prior and a map reference image.
//!
//! Supply an undistorted [`Frame`], a camera [`PosePrior`], and a [`ReferenceView`]
//! rendered with the frame's intrinsics. [`Localizer`] returns a camera pose or
//! a rejection. Cross-frame correlation is unspecified. The host owns map data,
//! camera calibration, coordinate conversion, and navigation fusion.
//!
//! The default matcher needs overlapping views with similar appearance. It
//! searches near the prior. It does not perform global location retrieval.
//! [`Estimate::geometry_covariance`] excludes map and calibration errors.
//! It must not be used as a navigation integrity bound.
//!
//! # Features
//!
//! The default build uses CPU matching. The optional `gpu` feature exposes
//! `GpuPyramidalMatcher` for hardware compute shaders. Build its documentation
//! with `cargo doc -p navigate-visual --features gpu --open`. Backend selection
//! is explicit. The GPU backend does not silently fall back to CPU matching.
//!
//! # Example
//!
//! A blank image must produce a rejection, even with a valid prior and depth.
//!
//! ```
//! use image::GrayImage;
//! use nalgebra::{UnitQuaternion, Vector3};
//! use navigate_visual::{CameraModel, CameraPose, Frame, FrameStamp, LocalFrame,
//!     Localizer, LocalizerConfig, MapRevision, PosePrior, PyramidalMatcher,
//!     ReferenceView, VisualError};
//!
//! let camera = CameraModel {
//!     width: 64, height: 64, fx: 55.0, fy: 55.0, cx: 31.5, cy: 31.5,
//! };
//! let pose = CameraPose {
//!     position: Vector3::new(0.0, 0.0, 1000.0),
//!     orientation: UnitQuaternion::identity(),
//! };
//! let prior = PosePrior { pose, position_radius_m: 100.0, attitude_radius_rad: 0.1 };
//! let frame = Frame {
//!     stamp: FrameStamp { sequence: 0, capture_time_ns: 1 },
//!     camera, image: GrayImage::new(64, 64),
//! };
//! let reference = ReferenceView {
//!     map: MapRevision { release_id: "selected-release".into(),
//!         manifest_sha256: "a".repeat(64) },
//!     frame: LocalFrame::anchor_mercator(47.0, 11.0)?,
//!     pose, image: GrayImage::new(64, 64), depth_m: vec![1000.0; 64 * 64],
//! };
//! let mut localizer = Localizer::new(PyramidalMatcher, LocalizerConfig::default())?;
//! assert!(matches!(localizer.estimate_blocking(&frame, &reference, &prior),
//!     Err(VisualError::InsufficientMatches { .. })));
//! # Ok::<(), VisualError>(())
//! ```

mod camera;
mod candidates;
mod error;
mod frame;
mod geometry;
mod local_frame;
mod local_scene;
mod localizer;
mod matching;
mod pose_solver;
mod retrieval;

#[cfg(feature = "gpu")]
mod gpu;

pub use camera::{CameraModel, CameraPose, PosePrior};
pub use candidates::{CandidateDecision, CandidateId, CandidateResults};
pub use error::VisualError;
pub use frame::{Frame, FrameStamp, MapRevision, ReferenceView};
pub use geometry::{
    CandidateEvaluation, PoseVerifier, SurfaceTrackUpdate, SurfaceTracks, TrackingMotion,
    TrackingProposal, TrackingReference,
};
pub use local_frame::{LocalFrame, MERCATOR_SPHERE_RADIUS_M};
pub use local_scene::{
    LocalScene, LocalSceneCamera, LocalSceneError, LocalScenePoint, LocalScenePose,
    SceneCoordinateGauge, ScenePointObservation, SceneRefinement, refine_local_scene,
    refine_local_scene_with_gauge,
};
pub use localizer::{Estimate, EstimateQuality, Localizer, LocalizerConfig};
pub use matching::{ImageMatcher, PixelMatch, PointTracker, PyramidalMatcher};
pub use retrieval::{GroundCorrespondence, RetrievalProposal, planar_proposal};

#[cfg(feature = "gpu")]
pub use gpu::GpuPyramidalMatcher;

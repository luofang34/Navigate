//! Pure-Rust visual recording pipeline for photo test sets, terrain pose search,
//! video tracking, semantic segmentation, and `sokoly-debrief` v1 publication.

pub mod clock;
pub mod error;
pub mod matcher_client;
pub mod metadata;
pub mod mosaic;
pub mod pose;
pub mod preview;
pub mod segment_cli;
pub mod stages;
pub mod surface;
pub mod testset;
pub mod util;
pub mod video_ops;

pub use clock::{
    PriorTrack, capture_anchor, movie_duration_s, rotation_from_heading, seed_live_prior,
};
pub use error::PipelineError;
pub use matcher_client::PipelineMatcher;
pub use metadata::{PhotoClaims, extract_photo_claims};
pub use mosaic::{compose_ground_mosaic, compose_survey_debrief, suitability};
pub use pose::{PoseCandidate, PoseScore, candidate_priority, fit_pose_ransac, spatial_holdout};
pub use preview::{export_preview, export_publication, freeze_anchors, reproduce_bundle};
pub use surface::{
    ContextSurface, Surface, build_reference_catalog, build_reference_catalog_zoom,
    horizon_attitude,
};
pub use testset::{build_cases, group_flights, interpolate_flight_priors, prepare_photo_testset};

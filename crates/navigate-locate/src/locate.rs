//! The locate pipeline: map scene, candidate views, matching, pose fit, refinement.
use crate::{
    LocateError,
    candidates::{LookAt, angles},
    scene::{self, SceneRequest, TileSource},
};
use image::GrayImage;
use navigate_visual::{
    AsyncImageMatcher, CameraModel, Estimate, Frame, FrameStamp, LocalizerConfig, PosePrior,
    ReferenceView,
};
use navigate_visual_render::{GpuRenderer, RenderLimits};
use navigate_visual_workflow::{CandidateReport, PairLocalizer, ResourceLimits};

/// One photo and where to look for it.
pub struct LocateRequest {
    /// Grayscale photo; its size must equal the camera size.
    pub image: GrayImage,
    /// Pinhole calibration of the photo.
    pub camera: CameraModel,
    /// Latitude and longitude of the ground at the photo centre.
    pub target: [f64; 2],
    /// Candidate grid around the target.
    pub look_at: LookAt,
    /// Map area and tile zooms; the anchor is the target.
    pub scene: SceneRequest,
    /// Prior radii for the pose verifier around each candidate.
    pub position_radius_m: f64,
    /// Attitude radius of the prior, radians.
    pub attitude_radius_rad: f64,
}

/// A stage of the pipeline, for a progress display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    /// Map tiles fetched and decoded.
    Tiles {
        /// Tiles finished.
        done: usize,
        /// Tiles in the area.
        total: usize,
    },
    /// Candidate views rendered.
    Rendering {
        /// Views finished.
        done: usize,
        /// Candidate views.
        total: usize,
    },
    /// Candidate views matched with the photo.
    Matching {
        /// Candidate views.
        total: usize,
    },
    /// The best pose is checked against a view rendered at that pose.
    Refining,
}

/// Maximum number of candidate views held in host memory per matching batch.
const CANDIDATE_BATCH_SIZE: usize = 8;

/// The located camera.
#[derive(Clone)]
pub struct Located {
    /// Accepted estimate, in the local frame of the target.
    pub estimate: Estimate,
    /// Other accepted candidate estimates sorted by descending inlier count.
    pub alternatives: Vec<Estimate>,
    /// Latitude, longitude and height (terrain datum) of the camera.
    pub geodetic: [f64; 3],
    /// Heading, downward pitch and roll of the optical axis, degrees.
    pub angles_deg: [f64; 3],
    /// True when a view rendered at the first estimate confirmed it.
    pub refined: bool,
}

/// Result of one localization.
#[derive(Clone)]
pub enum Outcome {
    /// An accepted pose.
    Found(Box<Located>),
    /// No candidate passed the geometry checks.
    NotFound {
        /// Candidates evaluated.
        candidates: usize,
        /// Most correspondences of one candidate before geometry checks.
        best_matches: usize,
    },
}

/// Locate the camera of one photo.
///
/// # Errors
///
/// Returns an error for an invalid request, a tile or GPU failure, or a matcher
/// failure. A pose that the geometry rejects is [`Outcome::NotFound`].
pub async fn locate<S: TileSource, M: AsyncImageMatcher>(
    source: &S,
    matcher: M,
    request: LocateRequest,
    mut progress: impl FnMut(Progress),
) -> Result<Outcome, LocateError> {
    if request.image.dimensions() != (request.camera.width, request.camera.height) {
        return Err(LocateError::Request(
            "image size differs from the camera size",
        ));
    }
    let mut scene_request = request.scene;
    scene_request.anchor = request.target;
    let scene = scene::build(source, &scene_request, |done, total| {
        progress(Progress::Tiles { done, total })
    })
    .await?;
    let ground = scene::height_at(&scene.terrain, request.target).ok_or(LocateError::NoTerrain)?;
    let frame = scene.frame;
    let target = frame.local([request.target[0], request.target[1], ground])?;
    let limits = RenderLimits {
        scene_bytes: request.scene.max_bytes,
        pixels: (request.camera.width * request.camera.height) as usize,
        far_m: 40_000.0,
        steps: 100,
        refinements: 12,
    };
    let mut renderer = GpuRenderer::new(scene, request.camera, limits).await?;
    let candidates = request.look_at.poses(target);
    let central = candidates
        .get(candidates.len() / 2)
        .map(|c| c.pose)
        .ok_or(LocateError::Request("no candidates"))?;
    let query = Frame {
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 0,
        },
        camera: request.camera,
        image: request.image,
    };
    let pixels = (request.camera.width * request.camera.height) as usize;
    let batch_cap = candidates.len().clamp(1, CANDIDATE_BATCH_SIZE);
    let resource = ResourceLimits {
        image_pixels: pixels,
        features_per_image: 1,
        candidates: batch_cap,
        // Each view holds one byte of image and four bytes of depth per pixel.
        reference_bytes: batch_cap * pixels * 5,
        query_feature_bytes: 1,
    };
    let mut localizer = PairLocalizer::new(matcher, LocalizerConfig::default(), resource)?;
    let prior = PosePrior {
        pose: central,
        position_radius_m: request.position_radius_m,
        attitude_radius_rad: request.attitude_radius_rad,
    };
    let total_candidates = candidates.len();
    let mut rendered_done = 0_usize;
    let mut all_reports = Vec::with_capacity(total_candidates);
    for chunk in candidates.chunks(CANDIDATE_BATCH_SIZE) {
        let mut batch_views = Vec::with_capacity(chunk.len());
        for candidate in chunk {
            batch_views.push(renderer.render(candidate.pose).await?);
            rendered_done += 1;
            progress(Progress::Rendering {
                done: rendered_done,
                total: total_candidates,
            });
        }
        progress(Progress::Matching {
            total: total_candidates,
        });
        let calibrated: Vec<(&ReferenceView, &CameraModel)> =
            batch_views.iter().map(|v| (v, &request.camera)).collect();
        let batch_report = localizer
            .estimate_calibrated(&query, &prior, &calibrated)
            .await?;
        all_reports.extend(batch_report.candidates);
    }
    let mut accepted = accepted_sorted(&all_reports);
    let Some(first) = accepted.first().cloned() else {
        let best_matches = all_reports.iter().map(|c| c.matches).max().unwrap_or(0);
        return Ok(Outcome::NotFound {
            candidates: all_reports.len(),
            best_matches,
        });
    };
    let alternatives = accepted.split_off(1);
    progress(Progress::Refining);
    let view = renderer.render(first.pose).await?;
    let refine_prior = PosePrior {
        pose: first.pose,
        ..prior
    };
    let refined = localizer
        .estimate_calibrated(&query, &refine_prior, &[(&view, &request.camera)])
        .await?;
    let (estimate, confirmed) = match accepted_sorted(&refined.candidates).into_iter().next() {
        Some(better) => (better, true),
        None => (first, false),
    };
    let geodetic = frame.geodetic(estimate.pose.position);
    let angles_deg = angles(&estimate.pose.orientation);
    Ok(Outcome::Found(Box::new(Located {
        estimate,
        alternatives,
        geodetic,
        angles_deg,
        refined: confirmed,
    })))
}

/// Accepted candidate estimates sorted by descending inliers and ascending RMS error.
fn accepted_sorted(candidates: &[CandidateReport]) -> Vec<Estimate> {
    let mut list: Vec<Estimate> = candidates
        .iter()
        .filter_map(|c| c.decision.as_ref().ok().cloned())
        .collect();
    list.sort_by(|a, b| {
        b.quality.inliers.cmp(&a.quality.inliers).then_with(|| {
            a.quality
                .reprojection_rms_px
                .total_cmp(&b.quality.reprojection_rms_px)
        })
    });
    list
}

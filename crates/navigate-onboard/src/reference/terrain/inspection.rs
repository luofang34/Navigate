//! What the terrain means did with the latest frame of one camera, kept
//! when the host asks for it so that it can show the match: the reference
//! view, the frame as it was matched, the correspondences with the inliers
//! of the fit, the solved pose, and the fix offered to the filter.
//!
//! Keeping it changes nothing that the navigator computes: the match runs
//! the same way, and the record only holds what the frame already made.

use std::sync::Arc;

use image::GrayImage;
use navigate_contract::{GeodeticPosition, MonotonicNanos, SymmetricCov3};
use navigate_fusion::{Observation, ObservationValue};
use navigate_visual::{CameraPose, Estimate, LocalFrame, MatchEvidence, ReferenceView};

use super::config::WorkingCamera;
use crate::Outcome;

/// One correspondence of a match, in pixel centres of the working camera:
/// the reference view and the frame as matched share its size, which is
/// not the size of the frame that the camera sent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchPair {
    /// Pixel in the reference view, `[x, y]`.
    pub reference_px: [f64; 2],
    /// Pixel in the frame, `[x, y]`.
    pub frame_px: [f64; 2],
    /// Whether the fit that solved the pose kept the pair. Every pair is
    /// `false` when the frame was refused before a fit.
    pub inlier: bool,
}

/// A camera pose in the local frame of a reference view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewPose {
    /// Anchor of the local frame: latitude and longitude, degrees.
    pub anchor_deg: [f64; 2],
    /// Position in the local frame (east, north, up), metres.
    pub position_m: [f64; 3],
    /// Rotation from the eye axes (right, up, back) into the local frame,
    /// as `[w, x, y, z]`.
    pub orientation: [f64; 4],
    /// Latitude and longitude in degrees and height above the ellipsoid
    /// in metres of the position.
    pub geodetic: [f64; 3],
}

impl ViewPose {
    fn of(frame: &LocalFrame, pose: &CameraPose) -> Self {
        let q = pose.orientation.quaternion();
        Self {
            anchor_deg: frame.anchor(),
            position_m: pose.position.into(),
            orientation: [q.w, q.i, q.j, q.k],
            geodetic: frame.geodetic(pose.position),
        }
    }
}

/// The camera pose that a match solved, with the evidence for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolvedPose {
    /// The solved camera, in the local frame of the reference view.
    pub camera: ViewPose,
    /// Matches with a reference depth.
    pub depth_matches: usize,
    /// Inliers of the final fit.
    pub inliers: usize,
    /// Inliers that are spatially separated.
    pub spatial_support: usize,
    /// Root mean square reprojection error of the inliers, pixels.
    pub reprojection_rms_px: f64,
}

/// The position fix that a match offered to the filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OfferedFix {
    /// The position of the body, the lever arm taken off.
    pub position: GeodeticPosition,
    /// Its covariance without the map error of its cell, local north,
    /// east, down, square metres: the relief, calibration and image terms.
    pub covariance_ned: SymmetricCov3,
    /// The map error of its cell, declared to the filter as a constant
    /// error that the fixes of the cell share: the 1-sigma of the
    /// registration north and east and of the terrain height, metres.
    pub map_sigma_m: Option<[f64; 3]>,
}

/// A grey image held by the record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Luma<'a> {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major pixels, one byte each.
    pub pixels: &'a [u8],
}

impl<'a> Luma<'a> {
    fn of(image: &'a GrayImage) -> Self {
        Self {
            width: image.width(),
            height: image.height(),
            pixels: image.as_raw(),
        }
    }
}

/// The two images of a match and its correspondences.
#[derive(Clone)]
struct Compared {
    view: Arc<ReferenceView>,
    frame: Arc<GrayImage>,
    working: WorkingCamera,
    pairs: Arc<[MatchPair]>,
}

impl std::fmt::Debug for Compared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compared")
            .field("view", &self.view.image.dimensions())
            .field("frame", &self.frame.dimensions())
            .field("working", &self.working)
            .field("pairs", &self.pairs.len())
            .finish()
    }
}

/// What one frame made, gathered while it is taken.
#[derive(Debug, Clone, Default)]
pub(super) struct Pending {
    compared: Option<Compared>,
    solved: Option<SolvedPose>,
    fix: Option<OfferedFix>,
}

impl Pending {
    /// Keeps the images of a match and its correspondences.
    pub(super) fn compared(
        &mut self,
        view: Arc<ReferenceView>,
        frame: GrayImage,
        working: WorkingCamera,
        evidence: &MatchEvidence,
    ) {
        let pairs = evidence
            .matches
            .iter()
            .enumerate()
            .map(|(index, pair)| MatchPair {
                reference_px: pair.reference.into(),
                frame_px: pair.query.into(),
                inlier: evidence.inliers.get(index).copied().unwrap_or(false),
            })
            .collect();
        self.compared = Some(Compared {
            view,
            frame: Arc::new(frame),
            working,
            pairs,
        });
    }

    /// Keeps the solved pose of an estimate.
    pub(super) fn solved(&mut self, estimate: &Estimate) {
        let quality = estimate.quality;
        self.solved = Some(SolvedPose {
            camera: ViewPose::of(&estimate.frame, &estimate.pose),
            depth_matches: quality.depth_matches,
            inliers: quality.inliers,
            spatial_support: quality.spatial_support,
            reprojection_rms_px: quality.reprojection_rms_px,
        });
    }

    /// Keeps the fix offered to the filter.
    pub(super) fn offered(&mut self, observation: &Observation) {
        if let ObservationValue::PositionFix {
            position,
            covariance,
        } = observation.value
        {
            self.fix = Some(OfferedFix {
                position,
                covariance_ned: covariance,
                map_sigma_m: observation.correlated.map(|c| c.stationary_sigma),
            });
        }
    }

    /// The record of the frame acquired at `acquired_at`, with what the
    /// navigator did with it.
    pub(super) fn finish(
        self,
        acquired_at: MonotonicNanos,
        outcome: Outcome,
        path: Option<&'static str>,
    ) -> TerrainMatch {
        TerrainMatch {
            acquired_at,
            outcome,
            path,
            compared: self.compared,
            solved: self.solved,
            fix: self.fix,
        }
    }
}

/// What the terrain means did with the latest frame of one camera. A frame
/// that was not matched, for example because each map cell that it can
/// show already gave a fix, has an outcome and no images.
#[derive(Debug, Clone)]
pub struct TerrainMatch {
    acquired_at: MonotonicNanos,
    outcome: Outcome,
    path: Option<&'static str>,
    compared: Option<Compared>,
    solved: Option<SolvedPose>,
    fix: Option<OfferedFix>,
}

impl TerrainMatch {
    /// Acquisition time of the frame.
    #[must_use]
    pub const fn acquired_at(&self) -> MonotonicNanos {
        self.acquired_at
    }

    /// What the navigator did with the frame, as its acceptance record
    /// states it.
    #[must_use]
    pub const fn outcome(&self) -> &Outcome {
        &self.outcome
    }

    /// The path of the reference view, a stored keyframe or a render, when
    /// one was made.
    #[must_use]
    pub const fn path(&self) -> Option<&'static str> {
        self.path
    }

    /// The camera of the reference view and of the match.
    #[must_use]
    pub fn working_camera(&self) -> Option<WorkingCamera> {
        self.compared.as_ref().map(|c| c.working)
    }

    /// The reference view that the frame was matched against.
    #[must_use]
    pub fn reference(&self) -> Option<Luma<'_>> {
        self.compared.as_ref().map(|c| Luma::of(&c.view.image))
    }

    /// The pose of the reference view's camera: the believed pose of the
    /// frame for a render, the stored pose for a keyframe.
    #[must_use]
    pub fn reference_pose(&self) -> Option<ViewPose> {
        self.compared
            .as_ref()
            .map(|c| ViewPose::of(&c.view.frame, &c.view.pose))
    }

    /// The frame as it was matched: resampled to the working camera.
    #[must_use]
    pub fn frame(&self) -> Option<Luma<'_>> {
        self.compared.as_ref().map(|c| Luma::of(&c.frame))
    }

    /// The correspondences of the match; empty when no match ran.
    #[must_use]
    pub fn pairs(&self) -> &[MatchPair] {
        self.compared.as_ref().map_or(&[], |c| &c.pairs)
    }

    /// The pose that the match solved, when the fit was accepted.
    #[must_use]
    pub const fn solved(&self) -> Option<SolvedPose> {
        self.solved
    }

    /// The fix offered to the filter, when the match gave one.
    #[must_use]
    pub const fn fix(&self) -> Option<OfferedFix> {
        self.fix
    }
}

//! Optional typed diagnostic records for CLI debriefing (`tools/visual-pipeline`).
//!
//! The `30 Hz` live flight path (`process_frame`) does not allocate these records.

use crate::types::{AnchorConfidence, FusionEligibility};
use navigate_locate::corridor::FilterIntegritySnapshot;
use navigate_visual::{PoseCandidate, PoseScore};

/// Typed record of one pose-solver or seed-refinement attempt for offline debriefing.
#[derive(Clone, Debug)]
pub struct AttemptRecord {
    /// Solver or matcher backend label.
    pub backend: &'static str,
    /// Seed index within the anchor sweep.
    pub seed: usize,
    /// Refinement step index (`0` = initial seed, `1` = reproject, `>= 2` = polish).
    pub step: usize,
    /// Sketch score or focal multiplier associated with the attempt.
    pub sketch_score: f32,
    /// Candidate correspondence pair count.
    pub pairs: usize,
    /// Pose score when `PnP` produced a candidate.
    pub score: Option<PoseScore>,
    /// Virtual-view landmark count when evaluated against the corridor bank.
    pub virtual_landmarks: Option<usize>,
    /// Visible tile count when evaluated against the corridor bank.
    pub visible_tiles: Option<usize>,
    /// Focal length multiplier when evaluated during focal sweep.
    pub focal_multiplier: Option<f64>,
}

/// Typed record of one completed background or synchronous anchor job for offline debriefing.
#[derive(Clone, Debug)]
pub struct AnchorArrivalRecord {
    /// Sequence number of the frame that captured the anchor query.
    pub sent_frame_seq: usize,
    /// Diagnostic label of the frame that captured the anchor query.
    pub sent_frame_label: String,
    /// Sequence number of the frame where the anchor arrived (`None` if drained at shutdown).
    pub arrived_frame_seq: Option<usize>,
    /// Diagnostic label of the frame where the anchor arrived (`None` if drained at shutdown).
    pub arrived_frame_label: Option<String>,
    /// Number of odometry frames across which the anchor pose was propagated.
    pub propagated_frames: usize,
    /// Capture lag (`t_arrival - t_capture`) in nanoseconds.
    pub capture_lag_ns: u64,
    /// True when the anchor passed geometry and physical plausibility gates.
    pub accepted: bool,
    /// Basin verification confidence when accepted.
    pub confidence: Option<AnchorConfidence>,
    /// True when the anchor was admitted as an independent EKF fix.
    pub independent_ekf_fix: bool,
    /// Fusion eligibility classification.
    pub fusion_eligibility: FusionEligibility,
    /// Classified map cell `[east_idx, north_idx]`.
    pub map_cell: Option<[i64; 2]>,
    /// Earlier frame identifier when sharing a visited map cell.
    pub earlier_anchor_id: Option<String>,
    /// Active foreground track count after applying the anchor.
    pub active_tracks: usize,
    /// True when landmarks were merged into an already healthy foreground track.
    pub merged_into_healthy_foreground: bool,
    /// Anchor worker execution latency in milliseconds.
    pub anchor_ms: f64,
    /// Attempt records collected during the anchor search.
    pub attempts: Vec<AttemptRecord>,
}

/// Full offline debriefing diagnostics for one frame (`process_frame_with_diagnostics`).
#[derive(Clone, Debug)]
pub struct FrameDiagnostics {
    /// Accepted or best rejected `PoseCandidate` in `render_size` coordinates.
    pub candidate: Option<(usize, usize, PoseCandidate)>,
    /// Render resolution `[width, height]` used for `candidate`.
    pub render_size: [u32; 2],
    /// Earlier frame identifier if this frame landed in a previously visited map cell.
    pub earlier_anchor_id: Option<String>,
    /// Locked pixel focal length if estimated by an uncalibrated sweep.
    pub locked_focal_px: Option<f64>,
    /// Navigation filter integrity snapshot after the frame.
    pub filter_snapshot: FilterIntegritySnapshot,
    /// Foreground or inline solver attempt records for this frame.
    pub attempts: Vec<AttemptRecord>,
    /// Background anchor arrivals processed on this frame.
    pub anchor_arrivals: Vec<AnchorArrivalRecord>,
    /// Tile build durations in milliseconds harvested on this frame.
    pub tile_build_ms: Vec<f64>,
    /// Count of tiles synthesized synchronously on this frame.
    pub sync_tiles_built: usize,
}

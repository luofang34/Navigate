//! The names in the acceptance records of the terrain means. A host counts
//! frames by these names, so a name never changes; a new outcome gets a
//! new name.

use navigate_visual::VisualError;

/// The measurement name of each record of a terrain frame.
pub(crate) const MEASUREMENT: &str = "terrain_position";

/// Context: the reference view was a stored keyframe.
pub(crate) const KEYFRAME: &str = "keyframe";
/// Context: the navigator rendered the reference view from the package.
pub(crate) const RENDER: &str = "render";

/// The frame has no structure that a match can use: it is flat, or it has
/// only the contrast of noise.
pub(crate) const FEATURELESS_FRAME: &str = "featureless_frame";
/// The frame has structure, but less contrast than the matcher states its
/// error for honestly.
pub(crate) const FAINT_FRAME: &str = "faint_frame";
/// The acquisition time of the frame is after its delivery time.
pub(crate) const ACQUIRED_AFTER_DELIVERY: &str = "acquired_after_delivery";
/// The frame came later after its exposure than the configured bound.
pub(crate) const CAPTURE_LAG: &str = "capture_lag";
/// The navigator has no solution with an attitude.
pub(crate) const NO_SOLUTION: &str = "no_solution";
/// The believed position is outside the range of a local frame.
pub(crate) const BELIEF_OUT_OF_RANGE: &str = "belief_out_of_range";
/// Too little of the reference view shows the map.
pub(crate) const OUTSIDE_COVERAGE: &str = "outside_coverage";
/// The renderer refused the reference view.
pub(crate) const RENDER_REFUSED: &str = "render_refused";
/// The matched ground is outside the grid scheme.
pub(crate) const OUTSIDE_GRID: &str = "outside_grid";
/// An earlier fix carries the map error of the cell.
pub(crate) const SHARED_MAP_ERROR: &str = "shared_map_error";
/// The ledger of map cells is full.
pub(crate) const UNTRACKED_MAP_CELL: &str = "untracked_map_cell";
/// The solved pose gave no usable position fix.
pub(crate) const FIX_REFUSED: &str = "fix_refused";
/// Another place matches the frame about as well as the best: the
/// acquisition of a wide belief does not choose.
pub(crate) const ACQUISITION_AMBIGUOUS: &str = "acquisition_ambiguous";
/// No place within the acquisition radius matches the frame.
pub(crate) const ACQUISITION_WEAK: &str = "acquisition_weak";
/// The acquisition of a wide belief goes on over the next frames.
pub(crate) const ACQUISITION_PENDING: &str = "acquisition_pending";
/// The fix of an acquisition waits for the next acquisition to agree.
pub(crate) const ACQUISITION_UNCONFIRMED: &str = "acquisition_unconfirmed";

/// The name of a localizer rejection.
pub(crate) const fn rejection(error: &VisualError) -> &'static str {
    match error {
        VisualError::InsufficientMatches { .. } => "insufficient_matches",
        VisualError::InsufficientSpatialSupport { .. } => "insufficient_spatial_support",
        VisualError::DegenerateGeometry => "degenerate_geometry",
        VisualError::OutsidePrior { .. } => "outside_prior",
        VisualError::FrameOrder { .. } => "frame_order",
        VisualError::Backend { .. } => "matcher_failed",
        VisualError::Invalid { .. } | VisualError::Dimensions { .. } => "invalid_view",
    }
}

#[cfg(test)]
mod tests;

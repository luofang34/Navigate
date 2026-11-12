//! Acquisition of a position that a frame-sized search cannot reach: the
//! frame, laid on the ground below the believed eye, is correlated with an
//! orthographic raster of the map about the belief.
//!
//! The tracker of a reference view reaches a few hundred metres; a belief
//! kilometres wide, as after a long gap of fixes, needs the offset found
//! first. Both rasters hold the mean luma of square ground cells, east and
//! north about their centre. The search scores each offset by normalized
//! cross-correlation, first at twice the cell over the whole radius, then
//! at the cell about the best coarse peaks, and finds an offset only when
//! its peak stands clear of every other: repeated ground gives a peak per
//! repetition, and the search then says so instead of choosing one.

//!
//! A believed heading that is off rotates the frame on the ground: the
//! search lays the frame at a few headings about the believed one and
//! finds the heading with the offset. Its work is counted in cell pairs
//! and can be spread over calls, so that a host bounds the time of one
//! frame.

mod raster;
mod search;

use nalgebra::Vector2;

pub use raster::GroundRaster;
pub use search::{AcquisitionSearch, MapPyramid};

use crate::VisualError;

/// What the search requires of a peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcquisitionConfig {
    /// Smallest correlation of an offset that is found.
    pub min_score: f64,
    /// Smallest lead of the best peak over the best peak farther than
    /// `exclusion_m` from it: a place within this of the best is another
    /// contender.
    pub min_margin: f64,
    /// Distance beyond which another peak is another place, metres.
    pub exclusion_m: f64,
    /// Coarse peaks searched again at the cell.
    pub peaks: usize,
}

impl Default for AcquisitionConfig {
    /// Thresholds measured over farmland and over regular and irregular
    /// town blocks from 1760 m: a true peak scores 0.43 to 0.98 with a
    /// lead of at least 0.2; a wrong one leads by a few hundredths.
    fn default() -> Self {
        Self {
            min_score: 0.3,
            min_margin: 0.15,
            exclusion_m: 300.0,
            peaks: 8,
        }
    }
}

/// One scored place of the frame on the map: the east and north offset
/// from the centre of the map raster to the point below the eye, and the
/// rotation of the frame about the local up from its believed heading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    /// East and north offset, metres.
    pub offset_m: Vector2<f64>,
    /// Rotation of the laid frame about the local up, radians,
    /// counter-clockwise seen from above.
    pub rotation_rad: f64,
    /// Normalized cross-correlation, `[-1, 1]`.
    pub score: f64,
}

/// What the search found.
#[derive(Clone, Debug, PartialEq)]
pub enum Acquisition {
    /// One place stands clear of every other.
    Found(Peak),
    /// Places farther apart than the exclusion distance score within the
    /// margin of the best, best first: the ground repeats, or the frame is
    /// not distinctive at the cell. Each is one place at its best rotation.
    Ambiguous(Vec<Peak>),
    /// No place correlates enough, or none overlaps the map.
    Weak(Option<Peak>),
}

/// Searches `map` for `frame` at the offsets within `radius_m` of the map
/// centre, with the frame as laid, in one call.
///
/// # Errors
/// As [`AcquisitionSearch::new`].
pub fn acquire(
    frame: &GroundRaster,
    map: &GroundRaster,
    radius_m: f64,
    config: &AcquisitionConfig,
) -> Result<Acquisition, VisualError> {
    let frames = vec![(0.0, frame.clone())];
    let map = std::sync::Arc::new(MapPyramid::new(map.clone()));
    let mut search = AcquisitionSearch::new(frames, map, Vector2::zeros(), radius_m, config)?;
    Ok(loop {
        if let Some(done) = search.advance(u64::MAX) {
            break done;
        }
    })
}

#[cfg(test)]
mod tests;

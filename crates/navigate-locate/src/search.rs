//! Search a region for the camera of one photo with a place index, then locate it.
//!
//! The search plans hypotheses from the priors, lays the photo on flat ground for
//! each, describes the projections with a place encoder, ranks the index records of
//! the planned cells, checks the best places against map windows, and runs the
//! locate pipeline once at the best place. A browser Worker and a native host run
//! the same code with their own encoder, index source and tile source.
mod check;
mod finish;
mod horizon;
mod project;
mod rank;
mod similarity;

pub use check::{Check, zoom_for};
pub use horizon::{Horizon, horizon};
pub use project::{View, canvas as search_canvas};

use crate::{LocateError, Outcome, Progress, TileSource};
use image::RgbImage;
use navigate_place::{
    Canvas, Fallback, IndexManifest, Ranked, ScalePlan, SearchRequest, Tokens, plan,
};
use navigate_visual::AsyncImageMatcher;
use std::future::Future;

/// Inliers that make a checked place acceptable.
const ACCEPT_INLIERS: usize = 12;
/// Checked places that the locate step tries, best first.
const LOCATE_TRIES: usize = 3;
/// Uncertainty of a camera angle read from the horizon, degrees.
const HORIZON_SIGMA_DEG: f64 = 4.0;
/// Inliers that end the checks early.
const STRONG_INLIERS: usize = 30;

/// Port to a place descriptor model, for example DINOv2 on burn or ONNX Runtime.
pub trait PlaceEncoder {
    /// The tokens of each canvas, in order.
    fn tokens(&mut self, canvases: &[Canvas]) -> impl Future<Output = Result<Vec<Tokens>, String>>;
}

/// Port to the files of one place index, by their path in the manifest.
pub trait IndexSource {
    /// The bytes of one file.
    fn read(&self, path: &str) -> impl Future<Output = Result<Vec<u8>, String>>;
}

/// A stage of the search, for a progress display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchProgress {
    /// Photo projections described.
    Describing {
        /// Projections finished.
        done: usize,
        /// Projections planned.
        total: usize,
    },
    /// Index shards read.
    Index {
        /// Shards read.
        done: usize,
        /// Shards planned.
        total: usize,
    },
    /// Ranked places checked against map windows.
    Checking {
        /// Places checked.
        done: usize,
        /// Places to check at most.
        total: usize,
    },
    /// The locate pipeline at the best place.
    Locating(Progress),
}

/// Result of one search.
#[derive(Clone)]
pub enum SearchOutcome {
    /// A place passed the check and the locate pipeline accepted a pose there.
    Found {
        /// The located camera.
        located: Box<crate::Located>,
        /// Inliers of the check at the place.
        inliers: usize,
        /// Places checked.
        checked: usize,
    },
    /// No place passed, or the locate pipeline rejected the best one.
    NotFound {
        /// Places ranked.
        ranked: usize,
        /// Places checked.
        checked: usize,
        /// Most inliers of one check.
        best_inliers: usize,
    },
    /// The index does not serve the request; the caller runs the fallback.
    Fallback(Fallback),
}

/// The map and index ports of a search.
pub struct SearchSources<'a, T, I> {
    /// Imagery and terrain tiles for the checks and the locate step.
    pub tiles: &'a T,
    /// The files of the place index.
    pub index: &'a I,
    /// The manifest of the place index.
    pub manifest: &'a IndexManifest,
}

/// One query hypothesis: the view, its crop size, and the footprint centre in metres
/// east and north of the point below the camera.
struct Hypothesis {
    scale: usize,
    view: View,
    footprint: [f64; 2],
}

/// Search the region of `request` for the camera of `photo`.
///
/// # Errors
///
/// Returns an error for an index file or tile failure, an encoder or matcher
/// failure, or an invalid index.
pub async fn search<
    T: TileSource,
    I: IndexSource,
    E: PlaceEncoder,
    M: AsyncImageMatcher + Clone,
>(
    sources: &SearchSources<'_, T, I>,
    encoder: &mut E,
    mut matcher: M,
    photo: &RgbImage,
    request: &SearchRequest,
    mut progress: impl FnMut(SearchProgress),
) -> Result<SearchOutcome, LocateError> {
    let SearchSources {
        tiles,
        index,
        manifest,
    } = *sources;
    let (request, planned) = planned(manifest, photo, request);
    let request = &request;
    if planned.scales.is_empty() {
        return Ok(SearchOutcome::Fallback(planned.fallback));
    }
    let (hypotheses, canvases) = hypotheses(photo, request, &planned.scales);
    let descriptors = rank::describe_all(
        index,
        manifest,
        &planned.scales,
        encoder,
        &hypotheses,
        &canvases,
        &mut progress,
    )
    .await?;
    let ranked = rank::rank_all(
        index,
        manifest,
        &planned.scales,
        request,
        &hypotheses,
        &descriptors,
        &mut progress,
    )
    .await?;
    let gray = image::DynamicImage::ImageRgb8(photo.clone()).to_luma8();
    let checks = Checks {
        ranked: &ranked,
        hypotheses: &hypotheses,
        scales: &planned.scales,
    };
    let (accepted, checked, best_inliers) = checks
        .run(tiles, &mut matcher, &gray, &mut progress)
        .await?;
    let (small, focal_px) = finish::working_size(&gray, request.focal_px);
    for (found, h) in accepted.iter().take(LOCATE_TRIES) {
        let outcome = finish::locate_at(
            tiles,
            matcher.clone(),
            &small,
            &hypotheses[*h],
            found,
            focal_px,
            &mut progress,
        )
        .await?;
        if let Outcome::Found(located) = outcome {
            return Ok(SearchOutcome::Found {
                located,
                inliers: found.inliers,
                checked,
            });
        }
    }
    Ok(SearchOutcome::NotFound {
        ranked: ranked.len(),
        checked,
        best_inliers,
    })
}

/// The request with a camera angle from the photo's horizon when it has none, and
/// its plan over the scales and cells of the index.
fn planned(
    manifest: &IndexManifest,
    photo: &RgbImage,
    request: &SearchRequest,
) -> (SearchRequest, navigate_place::SearchPlan) {
    let covered: Vec<(f64, Vec<navigate_place::Cell>)> = manifest
        .scales
        .iter()
        .map(|s| (s.side_m, s.shards.iter().map(|e| e.cell).collect()))
        .collect();
    let mut request = request.clone();
    if request.pitch.is_none()
        && let Some(h) = horizon::horizon(photo, request.focal_px)
    {
        request.pitch = Some(navigate_place::PitchPrior {
            deg: h.pitch_deg,
            sigma_deg: HORIZON_SIGMA_DEG,
        });
    }
    let plan = plan(&request, &covered);
    (request, plan)
}

/// Every view of the plan, with the photo laid on flat ground for it.
fn hypotheses(
    photo: &RgbImage,
    request: &SearchRequest,
    scales: &[ScalePlan],
) -> (Vec<Hypothesis>, Vec<Canvas>) {
    let size = photo.dimensions();
    let (mut hypotheses, mut canvases) = (Vec::new(), Vec::new());
    for (scale, part) in scales.iter().enumerate() {
        for &(height_m, pitch_deg) in &part.views {
            for &heading_deg in &part.headings_deg {
                let view = View {
                    height_m,
                    pitch_deg,
                    heading_deg,
                    focal_px: request.focal_px,
                };
                let Some(footprint) = view.footprint_centre(size) else {
                    continue;
                };
                canvases.push(project::canvas(photo, &view, footprint, part.side_m));
                hypotheses.push(Hypothesis {
                    scale,
                    view,
                    footprint,
                });
            }
        }
    }
    (hypotheses, canvases)
}

/// The ranked places and their hypotheses, to check against map windows.
struct Checks<'a> {
    ranked: &'a [(Ranked, usize)],
    hypotheses: &'a [Hypothesis],
    scales: &'a [ScalePlan],
}

impl Checks<'_> {
    /// The accepted checks, most inliers first, the places checked, and the most
    /// inliers of one check.
    async fn run<T: TileSource, M: AsyncImageMatcher>(
        &self,
        tiles: &T,
        matcher: &mut M,
        gray: &image::GrayImage,
        progress: &mut impl FnMut(SearchProgress),
    ) -> Result<(Vec<(check::Check, usize)>, usize, usize), LocateError> {
        let mut accepted: Vec<(check::Check, usize)> = Vec::new();
        let (mut checked, mut best_inliers) = (0, 0);
        for (place, h) in self.ranked {
            let hypothesis = &self.hypotheses[*h];
            let side_m = self.scales[hypothesis.scale].side_m;
            // An unknown heading leaves up to half a heading step of error, which an
            // oblique view does not tolerate: check the neighbouring headings too.
            let steps: &[f64] = if self.scales[hypothesis.scale].headings_deg.len() > 1 {
                &[0.0, -15.0, 15.0]
            } else {
                &[0.0]
            };
            let views: Vec<View> = steps
                .iter()
                .map(|d| View {
                    heading_deg: hypothesis.view.heading_deg + d,
                    ..hypothesis.view
                })
                .collect();
            let result = check::check(tiles, matcher, gray, &views, place.centre, side_m).await?;
            checked += 1;
            let inliers = result.map_or(0, |c| c.inliers);
            tracing::debug!(place = ?place.centre, score = place.score, heading = hypothesis.view.heading_deg,
                pitch = hypothesis.view.pitch_deg, height = hypothesis.view.height_m, inliers, "checked place");
            progress(SearchProgress::Checking {
                done: checked,
                total: self.ranked.len(),
            });
            best_inliers = best_inliers.max(inliers);
            if let Some(c) = result.filter(|c| c.inliers >= ACCEPT_INLIERS) {
                accepted.push((c, *h));
                if c.inliers >= STRONG_INLIERS {
                    break;
                }
            }
        }
        accepted.sort_by_key(|(c, _)| std::cmp::Reverse(c.inliers));
        Ok((accepted, checked, best_inliers))
    }
}

#[cfg(test)]
mod tests;

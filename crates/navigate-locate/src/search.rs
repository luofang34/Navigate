//! Search a region for the camera of one photo with a place index, then locate it.
//!
//! The search plans hypotheses from the priors, lays the photo on flat ground for
//! each, describes the projections with a place encoder, ranks the index records of
//! the planned cells, checks the best places against map windows, and runs the
//! locate pipeline once at the best place. A browser Worker and a native host run
//! the same code with their own encoder, index source and tile source.
mod finish;
mod horizon;
mod project;
mod rank;

pub use horizon::{Horizon, horizon};
pub use project::{View, canvas as search_canvas};

use crate::{LocateError, Outcome, Progress, TileSource};
use image::RgbImage;
use navigate_place::{
    Canvas, Fallback, IndexManifest, Ranked, ScalePlan, SearchRequest, Tokens, plan,
};
use navigate_visual::AsyncImageMatcher;
use std::future::Future;

/// Uncertainty of a camera angle read from the horizon, degrees.
const HORIZON_SIGMA_DEG: f64 = 4.0;

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
    matcher: M,
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
    let (small, focal_px) = finish::working_size(&gray, request.focal_px);
    let probes = Probes {
        ranked: &ranked,
        hypotheses: &hypotheses,
        scales: &planned.scales,
    };
    probes
        .run(tiles, matcher, &small, focal_px, &mut progress)
        .await
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

/// The ranked places and their hypotheses, each tested with a fast locate step.
struct Probes<'a> {
    ranked: &'a [(Ranked, usize)],
    hypotheses: &'a [Hypothesis],
    scales: &'a [ScalePlan],
}

impl Probes<'_> {
    /// Test the ranked places in order; at the first that gives a pose, run the full
    /// locate step there.
    async fn run<T: TileSource, M: AsyncImageMatcher + Clone>(
        &self,
        tiles: &T,
        matcher: M,
        gray: &image::GrayImage,
        focal_px: f64,
        progress: &mut impl FnMut(SearchProgress),
    ) -> Result<SearchOutcome, LocateError> {
        let total = self.ranked.len().min(self.budget());
        let mut best_matches = 0;
        for (checked, (place, h)) in self.ranked.iter().take(total).enumerate() {
            let hypothesis = &self.hypotheses[*h];
            // An unknown heading leaves up to half a heading step of error; the probe
            // covers ten degrees each side, so try the neighbouring steps too.
            let turns: &[f64] = if self.scales[hypothesis.scale].headings_deg.len() > 1 {
                &[0.0, -20.0, 20.0]
            } else {
                &[0.0]
            };
            for turn in turns {
                let aim = finish::Aim {
                    view: View {
                        heading_deg: hypothesis.view.heading_deg + turn,
                        ..hypothesis.view
                    },
                    ground: place.centre,
                    centre_back_m: hypothesis.footprint[0].hypot(hypothesis.footprint[1]),
                    slack_m: 0.75 * self.scales[hypothesis.scale].side_m,
                };
                let outcome =
                    finish::locate_at(tiles, matcher.clone(), gray, &aim, focal_px, progress)
                        .await?;
                match outcome {
                    Outcome::Found(located) => {
                        // The locate step already re-checks its estimate with a view
                        // rendered there; the probe's pose is the result.
                        let inliers = located.estimate.quality.inliers;
                        return Ok(SearchOutcome::Found {
                            located,
                            inliers,
                            checked: checked + 1,
                        });
                    }
                    Outcome::NotFound {
                        best_matches: m, ..
                    } => best_matches = best_matches.max(m),
                }
            }
            progress(SearchProgress::Checking {
                done: checked + 1,
                total,
            });
        }
        Ok(SearchOutcome::NotFound {
            ranked: self.ranked.len(),
            checked: total,
            best_inliers: best_matches,
        })
    }

    /// Ranked places to test: the plan's verification count of the first scale.
    fn budget(&self) -> usize {
        self.scales.iter().map(|s| s.verify).max().unwrap_or(0)
    }
}

//! Star sightings from a star camera image (ADR-0021).
//!
//! The crate makes fusion observations from one image in four steps:
//!
//! 1. [`detect_stars`] finds bright pixel groups above the local background
//!    and gives intensity-weighted [`StarCentroid`]s.
//! 2. [`identify_stars`] gives the [`StarMatch`]es. It projects the
//!    catalogue through an [`AttitudePrior`], or without a prior it finds
//!    the stars from their geometry in a [`PairCatalogue`].
//! 3. [`solve_attitude`] finds the rotation from the body axes to the
//!    Earth-centred, Earth-fixed (ECEF) axes with the Davenport q-method.
//!    The result is an [`AttitudeSolution`].
//! 4. [`line_of_sight_observations`] makes one
//!    `ObservationValue::LineOfSight` for each matched star.
//!    [`attitude_fix_local_ned`] makes an `AttitudeFix` in local NED axes
//!    from a position that the caller supplies.
//!
//! [`StarTracker`] chooses the identification mode of each frame: the
//! prior while the lock holds, lost-in-space at the start, after a run of
//! failed frames, or for a prior that is too wide.
//!
//! The crate does not read a clock or a file (ADR-0002). The results are a
//! function of the inputs only.

mod catalogue;
mod detect;
mod error;
mod fusion;
mod identify;
mod pairs;
mod solve;
#[cfg(test)]
mod synthetic;
mod tracker;

pub use catalogue::catalogue_ecef;
pub use detect::{DetectConfig, StarCentroid, detect_stars};
pub use error::StarTrackError;
pub use fusion::{attitude_fix_local_ned, line_of_sight_observations};
pub use identify::{AttitudePrior, IdentifyConfig, IdentifyMode, StarMatch, identify_stars};
pub use pairs::{MAX_PAIRS, PairCatalogue};
pub use solve::{AttitudeSolution, SolveConfig, solve_attitude};
pub use tracker::{StarTracker, TrackerConfig, TrackerLock};

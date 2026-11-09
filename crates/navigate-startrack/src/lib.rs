//! Star sightings from a star camera image (ADR-0021).
//!
//! The crate makes fusion observations from one image in four steps:
//!
//! 1. [`detect_stars`] finds bright pixel groups above the local background
//!    and gives intensity-weighted [`StarCentroid`]s.
//! 2. [`identify_stars`] projects the catalogue through an
//!    [`AttitudePrior`] and gives the [`StarMatch`]es. Identification
//!    without a prior (lost-in-space) is not supported.
//! 3. [`solve_attitude`] finds the rotation from the body axes to the
//!    Earth-centred, Earth-fixed (ECEF) axes with the Davenport q-method.
//!    The result is an [`AttitudeSolution`].
//! 4. [`line_of_sight_observations`] makes one
//!    `ObservationValue::LineOfSight` for each matched star.
//!    [`attitude_fix_local_ned`] makes an `AttitudeFix` in local NED axes
//!    from a position that the caller supplies.
//!
//! The crate does not read a clock or a file (ADR-0002). The results are a
//! function of the inputs only.

mod catalogue;
mod detect;
mod error;
mod fusion;
mod identify;
mod solve;
#[cfg(test)]
mod synthetic;

pub use catalogue::catalogue_ecef;
pub use detect::{DetectConfig, StarCentroid, detect_stars};
pub use error::StarTrackError;
pub use fusion::{attitude_fix_local_ned, line_of_sight_observations};
pub use identify::{AttitudePrior, IdentifyConfig, StarMatch, identify_stars};
pub use solve::{AttitudeSolution, SolveConfig, solve_attitude};

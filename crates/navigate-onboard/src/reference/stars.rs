//! The star camera channel of the reference navigator: Navigate's tracker
//! identifies each frame, lost in space or with the navigator's own
//! attitude as a prior, solves the attitude, and offers each star as a line
//! of sight.

use nalgebra::{Quaternion, UnitQuaternion};
use navigate_camera::{CameraMount, ImagerModel};
use navigate_celestial::{Duration, Epoch};
use navigate_contract::{MonotonicNanos, NavigationSolution};
use navigate_fusion::inertial::ned_to_body_fixed_dcm;
use navigate_startrack::{
    AttitudePrior, AttitudeSolution, IdentifyConfig, IdentifyMode, PairCatalogue, SolveConfig,
    StarCentroid, StarTracker, TrackerConfig, catalogue_ecef, identify_stars, solve_attitude,
};

use crate::{ErrorBudget, OnboardError, SensorSpec};

/// The star catalogue and the absolute time of the navigator clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StarCatalogueConfig {
    /// UTC as Unix seconds at navigator clock zero.
    pub utc_unix_s_at_zero: f64,
    /// Faintest star of the identification catalogue, Johnson V.
    pub catalogue_faintest_v: f64,
    /// Faintest star of the lost-in-space pair table, Johnson V.
    pub lost_in_space_faintest_v: f64,
}

impl StarCatalogueConfig {
    /// The catalogue limits of the reference composition: V 6.0 for
    /// identification and V 5.5 for the pair table.
    #[must_use]
    pub const fn with_utc_at_zero(utc_unix_s_at_zero: f64) -> Self {
        Self {
            utc_unix_s_at_zero,
            catalogue_faintest_v: 6.0,
            lost_in_space_faintest_v: 5.5,
        }
    }

    /// The epoch of a navigator time.
    #[must_use]
    pub fn epoch(&self, at: MonotonicNanos) -> Epoch {
        Epoch::from_unix_seconds(self.utc_unix_s_at_zero)
            + Duration::from_seconds(at.as_nanos() as f64 * 1e-9)
    }
}

/// The navigator's attitude belief as an identification prior.
#[must_use]
pub fn own_prior(solution: &NavigationSolution) -> Option<AttitudePrior> {
    let (q, covariance) = (solution.attitude?, solution.attitude_cov?);
    let body_to_ned = UnitQuaternion::from_quaternion(Quaternion::new(q.w, q.x, q.y, q.z));
    let p = &solution.position;
    let ned_to_ecef =
        UnitQuaternion::from_matrix(&ned_to_body_fixed_dcm(p.latitude_rad, p.longitude_rad));
    let (xx, yy, zz) = covariance.diagonal();
    Some(AttitudePrior {
        ecef_from_body: ned_to_ecef * body_to_ned,
        sigma_rad: (xx + yy + zz).sqrt(),
    })
}

/// The identification mode of a frame, as an acceptance record names it.
const fn mode_name(lost_in_space: bool) -> &'static str {
    if lost_in_space {
        "lost_in_space"
    } else {
        "tracking"
    }
}

/// One star camera.
#[derive(Debug, Clone)]
pub(crate) struct StarChannel {
    imager: ImagerModel,
    mount: CameraMount,
    pairs: PairCatalogue,
    catalogue: StarCatalogueConfig,
    pub(crate) mount_variance_rad2: f64,
    held_prior: Option<AttitudePrior>,
    tracker: StarTracker,
}

/// What one frame gave.
pub(crate) enum Sighting {
    /// Identified and solved, in the stated mode.
    Solved(AttitudeSolution, &'static str),
    /// Not solved, for the stated reason, in the stated mode.
    Unsolved(&'static str, &'static str),
}

impl StarChannel {
    /// A channel for one declared star camera.
    pub(crate) fn new(
        spec: &SensorSpec,
        catalogue: StarCatalogueConfig,
    ) -> Result<Self, OnboardError> {
        let invalid = |detail: String| OnboardError::Config {
            field: "star camera",
            detail,
        };
        let (
            Some(imager),
            Some(mount),
            ErrorBudget::StarCamera {
                mount_alignment_sigma_rad,
            },
        ) = (spec.imager, spec.mount, spec.budget)
        else {
            return Err(invalid(format!(
                "{:?} needs an imager, a mount, and a budget",
                spec.id
            )));
        };
        let i = imager.intrinsics;
        let diagonal_rad =
            2.0 * (f64::from(i.width).hypot(f64::from(i.height)) / 2.0 / i.fx).atan();
        let stars = catalogue_ecef(
            catalogue.epoch(MonotonicNanos::from_nanos(0)),
            catalogue.lost_in_space_faintest_v,
        )
        .map_err(|e| OnboardError::StarTrack(e.to_string()))?;
        let pairs = PairCatalogue::new(&stars, diagonal_rad, catalogue.lost_in_space_faintest_v)
            .map_err(|e| OnboardError::StarTrack(e.to_string()))?;
        Ok(Self {
            imager,
            mount,
            pairs,
            catalogue,
            mount_variance_rad2: mount_alignment_sigma_rad * mount_alignment_sigma_rad,
            held_prior: None,
            tracker: StarTracker::new(TrackerConfig::default())
                .map_err(|e| OnboardError::StarTrack(e.to_string()))?,
        })
    }

    /// Holds the attitude prior of the next frame.
    pub(crate) fn hold_prior(&mut self, solution: Option<&NavigationSolution>) {
        self.held_prior = solution.and_then(own_prior);
    }

    /// Identifies and solves one frame acquired at `acquired_at`.
    pub(crate) fn sight(
        &mut self,
        centroids: &[StarCentroid],
        acquired_at: MonotonicNanos,
    ) -> Result<Sighting, OnboardError> {
        let mode = self.tracker.mode(self.held_prior.take(), &self.pairs);
        let catalogue = catalogue_ecef(
            self.catalogue.epoch(acquired_at),
            self.catalogue.catalogue_faintest_v,
        )
        .map_err(|e| OnboardError::StarTrack(e.to_string()))?;
        let lost = matches!(mode, IdentifyMode::LostInSpace(_));
        let solved = identify_stars(
            centroids,
            &self.imager,
            &self.mount,
            mode,
            &catalogue,
            &IdentifyConfig::default(),
        )
        .and_then(|matches| {
            solve_attitude(
                &matches,
                centroids,
                &self.imager,
                &self.mount,
                &catalogue,
                &SolveConfig::default(),
            )
        });
        self.tracker.record(solved.is_ok());
        Ok(match solved {
            Ok(solution) => Sighting::Solved(solution, mode_name(lost)),
            Err(_) => Sighting::Unsolved("star frame not identified", mode_name(lost)),
        })
    }
}

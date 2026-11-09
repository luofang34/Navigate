//! Catalogue identification of star centroids.
//!
//! Identification with an attitude prior has two stages:
//!
//! 1. Each centroid matches the nearest catalogue star that the prior
//!    predicts inside the gate. The match must be unique in both
//!    directions, and the second-nearest star inside the gate must be
//!    clearly farther ([`IdentifyConfig::ambiguity_ratio`]).
//! 2. The angle between two stars does not change with the attitude. A
//!    pair of matches whose measured angle and catalogue angle differ by
//!    more than the pair tolerance has one wrong match. The stage removes
//!    the match with the most such conflicts until no conflict remains.
//!
//! A large prior error makes the gate wide. Then a wrong star can be the
//! nearest star. The second stage removes these matches.
//!
//! Without a prior, the pyramid search ([`pyramid`]) finds four stars from
//! the star geometry alone. Their attitude then acts as a prior for the two
//! stages above.

mod geometry;
mod pyramid;

use nalgebra::{UnitQuaternion, Vector3};
use navigate_camera::{CameraMount, ImagerModel};
use navigate_celestial::StarSight;

use crate::solve::davenport;
use crate::{PairCatalogue, StarCentroid, StarTrackError};
pub(crate) use geometry::{body_rays, match_nearest, predicted_in_view, ray_sigma_rad};

/// The prior estimate of the attitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttitudePrior {
    /// Rotation from the body axes (forward, right, down) into the ECEF
    /// axes.
    pub ecef_from_body: UnitQuaternion<f64>,
    /// One standard deviation of the attitude error angle in radians.
    pub sigma_rad: f64,
}

/// The settings of [`identify_stars`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdentifyConfig {
    /// One standard deviation of a centroid on each image axis in pixels.
    pub centroid_sigma_px: f64,
    /// One standard deviation of a catalogue direction in radians.
    pub catalogue_sigma_rad: f64,
    /// Gate radius in standard deviations of the predicted direction.
    pub gate_sigmas: f64,
    /// Largest ratio of the nearest to the second-nearest distance when
    /// two stars are inside the gate.
    pub ambiguity_ratio: f64,
    /// Pair tolerance in standard deviations of a measured angle.
    pub pair_sigmas: f64,
    /// Brightest centroids that identification uses.
    pub max_centroids: usize,
    /// Tolerance of the lost-in-space angle test in standard deviations
    /// of a measured angle.
    pub pyramid_sigmas: f64,
    /// Angle in radians that the lost-in-space tolerance adds for the
    /// difference between the pair table epoch and the acquisition epoch.
    pub pair_table_slack_rad: f64,
    /// Brightest centroids that the lost-in-space search uses.
    pub pyramid_centroids: usize,
    /// Most catalogue triangles for one centroid triangle. A triangle with
    /// more is not used.
    pub max_triangle_candidates: usize,
    /// Fewest matches at the attitude of a lost-in-space pyramid, the four
    /// pyramid stars included. It is at least 4.
    pub min_lost_in_space_matches: usize,
}

impl Default for IdentifyConfig {
    fn default() -> Self {
        Self {
            centroid_sigma_px: 0.3,
            catalogue_sigma_rad: 5.0e-6,
            gate_sigmas: 3.0,
            ambiguity_ratio: 0.5,
            pair_sigmas: 5.0,
            max_centroids: 40,
            pyramid_sigmas: 3.0,
            pair_table_slack_rad: 5.0e-5,
            pyramid_centroids: 10,
            max_triangle_candidates: 64,
            min_lost_in_space_matches: 5,
        }
    }
}

/// One centroid with its catalogue star.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StarMatch {
    /// Index of the centroid in the input slice.
    pub centroid_index: usize,
    /// Harvard Revised (HR) catalogue number.
    pub hr: u16,
    /// Measured unit direction in the body axes, after the lens
    /// distortion is removed.
    pub body_unit: Vector3<f64>,
    /// Catalogue unit direction in the ECEF axes.
    pub ecef_unit: Vector3<f64>,
    /// Angle in radians between the measured direction and the direction
    /// that the attitude of the matching step predicts.
    pub separation_rad: f64,
    /// One standard deviation of the measured direction in radians, with
    /// the catalogue error, as the step that made the match models it.
    /// [`crate::solve_attitude`] gives each star its own value from
    /// [`crate::StarCentroid::covariance_px2`].
    pub direction_sigma_rad: f64,
}

/// How [`identify_stars`] finds the first matches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IdentifyMode<'a> {
    /// Project the catalogue through an attitude prior.
    Prior(AttitudePrior),
    /// Find the attitude from the star geometry alone (lost-in-space),
    /// with the pyramid search in a pair table.
    LostInSpace(&'a PairCatalogue),
}

/// The stars of `centroids` that the catalogue identifies.
///
/// `catalogue` holds the reference directions at the acquisition time, for
/// example from [`crate::catalogue_ecef`]. Only the
/// [`IdentifyConfig::max_centroids`] first centroids take part; the output
/// of [`crate::detect_stars`] is brightest first. The result is in the
/// order of the centroids. A centroid where the lens distortion does not
/// invert does not match.
///
/// With [`IdentifyMode::LostInSpace`], the pyramid search uses the
/// [`IdentifyConfig::pyramid_centroids`] first centroids. The attitude of
/// the first confirmed pyramid then acts as a prior with the deviation of
/// the pyramid tolerance.
///
/// # Errors
/// Refuses an invalid camera model, configuration, or prior. Returns
/// [`StarTrackError::NotIdentified`] when the lost-in-space search finds no
/// unique pyramid that also matches the catalogue at its attitude.
pub fn identify_stars(
    centroids: &[StarCentroid],
    imager: &ImagerModel,
    mount: &CameraMount,
    mode: IdentifyMode<'_>,
    catalogue: &[StarSight],
    config: &IdentifyConfig,
) -> Result<Vec<StarMatch>, StarTrackError> {
    imager.validate()?;
    validate(config)?;
    let used = centroids.get(..config.max_centroids).unwrap_or(centroids);
    let measured = body_rays(used, imager, mount);
    let ray_sigma =
        ray_sigma_rad(imager, config.centroid_sigma_px).hypot(config.catalogue_sigma_rad);
    let scene = Scene {
        measured: &measured,
        imager,
        mount,
        catalogue,
        config,
        ray_sigma,
    };
    match mode {
        IdentifyMode::Prior(prior) => {
            if !(prior.sigma_rad.is_finite() && prior.sigma_rad > 0.0) {
                return Err(StarTrackError::Invalid {
                    field: "prior deviation",
                });
            }
            Ok(scene.with_prior(&prior))
        }
        IdentifyMode::LostInSpace(table) => scene.lost_in_space(table),
    }
}

/// The inputs of one identification.
struct Scene<'a> {
    measured: &'a [Option<Vector3<f64>>],
    imager: &'a ImagerModel,
    mount: &'a CameraMount,
    catalogue: &'a [StarSight],
    config: &'a IdentifyConfig,
    ray_sigma: f64,
}

impl Scene<'_> {
    fn with_prior(&self, prior: &AttitudePrior) -> Vec<StarMatch> {
        let gate = self.config.gate_sigmas * prior.sigma_rad.hypot(self.ray_sigma);
        let predicted = predicted_in_view(
            self.catalogue,
            self.imager,
            self.mount,
            &prior.ecef_from_body,
            gate,
        );
        let matches = match_nearest(
            self.measured,
            &predicted,
            gate,
            self.config.ambiguity_ratio,
            self.ray_sigma,
        );
        consistent_pairs(
            matches,
            std::f64::consts::SQRT_2 * self.config.pair_sigmas * self.ray_sigma,
        )
    }

    fn lost_in_space(&self, table: &PairCatalogue) -> Result<Vec<StarMatch>, StarTrackError> {
        let tolerance_rad = std::f64::consts::SQRT_2 * self.config.pyramid_sigmas * self.ray_sigma
            + self.config.pair_table_slack_rad;
        let rays: Vec<(usize, Vector3<f64>)> = self
            .measured
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|r| (i, r)))
            .take(self.config.pyramid_centroids)
            .collect();
        let limits = pyramid::Search {
            tolerance_rad,
            max_candidates: self.config.max_triangle_candidates,
        };
        pyramid::search(&rays, table, limits, |pyramid| {
            let seed: Vec<StarMatch> = pyramid
                .iter()
                .map(|&(centroid, star)| self.pairing_match(table, centroid, star))
                .collect::<Option<_>>()?;
            let prior = AttitudePrior {
                ecef_from_body: davenport(&seed).ok()?,
                sigma_rad: tolerance_rad,
            };
            let matches = self.with_prior(&prior);
            let all_seeds = seed.iter().all(|s| {
                matches
                    .iter()
                    .any(|m| m.centroid_index == s.centroid_index && m.hr == s.hr)
            });
            (all_seeds && matches.len() >= self.config.min_lost_in_space_matches).then_some(matches)
        })
        .ok_or(StarTrackError::NotIdentified)
    }

    /// The match of a centroid with a table star, with the direction of
    /// the same star in the catalogue of the acquisition time.
    fn pairing_match(
        &self,
        table: &PairCatalogue,
        centroid: usize,
        star: u16,
    ) -> Option<StarMatch> {
        let hr = table.star(star)?.hr;
        let sight = self.catalogue.iter().find(|s| s.hr == hr)?;
        Some(StarMatch {
            centroid_index: centroid,
            hr,
            body_unit: (*self.measured.get(centroid)?)?,
            ecef_unit: Vector3::from(sight.direction).normalize(),
            separation_rad: 0.0,
            direction_sigma_rad: self.ray_sigma,
        })
    }
}

fn validate(config: &IdentifyConfig) -> Result<(), StarTrackError> {
    let positive = [
        config.centroid_sigma_px,
        config.gate_sigmas,
        config.pair_sigmas,
        config.pyramid_sigmas,
    ]
    .iter()
    .all(|v| v.is_finite() && *v > 0.0);
    let ratio = config.ambiguity_ratio > 0.0 && config.ambiguity_ratio <= 1.0;
    let non_negative = [config.catalogue_sigma_rad, config.pair_table_slack_rad]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0);
    let counts = config.max_triangle_candidates > 0 && config.min_lost_in_space_matches >= 4;
    if positive && ratio && non_negative && counts {
        Ok(())
    } else {
        Err(StarTrackError::Invalid {
            field: "identification settings",
        })
    }
}

/// The matches that remain when no pair has a measured angle and a
/// catalogue angle that differ by more than `tolerance_rad`.
fn consistent_pairs(mut matches: Vec<StarMatch>, tolerance_rad: f64) -> Vec<StarMatch> {
    loop {
        let conflicts: Vec<usize> = matches
            .iter()
            .map(|a| {
                matches
                    .iter()
                    .filter(|b| {
                        let measured = a.body_unit.angle(&b.body_unit);
                        let catalogue = a.ecef_unit.angle(&b.ecef_unit);
                        (measured - catalogue).abs() > tolerance_rad
                    })
                    .count()
            })
            .collect();
        // The last match with the most conflicts goes first, so the result
        // does not depend on the sort algorithm.
        let worst = conflicts
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .max_by_key(|(i, n)| (**n, *i))
            .map(|(i, _)| i);
        match worst {
            Some(index) => {
                matches.remove(index);
            }
            None => return matches,
        }
    }
}

#[cfg(test)]
mod tests;

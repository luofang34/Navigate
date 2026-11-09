//! Catalogue identification of star centroids with an attitude prior.
//!
//! Identification has two stages:
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

mod geometry;

use nalgebra::{UnitQuaternion, Vector3};
use navigate_camera::{CameraMount, ImagerModel};
use navigate_celestial::StarSight;

use crate::{StarCentroid, StarTrackError};
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
/// # Errors
/// Returns [`StarTrackError::LostInSpaceUnsupported`] without a prior, and
/// refuses an invalid camera model or configuration.
pub fn identify_stars(
    centroids: &[StarCentroid],
    imager: &ImagerModel,
    mount: &CameraMount,
    prior: Option<&AttitudePrior>,
    catalogue: &[StarSight],
    config: &IdentifyConfig,
) -> Result<Vec<StarMatch>, StarTrackError> {
    let prior = prior.ok_or(StarTrackError::LostInSpaceUnsupported)?;
    imager.validate()?;
    validate(config, prior)?;
    let used = centroids.get(..config.max_centroids).unwrap_or(centroids);
    let measured = body_rays(used, imager, mount);
    let ray_sigma =
        ray_sigma_rad(imager, config.centroid_sigma_px).hypot(config.catalogue_sigma_rad);
    let gate = config.gate_sigmas * prior.sigma_rad.hypot(ray_sigma);
    let predicted = predicted_in_view(catalogue, imager, mount, &prior.ecef_from_body, gate);
    let matches = match_nearest(&measured, &predicted, gate, config.ambiguity_ratio);
    Ok(consistent_pairs(
        matches,
        std::f64::consts::SQRT_2 * config.pair_sigmas * ray_sigma,
    ))
}

fn validate(config: &IdentifyConfig, prior: &AttitudePrior) -> Result<(), StarTrackError> {
    let positive = [
        config.centroid_sigma_px,
        config.gate_sigmas,
        config.pair_sigmas,
        prior.sigma_rad,
    ]
    .iter()
    .all(|v| v.is_finite() && *v > 0.0);
    let ratio = config.ambiguity_ratio > 0.0 && config.ambiguity_ratio <= 1.0;
    let catalogue = config.catalogue_sigma_rad.is_finite() && config.catalogue_sigma_rad >= 0.0;
    if positive && ratio && catalogue {
        Ok(())
    } else {
        Err(StarTrackError::Invalid {
            field: "identification settings or prior deviation",
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

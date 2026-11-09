//! The attitude from matched stars with the Davenport q-method.
//!
//! [`solve_attitude`] solves the attitude from the matches of
//! identification. Then it matches all centroids again with the solved
//! attitude and a tight gate, removes the residual outliers one at a
//! time, and solves again.
//!
//! Each star has its own direction deviation `σᵢ`: the largest axis of
//! [`StarCentroid::covariance_px2`], with a floor for the errors of the
//! model, over the focal length, with the catalogue error. The q-method
//! weights each star by `1/σᵢ²`, and the outlier test reads each residual
//! in its own `σᵢ`. The covariance keeps the shape of each centroid
//! covariance `Cᵢ`, which a streak makes long: it is `(Σ Hᵢᵀ·Cᵢ⁻¹·Hᵢ)⁻¹`,
//! with `Hᵢ` the change of the pinhole pixel of star `i` with a rotation
//! of the body axes.

use nalgebra::{Matrix2, Matrix3, Matrix4, Quaternion, SymmetricEigen, UnitQuaternion};
use navigate_camera::{CameraMount, ImagerModel};
use navigate_celestial::StarSight;

use crate::identify::{match_nearest, predicted_in_view};
use crate::{StarCentroid, StarMatch, StarTrackError, identify};

/// The settings of [`solve_attitude`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveConfig {
    /// Smallest standard deviation of a centroid on each image axis in
    /// pixels. It holds the errors that the pixel noise does not, such as
    /// the sampling of the point spread function and the calibration of
    /// the lens, and adds to the noise of the centroid in quadrature.
    pub min_centroid_sigma_px: f64,
    /// The same floor for a saturated star, whose flat core biases the
    /// centroid.
    pub saturated_centroid_sigma_px: f64,
    /// One standard deviation of a catalogue direction in radians.
    pub catalogue_sigma_rad: f64,
    /// Gate radius of the second match in standard deviations of a
    /// measured direction.
    pub rematch_gate_sigmas: f64,
    /// Largest ratio of the nearest to the second-nearest distance in the
    /// second match.
    pub ambiguity_ratio: f64,
    /// Residual angle in standard deviations above which a star is an
    /// outlier.
    pub outlier_sigmas: f64,
    /// Fewest stars of a solution. It is at least 3.
    pub min_stars: usize,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            min_centroid_sigma_px: 0.02,
            saturated_centroid_sigma_px: 0.3,
            catalogue_sigma_rad: 5.0e-6,
            rematch_gate_sigmas: 6.0,
            ambiguity_ratio: 0.5,
            outlier_sigmas: 3.0,
            min_stars: 3,
        }
    }
}

/// An attitude from star sightings.
#[derive(Debug, Clone, PartialEq)]
pub struct AttitudeSolution {
    /// Rotation from the body axes (forward, right, down) into the ECEF
    /// axes at the acquisition time.
    pub ecef_from_body: UnitQuaternion<f64>,
    /// Covariance of the body-axis rotation vector in radians²:
    /// `(Σ Hᵢᵀ·Cᵢ⁻¹·Hᵢ)⁻¹` over the stars, with `Cᵢ` the centroid
    /// covariance with its floor and the catalogue error in pixels², and
    /// `Hᵢ` the pinhole Jacobian of the pixel of star `i` with respect to
    /// the rotation vector. The lens distortion is left out of `Hᵢ`. The
    /// true attitude is `ecef_from_body · exp(δθ)`.
    pub covariance_body_rad2: Matrix3<f64>,
    /// The stars of the solution. [`StarMatch::separation_rad`] is the
    /// residual angle at the solved attitude, and
    /// [`StarMatch::direction_sigma_rad`] the deviation of the star.
    pub matches: Vec<StarMatch>,
    /// Root mean square of the residual angles in radians.
    pub rms_residual_rad: f64,
    /// Matches that the outlier test removed.
    pub rejected_outliers: usize,
}

/// The attitude from `matches`, refined with all `centroids`.
///
/// `centroids` and `catalogue` are the inputs of [`crate::identify_stars`]
/// that gave `matches`.
///
/// # Errors
/// Refuses an invalid camera model or configuration. Returns
/// [`StarTrackError::TooFewStars`] when fewer than
/// [`SolveConfig::min_stars`] stars are matched, at the start or after the
/// outlier test, and [`StarTrackError::Degenerate`] when the directions do
/// not define the attitude.
pub fn solve_attitude(
    matches: &[StarMatch],
    centroids: &[StarCentroid],
    imager: &ImagerModel,
    mount: &CameraMount,
    catalogue: &[StarSight],
    config: &SolveConfig,
) -> Result<AttitudeSolution, StarTrackError> {
    imager.validate()?;
    validate(config)?;
    enough(matches.len(), config)?;
    let first = davenport(matches)?;
    let sigmas: Vec<f64> = centroids
        .iter()
        .map(|c| direction_sigma_rad(c, imager, config))
        .collect();
    let widest = sigmas.iter().copied().fold(0.0_f64, f64::max);
    let gate = config.rematch_gate_sigmas * widest;
    let measured = identify::body_rays(centroids, imager, mount);
    let predicted = predicted_in_view(catalogue, imager, mount, &first, gate);
    let mut matched = match_nearest(&measured, &predicted, gate, config.ambiguity_ratio, widest);
    for m in &mut matched {
        m.direction_sigma_rad = sigmas.get(m.centroid_index).copied().unwrap_or(widest);
    }
    enough(matched.len(), config)?;
    let mut attitude = davenport(&matched)?;
    let mut rejected_outliers = 0_usize;
    loop {
        set_residuals(&mut matched, &attitude);
        let worst = matched
            .iter()
            .enumerate()
            .map(|(index, m)| (index, m.separation_rad / m.direction_sigma_rad))
            .max_by(|(_, a), (_, b)| a.total_cmp(b));
        match worst {
            Some((index, normalized)) if normalized > config.outlier_sigmas => {
                matched.remove(index);
                rejected_outliers = rejected_outliers.saturating_add(1);
                enough(matched.len(), config)?;
                attitude = davenport(&matched)?;
            }
            _ => break,
        }
    }
    let covariance = information(&matched, centroids, imager, mount, config)
        .and_then(|i| i.try_inverse())
        .ok_or(StarTrackError::Degenerate)?;
    let sum_sq: f64 = matched.iter().map(|m| m.separation_rad.powi(2)).sum();
    Ok(AttitudeSolution {
        ecef_from_body: attitude,
        covariance_body_rad2: covariance,
        rms_residual_rad: (sum_sq / matched.len() as f64).sqrt(),
        matches: matched,
        rejected_outliers,
    })
}

fn validate(config: &SolveConfig) -> Result<(), StarTrackError> {
    let positive = [
        config.min_centroid_sigma_px,
        config.saturated_centroid_sigma_px,
        config.rematch_gate_sigmas,
        config.outlier_sigmas,
    ]
    .iter()
    .all(|v| v.is_finite() && *v > 0.0);
    let ratio = config.ambiguity_ratio > 0.0 && config.ambiguity_ratio <= 1.0;
    let catalogue = config.catalogue_sigma_rad.is_finite() && config.catalogue_sigma_rad >= 0.0;
    if positive && ratio && catalogue && config.min_stars >= 3 {
        Ok(())
    } else {
        Err(StarTrackError::Invalid {
            field: "solution settings",
        })
    }
}

fn enough(found: usize, config: &SolveConfig) -> Result<(), StarTrackError> {
    if found < config.min_stars {
        Err(StarTrackError::TooFewStars {
            found,
            needed: config.min_stars,
        })
    } else {
        Ok(())
    }
}

/// The information of the body-axis rotation vector from `matched`:
/// `Σ Hᵢᵀ·Cᵢ⁻¹·Hᵢ` with `Hᵢ = J·R·[bᵢ×]`, where `J` is the pinhole
/// Jacobian at the camera ray of star `i` and `R` the rotation from the
/// body axes into the camera axes. `None` for a star behind the camera or
/// a singular `Cᵢ`.
fn information(
    matched: &[StarMatch],
    centroids: &[StarCentroid],
    imager: &ImagerModel,
    mount: &CameraMount,
    config: &SolveConfig,
) -> Option<Matrix3<f64>> {
    let k = &imager.intrinsics;
    let cam_from_body = mount.cam_from_body.to_rotation_matrix();
    let catalogue = Matrix2::from_diagonal(&nalgebra::Vector2::new(
        (config.catalogue_sigma_rad * k.fx).powi(2),
        (config.catalogue_sigma_rad * k.fy).powi(2),
    ));
    matched.iter().try_fold(Matrix3::zeros(), |sum, m| {
        let centroid = centroids.get(m.centroid_index)?;
        let floor = centroid_floor_px(centroid, config);
        let c = centroid.covariance_px2 + Matrix2::identity() * floor * floor + catalogue;
        let ray = cam_from_body * m.body_unit;
        if ray.z <= 0.0 {
            return None;
        }
        let (x, y, z) = (ray.x / ray.z, ray.y / ray.z, ray.z);
        let projection =
            nalgebra::Matrix2x3::new(k.fx / z, 0.0, -k.fx * x / z, 0.0, k.fy / z, -k.fy * y / z);
        let h = projection * cam_from_body.matrix() * m.body_unit.cross_matrix();
        Some(sum + h.transpose() * c.try_inverse()? * h)
    })
}

/// The centroid floor of `centroid` in pixels: the errors of the model that
/// the pixel noise does not hold.
fn centroid_floor_px(centroid: &StarCentroid, config: &SolveConfig) -> f64 {
    if centroid.saturated {
        config.saturated_centroid_sigma_px
    } else {
        config.min_centroid_sigma_px
    }
}

/// One standard deviation of the measured direction of `centroid` in
/// radians, with the catalogue error.
fn direction_sigma_rad(centroid: &StarCentroid, imager: &ImagerModel, config: &SolveConfig) -> f64 {
    let floor = centroid_floor_px(centroid, config);
    let noise = largest_eigenvalue(&centroid.covariance_px2).max(0.0).sqrt();
    identify::ray_sigma_rad(imager, noise.hypot(floor)).hypot(config.catalogue_sigma_rad)
}

/// The largest eigenvalue of a symmetric 2×2 matrix.
fn largest_eigenvalue(m: &Matrix2<f64>) -> f64 {
    let mean = 0.5 * (m[(0, 0)] + m[(1, 1)]);
    let half_difference = 0.5 * (m[(0, 0)] - m[(1, 1)]);
    mean + half_difference.hypot(m[(0, 1)])
}

/// Sets the separation of each match to its residual angle at `attitude`.
fn set_residuals(matches: &mut [StarMatch], attitude: &UnitQuaternion<f64>) {
    let body_from_ecef = attitude.inverse();
    for m in matches {
        m.separation_rad = m.body_unit.angle(&(body_from_ecef * m.ecef_unit));
    }
}

/// The rotation `R` from the body axes into the ECEF axes that minimizes
/// `Σ wᵢ·|e − R·b|²` over the matched pairs (Wahba's problem), from the
/// Davenport q-method. The weight of a pair is `1/σ²` of its
/// [`StarMatch::direction_sigma_rad`], scaled so that the largest weight
/// is one.
///
/// With `B = Σ wᵢ·e·bᵀ`, `S = B + Bᵀ`, `s = tr B`, and
/// `z = Σ e × b`, the eigenvector `(q, q₄)` of the largest eigenvalue of
/// `K = [[S − s·I, z], [zᵀ, s]]` gives `R = (q₄² − |q|²)·I + 2·q·qᵀ −
/// 2·q₄·[q×]`. That is the active rotation of the quaternion `(q₄, −q)`.
pub(crate) fn davenport(matches: &[StarMatch]) -> Result<UnitQuaternion<f64>, StarTrackError> {
    let tightest = matches
        .iter()
        .map(|m| m.direction_sigma_rad)
        .fold(f64::INFINITY, f64::min);
    if !(tightest.is_finite() && tightest > 0.0) {
        return Err(StarTrackError::Degenerate);
    }
    let b: Matrix3<f64> = matches
        .iter()
        .map(|m| m.ecef_unit * m.body_unit.transpose() * (tightest / m.direction_sigma_rad).powi(2))
        .sum();
    let s = b + b.transpose();
    let trace = b.trace();
    let z = nalgebra::Vector3::new(
        b[(1, 2)] - b[(2, 1)],
        b[(2, 0)] - b[(0, 2)],
        b[(0, 1)] - b[(1, 0)],
    );
    let mut k = Matrix4::zeros();
    k.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(s - Matrix3::identity() * trace));
    k.fixed_view_mut::<3, 1>(0, 3).copy_from(&z);
    k.fixed_view_mut::<1, 3>(3, 0).copy_from(&z.transpose());
    k[(3, 3)] = trace;
    let eigen = SymmetricEigen::new(k);
    let mut order: Vec<(f64, usize)> = eigen.eigenvalues.iter().copied().zip(0..).collect();
    order.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (Some(&(top_value, top)), Some(&(next_value, _))) = (order.first(), order.get(1)) else {
        return Err(StarTrackError::Degenerate);
    };
    // Equal largest eigenvalues leave a rotation about one direction free.
    let gap = top_value - next_value;
    if gap.is_nan() || gap <= 1e-9 * matches.len() as f64 {
        return Err(StarTrackError::Degenerate);
    }
    let v = eigen.eigenvectors.column(top);
    let (Some(&x), Some(&y), Some(&z), Some(&w)) = (v.get(0), v.get(1), v.get(2), v.get(3)) else {
        return Err(StarTrackError::Degenerate);
    };
    let q = Quaternion::new(w, -x, -y, -z);
    if !q.norm().is_finite() || q.norm() == 0.0 {
        return Err(StarTrackError::Degenerate);
    }
    Ok(UnitQuaternion::from_quaternion(q))
}

#[cfg(test)]
mod tests;

//! Map anchors: camera poses matched against map imagery, and their error budget.

use crate::{FrameKey, pose::Pose, pose::from_camera};
use nalgebra::{Matrix3, Matrix6, Vector3};
use navigate_visual::{Estimate, LocalFrame, MapRevision};

/// The surface that the map depth describes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SurfaceModel {
    /// Bare-earth elevation. Roofs and trees are not in the depth.
    ///
    /// Features on objects above the ground get the ground depth. The pose
    /// error from this parallax grows with object height and view angle.
    BareEarth {
        /// Height of the tallest objects that the camera can match, in metres.
        max_object_height_m: f64,
    },
    /// A surface model that includes buildings and vegetation.
    Surface,
}

/// What the host knows about change in the mapped area since the imagery date.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionChange {
    /// No known change.
    Unchanged,
    /// The area has known change, such as construction. Anchors are refused.
    Changed,
    /// Change is not known.
    Unknown,
}

/// The declared error of the map near one anchor.
///
/// The host supplies the imagery age from source metadata, or `None` when
/// the age is not known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapReliability {
    /// Horizontal error of the orthoimage, in metres (one sigma).
    pub horizontal_m: f64,
    /// Vertical error of the elevation model, in metres (one sigma).
    pub vertical_m: f64,
    /// Imagery age at the time of the flight, in years.
    pub imagery_age_years: Option<f64>,
    /// Added horizontal error per year of age, for example from re-survey shifts.
    pub age_growth_m_per_year: f64,
    /// Age used when the age is not known, in years.
    pub unknown_age_years: f64,
    /// Surface in the map depth.
    pub surface: SurfaceModel,
    /// Change in the area.
    pub change: RegionChange,
}

impl MapReliability {
    pub(crate) fn validate(&self) -> bool {
        let object = match self.surface {
            SurfaceModel::BareEarth {
                max_object_height_m,
            } => max_object_height_m,
            SurfaceModel::Surface => 0.0,
        };
        [
            self.horizontal_m,
            self.vertical_m,
            self.age_growth_m_per_year,
            self.unknown_age_years,
            object,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0)
            && self.horizontal_m > 0.0
            && self.vertical_m > 0.0
            && self
                .imagery_age_years
                .is_none_or(|a| a.is_finite() && a >= 0.0)
    }
}

/// A camera pose matched against map imagery for one frame.
///
/// Only a map match makes an anchor. A tracking result cannot become an
/// anchor, because its pose is conditional on an earlier pose.
#[derive(Clone, Debug)]
pub struct AnchorObservation {
    /// Frame of the matched image.
    pub frame: FrameKey,
    /// Evidence digest of the matched image.
    pub observation_sha256: String,
    /// Map revision of the reference.
    pub map: MapRevision,
    /// Local frame of the reference.
    pub local_frame: LocalFrame,
    /// Camera pose in the local frame.
    pub pose: Pose,
    /// Image-geometry covariance of the pose. The first three axes are the
    /// position in the local frame, in metres. The last three are the
    /// rotation `R = R_pose Exp(theta)` in camera axes, in radians.
    ///
    /// Over flat ground, tilt and horizontal position errors are strongly
    /// correlated: together they keep the matched ground under the same
    /// pixels. The cross terms carry that constraint.
    pub geometry_covariance: Matrix6<f64>,
    /// Matcher identity.
    pub backend: String,
    /// Declared map error near the anchor.
    pub reliability: MapReliability,
}

impl AnchorObservation {
    /// An anchor from an accepted map-match estimate.
    pub fn from_estimate(
        frame: FrameKey,
        estimate: &Estimate,
        reliability: MapReliability,
    ) -> Self {
        Self {
            frame,
            observation_sha256: estimate.observation_sha256.clone(),
            map: estimate.map.clone(),
            local_frame: estimate.frame,
            pose: from_camera(&estimate.pose),
            geometry_covariance: estimate.geometry_covariance,
            backend: estimate.backend.clone(),
            reliability,
        }
    }
}

/// A pose covariance after a change of horizontal scale between local
/// frames. The horizontal position axes take `ratio`; rotation and height
/// keep their values.
pub(crate) fn rescaled(covariance: &Matrix6<f64>, ratio: f64) -> Matrix6<f64> {
    let transport =
        Matrix6::from_diagonal(&nalgebra::Vector6::new(ratio, ratio, 1.0, 1.0, 1.0, 1.0));
    transport * covariance * transport.transpose()
}

/// The error budget of one anchor, split into independent and shared parts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchorBudget {
    /// Independent pose covariance: image geometry, attitude floor, and
    /// parallax. Axes as in [`AnchorObservation::geometry_covariance`].
    pub independent: Matrix6<f64>,
    /// Horizontal map error that all anchors in one map cell share, in metres.
    pub shared_horizontal_m: f64,
    /// Vertical map error that all anchors in one map cell share, in metres.
    pub shared_vertical_m: f64,
    /// Heading error, the rotation about the map vertical, in radians. It is
    /// the image-geometry error scaled with the attitude floor. The parallax
    /// term of this budget is a tilt about the matched ground, so it adds no
    /// heading error.
    pub heading_rad: f64,
    /// Tilt error, the rotation about the map horizontal axes, with the
    /// attitude floor and parallax, in radians.
    pub tilt_rad: f64,
}

impl AnchorBudget {
    /// Total horizontal one-sigma error of the camera position, in metres.
    pub fn horizontal_m(&self) -> f64 {
        let i = &self.independent;
        ((i[(0, 0)] + i[(1, 1)]) * 0.5 + self.shared_horizontal_m.powi(2)).sqrt()
    }
}

const MAX_PARALLAX_ANGLE_RAD: f64 = 1.31;
/// A map match fits the camera to bare-earth depth. Roofs, trees, elevation
/// model tilt, and calibration limit its attitude to about two degrees, even
/// when the image residuals are small.
pub(crate) const MIN_ROTATION_RAD: f64 = 0.035;

/// Combine image geometry, parallax, imagery age, and map accuracy.
///
/// Parallax uses the angle between the view axis and straight down, plus
/// half of the diagonal field of view. The result is a declared bound, not a
/// measured error.
///
/// The attitude floor scales the whole image-geometry covariance. Lens
/// distortion and elevation-model tilt move image points as image noise
/// does, so a scale keeps the correlation that holds the matched ground
/// under the image. Independent position and rotation terms would let that
/// ground slide by the full position bound.
///
/// Objects above the bare earth move image points away from the nadir
/// point. The fitted camera then turns about the matched ground and moves
/// with the turn; see [`pivot_parallax`]. Objects of one height over the
/// whole image are the same as a lower camera, so that part is vertical.
pub(crate) fn budget(
    anchor: &AnchorObservation,
    half_diagonal_fov_rad: f64,
    min_rotation_rad: f64,
) -> AnchorBudget {
    let r = &anchor.reliability;
    let (parallax_h, parallax_v) = match r.surface {
        SurfaceModel::BareEarth {
            max_object_height_m,
        } => {
            let angle = (crate::pose::off_nadir(&anchor.pose) + half_diagonal_fov_rad)
                .min(MAX_PARALLAX_ANGLE_RAD);
            (max_object_height_m * angle.tan(), max_object_height_m)
        }
        SurfaceModel::Surface => (0.0, 0.0),
    };
    let age = r.imagery_age_years.unwrap_or(r.unknown_age_years);
    let shared_h = (r.horizontal_m.powi(2) + (age * r.age_growth_m_per_year).powi(2)).sqrt();
    let geometry = (anchor.geometry_covariance + anchor.geometry_covariance.transpose()) * 0.5;
    let rotation_rad2 = geometry[(3, 3)] + geometry[(4, 4)] + geometry[(5, 5)];
    let floor = if rotation_rad2 > 0.0 {
        (min_rotation_rad.powi(2) / rotation_rad2).max(1.0)
    } else {
        1.0
    };
    let to_map = anchor.pose.rotation.to_rotation_matrix().into_inner();
    let floored = geometry * floor;
    let mut independent = match pivot_parallax(&floored, &to_map, parallax_h) {
        Some(parallax) => floored + parallax,
        // Without a pivot, the parallax scales the whole covariance.
        None => {
            let horizontal_m2 = (floored[(0, 0)] + floored[(1, 1)]) * 0.5;
            let scale = if horizontal_m2 > 0.0 {
                (parallax_h.powi(2) / horizontal_m2).max(1.0)
            } else {
                1.0
            };
            floored * scale
        }
    };
    independent[(2, 2)] += parallax_v.powi(2);
    // Rotation errors in map axes: heading about the vertical, tilt about
    // the horizontal axes.
    let rotation = to_map * independent.fixed_view::<3, 3>(3, 3) * to_map.transpose();
    AnchorBudget {
        independent,
        shared_horizontal_m: shared_h,
        shared_vertical_m: r.vertical_m,
        heading_rad: rotation[(2, 2)].max(0.0).sqrt(),
        tilt_rad: (rotation[(0, 0)] + rotation[(1, 1)])
            .max(0.0)
            .sqrt()
            .max(min_rotation_rad),
    }
}

/// Covariance of a turn about the horizontal map axes through the matched
/// ground that moves the camera horizontally by `parallax_m` (one sigma on
/// each horizontal axis).
///
/// The pivot comes from the image-geometry covariance. Image noise turns
/// the fitted camera about the matched ground, so the position error that
/// goes with a rotation error `phi` (map axes) is `phi x v`, where `v` is
/// the camera position minus the pivot. A turn about a horizontal axis then
/// moves the camera horizontally by `phi v.z`. `None` when the covariance
/// does not fit a pivot below the camera.
fn pivot_parallax(
    geometry: &Matrix6<f64>,
    to_map: &Matrix3<f64>,
    parallax_m: f64,
) -> Option<Matrix6<f64>> {
    let cross = geometry.fixed_view::<3, 3>(0, 3);
    let rotation = geometry.fixed_view::<3, 3>(3, 3).into_owned();
    // Position error per map-axis rotation error: `-[v]x` for a pure pivot.
    let lever = cross * rotation.try_inverse()? * to_map.transpose();
    let skew = (lever - lever.transpose()) * 0.5;
    let symmetric = (lever + lever.transpose()) * 0.5;
    let v = -Vector3::new(skew[(2, 1)], skew[(0, 2)], skew[(1, 0)]);
    if !v.iter().all(|x| x.is_finite())
        || v.z < MIN_PIVOT_DISTANCE_M
        || symmetric.norm() > MAX_NON_PIVOT_RATIO * skew.norm()
    {
        return None;
    }
    let sigma = parallax_m / v.z;
    let mut parallax = Matrix6::zeros();
    for axis in [Vector3::x(), Vector3::y()] {
        let mut turn = nalgebra::Vector6::zeros();
        turn.fixed_rows_mut::<3>(0).copy_from(&axis.cross(&v));
        turn.fixed_rows_mut::<3>(3)
            .copy_from(&(to_map.transpose() * axis));
        parallax += turn * turn.transpose() * sigma.powi(2);
    }
    Some(parallax)
}

/// A pivot closer below the camera than this is not a ground pivot, in metres.
const MIN_PIVOT_DISTANCE_M: f64 = 1.0;
/// Largest part of the position-rotation relation that a pivot does not
/// explain.
const MAX_NON_PIVOT_RATIO: f64 = 0.5;

#[cfg(test)]
mod tests;

//! Map anchors: camera poses matched against map imagery, and their error budget.

use crate::{FrameKey, pose::Pose, pose::from_camera};
use nalgebra::Matrix3;
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
    /// Image-geometry covariance of the position, in square metres.
    pub geometry_position_m2: Matrix3<f64>,
    /// Image-geometry rotation error, in radians.
    pub geometry_rotation_rad: f64,
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
        let c = &estimate.geometry_covariance;
        let rotation = (c[(3, 3)] + c[(4, 4)] + c[(5, 5)]).max(0.0).sqrt();
        Self {
            frame,
            observation_sha256: estimate.observation_sha256.clone(),
            map: estimate.map.clone(),
            local_frame: estimate.frame,
            pose: from_camera(&estimate.pose),
            geometry_position_m2: c.fixed_view::<3, 3>(0, 0).into_owned(),
            geometry_rotation_rad: rotation,
            backend: estimate.backend.clone(),
            reliability,
        }
    }
}

/// The error budget of one anchor, split into independent and shared parts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchorBudget {
    /// Independent position covariance: image geometry and parallax, in square metres.
    pub independent_m2: Matrix3<f64>,
    /// Horizontal map error that all anchors in one map cell share, in metres.
    pub shared_horizontal_m: f64,
    /// Vertical map error that all anchors in one map cell share, in metres.
    pub shared_vertical_m: f64,
    /// Rotation error, in radians.
    pub rotation_rad: f64,
}

impl AnchorBudget {
    /// Total horizontal one-sigma error, in metres.
    pub fn horizontal_m(&self) -> f64 {
        let i = &self.independent_m2;
        ((i[(0, 0)] + i[(1, 1)]) * 0.5 + self.shared_horizontal_m.powi(2)).sqrt()
    }
}

const MAX_PARALLAX_ANGLE_RAD: f64 = 1.31;
/// A map match fits the camera to bare-earth depth. Roofs, trees, elevation
/// model tilt, and calibration limit its attitude to about two degrees, even
/// when the image residuals are small.
const MIN_ROTATION_RAD: f64 = 0.035;

/// Combine image geometry, parallax, imagery age, and map accuracy.
///
/// Parallax uses the angle between the view axis and straight down, plus
/// half of the diagonal field of view. The result is a declared bound, not a
/// measured error.
pub(crate) fn budget(anchor: &AnchorObservation, half_diagonal_fov_rad: f64) -> AnchorBudget {
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
    let mut independent = anchor.geometry_position_m2;
    independent[(0, 0)] += parallax_h.powi(2);
    independent[(1, 1)] += parallax_h.powi(2);
    independent[(2, 2)] += parallax_v.powi(2);
    AnchorBudget {
        independent_m2: independent,
        shared_horizontal_m: shared_h,
        shared_vertical_m: r.vertical_m,
        rotation_rad: anchor.geometry_rotation_rad.max(MIN_ROTATION_RAD),
    }
}

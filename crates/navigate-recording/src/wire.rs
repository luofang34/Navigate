//! Stable records at the boundary between media preparation and navigation.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use navigate_visual::{CameraModel, CameraPose};
use serde::{Deserialize, Serialize};

use crate::Error;

/// Pinhole intrinsics at the processed image resolution.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Camera {
    /// Image size in pixels.
    pub size: [u32; 2],
    /// Focal lengths, in pixels.
    pub focal: [f64; 2],
    /// Principal point, in pixel-centre coordinates.
    pub principal: [f64; 2],
}

impl Camera {
    /// Convert the wire calibration into the geometry model.
    pub fn model(self) -> CameraModel {
        CameraModel {
            width: self.size[0],
            height: self.size[1],
            fx: self.focal[0],
            fy: self.focal[1],
            cx: self.principal[0],
            cy: self.principal[1],
        }
    }
}

/// Camera optical-centre pose in the map's declared local frame.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Pose {
    /// East, north, and map-datum altitude, in metres.
    pub position: [f64; 3],
    /// Eye-to-world unit quaternion in x, y, z, w order.
    pub rotation: [f64; 4],
}

impl Pose {
    /// Convert the wire pose into a validated camera pose.
    ///
    /// # Errors
    /// Rejects non-finite coordinates and invalid unit quaternions.
    pub fn camera(self) -> Result<CameraPose, Error> {
        let [x, y, z, w] = self.rotation;
        let q = Quaternion::new(w, x, y, z);
        if !q.coords.iter().all(|v| v.is_finite()) || (q.norm_squared() - 1.0).abs() > 1e-6 {
            return Err(Error::Invalid(
                "camera rotation is not a unit quaternion".into(),
            ));
        }
        let pose = CameraPose {
            position: Vector3::from(self.position),
            orientation: UnitQuaternion::new_normalize(q),
        };
        pose.validate()?;
        Ok(pose)
    }
}

impl From<CameraPose> for Pose {
    fn from(pose: CameraPose) -> Self {
        Self {
            position: pose.position.into(),
            rotation: pose.orientation.coords.into(),
        }
    }
}

/// Configuration retained with the evidence so the processing is reproducible.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    /// Processed-image calibration. Lens and mounting remain explicitly unknown.
    pub camera: Camera,
    /// Latitude and longitude of the local Mercator origin.
    pub anchor: [f64; 2],
    /// Immutable reference-map release.
    pub map_release: String,
    /// Reference package identifier.
    pub map_sha256: String,
    /// Search prior, separate from an accepted map observation.
    pub prior: Pose,
    /// Search radius around the externally supplied prior.
    pub position_radius_m: f64,
    /// Horizontal and vertical map uncertainty declarations, in metres.
    pub map_sigma_m: [f64; 2],
    /// Maximum height of objects absent from the bare-earth DEM.
    pub object_height_m: f64,
    /// Exposure timestamp uncertainty in the recording-local clock.
    pub capture_error_ns: u64,
    /// Origin and limitations of the recording clock.
    pub clock_description: String,
    /// Actual correspondence backend identity.
    pub matcher: String,
}

/// One rendered map view. Missing imagery or terrain must have zero depth.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    /// Grayscale reference PNG, relative to the evidence directory.
    pub image: String,
    /// Little-endian f32 optical-axis depths, in metres, row-major.
    pub depth: String,
    /// Camera that rendered this view.
    pub pose: Pose,
}

/// An immutable video observation and the correspondence evidence processed for it.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    /// Frame index in the source recording.
    pub sequence: u64,
    /// Exposure timestamp in the declared recording-local clock.
    pub capture_ns: u64,
    /// Container presentation timestamp, in nanoseconds.
    pub media_ns: u64,
    /// Processed grayscale PNG, relative to the evidence directory.
    pub image: String,
    /// Independently rendered map imagery for geographic anchoring.
    pub map_reference: Option<Reference>,
    /// Reference and query pixel centres for each map correspondence.
    pub map_matches: Vec<[[f64; 2]; 2]>,
    /// Map depth rendered at the preceding accepted camera pose.
    pub tracking_reference: Option<Reference>,
    /// Previous video and current video correspondences.
    pub tracking_matches: Vec<[[f64; 2]; 2]>,
    /// Terrain elevation below the camera, in the same map datum.
    pub terrain_m: f64,
    /// Revisit evidence for the preceding frame, checked before this frame.
    #[serde(default)]
    pub revisits: Vec<Revisit>,
}

/// Correspondences between an earlier keyframe and a later frame that sees
/// the same ground again. The session verifies the geometry.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Revisit {
    /// Later frame index; it must be the preceding frame of the stream.
    pub later: u64,
    /// Earlier keyframe index.
    pub earlier: u64,
    /// Exposure timestamp of the earlier keyframe.
    pub earlier_capture_ns: u64,
    /// Processed grayscale PNG of the earlier keyframe, relative to the evidence directory.
    pub earlier_image: String,
    /// Map depth rendered at the session estimate of the earlier keyframe.
    pub earlier_reference: Reference,
    /// Earlier (reference) and later (query) pixel centres.
    pub matches: Vec<[[f64; 2]; 2]>,
}

/// An earlier keyframe that the preceding frame may see again.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RevisitRequest {
    /// Later frame index.
    pub later: u64,
    /// Earlier keyframe index.
    pub earlier: u64,
    /// Session estimate of the earlier keyframe, for the depth render.
    pub earlier_pose: Pose,
    /// Predicted pose of the later frame. It is a search hint, not a measurement.
    pub later_pose: Pose,
}

/// A session result at one trajectory revision, not a body-position measurement.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResultFrame {
    /// Source frame identity.
    pub sequence: u64,
    /// Presentation timestamp in nanoseconds.
    pub media_ns: u64,
    /// Evidence digest of the processed pixels, intrinsics and capture stamp.
    pub observation_sha256: String,
    /// Revised map camera pose, absent when the session cannot locate the frame.
    pub pose: Option<Pose>,
    /// Latitude, longitude and map-datum camera altitude.
    pub geodetic: Option<[f64; 3]>,
    /// Camera AGL and its uncertainty, both in metres.
    pub camera_agl: Option<[f64; 2]>,
    /// Declared position uncertainty bound, in metres.
    pub position_bound_m: Option<f64>,
    /// Whether more than one anchor supports this trajectory component.
    pub confirmed: bool,
    /// This observation supplied an admitted map match, not just relative tracking.
    #[serde(default)]
    pub map_observation_accepted: bool,
    /// Evidence bounds and permitted uses; absent in legacy recordings.
    #[serde(default)]
    pub support: Option<OutputSupport>,
    /// Trajectory revision used by this result.
    pub revision: u64,
    /// Admission or geometry diagnostics; these are not navigation fixes.
    pub diagnostics: Vec<String>,
}

/// Evidence bounds and permitted uses of a located camera pose.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSupport {
    /// Whether recent map evidence supports the pose.
    pub recent_map_support: bool,
    /// Whether the position bound permits navigation use.
    pub navigation: bool,
    /// Whether position, tilt and map support permit ground projection.
    pub ground_projection: bool,
    /// Conservative tilt bound, in degrees.
    pub tilt_bound_deg: Option<f64>,
    /// Capture-time distance to the nearest map anchor, in seconds.
    pub nearest_anchor_s: Option<f64>,
}

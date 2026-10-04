//! Per-frame camera and body height above the vertical reference and above ground.
//!
//! Image matches and intrinsics alone cannot give metric height: a scene
//! twice as far away with twice the size makes the same image. Height comes
//! from the map anchors (map imagery and elevation), from a host sensor, or
//! from a range measurement. When none of these exists, the result states
//! the reason and contains no value.

use crate::{Calibration, FrameKey, FramePose, MapPose, Unlocated, VerticalReference, pose::Pose};
use nalgebra::UnitQuaternion;
use navigate_visual::MapRevision;

/// The origin of a terrain elevation.
#[derive(Clone, Debug, PartialEq)]
pub enum TerrainSource {
    /// The elevation model of the map package.
    MapElevation {
        /// Map revision.
        map: MapRevision,
    },
    /// Another host terrain source.
    Host {
        /// Host name of the source.
        name: String,
    },
}

/// Terrain elevation below the camera, at its horizontal position.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainSample {
    /// Terrain elevation, in metres.
    pub elevation_m: f64,
    /// One-sigma error, in metres.
    pub sigma_m: f64,
    /// Vertical reference of `elevation_m`.
    pub vertical: VerticalReference,
    /// Origin of the value.
    pub source: TerrainSource,
}

/// The origin of a host height measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeightSensor {
    /// Satellite navigation height.
    Gnss,
    /// Barometric height with a known reference pressure.
    Barometer,
    /// Another host source.
    Host(String),
}

/// Host measurement of the body-origin height above a vertical reference.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightSample {
    /// Height, in metres.
    pub height_m: f64,
    /// One-sigma error, in metres.
    pub sigma_m: f64,
    /// Vertical reference of `height_m`.
    pub vertical: VerticalReference,
    /// Sensor.
    pub sensor: HeightSensor,
}

/// The origin of a range along the optical axis.
#[derive(Clone, Debug, PartialEq)]
pub enum RangeSource {
    /// Depth at the principal point, rendered from the map at this frame pose.
    RenderedDepth {
        /// Map revision.
        map: MapRevision,
    },
    /// A rangefinder aligned with the optical axis.
    Rangefinder,
}

/// Distance from the camera to the surface along the optical axis.
#[derive(Clone, Debug, PartialEq)]
pub struct AxisRange {
    /// Range, in metres.
    pub range_m: f64,
    /// One-sigma error, in metres.
    pub sigma_m: f64,
    /// Origin of the value.
    pub source: RangeSource,
}

/// Host inputs for one frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HeightInputs {
    /// Terrain elevation below the camera.
    pub terrain: Option<TerrainSample>,
    /// Host height measurement at the frame capture time.
    pub sensor_height: Option<HeightSample>,
    /// Range along the optical axis.
    pub axis_range: Option<AxisRange>,
    /// Gimbal attitude relative to its base at the frame capture time.
    pub gimbal: Option<UnitQuaternion<f64>>,
}

/// Where a height value comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum HeightBasis {
    /// Map-anchored camera pose.
    MapPose,
    /// Host height sensor.
    Sensor(HeightSensor),
    /// Height above the reference minus the terrain elevation.
    TerrainDifference,
    /// Range along the optical axis.
    Range(RangeSource),
}

/// Why a height is not available.
#[derive(Clone, Debug, PartialEq)]
pub enum Unobservable {
    /// The frame has no map pose and no host height.
    NoHeight(Unlocated),
    /// No terrain elevation for the frame.
    NoTerrain,
    /// Height and terrain use different vertical references.
    VerticalMismatch,
    /// A vertical reference is unknown, and the values do not share one map.
    VerticalUnknown,
    /// The camera mount is unknown.
    MountUnknown,
    /// The gimbal attitude of the frame is missing.
    GimbalMissing,
    /// No range along the optical axis.
    NoRange,
    /// An input value is not finite or its error is not positive.
    InvalidInput,
}

/// One height result.
#[derive(Clone, Debug, PartialEq)]
pub enum Height {
    /// A value with a declared one-sigma error.
    Estimated {
        /// Value, in metres.
        value_m: f64,
        /// One-sigma error, in metres.
        sigma_m: f64,
        /// Origin of the value.
        basis: HeightBasis,
    },
    /// No value is available.
    Unobservable(Unobservable),
}

/// Heights of one frame. Each value is a separate quantity.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightEstimate {
    /// Frame.
    pub frame: FrameKey,
    /// Camera optical-centre height above `vertical`.
    pub camera_height: Height,
    /// Body-origin height above `vertical`.
    pub body_height: Height,
    /// Vertical distance from the camera to the terrain below it.
    pub camera_agl: Height,
    /// Vertical distance from the body origin to the terrain below it.
    pub body_agl: Height,
    /// Distance from the camera to the surface along the optical axis. It is
    /// a slant range unless the camera looks straight down.
    pub axis_range: Height,
    /// Vertical reference of the heights above the reference.
    pub vertical: VerticalReference,
}

struct Reference {
    camera: Height,
    body: Height,
    vertical: VerticalReference,
    from_map: bool,
}

/// Compute the heights of one frame from its pose and host inputs.
pub fn estimate_height(
    pose: &FramePose,
    calibration: &Calibration,
    inputs: &HeightInputs,
) -> HeightEstimate {
    let body_from_camera = calibration.mount.body_from_camera(inputs.gimbal.as_ref());
    let mount_reason = match (&calibration.mount, &body_from_camera) {
        (_, Some(_)) => None,
        (crate::CameraMount::Gimbal { .. }, None) => Some(Unobservable::GimbalMissing),
        _ => Some(Unobservable::MountUnknown),
    };
    let reference = reference_heights(
        pose,
        calibration,
        inputs,
        body_from_camera.as_ref(),
        mount_reason.clone(),
    );
    let camera_agl = above_terrain(&reference.camera, &reference, inputs);
    let body_agl = above_terrain(&reference.body, &reference, inputs);
    let axis_range = match &inputs.axis_range {
        Some(r)
            if r.range_m.is_finite()
                && r.range_m > 0.0
                && r.sigma_m.is_finite()
                && r.sigma_m > 0.0 =>
        {
            Height::Estimated {
                value_m: r.range_m,
                sigma_m: r.sigma_m,
                basis: HeightBasis::Range(r.source.clone()),
            }
        }
        Some(_) => Height::Unobservable(Unobservable::InvalidInput),
        None => Height::Unobservable(Unobservable::NoRange),
    };
    HeightEstimate {
        frame: pose.frame,
        camera_height: reference.camera,
        body_height: reference.body,
        camera_agl,
        body_agl,
        axis_range,
        vertical: reference.vertical,
    }
}

fn reference_heights(
    pose: &FramePose,
    calibration: &Calibration,
    inputs: &HeightInputs,
    body_from_camera: Option<&Pose>,
    mount_reason: Option<Unobservable>,
) -> Reference {
    let missing = |reason: Unobservable| Height::Unobservable(reason);
    if let MapPose::Located {
        pose: camera,
        position_bound_m,
        ..
    } = &pose.map
    {
        let body = match (body_from_camera, mount_reason) {
            (Some(b), _) => estimated(
                (camera * b.inverse()).translation.vector.z,
                *position_bound_m,
                HeightBasis::MapPose,
            ),
            (None, reason) => missing(reason.unwrap_or(Unobservable::MountUnknown)),
        };
        return Reference {
            camera: estimated(
                camera.translation.vector.z,
                *position_bound_m,
                HeightBasis::MapPose,
            ),
            body,
            vertical: calibration.map_vertical.clone(),
            from_map: true,
        };
    }
    let unlocated = match &pose.map {
        MapPose::Unlocated(reason) => *reason,
        MapPose::Located { .. } => Unlocated::NoAnchor,
    };
    let Some(sample) = &inputs.sensor_height else {
        return Reference {
            camera: missing(Unobservable::NoHeight(unlocated)),
            body: missing(Unobservable::NoHeight(unlocated)),
            vertical: VerticalReference::Unknown,
            from_map: false,
        };
    };
    if !(sample.height_m.is_finite() && sample.sigma_m.is_finite() && sample.sigma_m > 0.0) {
        return Reference {
            camera: missing(Unobservable::InvalidInput),
            body: missing(Unobservable::InvalidInput),
            vertical: sample.vertical.clone(),
            from_map: false,
        };
    }
    let basis = HeightBasis::Sensor(sample.sensor.clone());
    let body = estimated(sample.height_m, sample.sigma_m, basis.clone());
    // The camera offset needs the vertical direction in the body frame, which
    // only a located camera attitude gives.
    let camera = match mount_reason {
        Some(reason) => missing(reason),
        None => missing(Unobservable::NoHeight(unlocated)),
    };
    Reference {
        camera,
        body,
        vertical: sample.vertical.clone(),
        from_map: false,
    }
}

fn above_terrain(height: &Height, reference: &Reference, inputs: &HeightInputs) -> Height {
    let Height::Estimated {
        value_m, sigma_m, ..
    } = height
    else {
        return height.clone();
    };
    let Some(terrain) = &inputs.terrain else {
        return Height::Unobservable(Unobservable::NoTerrain);
    };
    if !(terrain.elevation_m.is_finite() && terrain.sigma_m.is_finite() && terrain.sigma_m > 0.0) {
        return Height::Unobservable(Unobservable::InvalidInput);
    }
    let shared_map =
        reference.from_map && matches!(terrain.source, TerrainSource::MapElevation { .. });
    match (&reference.vertical, &terrain.vertical) {
        (VerticalReference::Unknown, _) | (_, VerticalReference::Unknown) if !shared_map => {
            return Height::Unobservable(Unobservable::VerticalUnknown);
        }
        (a, b) if a != b && !shared_map => {
            return Height::Unobservable(Unobservable::VerticalMismatch);
        }
        _ => {}
    }
    estimated(
        value_m - terrain.elevation_m,
        (sigma_m.powi(2) + terrain.sigma_m.powi(2)).sqrt(),
        HeightBasis::TerrainDifference,
    )
}

fn estimated(value_m: f64, sigma_m: f64, basis: HeightBasis) -> Height {
    if value_m.is_finite() && sigma_m.is_finite() && sigma_m > 0.0 {
        Height::Estimated {
            value_m,
            sigma_m,
            basis,
        }
    } else {
        Height::Unobservable(Unobservable::InvalidInput)
    }
}

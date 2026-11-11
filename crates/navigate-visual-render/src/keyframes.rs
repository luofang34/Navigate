//! Route keyframes (ADR-0022, representation K): reference views rendered
//! at planned poses, stored as a derived product with a validity envelope.

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use navigate_data::product::{
    Calibration, PlannedPose, ProductKind, ProductWriter, Recipe, ValidityEnvelope,
};
use navigate_data::resource::{great_circle_m, point_along};
use navigate_data::{Package, PlannedLeg, ReadPort, Sha256Digest, WritePort};
use navigate_visual::{CameraModel, CameraPose, ReferenceView};

use crate::package_scene::{Geoid, SceneRequest, scene_around};
use crate::{RenderError, SparseSceneSampler};

/// Magic of one encoded keyframe view.
pub const KEYFRAME_MAGIC: &[u8; 8] = b"NVKEYV01";

/// How keyframes are made. Each value goes into the product recipe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyframeSpec<'a> {
    /// Imagery layer identifier.
    pub imagery_layer: &'a str,
    /// Imagery zoom.
    pub imagery_zoom: u32,
    /// Elevation layer identifier.
    pub elevation_layer: &'a str,
    /// Elevation zoom.
    pub elevation_zoom: u32,
    /// The camera.
    pub camera: CameraModel,
    /// Angle of the optical axis below the horizontal, radians.
    pub depression_rad: f64,
    /// Distance between keyframes along a leg, metres.
    pub spacing_m: f64,
    /// Validity envelope of each keyframe.
    pub envelope: ValidityEnvelope,
    /// Farthest ray distance, metres.
    pub far_m: f64,
    /// Ray-march steps.
    pub steps: u32,
    /// Bisection refinements of a bracketed hit.
    pub refinements: u32,
    /// Grid stride of the adaptive renderer, pixels.
    pub stride: u32,
    /// Luma step that the renderer refines, not interpolates.
    pub luma_edge: u8,
    /// Relative depth step that the renderer refines.
    pub depth_edge: f32,
    /// Altitude band of the legs.
    pub altitude_band: u32,
}

impl KeyframeSpec<'_> {
    /// The complete recipe of the product.
    #[must_use]
    pub fn recipe(&self) -> Recipe {
        let parameters = BTreeMap::from([
            ("imagery_layer".into(), self.imagery_layer.to_owned()),
            ("imagery_zoom".into(), self.imagery_zoom.to_string()),
            ("elevation_layer".into(), self.elevation_layer.to_owned()),
            ("elevation_zoom".into(), self.elevation_zoom.to_string()),
            ("depression_rad".into(), self.depression_rad.to_string()),
            ("spacing_m".into(), self.spacing_m.to_string()),
            ("far_m".into(), self.far_m.to_string()),
            ("steps".into(), self.steps.to_string()),
            ("refinements".into(), self.refinements.to_string()),
            ("stride".into(), self.stride.to_string()),
            ("luma_edge".into(), self.luma_edge.to_string()),
            ("depth_edge".into(), self.depth_edge.to_string()),
            (
                "encoding".into(),
                "NVKEYV01 u32 width, u32 height, luma, f32 depth".into(),
            ),
        ]);
        Recipe {
            algorithm: "navigate-visual-render sparse adaptive ray march".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            weights: None,
            precision: "f32".into(),
            parameters,
        }
    }
}

/// The eye-to-world rotation of a camera at `heading_rad` (clockwise from
/// north) and `depression_rad` (below the horizontal) in east, north, up
/// axes. The eye axes are right, up, and back.
#[must_use]
pub fn camera_orientation(heading_rad: f64, depression_rad: f64) -> UnitQuaternion<f64> {
    let (sh, ch) = heading_rad.sin_cos();
    let (sd, cd) = depression_rad.sin_cos();
    let forward = Vector3::new(sh * cd, ch * cd, -sd);
    let right = Vector3::new(ch, -sh, 0.0);
    let up = right.cross(&forward);
    let matrix = Matrix3::from_columns(&[right, up, -forward]);
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(matrix))
}

/// Encodes one reference view: magic, width, height, luma, depth.
#[must_use]
pub fn encode_view(view: &ReferenceView) -> Vec<u8> {
    let (width, height) = view.image.dimensions();
    let mut out = Vec::with_capacity(16 + view.image.as_raw().len() + view.depth_m.len() * 4);
    out.extend_from_slice(KEYFRAME_MAGIC);
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(view.image.as_raw());
    for d in &view.depth_m {
        out.extend_from_slice(&d.to_le_bytes());
    }
    out
}

/// Decodes one keyframe view: `(width, height, luma, depth)`.
///
/// # Errors
///
/// Refuses bytes with another magic or another length.
pub fn decode_view(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>, Vec<f32>), RenderError> {
    let bad = || RenderError::Invalid("keyframe view encoding");
    let (magic, rest) = bytes.split_first_chunk::<8>().ok_or_else(bad)?;
    if magic != KEYFRAME_MAGIC {
        return Err(bad());
    }
    let (w, rest) = rest.split_first_chunk::<4>().ok_or_else(bad)?;
    let (h, rest) = rest.split_first_chunk::<4>().ok_or_else(bad)?;
    let (width, height) = (u32::from_le_bytes(*w), u32::from_le_bytes(*h));
    let pixels = width as usize * height as usize;
    if rest.len() != pixels * 5 {
        return Err(bad());
    }
    let (luma, depth) = rest.split_at(pixels);
    let depth = depth
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok((width, height, luma.to_vec(), depth))
}

/// The planned poses along the legs, at `spacing_m`, each with the heading
/// of the leg at that point.
fn planned_poses(legs: &[PlannedLeg], spacing_m: f64) -> Vec<PlannedPose> {
    let mut poses = Vec::new();
    for leg in legs {
        let (a, b) = ([leg.from[0], leg.from[1]], [leg.to[0], leg.to[1]]);
        let length = great_circle_m(a, b);
        let count = (length / spacing_m).floor().max(0.0) as u32;
        let step = 1.0 / length.max(1.0);
        for i in 0..=count {
            let t = (f64::from(i) * spacing_m / length.max(1.0)).min(1.0);
            let [lat, lon] = point_along(a, b, t);
            let before = point_along(a, b, (t - step).max(0.0));
            let after = point_along(a, b, (t + step).min(1.0));
            let heading = bearing(before, after);
            let altitude = leg.from[2] + (leg.to[2] - leg.from[2]) * t;
            poses.push(PlannedPose {
                position: [lat, lon, altitude],
                heading_rad: heading,
            });
        }
    }
    poses
}

fn bearing([lat1, lon1]: [f64; 2], [lat2, lon2]: [f64; 2]) -> f64 {
    let (p1, p2, dl) = (
        lat1.to_radians(),
        lat2.to_radians(),
        (lon2 - lon1).to_radians(),
    );
    (dl.sin() * p2.cos()).atan2(p1.cos() * p2.sin() - p1.sin() * p2.cos() * dl.cos())
}

/// Renders a keyframe at each planned pose along `legs` from the package
/// and writes them as a sealed product. The leg altitudes are heights above
/// the WGS84 ellipsoid. Returns the product identity and the keyframe
/// count.
///
/// # Errors
///
/// Refuses an unusable spec, a scene that the package cannot supply, a
/// failed render, and a failed write.
pub async fn build_keyframes<S: ReadPort + WritePort + ?Sized>(
    package: &Package,
    store: &mut S,
    legs: &[PlannedLeg],
    spec: &KeyframeSpec<'_>,
    geoid: Geoid<'_>,
) -> Result<(Sha256Digest, usize), RenderError> {
    spec.camera.validate()?;
    if !(spec.spacing_m.is_finite() && spec.spacing_m > 0.0) {
        return Err(RenderError::Invalid("keyframe spacing"));
    }
    let c = &spec.camera;
    let mut writer = ProductWriter::new(ProductKind::Keyframes, package.identity(), spec.recipe())
        .with_calibration(Calibration {
            width: c.width,
            height: c.height,
            fx: c.fx,
            fy: c.fy,
            cx: c.cx,
            cy: c.cy,
        })
        .with_altitude_band(spec.altitude_band)
        .with_envelope(spec.envelope);
    let poses = planned_poses(legs, spec.spacing_m);
    for (index, pose) in poses.iter().enumerate() {
        let view = render_at(package, &*store, spec, pose, geoid).await?;
        let key = format!("k{index:06}");
        writer
            .add_entry(store, &key, Some(*pose), &encode_view(&view))
            .await?;
    }
    let identity = writer.seal(store).await?;
    Ok((identity, poses.len()))
}

/// Renders the reference view of one planned pose.
///
/// # Errors
///
/// Refuses a scene that the package cannot supply and a failed render.
pub async fn render_at<S: ReadPort + ?Sized>(
    package: &Package,
    store: &S,
    spec: &KeyframeSpec<'_>,
    pose: &PlannedPose,
    geoid: Geoid<'_>,
) -> Result<ReferenceView, RenderError> {
    let [lat, lon, height] = pose.position;
    let request = SceneRequest {
        imagery_layer: spec.imagery_layer,
        imagery_zoom: spec.imagery_zoom,
        elevation_layer: spec.elevation_layer,
        elevation_zoom: spec.elevation_zoom,
        centre: [lat, lon],
        half_side_m: spec.far_m,
    };
    let window = scene_around(package, store, &request, geoid).await?;
    let camera_pose = CameraPose {
        position: window.scene.frame.local([lat, lon, height])?,
        orientation: camera_orientation(pose.heading_rad, spec.depression_rad),
    };
    let sampler = SparseSceneSampler::new(&window.scene, spec.far_m, spec.steps, spec.refinements)?;
    let (view, _) = sampler.render_adaptive(
        &spec.camera,
        &camera_pose,
        spec.stride,
        spec.luma_edge,
        spec.depth_edge,
    )?;
    Ok(view)
}

#[cfg(test)]
mod tests;

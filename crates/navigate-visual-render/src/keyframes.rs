//! Route keyframes (ADR-0022, representation K): reference views rendered
//! at planned poses, stored as a derived product with a validity envelope.

use std::collections::BTreeSet;

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use navigate_data::product::{
    Calibration, ChunkBinding, PlannedPose, Precision, ProductKind, ProductWriter, Recipe,
    ValidityEnvelope, route_digest,
};
use navigate_data::resource::{great_circle_m, point_along};
use navigate_data::{GeoidModel, LayerKey, Package, PlannedLeg, ReadPort, Sha256Digest, WritePort};
use navigate_visual::{CameraModel, CameraPose, LocalFrame, MapRevision, ReferenceView};

use crate::package_scene::{Geoid, SceneRequest, scene_around};
use crate::{RenderError, SparseSceneSampler};

/// The recipe algorithm of the keyframes that this module builds and reads.
pub const KEYFRAME_ALGORITHM: &str = "navigate-visual-render/sparse-adaptive-ray-march";

/// Magic of one encoded keyframe view.
pub const KEYFRAME_MAGIC: &[u8; 8] = b"NVKEYV01";

/// How keyframes are made. Each value goes into the product recipe as an
/// integer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyframeSpec {
    /// Imagery layer.
    pub imagery_layer: LayerKey,
    /// Imagery zoom.
    pub imagery_zoom: u32,
    /// Elevation layer.
    pub elevation_layer: LayerKey,
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

fn micro(v: f64) -> i64 {
    (v * 1e6).round() as i64
}

/// How the keyframes of a chunk were rendered, as its recipe states.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyframeRendering {
    /// Angle of the optical axis of each view below the horizontal, radians.
    pub depression_rad: f64,
    /// The geoid model that the builder had for the heights of an
    /// orthometric elevation layer. The depth of each view follows from
    /// it, so a reader with another model, or with none, has other
    /// terrain heights than the views.
    pub geoid: Option<GeoidModel>,
}

/// The recipe value of a geoid model: zero for none.
const fn geoid_code(model: Option<GeoidModel>) -> i64 {
    match model {
        Some(GeoidModel::Egm96) => 1,
        Some(GeoidModel::Egm2008) => 2,
        Some(GeoidModel::Unsupported) | None => 0,
    }
}

impl KeyframeSpec {
    /// The complete recipe of the product, in integer units, for a builder
    /// with the geoid model `geoid`.
    #[must_use]
    pub fn recipe(&self, geoid: Option<GeoidModel>) -> Recipe {
        Recipe::new(
            KEYFRAME_ALGORITHM,
            env!("CARGO_PKG_VERSION"),
            None,
            Precision::F32,
        )
        .with("imagery_zoom", i64::from(self.imagery_zoom))
        .with("elevation_zoom", i64::from(self.elevation_zoom))
        .with("depression_urad", micro(self.depression_rad))
        .with("spacing_mm", (self.spacing_m * 1e3).round() as i64)
        .with("far_mm", (self.far_m * 1e3).round() as i64)
        .with("steps", i64::from(self.steps))
        .with("refinements", i64::from(self.refinements))
        .with("stride_px", i64::from(self.stride))
        .with("luma_edge", i64::from(self.luma_edge))
        .with("depth_edge_ppm", micro(f64::from(self.depth_edge)))
        .with("encoding_nvkeyv01", 1)
        .with("geoid_model", geoid_code(geoid))
    }

    /// What a recipe of [`Self::recipe`] states about the views, or `None`
    /// for a recipe of another algorithm or encoding.
    #[must_use]
    pub fn rendering(recipe: &Recipe) -> Option<KeyframeRendering> {
        let known = recipe.algorithm == KEYFRAME_ALGORITHM
            && recipe.parameters.get("encoding_nvkeyv01") == Some(&1);
        let depression = recipe.parameters.get("depression_urad")?;
        let geoid = match recipe.parameters.get("geoid_model")? {
            0 => None,
            1 => Some(GeoidModel::Egm96),
            2 => Some(GeoidModel::Egm2008),
            _ => return None,
        };
        known.then_some(KeyframeRendering {
            depression_rad: *depression as f64 * 1e-6,
            geoid,
        })
    }

    /// The camera calibration in micro-pixels.
    #[must_use]
    pub fn calibration(&self) -> Calibration {
        let c = &self.camera;
        Calibration {
            width: c.width,
            height: c.height,
            fx_upx: micro(c.fx),
            fy_upx: micro(c.fy),
            cx_upx: micro(c.cx),
            cy_upx: micro(c.cy),
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

/// The reference view of one stored keyframe: the view that
/// [`build_keyframes`] rendered at `pose` with a camera `depression_rad`
/// below the horizontal, from its entry blob `bytes`. The frame is the
/// WGS84 east, north, up frame at the planned position.
///
/// # Errors
///
/// Refuses bytes that [`decode_view`] refuses and a pose outside the frame
/// range.
pub fn reference_view(
    map: MapRevision,
    pose: &PlannedPose,
    depression_rad: f64,
    bytes: &[u8],
) -> Result<ReferenceView, RenderError> {
    let (width, height, luma, depth_m) = decode_view(bytes)?;
    let image = image::GrayImage::from_raw(width, height, luma)
        .ok_or(RenderError::Invalid("keyframe view encoding"))?;
    let ([lat, lon], altitude, heading) = pose.to_si();
    let frame = LocalFrame::wgs84_enu(lat, lon)?;
    Ok(ReferenceView {
        map,
        frame,
        pose: CameraPose {
            position: frame.local([lat, lon, altitude])?,
            orientation: camera_orientation(heading, depression_rad),
        },
        image,
        depth_m,
    })
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
            poses.push(PlannedPose::from_si([lat, lon], altitude, heading));
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
/// and writes them as one sealed product chunk. The leg altitudes are
/// heights above the WGS84 ellipsoid. The chunk binds the route digest, the
/// envelope, the altitude band, the calibration, and the digests of the
/// tile blobs that the views were rendered from. Returns the chunk digest
/// and the keyframe count.
///
/// # Errors
///
/// Refuses an unusable spec, a scene that the package cannot supply, a
/// failed render, and a failed write.
pub async fn build_keyframes<S: ReadPort + WritePort + ?Sized>(
    package: &Package<S::File>,
    store: &mut S,
    legs: &[PlannedLeg],
    spec: &KeyframeSpec,
    geoid: Geoid<'_>,
) -> Result<(Sha256Digest, usize), RenderError> {
    spec.camera.validate()?;
    if !(spec.spacing_m.is_finite() && spec.spacing_m > 0.0) {
        return Err(RenderError::Invalid("keyframe spacing"));
    }
    let poses = planned_poses(legs, spec.spacing_m);
    let mut views = Vec::with_capacity(poses.len());
    let mut inputs = BTreeSet::new();
    for pose in &poses {
        let (view, used) = render_with_inputs(package, &*store, spec, pose, geoid).await?;
        inputs.extend(used);
        views.push(encode_view(&view));
    }
    let mut points: Vec<[f64; 3]> = legs.iter().map(|l| l.from).collect();
    points.extend(legs.last().map(|l| l.to));
    let binding = ChunkBinding {
        cell: None,
        altitude_band: Some(spec.altitude_band),
        inputs: inputs.into_iter().collect(),
        route: Some(route_digest(&points)),
        envelope: Some(spec.envelope),
        calibration: Some(spec.calibration()),
    };
    let recipe = spec.recipe(geoid.map(|(model, _)| model));
    let mut writer = ProductWriter::new(ProductKind::Keyframes, recipe, binding);
    for (key, (pose, view)) in (0_u32..).zip(poses.iter().zip(&views)) {
        writer.add_entry(store, key, Some(*pose), view).await?;
    }
    let digest = writer.seal(store).await?;
    Ok((digest, poses.len()))
}

/// Renders the reference view of one planned pose.
///
/// # Errors
///
/// Refuses a scene that the package cannot supply and a failed render.
pub async fn render_at<S: ReadPort + ?Sized>(
    package: &Package<S::File>,
    store: &S,
    spec: &KeyframeSpec,
    pose: &PlannedPose,
    geoid: Geoid<'_>,
) -> Result<ReferenceView, RenderError> {
    render_with_inputs(package, store, spec, pose, geoid)
        .await
        .map(|(view, _)| view)
}

async fn render_with_inputs<S: ReadPort + ?Sized>(
    package: &Package<S::File>,
    store: &S,
    spec: &KeyframeSpec,
    pose: &PlannedPose,
    geoid: Geoid<'_>,
) -> Result<(ReferenceView, BTreeSet<Sha256Digest>), RenderError> {
    let ([lat, lon], height, heading) = pose.to_si();
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
        orientation: camera_orientation(heading, spec.depression_rad),
    };
    let sampler = SparseSceneSampler::new(&window.scene, spec.far_m, spec.steps, spec.refinements)?;
    let (view, _) = sampler.render_adaptive(
        &spec.camera,
        &camera_pose,
        spec.stride,
        spec.luma_edge,
        spec.depth_edge,
    )?;
    Ok((view, window.inputs))
}

#[cfg(test)]
mod tests;

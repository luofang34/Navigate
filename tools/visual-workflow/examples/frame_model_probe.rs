//! Synthetic physical-model controls through the public geometric acceptance path.
use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector2, Vector3};
use navigate_visual::{
    CameraModel, CameraPose, Frame, FrameStamp, LocalFrame, LocalizerConfig, MapRevision,
    PixelMatch, PosePrior, PoseVerifier, ReferenceView,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("probe I/O at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("probe JSON at {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(transparent)]
    Visual(#[from] navigate_visual::VisualError),
    #[error("invalid probe input: {0}")]
    Input(&'static str),
}
#[derive(Deserialize)]
struct Input {
    #[serde(rename = "K")]
    k: [[f64; 3]; 3],
    image_width: u32,
    image_height: u32,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    anchor_deg: [f64; 2],
    height_m: f64,
    eye_to_world_rotation: String,
    true_camera_enu_m: [f64; 3],
    points: Vec<Point>,
}
#[derive(Deserialize)]
struct Point {
    geodetic_deg_m: [f64; 3],
    wgs84_enu_m: [f64; 3],
    sphere_enu_m: [f64; 3],
    query_original_pixel: [f64; 2],
}
fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(args.next().ok_or(Error::Input("fixture path required"))?);
    let output = PathBuf::from(args.next().ok_or(Error::Input("output path required"))?);
    if args.next().is_some() {
        return Err(Error::Input("only input and output paths are accepted"));
    }
    let bytes = read_blocking(&input)?;
    let data: Input = serde_json::from_slice(&bytes).map_err(|source| Error::Json {
        path: input.clone(),
        source,
    })?;
    let camera = CameraModel {
        width: data.image_width,
        height: data.image_height,
        fx: data.k[0][0],
        fy: data.k[1][1],
        cx: data.k[0][2],
        cy: data.k[1][2],
    };
    camera.validate()?;
    if camera.width > 2048
        || camera.height > 2048
        || data.cases.is_empty()
        || data.cases.len() > 100
    {
        return Err(Error::Input("fixture dimensions or count"));
    }
    let mut rows = Vec::new();
    for case in data.cases {
        for sphere in [true, false] {
            rows.push(run_case(&case, camera, sphere)?);
        }
    }
    let report = json!({"fixture_sha256":format!("{:x}",Sha256::digest(bytes)),"cases":rows,
        "scope":"Synthetic WGS84 image observations, supplied independently of the Rust fitter. Sphere map points use LocalFrame::local. Both point sets use verify_calibrated_reference with default acceptance. Reference depth is f32; its reconstruction error is recorded. WGS84 XYZ is a solver control only: the ReferenceView frame tag is not used to convert its pose or claim a supported ellipsoidal deployment frame. No image matching, real-flight accuracy, or covariance coverage claim."});
    write_blocking(&output, &report)?;
    tracing::info!(path=%output.display(),"Saved geometric model probe");
    Ok(())
}
fn run_case(case: &Case, camera: CameraModel, sphere: bool) -> Result<Value, Error> {
    if case.eye_to_world_rotation
        != "identity in Navigate right/up/back convention (camera looks along -Z)"
    {
        return Err(Error::Input("fixture orientation must be identity"));
    }
    if case.points.len() < 20
        || case.points.len() > 4096
        || !case.height_m.is_finite()
        || case.height_m <= 0.0
    {
        return Err(Error::Input("case bounds"));
    }
    let frame_model = LocalFrame::spherical_enu(case.anchor_deg[0], case.anchor_deg[1])?;
    let truth = CameraPose {
        position: case.true_camera_enu_m.into(),
        orientation: UnitQuaternion::identity(),
    };
    truth.validate()?;
    let seed = CameraPose {
        position: truth.position + Vector3::new(37.0, -23.0, 41.0),
        orientation: UnitQuaternion::from_euler_angles(0.015, -0.012, 0.008),
    };
    let frame = Frame {
        camera,
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 1,
        },
        image: GrayImage::new(camera.width, camera.height),
    };
    let mut reference = ReferenceView {
        frame: frame_model,
        pose: seed,
        map: MapRevision {
            release_id: "synthetic-physical-model-control".into(),
            manifest_sha256: "a".repeat(64),
        },
        image: frame.image.clone(),
        depth_m: vec![0.0; (camera.width * camera.height) as usize],
    };
    let (pairs, sphere_fixture_delta, depth_roundtrip, max_map_displacement) =
        fill_reference(case, camera, sphere, &mut reference)?;
    let prior = PosePrior {
        pose: seed,
        position_radius_m: 1000.0,
        attitude_radius_rad: 0.3,
    };
    let verifier = PoseVerifier::new(LocalizerConfig::default())?;
    let start = Instant::now();
    let result = verifier.verify_calibrated_reference(
        &frame,
        &reference,
        &camera,
        &prior,
        &pairs,
        "synthetic-oracle",
    );
    let mut row = json!({"case":case.id,"height_m":case.height_m,"anchor_deg":case.anchor_deg,"geometry":if sphere {"sphere"} else {"wgs84_xyz_solver_control"},"accepted":result.is_ok(),"solve_ms":start.elapsed().as_secs_f64()*1000.0,"sphere_fixture_max_difference_m":sphere_fixture_delta,"reference_depth_max_roundtrip_error_m":depth_roundtrip,"max_map_point_displacement_m":max_map_displacement});
    match result {
        Ok(e) => {
            row["position"] = json!([e.pose.position.x, e.pose.position.y, e.pose.position.z]);
            row["position_error_m"] = json!((e.pose.position - truth.position).norm());
            row["height_error_m"] = json!(e.pose.position.z - truth.position.z);
            row["rms_px"] = json!(e.quality.reprojection_rms_px);
            row["inliers"] = json!(e.quality.inliers);
            row["spatial_support"] = json!(e.quality.spatial_support);
            row["occupied_cells"] = json!(e.quality.occupied_cells);
            row["condition_number"] = json!(e.quality.condition_number);
            row["position_covariance_diagonal_m2"] = json!([
                e.geometry_covariance[(0, 0)],
                e.geometry_covariance[(1, 1)],
                e.geometry_covariance[(2, 2)]
            ]);
            if sphere {
                row["camera_point_model_error_m"] =
                    json!(frame_model.model_error_m(e.pose.position));
            }
        }
        Err(e) => row["rejection"] = json!(e.to_string()),
    }
    Ok(row)
}
fn fill_reference(
    case: &Case,
    camera: CameraModel,
    sphere: bool,
    reference: &mut ReferenceView,
) -> Result<(Vec<PixelMatch>, f64, f64, f64), Error> {
    let mut pairs = Vec::new();
    let mut fixture_delta = 0.0_f64;
    let mut roundtrip = 0.0_f64;
    let mut displacement = 0.0_f64;
    for p in &case.points {
        let spherical = reference.frame.local(p.geodetic_deg_m)?;
        let oracle = Vector3::from(p.wgs84_enu_m);
        fixture_delta = fixture_delta.max((spherical - Vector3::from(p.sphere_enu_m)).norm());
        displacement = displacement.max((spherical - oracle).norm());
        let world = if sphere { spherical } else { oracle };
        let pixel = camera
            .project(&reference.pose, world)
            .ok_or(Error::Input("point behind reference"))?;
        let depth = -(reference.pose.orientation.inverse() * (world - reference.pose.position)).z;
        if pixel.x < 0.0
            || pixel.y < 0.0
            || pixel.x >= f64::from(camera.width - 1)
            || pixel.y >= f64::from(camera.height - 1)
        {
            return Err(Error::Input("point outside reference"));
        }
        let index = pixel.y.round() as usize * camera.width as usize + pixel.x.round() as usize;
        if reference.depth_m[index] != 0.0 {
            return Err(Error::Input("colliding reference pixels"));
        }
        reference.depth_m[index] = depth as f32;
        roundtrip = roundtrip.max(
            (camera.unproject(&reference.pose, pixel, f64::from(depth as f32)) - world).norm(),
        );
        pairs.push(PixelMatch {
            reference: pixel,
            query: Vector2::from(p.query_original_pixel),
        });
    }
    Ok((pairs, fixture_delta, roundtrip, displacement))
}

fn read_blocking(path: &Path) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|f| f.take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes))
        .map_err(|source| Error::Io {
            path: path.into(),
            source,
        })?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err(Error::Input("fixture exceeds limit"));
    }
    Ok(bytes)
}
fn write_blocking(path: &Path, report: &Value) -> Result<(), Error> {
    let bytes = serde_json::to_vec_pretty(report).map_err(|source| Error::Json {
        path: path.into(),
        source,
    })?;
    File::options()
        .create_new(true)
        .write(true)
        .open(path)
        .and_then(|mut f| f.write_all(&bytes))
        .map_err(|source| Error::Io {
            path: path.into(),
            source,
        })
}

#[cfg(test)]
#[path = "frame_model_probe/tests.rs"]
mod tests;

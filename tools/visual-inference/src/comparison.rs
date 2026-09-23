//! Fixed-reference quality comparison through the shared Rust verifier.
#[path = "comparison/guardrails.rs"]
mod guardrails;
#[path = "comparison/input.rs"]
mod input;
use input::{Case, Suite};
use navigate_visual::{
    Frame, FrameStamp, ImageMatcher, LocalFrame, LocalizerConfig, MapRevision, PoseVerifier,
    ReferenceView,
};
use navigate_visual_onnx::{OnnxMatcher, initialize_blocking};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};
type Error = Box<dyn std::error::Error>;

pub(super) fn run_blocking() -> Result<(), Error> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("Usage: matcher_compare SUITE.json OUTPUT.json".into());
    }
    let input = Path::new(&args[1]);
    let root = input.parent().ok_or("suite has no parent")?;
    let output = Path::new(&args[2]);
    if output.exists() {
        return Err("output already exists".into());
    }
    let suite: Suite = serde_json::from_slice(&std::fs::read(input)?)?;
    initialize_blocking(&root.join(&suite.library))?;
    let mut reports = Vec::new();
    let mut models = Vec::new();
    for model in &suite.models {
        let start = Instant::now();
        let mut matcher = OnnxMatcher::load_blocking(
            model.files(root),
            suite.provider.execution(),
            suite.keypoints,
        )?;
        let load_ms = start.elapsed().as_secs_f64() * 1000.0;
        let first = suite.cases.first().ok_or("suite has no cases")?;
        let image = image::open(root.join(&first.reference))?.to_luma8();
        let controls = guardrails::check_blocking(&mut matcher, &image)?;
        models.push(json!({"backend":matcher.identity(),"load_ms":load_ms,"controls":controls}));
        for case in &suite.cases {
            let report = match evaluate_blocking(case, root, &mut matcher) {
                Ok(report) => report,
                Err(error) => {
                    json!({"case":case.id,"backend":matcher.identity(),"error":error_chain(error.as_ref())})
                }
            };
            tracing::info!(case=case.id,accepted=?report["geometric_acceptance"],pairs=?report["matches"],error=?report["error"],"native comparison");
            reports.push(report);
            std::fs::write(
                output,
                serde_json::to_vec_pretty(
                    &json!({"scope":"fixed reference candidates; does not measure geographic retrieval recall","accuracy":"not independently measured","models":models,"cases":reports}),
                )?,
            )?;
        }
    }
    if reports.iter().any(|r| r.get("error").is_some()) {
        return Err("comparison contains execution errors".into());
    }
    Ok(())
}
fn evaluate_blocking(case: &Case, root: &Path, matcher: &mut OnnxMatcher) -> Result<Value, Error> {
    let query = image::open(root.join(&case.query))?.to_luma8();
    let reference_image = image::open(root.join(&case.reference))?.to_luma8();
    let depth = std::fs::read(root.join(&case.depth))?;
    check_digest(query.as_raw(), &case.query_image_sha256)?;
    check_digest(reference_image.as_raw(), &case.reference_image_sha256)?;
    check_digest(&depth, &case.reference_depth_sha256)?;
    if depth.len() % 4 != 0 {
        return Err("incomplete depth sample".into());
    }
    let frame = Frame {
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 0,
        },
        camera: case.camera.model(),
        image: query,
    };
    let reference = ReferenceView {
        map: MapRevision {
            release_id: case.map_release_id.clone(),
            manifest_sha256: case.map_manifest_sha256.clone(),
        },
        frame: LocalFrame::anchor_mercator(case.anchor_lat_lon[0], case.anchor_lat_lon[1])?,
        pose: case.reference_pose.model()?,
        image: reference_image,
        depth_m: depth
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    };
    let started = Instant::now();
    let pairs = matcher.match_images_blocking(&reference.image, &frame.image)?;
    let match_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let outcome = PoseVerifier::new(LocalizerConfig::default())?.verify(
        &frame,
        &reference,
        &case.prior.model()?,
        &pairs,
        matcher.identity(),
    );
    let geometry_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut report = json!({"case":case.id,"backend":matcher.identity(),"matches":pairs.len(),"match_ms":match_ms,"geometry_ms":geometry_ms,"observation_sha256":frame.evidence_sha256(),"reference_image_sha256":case.reference_image_sha256,"reference_depth_sha256":case.reference_depth_sha256,"map_manifest_sha256":case.map_manifest_sha256,"geometric_acceptance":outcome.is_ok(),"surface_geometry":"rendered depth; not independently verified"});
    match outcome {
        Ok(e) => {
            report["inliers"] = json!(e.quality.inliers);
            report["rms_px"] = json!(e.quality.reprojection_rms_px);
            report["occupied_cells"] = json!(e.quality.occupied_cells);
            report["position_enu_m"] =
                json!([e.pose.position.x, e.pose.position.y, e.pose.position.z]);
        }
        Err(e) => report["rejection"] = json!(e.to_string()),
    }
    Ok(report)
}
fn check_digest(bytes: &[u8], expected: &str) -> Result<(), Error> {
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        return Err(format!("evidence digest mismatch: expected {expected}, got {actual}").into());
    }
    Ok(())
}

fn error_chain(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(error) = source {
        message.push_str(": ");
        message.push_str(&error.to_string());
        source = error.source();
    }
    message
}

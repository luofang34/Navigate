//! Compare independent visual observations with withheld camera poses.

mod cases;
use crate::backend::{Backend, BackendKind};
use crate::{
    BenchError, directory_blocking,
    package::MapPackage,
    renderer::ReferenceRenderer,
    stream::{FrameRecord, write_record_blocking},
    write_blocking,
};
use navigate_visual::{
    CameraModel, CameraPose, Estimate, ImageMatcher, Localizer, LocalizerConfig,
};
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
struct CaseResult {
    name: String,
    truth_position_enu_m: [f64; 3],
    prior_position_enu_m: [f64; 3],
    estimated_position_enu_m: Option<[f64; 3]>,
    prior_position_error_m: f64,
    prior_attitude_error_deg: f64,
    position_error_m: Option<f64>,
    attitude_error_deg: Option<f64>,
    reprojection_rms_px: Option<f64>,
    inliers: usize,
    matcher_solver_ms: f64,
    total_ms: f64,
    observation_accepted: bool,
    expected_observation: bool,
    passed: bool,
    error: Option<String>,
}

pub(crate) async fn evaluate_blocking(
    root: &Path,
    output: &Path,
    backend: BackendKind,
) -> Result<(), BenchError> {
    directory_blocking(output)?;
    let package = MapPackage::open_blocking(root)?;
    if package.manifest.release_id != "synthetic-terrain-v1" {
        return Err(BenchError::Package {
            reason: "evaluate needs the package from prepare".into(),
        });
    }
    let revision = package.revision.clone();
    let anchor = package.manifest.anchor_lat_lon;
    let camera = CameraModel {
        width: 640,
        height: 480,
        fx: 550.0,
        fy: 550.0,
        cx: 319.5,
        cy: 239.5,
    };
    let mut renderer = ReferenceRenderer::new(package, camera).await?;
    let matcher = Backend::new(backend).await?;
    let identity = matcher.identity().to_owned();
    let mut localizer = Localizer::new(matcher, LocalizerConfig::default())?;
    let mut results = Vec::new();
    let mut records = Vec::new();
    for (index, spec) in cases::suite().into_iter().enumerate() {
        results.push(run_blocking(
            spec,
            index as u64,
            camera,
            output,
            &mut renderer,
            &mut localizer,
            &mut records,
        )?);
    }
    write_blocking(&output.join("frames.jsonl"), &records)?;
    let failed = results.iter().filter(|result| !result.passed).count();
    let report = serde_json::json!({"map_release":revision.release_id,
        "map_manifest_sha256":revision.manifest_sha256,"anchor_lat_lon":anchor,
        "backend":identity,"position_limit_m":10.0,"attitude_limit_deg":0.5,
        "cases":results,"failed":failed});
    let path = output.join("metrics.json");
    let bytes = serde_json::to_vec_pretty(&report).map_err(|source| BenchError::Json {
        path: path.clone(),
        source,
    })?;
    write_blocking(&path, &bytes)?;
    if failed > 0 {
        return Err(BenchError::Evaluation { failed });
    }
    Ok(())
}

fn run_blocking(
    spec: cases::Case,
    sequence: u64,
    camera: CameraModel,
    output: &Path,
    renderer: &mut ReferenceRenderer,
    localizer: &mut Localizer<Backend>,
    records: &mut Vec<u8>,
) -> Result<CaseResult, BenchError> {
    let (frame, prior, truth) = spec.frame_blocking(renderer, camera, sequence)?;
    let name = spec.name();
    let image_name = format!("{name}-query.png");
    save_blocking(output, &image_name, &frame.image)?;
    let record = FrameRecord {
        sequence,
        capture_time_ns: frame.stamp.capture_time_ns,
        image: image_name.into(),
        camera: camera.into(),
        prior: prior.into(),
    };
    if sequence == 0 {
        let path = output.join("prior.json");
        let value = serde_json::json!({"camera":record.camera,"prior":record.prior});
        let bytes = serde_json::to_vec_pretty(&value).map_err(|source| BenchError::Json {
            path: path.clone(),
            source,
        })?;
        write_blocking(&path, &bytes)?;
    }
    write_record_blocking(records, &output.join("frames.jsonl"), &record)?;
    let total = std::time::Instant::now();
    let reference = renderer.render_blocking(prior.pose)?;
    let started = std::time::Instant::now();
    let estimate = localizer.estimate_blocking(&frame, &reference, &prior);
    let matcher_solver_ms = started.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total.elapsed().as_secs_f64() * 1000.0;
    save_blocking(output, &format!("{name}-prior.png"), &reference.image)?;
    let mut result = CaseResult {
        name,
        truth_position_enu_m: truth.position.into(),
        prior_position_enu_m: prior.pose.position.into(),
        estimated_position_enu_m: None,
        prior_position_error_m: (prior.pose.position - truth.position).norm(),
        prior_attitude_error_deg: prior
            .pose
            .orientation
            .angle_to(&truth.orientation)
            .to_degrees(),
        position_error_m: None,
        attitude_error_deg: None,
        reprojection_rms_px: None,
        inliers: 0,
        matcher_solver_ms,
        total_ms,
        observation_accepted: false,
        expected_observation: spec.expects_observation(),
        passed: false,
        error: None,
    };
    match estimate {
        Ok(estimate) => {
            result.measure(&estimate, truth);
            let rendered = renderer.render_blocking(estimate.pose)?;
            save_blocking(
                output,
                &format!("{}-estimate.png", result.name),
                &rendered.image,
            )?;
        }
        Err(error) => {
            result.passed = !result.expected_observation;
            result.error = Some(error.to_string());
        }
    }
    tracing::info!(name = result.name, position_error_m = ?result.position_error_m,
        attitude_error_deg = ?result.attitude_error_deg, result.inliers, result.matcher_solver_ms,
        result.passed, error = ?result.error, "synthetic evaluation");
    Ok(result)
}

impl CaseResult {
    fn measure(&mut self, estimate: &Estimate, truth: CameraPose) {
        let error_m = (estimate.pose.position - truth.position).norm();
        let error_deg = estimate
            .pose
            .orientation
            .angle_to(&truth.orientation)
            .to_degrees();
        self.estimated_position_enu_m = Some(estimate.pose.position.into());
        self.position_error_m = Some(error_m);
        self.attitude_error_deg = Some(error_deg);
        self.reprojection_rms_px = Some(estimate.quality.reprojection_rms_px);
        self.inliers = estimate.quality.inliers;
        self.observation_accepted = true;
        self.passed = self.expected_observation && error_m < 10.0 && error_deg < 0.5;
    }
}

fn save_blocking(root: &Path, name: &str, image: &image::GrayImage) -> Result<(), BenchError> {
    let path = root.join(name);
    image
        .save(&path)
        .map_err(|source| BenchError::Image { path, source })
}

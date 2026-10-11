//! Image-only temporal checks of frozen pose outputs, without fitting a new pose.
#![cfg(feature = "gpu")]
mod geometry;
mod input;
use crate::{Error, assets};
use input::{Manifest, Run};
use navigate_visual::{Frame, FrameStamp, ImageMatcher, PointTracker};
use serde_json::json;
use std::{path::Path, time::Instant};

#[cfg(feature = "gpu")]
pub fn run_blocking(manifest: &Path, output: &Path) -> Result<(), Error> {
    let spec: Manifest = assets::read_json_blocking(manifest)?;
    spec.validate()?;
    let root = manifest.parent().unwrap_or(Path::new("."));
    let runs = spec
        .runs
        .iter()
        .map(|a| Run::load_blocking(a, root, spec.frames.len()))
        .collect::<Result<Vec<_>, _>>()?;
    let job_path = root.join(&spec.job.path);
    let job_bytes = assets::read_blocking(&job_path, 4 * 1024 * 1024)?;
    assets::verify(&job_bytes, &spec.job.sha256)?;
    let job: crate::job::Job =
        serde_json::from_slice(&job_bytes).map_err(|source| Error::Json {
            path: job_path,
            source,
        })?;
    job.frame()?;
    for run in &runs {
        run.validate_camera(&job)?;
    }
    if runs.windows(2).any(|p| p[0].camera() != p[1].camera()) {
        return Err(Error::Input(
            "motion comparison requires one effective camera".into(),
        ));
    }
    let mut tracker = pollster::block_on(navigate_visual::GpuPyramidalMatcher::new())?;
    let mut rows = Vec::new();
    for index in 0..spec.frames.len() - spec.lag {
        let target = index + spec.lag;
        let a = spec.image_blocking(root, index)?;
        let b = spec.image_blocking(root, target)?;
        for run in &runs {
            run.bind(&spec, index, &a)?;
            run.bind(&spec, target, &b)?;
        }
        let start = Instant::now();
        let points = tracker.features_blocking(&a)?;
        let selected_points = points.len();
        let found = tracker.track_points_blocking(&a, &b, &points)?;
        let pairs: Vec<_> = points
            .into_iter()
            .zip(found)
            .filter_map(|(p, q)| q.map(|q| (p, q)))
            .collect();
        let tracking_ms = start.elapsed().as_secs_f64() * 1000.0;
        let scores = runs
            .iter()
            .map(|run| run.score(index, target, &pairs))
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(json!({"source_index":index,"target_index":target,
            "delta_time_ns":spec.frames[target].capture_time_ns-spec.frames[index].capture_time_ns,
            "tracking_ms":tracking_ms,"selected_points":selected_points,"source_grid_cells_4x4":coverage(&pairs,a.width(),a.height()),"pairs":pairs.iter().map(|(p,q)|[p.x,p.y,q.x,q.y]).collect::<Vec<_>>(),
            "scores":scores}));
    }
    assets::write_json_blocking(
        output,
        &json!({"tracker":ImageMatcher::identity(&tracker),"lag":spec.lag,"runs":spec.runs,"job":spec.job,"anchor_lat_lon":job.anchor_lat_lon,"coordinate_frame":job.coordinate_frame,"pairs":rows,
        "scope":"Processed image corners and forward-backward optical flow only; source result/job association is declared by this manifest. Camera models are checked against the job and result transforms. Pairs spanning a rejected pose are excluded; map refreshes are labeled. no map matches, pose fitting, or pose-based track rejection. The LK implementation is shared with temporal navigation. Correlated image/tracker/calibration errors remain. Epipolar residuals cannot establish absolute position, global attitude, translation scale or baseline accuracy. Pure rotation has no epipolar score. Rotation-only residual is a diagnostic, not a translation error. No selection or navigation state is changed."}),
    )?;
    tracing::info!(path=%output.display(),"Saved image motion consistency probe");
    Ok(())
}
fn digest(
    image: &image::GrayImage,
    camera: navigate_visual::CameraModel,
    index: usize,
    time: u64,
) -> String {
    Frame {
        image: image.clone(),
        camera,
        stamp: FrameStamp {
            sequence: index as u64,
            capture_time_ns: time,
        },
    }
    .evidence_sha256()
}

fn coverage(
    pairs: &[(nalgebra::Vector2<f64>, nalgebra::Vector2<f64>)],
    width: u32,
    height: u32,
) -> usize {
    pairs
        .iter()
        .map(|(p, _)| {
            (
                (p.x * 4.0 / f64::from(width)) as u32,
                (p.y * 4.0 / f64::from(height)) as u32,
            )
        })
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

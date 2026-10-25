//! Compare fixed poses on the same map patches without new correspondences.
use super::job::SurfaceJob;
use crate::{Error, assets, config::Config};
use image::GrayImage;
use nalgebra::Vector2;
use navigate_recording::Pose;
use navigate_visual::{CameraPose, Frame, OrthographicReference};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

const STRIDE: u32 = 32;
const OFFSETS: [f64; 5] = [-6.0, -3.0, 0.0, 3.0, 6.0];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    job: PathBuf,
    poses: Vec<Candidate>,
    excluded_map_pixels: Vec<[f64; 2]>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    id: String,
    pose: Pose,
}
struct Patch {
    centre: [u32; 2],
    fold: u32,
    scores: Vec<f64>,
}
struct Counts {
    centres: usize,
    invalid_centres: usize,
    outside: usize,
    excluded: usize,
    low_texture: usize,
    invalid_patch: usize,
}

pub fn run_blocking(config: &Config, input: &Path, output: &Path) -> Result<(), Error> {
    let started = Instant::now();
    let spec: Input = assets::read_json_blocking(input)?;
    validate(&spec)?;
    let job_path = input.parent().unwrap_or(Path::new(".")).join(&spec.job);
    let job: SurfaceJob = assets::read_json_blocking(&job_path)?;
    if job.reference.window.is_some() {
        return Err(Error::Input(
            "appearance probe needs full-map coordinates".into(),
        ));
    }
    let (frame, reference, _) =
        job.load_blocking(job_path.parent().unwrap_or(Path::new(".")), config)?;
    let poses = spec
        .poses
        .iter()
        .map(|p| p.pose.camera())
        .collect::<Result<Vec<_>, _>>()?;
    let loading_ms = started.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let (patches, counts) = compare(&frame, &reference, &poses, &spec.excluded_map_pixels);
    let scoring_ms = start.elapsed().as_secs_f64() * 1000.0;
    let summaries: Vec<_> = spec
        .poses
        .iter()
        .enumerate()
        .map(|(i, p)| {
            json!({"id":p.id,"pose":p.pose,
        "all":statistics(patches.iter().map(|p|p.scores[i]).collect()),
        "even":statistics(patches.iter().filter(|p|p.fold==0).map(|p|p.scores[i]).collect()),
        "odd":statistics(patches.iter().filter(|p|p.fold==1).map(|p|p.scores[i]).collect())})
        })
        .collect();
    let rows: Vec<_> = patches
        .iter()
        .map(|p| json!({"centre":p.centre,"fold":p.fold,"scores":p.scores}))
        .collect();
    assets::write_json_blocking(
        output,
        &json!({"candidates":summaries,"patches":rows,"loading_ms":loading_ms,"scoring_ms":scoring_ms,
        "grid_centres":counts.centres,"invalid_height_centres":counts.invalid_centres,"outside_common_view":counts.outside,"excluded_centres":counts.excluded,
        "low_map_variance":counts.low_texture,"invalid_common_patches":counts.invalid_patch,"common_patches":patches.len(),
        "resident_source_bytes":reference.image.as_raw().len()*5,"query_image_bytes":frame.image.as_raw().len(),
        "observation_sha256":frame.evidence_sha256(),"job":job,
        "scope":"Scalar Rust diagnostic, not an accelerator performance result. Fixed map grid: stride32, offset16, five offsets -6,-3,0,3,6 on each axis. Exclude centres within16 map pixels of supplied fitting points. All poses use the same complete25-sample patches with finite heights and valid query projections. Require map intensity variance at least9. Query zero variance gives correlation0. Even and odd grid cells are separate spatial checks. No occlusion model, pose optimization, acceptance threshold, truth selection, independent-evidence claim, or integrity probability."}),
    )?;
    tracing::info!(path=%output.display(), "Saved shared-patch appearance diagnostic");
    Ok(())
}
fn validate(input: &Input) -> Result<(), Error> {
    if !(1..=16).contains(&input.poses.len())
        || input.excluded_map_pixels.len() > 32768
        || input
            .excluded_map_pixels
            .iter()
            .flatten()
            .any(|x| !x.is_finite())
        || input.poses.iter().enumerate().any(|(i, p)| {
            p.id.is_empty() || p.id.len() > 80 || input.poses[..i].iter().any(|q| q.id == p.id)
        })
    {
        return Err(Error::Input("appearance probe needs 1 through16 unique pose IDs and at most32768 finite excluded points".into()));
    }
    Ok(())
}
fn compare(
    frame: &Frame,
    reference: &OrthographicReference,
    poses: &[CameraPose],
    excluded: &[[f64; 2]],
) -> (Vec<Patch>, Counts) {
    let mut patches = Vec::new();
    let mut counts = Counts {
        centres: 0,
        invalid_centres: 0,
        outside: 0,
        excluded: 0,
        low_texture: 0,
        invalid_patch: 0,
    };
    for y in (16..reference.image.height()).step_by(STRIDE as usize) {
        for x in (16..reference.image.width()).step_by(STRIDE as usize) {
            counts.centres = counts.centres.wrapping_add(1);
            let centre = Vector2::new(f64::from(x), f64::from(y));
            let Some(world) = reference.world_at(centre) else {
                counts.invalid_centres = counts.invalid_centres.wrapping_add(1);
                continue;
            };
            if poses.iter().any(|p| {
                frame
                    .camera
                    .project(p, world)
                    .and_then(|q| sample(&frame.image, q))
                    .is_none()
            }) {
                counts.outside = counts.outside.wrapping_add(1);
                continue;
            }
            if excluded
                .iter()
                .any(|p| (centre - Vector2::from(*p)).norm_squared() <= 256.0)
            {
                counts.excluded = counts.excluded.wrapping_add(1);
                continue;
            }
            let Some((source, targets)) = patch(frame, reference, poses, centre) else {
                counts.invalid_patch = counts.invalid_patch.wrapping_add(1);
                continue;
            };
            let (_, variance) = moments(&source);
            if variance / 25.0 < 9.0 {
                counts.low_texture = counts.low_texture.wrapping_add(1);
                continue;
            }
            patches.push(Patch {
                centre: [x, y],
                fold: (x / STRIDE + y / STRIDE) % 2,
                scores: targets.iter().map(|q| correlation(&source, q)).collect(),
            });
        }
    }
    (patches, counts)
}
fn patch(
    frame: &Frame,
    reference: &OrthographicReference,
    poses: &[CameraPose],
    centre: Vector2<f64>,
) -> Option<([f64; 25], Vec<[f64; 25]>)> {
    let mut source = [0.0; 25];
    let mut targets = vec![[0.0; 25]; poses.len()];
    for (row, dy) in OFFSETS.iter().enumerate() {
        for (column, dx) in OFFSETS.iter().enumerate() {
            let index = row * 5 + column;
            let pixel = centre + Vector2::new(*dx, *dy);
            let world = reference.world_at(pixel)?;
            source[index] = sample(&reference.image, pixel)?;
            for (i, pose) in poses.iter().enumerate() {
                targets[i][index] = sample(&frame.image, frame.camera.project(pose, world)?)?;
            }
        }
    }
    Some((source, targets))
}
fn sample(image: &GrayImage, p: Vector2<f64>) -> Option<f64> {
    if !p.iter().all(|v| v.is_finite())
        || p.x < 0.0
        || p.y < 0.0
        || p.x >= f64::from(image.width().saturating_sub(1))
        || p.y >= f64::from(image.height().saturating_sub(1))
    {
        return None;
    }
    let x = p.x.floor() as u32;
    let y = p.y.floor() as u32;
    let dx = p.x - f64::from(x);
    let dy = p.y - f64::from(y);
    Some(
        (1.0 - dy)
            * ((1.0 - dx) * f64::from(image[(x, y)][0]) + dx * f64::from(image[(x + 1, y)][0]))
            + dy * ((1.0 - dx) * f64::from(image[(x, y + 1)][0])
                + dx * f64::from(image[(x + 1, y + 1)][0])),
    )
}
fn moments(values: &[f64; 25]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / 25.0;
    (mean, values.iter().map(|x| (x - mean).powi(2)).sum())
}
fn correlation(a: &[f64; 25], b: &[f64; 25]) -> f64 {
    let (ma, va) = moments(a);
    let (mb, vb) = moments(b);
    if va <= 1e-12 || vb <= 1e-12 {
        return 0.0;
    }
    (a.iter()
        .zip(b)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / (va * vb).sqrt())
    .clamp(-1.0, 1.0)
}
fn statistics(mut values: Vec<f64>) -> Value {
    if values.is_empty() {
        return json!({"count":0,"mean":null,"median":null});
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    json!({"count":n,"mean":values.iter().sum::<f64>()/n as f64,"median":(values[(n-1)/2]+values[n/2])/2.0})
}
#[cfg(test)]
mod tests;

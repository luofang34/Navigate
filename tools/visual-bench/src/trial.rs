//! Single-image trials share the same session and output contract as video.

use crate::{
    BenchError,
    backend::BackendKind,
    cli::TrialArgs,
    package::MapPackage,
    read_blocking,
    session::Session,
    stream::{CameraRecord, PriorRecord, write_record_blocking},
};
use navigate_visual::{Frame, FrameStamp};
use serde::Deserialize;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrialInput {
    pub camera: CameraRecord,
    pub prior: PriorRecord,
}

pub(crate) fn config_blocking(path: &Path) -> Result<TrialInput, BenchError> {
    serde_json::from_slice(&read_blocking(path)?).map_err(|source| BenchError::Json {
        path: path.to_owned(),
        source,
    })
}

pub(crate) fn writer_blocking(path: &Path) -> Result<BufWriter<File>, BenchError> {
    File::create_new(path)
        .map(BufWriter::new)
        .map_err(|source| BenchError::Io {
            path: path.to_owned(),
            source,
        })
}

pub(crate) async fn image_blocking(
    args: &TrialArgs,
    backend: BackendKind,
) -> Result<(), BenchError> {
    let input = config_blocking(&args.prior)?;
    let mut session = Session::new(
        MapPackage::open_blocking(&args.package)?,
        input.camera.model(),
        backend,
    )
    .await?;
    let prior = input.prior.prior(session.frame)?;
    let image = image::load_from_memory(&read_blocking(&args.input)?)
        .map_err(|source| BenchError::Image {
            path: args.input.clone(),
            source,
        })?
        .to_luma8();
    let frame = Frame {
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 0,
        },
        camera: session.camera,
        image,
    };
    let report = session.observe_blocking(&frame, &prior)?;
    let mut writer = writer_blocking(&args.output)?;
    write_record_blocking(&mut writer, &args.output, &report)?;
    tracing::info!(accepted=?report["accepted"],latitude=?report["latitude_deg"],longitude=?report["longitude_deg"],
        altitude=?report["altitude_m"],reason=?report["reason"],output=%args.output.display(),"image result");
    if let Some(path) = &args.track {
        crate::track::export_blocking(&args.output, path)?;
    }
    Ok(())
}

pub(crate) async fn refine_blocking(
    args: &TrialArgs,
    reference_prior: &Path,
    matches: &Path,
) -> Result<(), BenchError> {
    let input = config_blocking(&args.prior)?;
    let candidate = config_blocking(reference_prior)?;
    if input.camera.model() != candidate.camera.model() {
        return Err(BenchError::ChangedCamera);
    }
    let backend =
        crate::backend::Backend::Matches(crate::matches::VerifiedMatches::open_blocking(matches)?);
    let mut session = Session::with_backend(
        MapPackage::open_blocking(&args.package)?,
        input.camera.model(),
        backend,
    )
    .await?;
    let prior = input.prior.prior(session.frame)?;
    let candidate_pose = candidate.prior.prior(session.frame)?.pose;
    let image = image::load_from_memory(&read_blocking(&args.input)?)
        .map_err(|source| BenchError::Image {
            path: args.input.clone(),
            source,
        })?
        .to_luma8();
    let frame = Frame {
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 0,
        },
        camera: session.camera,
        image,
    };
    let report = session.observe_candidate_blocking(&frame, &prior, candidate_pose)?;
    let mut writer = writer_blocking(&args.output)?;
    write_record_blocking(&mut writer, &args.output, &report)?;
    tracing::info!(accepted=?report["accepted"], reason=?report["reason"], output=%args.output.display(), "matched image result");
    if let Some(path) = &args.track {
        crate::track::export_blocking(&args.output, path)?;
    }
    Ok(())
}

/// Export the exact reference used by the localizer for inspection.
pub(crate) async fn render_blocking(
    package: &Path,
    prior: &Path,
    output: &Path,
) -> Result<(), BenchError> {
    let input = config_blocking(prior)?;
    let package = MapPackage::open_blocking(package)?;
    let pose = input.prior.prior(package.frame)?.pose;
    let mut renderer =
        crate::renderer::ReferenceRenderer::new(package, input.camera.model()).await?;
    let reference = renderer.render_blocking(pose)?;
    export_reference_blocking(&reference, input.camera.model(), output)
}

pub(crate) fn export_reference_blocking(
    reference: &navigate_visual::ReferenceView,
    camera: navigate_visual::CameraModel,
    output: &Path,
) -> Result<(), BenchError> {
    let pose = reference.pose;
    let depth_path = output.with_extension("depth.bin");
    let mut depth_writer = writer_blocking(&depth_path)?;
    let depth_bytes: Vec<u8> = reference
        .depth_m
        .iter()
        .flat_map(|d| d.to_le_bytes())
        .collect();
    depth_writer
        .write_all(&depth_bytes)
        .and_then(|()| depth_writer.flush())
        .map_err(|source| BenchError::Io {
            path: depth_path,
            source,
        })?;
    let mut samples = Vec::new();
    for y in (0..camera.height).step_by(40) {
        for x in (0..camera.width).step_by(40) {
            let depth = reference.depth_m[(y * camera.width + x) as usize];
            if depth > 0.0 {
                let point = camera.unproject(
                    &pose,
                    nalgebra::Vector2::new(f64::from(x), f64::from(y)),
                    f64::from(depth),
                );
                samples.push(serde_json::json!({"pixel":[x,y],"depth_m":depth,"world_xyz_m":[point.x,point.y,point.z]}));
            }
        }
    }
    let samples_path = output.with_extension("depth.json");
    let mut samples_writer = writer_blocking(&samples_path)?;
    write_record_blocking(
        &mut samples_writer,
        &samples_path,
        &serde_json::json!({"samples":samples, "reference_image_sha256": crate::package::digest(reference.image.as_raw()), "reference_depth_sha256": crate::matches::depth_digest(&reference.depth_m)}),
    )?;
    let mut writer = writer_blocking(output)?;
    reference
        .image
        .write_to(&mut writer, image::ImageFormat::Png)
        .map_err(|source| BenchError::Image {
            path: output.to_owned(),
            source,
        })?;
    writer.flush().map_err(|source| BenchError::Io {
        path: output.to_owned(),
        source,
    })?;
    let valid = reference
        .depth_m
        .iter()
        .filter(|depth| **depth > 0.0)
        .count();
    tracing::info!(valid_depth_pixels=valid, total_pixels=reference.depth_m.len(), output=%output.display(), "reference exported");
    Ok(())
}

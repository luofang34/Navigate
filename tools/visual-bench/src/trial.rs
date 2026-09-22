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
use std::{fs::File, io::BufWriter, path::Path};

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

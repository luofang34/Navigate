//! Video trials use decoded presentation times and explicit pose priors.
//!
//! Each frame receives an independent visual estimate. A fixed prior is useful
//! only while the camera remains in its capture range. A prior sidecar has one
//! `FrameRecord` per decoded frame. Its sequence and relative timestamp must
//! match that frame. Its image path is unused. No trajectory is extrapolated
//! across a visual rejection.

mod decoder;
mod probe;

use crate::{
    BenchError,
    backend::BackendKind,
    cli::TrialArgs,
    package::MapPackage,
    session::Session,
    stream::{FrameRecord, write_record_blocking},
    trial::{config_blocking, writer_blocking},
};
use navigate_visual::{Frame, FrameStamp};
use std::{
    io::{BufRead, BufReader},
    path::Path,
};

pub(crate) async fn run_blocking(
    args: &TrialArgs,
    priors: Option<&Path>,
    backend: BackendKind,
) -> Result<(), BenchError> {
    let input = config_blocking(&args.prior)?;
    let camera = input.camera.model();
    camera.validate()?;
    let timestamps = probe::timestamps_blocking(&args.input, camera)?;
    let mut session =
        Session::new(MapPackage::open_blocking(&args.package)?, camera, backend).await?;
    let fixed_prior = input.prior.prior(session.frame)?;
    let mut sidecar = priors.map(open_priors_blocking).transpose()?;
    let mut decoder = decoder::Decoder::new_blocking(&args.input, camera)?;
    let mut writer = writer_blocking(&args.output)?;
    for (index, time) in timestamps.into_iter().enumerate() {
        let stamp = FrameStamp {
            sequence: index as u64,
            capture_time_ns: time,
        };
        let prior = match &mut sidecar {
            Some(records) => next_prior_blocking(records, stamp, &session)?,
            None => fixed_prior,
        };
        let frame = Frame {
            stamp,
            camera,
            image: decoder.next_blocking()?,
        };
        let mut report = session.observe_blocking(&frame, &prior)?;
        report["clock_domain"] = "video-relative-pts".into();
        write_record_blocking(&mut writer, &args.output, &report)?;
    }
    decoder.finish_blocking()?;
    if let Some(line) = sidecar.as_mut().and_then(Iterator::next) {
        line.map_err(|source| BenchError::Io {
            path: priors.unwrap_or(Path::new("<video-priors>")).to_owned(),
            source,
        })?;
        return Err(BenchError::Record {
            reason: "prior sidecar has extra frames".into(),
        });
    }
    if let Some(path) = &args.track {
        crate::track::export_blocking(&args.output, path)?;
    }
    tracing::info!(output=%args.output.display(),"video observations ready");
    Ok(())
}

type PriorLines = std::io::Lines<BufReader<std::fs::File>>;

fn open_priors_blocking(path: &Path) -> Result<PriorLines, BenchError> {
    let file = std::fs::File::open(path).map_err(|source| BenchError::Io {
        path: path.to_owned(),
        source,
    })?;
    Ok(BufReader::new(file).lines())
}

fn next_prior_blocking(
    lines: &mut PriorLines,
    stamp: FrameStamp,
    session: &Session,
) -> Result<navigate_visual::PosePrior, BenchError> {
    let line = lines
        .next()
        .ok_or_else(|| BenchError::Record {
            reason: "prior sidecar ended before video".into(),
        })?
        .map_err(|source| BenchError::Io {
            path: "<video-priors>".into(),
            source,
        })?;
    let record: FrameRecord = serde_json::from_str(&line).map_err(|source| BenchError::Json {
        path: "<video-priors>".into(),
        source,
    })?;
    if record.sequence != stamp.sequence
        || record.capture_time_ns.abs_diff(stamp.capture_time_ns) > 1_000_000
        || record.camera.model() != session.camera
    {
        return Err(BenchError::Record {
            reason: format!(
                "prior does not match video frame {} at {} ns",
                stamp.sequence, stamp.capture_time_ns
            ),
        });
    }
    Ok(record.prior.prior(session.frame)?)
}

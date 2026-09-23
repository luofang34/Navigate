//! Read calibrated frames from a file or a live JSON Lines stream.
//!
//! Each line is a [`FrameRecord`]. Image paths are relative to the input file,
//! or the current directory for standard input. The output contains one record
//! per frame. Visual rejections contain no position. Input or file errors stop
//! the stream. A valid frame consumes its stamp even when localization rejects.
//! Frames do not inherit a pose from earlier estimates.

mod prior;
mod record;
pub(crate) use prior::PriorRecord;
pub(crate) use record::{CameraRecord, FrameRecord};

use crate::{BenchError, backend::BackendKind, package::MapPackage, session::Session};
use std::{
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

pub(crate) async fn localize_blocking(
    package: &Path,
    input: &Path,
    output: &Path,
    backend: BackendKind,
) -> Result<(), BenchError> {
    let reader: Box<dyn BufRead> = if input == Path::new("-") {
        Box::new(BufReader::new(std::io::stdin()))
    } else {
        Box::new(BufReader::new(std::fs::File::open(input).map_err(
            |source| BenchError::Io {
                path: input.to_owned(),
                source,
            },
        )?))
    };
    let file = std::fs::File::create_new(output).map_err(|source| BenchError::Io {
        path: output.to_owned(),
        source,
    })?;
    let mut writer = BufWriter::new(file);
    let mut lines = reader.lines();
    let first = lines
        .next()
        .ok_or(BenchError::EmptyStream)?
        .map_err(|source| BenchError::Io {
            path: input.to_owned(),
            source,
        })?;
    let first = decode(&first, input)?;
    let camera = first.camera.model();
    let package = MapPackage::open_blocking(package)?;
    let mut session = Session::new(package, camera, backend).await?;
    process_blocking(first, input, output, &mut writer, &mut session)?;
    for line in lines {
        let line = line.map_err(|source| BenchError::Io {
            path: input.to_owned(),
            source,
        })?;
        let frame = decode(&line, input)?;
        if frame.camera.model() != camera {
            return Err(BenchError::ChangedCamera);
        }
        process_blocking(frame, input, output, &mut writer, &mut session)?;
    }
    Ok(())
}

fn decode(line: &str, path: &Path) -> Result<FrameRecord, BenchError> {
    serde_json::from_str(line).map_err(|source| BenchError::Json {
        path: path.to_owned(),
        source,
    })
}

fn process_blocking(
    record: FrameRecord,
    input: &Path,
    output: &Path,
    writer: &mut impl Write,
    session: &mut Session,
) -> Result<(), BenchError> {
    let base = input.parent().unwrap_or(Path::new("."));
    let prior = record.prior.prior(session.frame)?;
    let frame = record.load_blocking(base)?;
    let report = session.observe_blocking(&frame, &prior)?;
    write_record_blocking(writer, output, &report)
}

pub(crate) fn write_record_blocking(
    writer: &mut impl Write,
    path: &Path,
    value: &impl serde::Serialize,
) -> Result<(), BenchError> {
    serde_json::to_writer(&mut *writer, value).map_err(|source| BenchError::Json {
        path: path.to_owned(),
        source,
    })?;
    writer
        .write_all(b"\n")
        .and_then(|()| writer.flush())
        .map_err(|source| BenchError::Io {
            path: PathBuf::from(path),
            source,
        })
}

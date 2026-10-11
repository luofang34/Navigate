//! Bounded records and incremental results keep replay memory independent of length.
use crate::Error;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const RECORD_BYTES: usize = 16 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Record {
    pub job: PathBuf,
    pub capture_time_ns: u64,
}

#[derive(Default)]
struct Counts {
    records: u64,
    failed: u64,
}

pub(super) fn run_blocking(
    manifest: &Path,
    output: &Path,
    load_ms: f64,
    process: impl FnMut(&Record, u64) -> Result<Value, Error>,
) -> Result<(), Error> {
    let input = File::open(manifest).map_err(|source| io_error("open replay", manifest, source))?;
    let file =
        File::create_new(output).map_err(|source| io_error("create replay", output, source))?;
    let mut writer = BufWriter::new(file);
    let root = manifest.parent().unwrap_or(Path::new("."));
    let counts = execute_blocking(
        &mut BufReader::new(input),
        &mut writer,
        root,
        load_ms,
        process,
    )
    .map_err(|source| io_error("stream replay", output, source))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|source| io_error("sync replay", output, source))?;
    tracing::info!(
        records = counts.records,
        failed = counts.failed,
        "replay complete"
    );
    Ok(())
}

fn execute_blocking(
    reader: &mut impl BufRead,
    writer: &mut impl Write,
    root: &Path,
    load_ms: f64,
    mut process: impl FnMut(&Record, u64) -> Result<Value, Error>,
) -> std::io::Result<Counts> {
    write_record_blocking(writer, &json!({"kind":"start", "backend_load_ms":load_ms}))?;
    let mut counts = Counts::default();
    let mut previous_time = None;
    while let Some(line) = read_record_blocking(reader)? {
        let start = Instant::now();
        let sequence = counts.records;
        counts.records = counts.records.wrapping_add(1);
        let record = parse_record(line, root, &mut previous_time);
        let metadata = record.as_ref().ok().map(|r| {
            json!({
                "job":r.job, "capture_time_ns":r.capture_time_ns,
            })
        });
        let result = record.and_then(|r| process(&r, sequence));
        let value = match result {
            Ok(value) => value,
            Err(error) => {
                counts.failed = counts.failed.wrapping_add(1);
                tracing::warn!(sequence, %error, "replay frame failed");
                json!({"status":"failed", "error":error.to_string()})
            }
        };
        write_record_blocking(
            writer,
            &json!({"kind":"frame", "sequence":sequence,
            "input":metadata, "elapsed_ms":start.elapsed().as_secs_f64()*1000.0,
            "result":value}),
        )?;
    }
    write_record_blocking(
        writer,
        &json!({"kind":"end", "records":counts.records,
        "failed":counts.failed}),
    )?;
    Ok(counts)
}

fn parse_record(
    line: Result<Vec<u8>, ()>,
    root: &Path,
    previous_time: &mut Option<u64>,
) -> Result<Record, Error> {
    let bytes =
        line.map_err(|()| Error::Input(format!("replay record exceeds {RECORD_BYTES} bytes")))?;
    let mut record: Record = serde_json::from_slice(&bytes).map_err(|source| Error::Json {
        path: root.join("replay record"),
        source,
    })?;
    if previous_time.is_some_and(|previous| record.capture_time_ns <= previous) {
        return Err(Error::Input(
            "capture timestamps must strictly increase".into(),
        ));
    }
    *previous_time = Some(record.capture_time_ns);
    record.job = root.join(record.job);
    Ok(record)
}

fn read_record_blocking(reader: &mut impl BufRead) -> std::io::Result<Option<Result<Vec<u8>, ()>>> {
    let mut bytes = Vec::new();
    let mut oversized = false;
    let mut observed = false;
    loop {
        let chunk = reader.fill_buf()?;
        if chunk.is_empty() {
            return Ok(observed.then_some(if oversized { Err(()) } else { Ok(bytes) }));
        }
        observed = true;
        let end = chunk.iter().position(|b| *b == b'\n');
        let length = end.unwrap_or(chunk.len());
        if bytes.len().saturating_add(length) > RECORD_BYTES {
            oversized = true;
        }
        if !oversized {
            bytes.extend_from_slice(&chunk[..length]);
        }
        reader.consume(length + usize::from(end.is_some()));
        if end.is_some() {
            return Ok(Some(if oversized { Err(()) } else { Ok(bytes) }));
        }
    }
}

fn write_record_blocking(writer: &mut impl Write, value: &Value) -> std::io::Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

fn io_error(operation: &'static str, path: &Path, source: std::io::Error) -> Error {
    Error::Io {
        operation,
        path: path.into(),
        source,
    }
}

#[cfg(test)]
mod tests;

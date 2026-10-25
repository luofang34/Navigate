use super::*;
use std::io::Cursor;

#[test]
fn processing_errors_do_not_skip_later_frames_or_reuse_results() {
    let input = b"{\"job\":\"first\",\"capture_time_ns\":1}\n{\"job\":\"broken\",\"capture_time_ns\":2}\n{\"job\":\"third\",\"capture_time_ns\":3}";
    let mut output = Vec::new();
    let mut calls = Vec::new();
    let counts = execute_blocking(
        &mut Cursor::new(input),
        &mut output,
        Path::new("jobs"),
        12.0,
        |record, sequence| {
            calls.push((record.job.clone(), sequence));
            if record.job.ends_with("broken") {
                return Err(Error::Input("bad image".into()));
            }
            Ok(json!({"status":"processed", "value":sequence}))
        },
    )
    .unwrap();
    assert_eq!((counts.records, counts.failed), (3, 1));
    assert_eq!(calls[2], (PathBuf::from("jobs/third"), 2));
    let lines: Vec<Value> = output
        .split(|b| *b == b'\n')
        .filter(|b| !b.is_empty())
        .map(|b| serde_json::from_slice(b).unwrap())
        .collect();
    assert_eq!(lines[2]["result"]["status"], "failed");
    assert!(lines[2]["result"].get("value").is_none());
    assert_eq!(lines[3]["result"]["value"], 2);
    assert_eq!(lines[4]["kind"], "end");
}

#[test]
fn malformed_oversized_and_out_of_order_records_are_bounded_and_recoverable() {
    let mut input = b"{\"job\":\"one\",\"capture_time_ns\":5}\n".to_vec();
    input.extend(vec![b'x'; RECORD_BYTES * 3]);
    input.extend(b"\ninvalid\n{\"job\":\"late\",\"capture_time_ns\":4}\n{\"job\":\"same\",\"capture_time_ns\":5}\n{\"job\":\"next\",\"capture_time_ns\":6}\n");
    let mut calls = Vec::new();
    let counts = execute_blocking(
        &mut BufReader::with_capacity(3, Cursor::new(input)),
        &mut Vec::new(),
        Path::new("."),
        0.0,
        |record, sequence| {
            calls.push((record.capture_time_ns, sequence));
            Ok(json!({}))
        },
    )
    .unwrap();
    assert_eq!((counts.records, counts.failed), (6, 4));
    assert_eq!(calls, [(5, 0), (6, 5)]);
}

#[test]
fn output_failure_stops_before_processing_another_frame() {
    struct FailingWriter;
    impl Write for FailingWriter {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let input = b"{\"job\":\"one\",\"capture_time_ns\":1}\n";
    let mut calls = 0_u64;
    assert!(
        execute_blocking(
            &mut Cursor::new(input),
            &mut FailingWriter,
            Path::new("."),
            0.0,
            |_, _| {
                calls = calls.wrapping_add(1);
                Ok(json!({}))
            }
        )
        .is_err()
    );
    assert_eq!(calls, 0);
}

#[test]
fn existing_output_is_preserved() {
    let directory = tempfile::tempdir().unwrap();
    let manifest = directory.path().join("jobs.jsonl");
    let output = directory.path().join("results.jsonl");
    std::fs::write(&manifest, b"").unwrap();
    std::fs::write(&output, b"existing").unwrap();
    assert!(run_blocking(&manifest, &output, 0.0, |_, _| Ok(json!({}))).is_err());
    assert_eq!(std::fs::read(output).unwrap(), b"existing");
}

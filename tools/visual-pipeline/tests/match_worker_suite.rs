//! JSON Lines handshake between `PipelineMatcher::spawn_onnx_worker` and a fake worker.
//!
//! All cases run in one test so that no other thread forks while a fake
//! worker script is open for writing.

#![cfg(unix)]

use image::GrayImage;
use navigate_visual_pipeline::PipelineMatcher;
use serde_json::Value;
use std::{fs, os::unix::fs::PermissionsExt, path::Path};

/// Writes an executable worker script that checks the arguments, prints
/// `banner`, records the first request, and answers it with `response`.
#[cfg(test)]
fn fake_worker(dir: &Path, name: &str, banner: &str, response: &str) -> std::path::PathBuf {
    let script = dir.join(name);
    let request_log = dir.join(format!("{name}.request"));
    let body = format!(
        r#"#!/bin/sh
if [ "$#" -ne 5 ] || [ "$1" != match-worker ] || [ "$2" != --model ] \
    || [ "$3" != "{model}" ] || [ "$4" != --runtime ] || [ "$5" != "{runtime}" ]; then
    echo "unexpected arguments: $*" >&2
    exit 2
fi
printf '%s\n' '{banner}'
IFS= read -r line || exit 3
printf '%s\n' "$line" > "{log}"
printf '%s\n' '{response}'
IFS= read -r line
exit 0
"#,
        model = dir.join("model.onnx").display(),
        runtime = dir.join("libonnxruntime.dylib").display(),
        log = request_log.display(),
    );
    fs::write(&script, body).unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    script
}

#[cfg(test)]
fn spawn(
    dir: &Path,
    script: &Path,
) -> Result<PipelineMatcher, navigate_visual_pipeline::PipelineError> {
    PipelineMatcher::spawn_onnx_worker(
        script,
        &dir.join("model.onnx"),
        &dir.join("libonnxruntime.dylib"),
    )
}

#[cfg(test)]
fn assert_request_fits_worker_schema(dir: &Path, name: &str) {
    let line = fs::read_to_string(dir.join(format!("{name}.request"))).unwrap();
    let request: Value = serde_json::from_str(&line).unwrap();
    let object = request.as_object().unwrap();
    assert!(object["reference"].is_string() && object["query"].is_string());
    for key in object.keys() {
        assert!(
            ["reference", "query", "dense", "refinement"].contains(&key.as_str()),
            "match-worker rejects unknown request field {key}"
        );
    }
}

#[test]
fn onnx_worker_handshake_and_pair_round_trip() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let image = GrayImage::from_pixel(8, 8, image::Luma([128]));

    let current = fake_worker(
        dir,
        "current",
        r#"{"status":"ready","backend":"loftr-fake"}"#,
        r#"{"backend":"loftr-fake","pairs":[[[1.0,2.0],[3.0,4.0]]],"matches":[[[1.0,2.0],[3.0,4.0]]]}"#,
    );
    let mut matcher = spawn(dir, &current).unwrap();
    assert_eq!(matcher.identity(), "loftr-fake");
    let pairs = matcher.match_gray(&image, &image).unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].reference.x, 1.0);
    assert_eq!(pairs[0].reference.y, 2.0);
    assert_eq!(pairs[0].query.x, 3.0);
    assert_eq!(pairs[0].query.y, 4.0);
    drop(matcher);
    assert_request_fits_worker_schema(dir, "current");

    let pairs_only = fake_worker(
        dir,
        "pairs-only",
        r#"{"status":"ready","backend":"loftr-fake"}"#,
        r#"{"backend":"loftr-fake","pairs":[[[5.0,6.0],[7.0,8.0]]]}"#,
    );
    let mut matcher = spawn(dir, &pairs_only).unwrap();
    let pairs = matcher.match_gray(&image, &image).unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].query.y, 8.0);
    drop(matcher);

    let not_ready = fake_worker(dir, "not-ready", r#"{"status":"loading"}"#, "{}");
    assert!(spawn(dir, &not_ready).is_err());

    let wrong_args = fake_worker(dir, "wrong-args", "{}", "{}");
    let missing_model = PipelineMatcher::spawn_onnx_worker(
        &wrong_args,
        &dir.join("other.onnx"),
        &dir.join("libonnxruntime.dylib"),
    );
    assert!(missing_model.is_err());
}

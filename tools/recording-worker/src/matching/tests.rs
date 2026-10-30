use super::{Request, WorkerArgs, parse_args};
use std::{ffi::OsString, path::PathBuf};

fn os(args: &[&str]) -> Vec<OsString> {
    args.iter().map(OsString::from).collect()
}

fn expected() -> WorkerArgs {
    WorkerArgs {
        model: PathBuf::from("loftr.onnx"),
        runtime: PathBuf::from("libonnxruntime.dylib"),
    }
}

#[test]
fn dense_requests_keep_refinement_opt_in() {
    let ordinary: Request =
        serde_json::from_str(r#"{"reference":"reference.png","query":"query.png","dense":true}"#)
            .unwrap();
    assert!(!ordinary.refinement);
    let refinement: Request = serde_json::from_str(
        r#"{"reference":"reference.png","query":"query.png","dense":true,"refinement":true}"#,
    )
    .unwrap();
    assert!(refinement.dense && refinement.refinement);
}

#[test]
fn requests_without_dense_use_the_dense_matcher() {
    let request: Request =
        serde_json::from_str(r#"{"reference":"reference.png","query":"query.png"}"#).unwrap();
    assert!(request.dense);
    assert!(!request.refinement);
    let flow: Request =
        serde_json::from_str(r#"{"reference":"r.png","query":"q.png","dense":false}"#).unwrap();
    assert!(!flow.dense);
}

#[test]
fn requests_reject_unknown_fields() {
    assert!(
        serde_json::from_str::<Request>(r#"{"reference":"r.png","query":"q.png","extra":1}"#)
            .is_err()
    );
}

#[test]
fn positional_arguments_parse() {
    assert_eq!(
        parse_args(&os(&["loftr.onnx", "libonnxruntime.dylib"])),
        Ok(expected())
    );
}

#[test]
fn named_arguments_parse_in_any_order() {
    assert_eq!(
        parse_args(&os(&[
            "--model",
            "loftr.onnx",
            "--runtime",
            "libonnxruntime.dylib"
        ])),
        Ok(expected())
    );
    assert_eq!(
        parse_args(&os(&[
            "--runtime",
            "libonnxruntime.dylib",
            "--model",
            "loftr.onnx"
        ])),
        Ok(expected())
    );
}

#[test]
fn incomplete_or_unknown_arguments_fail() {
    assert!(parse_args(&os(&[])).is_err());
    assert!(parse_args(&os(&["loftr.onnx"])).is_err());
    assert!(parse_args(&os(&["--model", "loftr.onnx"])).is_err());
    assert!(parse_args(&os(&["--model", "loftr.onnx", "--runtime"])).is_err());
    assert!(parse_args(&os(&["--model", "a", "--model", "b"])).is_err());
    assert!(parse_args(&os(&["--model", "a", "--device", "b"])).is_err());
    assert!(parse_args(&os(&["a", "b", "c"])).is_err());
}

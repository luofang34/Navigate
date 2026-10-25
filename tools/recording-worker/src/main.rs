//! JSON Lines workers for camera matching and recording reconstruction.

mod matching;
mod navigation;

use std::{ffi::OsString, io::Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    match args.first().and_then(|arg| arg.to_str()) {
        Some("navigate-worker") => navigation::run_blocking(&args[1..]),
        Some("match-worker") => matching::run_blocking(&args[1..]),
        None | Some("--help" | "-h") => {
            std::io::stdout().write_all(
                b"navigate-recording-worker navigate-worker EVIDENCE_DIRECTORY CONFIGURATION_JSON\nnavigate-recording-worker match-worker LOFTR_MODEL ONNX_RUNTIME_LIBRARY\n",
            )?;
            Ok(())
        }
        _ => Err("expected navigate-worker or match-worker".into()),
    }
}

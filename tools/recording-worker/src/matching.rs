//! Resident matching uses Navigate's ONNX and optical-flow adapters.

use navigate_visual::{ImageMatcher, PyramidalMatcher};
use navigate_visual_onnx::{ExecutionConfig, LoFtrMatcher, initialize_blocking};
use serde::Deserialize;
use std::{
    ffi::OsString,
    io::{BufRead, Write},
    path::PathBuf,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    reference: PathBuf,
    query: PathBuf,
    dense: bool,
    #[serde(default)]
    refinement: bool,
}

pub(crate) fn run_blocking(args: &[OsString]) -> Result<(), Box<dyn std::error::Error>> {
    let [model, runtime] = args else {
        return Err("match-worker requires LOFTR_MODEL ONNX_RUNTIME_LIBRARY".into());
    };
    initialize_blocking(&PathBuf::from(runtime))?;
    let mut dense =
        LoFtrMatcher::load_blocking(&PathBuf::from(model), &ExecutionConfig::default())?;
    let mut refinement =
        LoFtrMatcher::load_blocking(&PathBuf::from(model), &ExecutionConfig::default())?
            .with_refinement_patches();
    let mut flow = PyramidalMatcher;
    let input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    for line in input.lines() {
        let request: Request = serde_json::from_str(&line?)?;
        let reference = image::open(&request.reference)?.to_luma8();
        let query = image::open(&request.query)?.to_luma8();
        let matcher: &mut dyn ImageMatcher = if request.dense && request.refinement {
            &mut refinement
        } else if request.dense {
            &mut dense
        } else {
            &mut flow
        };
        let pairs: Vec<[[f64; 2]; 2]> = matcher
            .match_images_blocking(&reference, &query)?
            .iter()
            .map(|pair| [pair.reference.into(), pair.query.into()])
            .collect();
        serde_json::to_writer(
            &mut output,
            &serde_json::json!({"backend":matcher.identity(),"pairs":pairs}),
        )?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;

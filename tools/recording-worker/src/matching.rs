//! Resident matching uses Navigate's ONNX and optical-flow adapters.

use navigate_visual::{ImageMatcher, PyramidalMatcher};
use navigate_visual_onnx::{ExecutionConfig, LoFtrMatcher, initialize_blocking};
use serde::Deserialize;
use std::{
    ffi::OsString,
    io::{BufRead, Write},
    path::PathBuf,
};

const USAGE: &str = "match-worker requires LOFTR_MODEL ONNX_RUNTIME_LIBRARY \
    or --model LOFTR_MODEL --runtime ONNX_RUNTIME_LIBRARY";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    reference: PathBuf,
    query: PathBuf,
    #[serde(default = "default_true")]
    dense: bool,
    #[serde(default)]
    refinement: bool,
}

fn default_true() -> bool {
    true
}

/// Model and runtime paths for one worker process.
#[derive(Debug, PartialEq, Eq)]
struct WorkerArgs {
    model: PathBuf,
    runtime: PathBuf,
}

/// Accepts the positional form and the named form that
/// `navigate_visual_pipeline::PipelineMatcher::spawn_onnx_worker` sends.
fn parse_args(args: &[OsString]) -> Result<WorkerArgs, String> {
    if let [model, runtime] = args
        && model.to_str().is_none_or(|m| !m.starts_with("--"))
    {
        return Ok(WorkerArgs {
            model: PathBuf::from(model),
            runtime: PathBuf::from(runtime),
        });
    }
    let mut model = None;
    let mut runtime = None;
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        let slot = match flag.to_str() {
            Some("--model") => &mut model,
            Some("--runtime") => &mut runtime,
            _ => return Err(format!("{USAGE}; unexpected argument {flag:?}")),
        };
        let value = iter
            .next()
            .ok_or_else(|| format!("{USAGE}; {flag:?} needs a value"))?;
        if slot.replace(PathBuf::from(value)).is_some() {
            return Err(format!("{USAGE}; {flag:?} given more than once"));
        }
    }
    match (model, runtime) {
        (Some(model), Some(runtime)) => Ok(WorkerArgs { model, runtime }),
        _ => Err(USAGE.to_owned()),
    }
}

pub(crate) fn run_blocking(args: &[OsString]) -> Result<(), Box<dyn std::error::Error>> {
    let WorkerArgs { model, runtime } = parse_args(args)?;
    initialize_blocking(&runtime)?;
    let mut dense = LoFtrMatcher::load_blocking(&model, &ExecutionConfig::default())?;
    let mut refinement =
        LoFtrMatcher::load_blocking(&model, &ExecutionConfig::default())?.with_refinement_patches();
    let mut flow = PyramidalMatcher;
    let input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    // Clients block on this banner, so it is sent only after every model loads.
    serde_json::to_writer(
        &mut output,
        &serde_json::json!({"status":"ready","backend":dense.identity()}),
    )?;
    output.write_all(b"\n")?;
    output.flush()?;
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
            &serde_json::json!({"backend":matcher.identity(),"pairs":pairs,"matches":pairs}),
        )?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;

//! Explicit runtime selection and persistent model timing.
use crate::{Args, ProbeError, Provider, context};
use ort::{
    ep::{
        self,
        coreml::{ComputeUnits, ModelFormat},
    },
    session::Session,
    value::{DynValue, Tensor},
};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::Path, time::Instant};
#[derive(Deserialize)]
struct Input {
    name: String,
    shape: Vec<usize>,
    path: String,
}
fn read_blocking(path: &Path) -> Result<Vec<u8>, ProbeError> {
    fs::read(path).map_err(|source| context(format!("read {}", path.display()), source))
}
fn inputs_blocking(path: &Path) -> Result<Vec<(String, DynValue)>, ProbeError> {
    let inputs: Vec<Input> = serde_json::from_slice(&read_blocking(path)?)
        .map_err(|source| context(format!("parse {}", path.display()), source))?;
    let directory = path
        .parent()
        .ok_or_else(|| ProbeError::Input("input manifest has no parent".into()))?;
    inputs
        .into_iter()
        .map(|input| {
            let data = read_blocking(&directory.join(input.path))?;
            if data.len() % 4 != 0 {
                return Err(ProbeError::Input(format!(
                    "{} is not float32 data",
                    input.name
                )));
            }
            let data: Vec<f32> = data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let tensor = Tensor::from_array((input.shape, data))
                .map_err(|source| context(format!("build tensor {}", input.name), source))?;
            Ok((input.name, tensor.into_dyn()))
        })
        .collect()
}
pub(super) fn run_blocking(args: &Args) -> Result<(), ProbeError> {
    ort::init_from(&args.library)
        .map_err(|source| context("load ONNX Runtime", source))?
        .with_telemetry(false)
        .commit();
    let inputs = inputs_blocking(&args.inputs)?;
    fs::create_dir(&args.output)
        .map_err(|source| context(format!("create {}", args.output.display()), source))?;
    let started = Instant::now();
    let mut session = session_blocking(args)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let (times, values) = measure(&mut session, &inputs, args.repetitions)?;
    let profile = session
        .end_profiling()
        .map_err(|source| context("finish provider profile", source))?;
    let nodes = provider_nodes_blocking(Path::new(&profile))?;
    let report = serde_json::json!({"requested_provider":format!("{:?}",args.provider),"load_ms":load_ms,"warm_inference_ms":times,"profile":profile,"provider_node_events":nodes,"outputs":values,"scope":"Model only. Core ML selection permits CPU execution. Provider node placement does not prove exclusive GPU or ANE use."});
    let output = args.output.join("result.json");
    fs::write(
        &output,
        serde_json::to_vec_pretty(&report).map_err(|source| context("serialize result", source))?,
    )
    .map_err(|source| context(format!("write {}", output.display()), source))?;
    tracing::info!(output=%output.display(),load_ms,provider=?args.provider,"native inference complete");
    Ok(())
}
fn session_blocking(args: &Args) -> Result<Session, ProbeError> {
    let mut builder = Session::builder()
        .map_err(|source| context("create ONNX session", source))?
        .with_intra_threads(args.threads as usize)
        .map_err(|source| context("set CPU threads", source))?
        .with_profiling(args.output.join("profile"))
        .map_err(|source| context("enable provider profiling", source))?;
    if !matches!(args.provider, Provider::Cpu) {
        let units = if matches!(args.provider, Provider::CoremlAne) {
            ComputeUnits::CPUAndNeuralEngine
        } else {
            ComputeUnits::CPUAndGPU
        };
        let provider = ep::CoreML::default()
            .with_compute_units(units)
            .with_model_format(ModelFormat::MLProgram)
            .with_profile_compute_plan(true)
            .with_model_cache_dir(args.output.join("coreml-cache").display())
            .build()
            .error_on_failure();
        builder = builder
            .with_execution_providers([provider])
            .map_err(|source| context("register requested Core ML provider", source))?;
    }
    let session = builder
        .commit_from_file(&args.model)
        .map_err(|source| context(format!("load {}", args.model.display()), source))?;
    Ok(session)
}
fn measure(
    session: &mut Session,
    inputs: &[(String, DynValue)],
    repetitions: u32,
) -> Result<(Vec<f64>, serde_json::Map<String, serde_json::Value>), ProbeError> {
    let mut times = Vec::new();
    let mut values = serde_json::Map::new();
    for index in 0..=repetitions {
        let start = Instant::now();
        let supplied: Vec<_> = inputs
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect();
        let output = session
            .run(supplied)
            .map_err(|source| context("run ONNX inference", source))?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if index > 0 {
            times.push(elapsed);
        }
        if index == repetitions {
            for (name, value) in output.iter() {
                let data = if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
                    serde_json::json!({"shape":shape.to_vec(),"data":data})
                } else if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                    serde_json::json!({"shape":shape.to_vec(),"data":if data.len()<=4096 {Some(data)}else{None},"elements":data.len(),"sum":data.iter().map(|x|f64::from(*x)).sum::<f64>()})
                } else {
                    serde_json::json!({"unsupported_output_type":true})
                };
                values.insert(name.to_owned(), data);
            }
        }
    }
    Ok((times, values))
}
fn provider_nodes_blocking(path: &Path) -> Result<BTreeMap<String, u64>, ProbeError> {
    let events: Vec<serde_json::Value> = serde_json::from_slice(&read_blocking(path)?)
        .map_err(|source| context("read provider profile", source))?;
    let mut counts = BTreeMap::new();
    for event in events {
        if let Some(provider) = event["args"]["provider"].as_str() {
            let count = counts.entry(provider.to_owned()).or_insert(0_u64);
            *count = count.wrapping_add(1);
        }
    }
    Ok(counts)
}

//! Measure native ONNX execution and record provider placement.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod runtime;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
enum ProbeError {
    #[error("{operation}: {source}")]
    Operation {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error>,
    },
    #[error("invalid probe input: {0}")]
    Input(String),
}
fn context<E: std::error::Error + 'static>(operation: impl Into<String>, source: E) -> ProbeError {
    ProbeError::Operation {
        operation: operation.into(),
        source: Box::new(source),
    }
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Provider {
    Cpu,
    CoremlGpu,
    CoremlAne,
    Cuda,
    TensorRt,
}
#[derive(Parser)]
#[command(
    about = "Measure native Rust ONNX inference. Accelerator placement can include CPU operations."
)]
struct Args {
    #[arg(long)]
    library: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    inputs: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// NVIDIA device index for CUDA or TensorRT.
    #[arg(long, default_value_t=0, value_parser=clap::value_parser!(i32).range(0..))]
    device_id: i32,
    /// TensorRT builder workspace limit. This is not total GPU memory.
    #[arg(long, default_value_t=256, value_parser=clap::value_parser!(u32).range(1..=65536))]
    workspace_mib: u32,
    /// Save float32 output tensors for numerical comparison.
    #[arg(long)]
    save_outputs: bool,
    #[arg(long, value_enum, default_value = "cpu")]
    provider: Provider,
    #[arg(long,default_value_t=10,value_parser=clap::value_parser!(u32).range(1..=10000))]
    repetitions: u32,
    #[arg(long,default_value_t=4,value_parser=clap::value_parser!(u32).range(1..=64))]
    threads: u32,
}
fn main() -> Result<(), ProbeError> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let args = match Args::try_parse() {
        Ok(args) => args,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            error
                .print()
                .map_err(|source| context("write command help", source))?;
            return Ok(());
        }
        Err(error) => return Err(context("parse command arguments", error)),
    };
    runtime::run_blocking(&args)
}

use crate::ProbeError;
use clap::{Arg, ArgAction, ArgMatches, Command, ValueEnum, builder::PossibleValue, value_parser};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Provider {
    Cpu,
    CoremlGpu,
    CoremlAne,
    Cuda,
    TensorRt,
}
impl ValueEnum for Provider {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            Self::Cpu,
            Self::CoremlGpu,
            Self::CoremlAne,
            Self::Cuda,
            Self::TensorRt,
        ]
    }
    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(PossibleValue::new(match self {
            Self::Cpu => "cpu",
            Self::CoremlGpu => "coreml-gpu",
            Self::CoremlAne => "coreml-ane",
            Self::Cuda => "cuda",
            Self::TensorRt => "tensor-rt",
        }))
    }
}
pub(crate) struct Args {
    pub(crate) library: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) inputs: PathBuf,
    pub(crate) output: PathBuf,
    pub(crate) device_id: i32,
    pub(crate) workspace_mib: u32,
    pub(crate) save_outputs: bool,
    pub(crate) provider: Provider,
    pub(crate) repetitions: u32,
    pub(crate) threads: u32,
}
fn path_arg(id: &'static str) -> Arg {
    Arg::new(id)
        .long(id)
        .required(true)
        .value_parser(value_parser!(PathBuf))
}
pub(crate) fn command() -> Command {
    Command::new(env!("CARGO_PKG_NAME"))
        .about(
            "Measure native Rust ONNX inference. Accelerator placement can include CPU operations.",
        )
        .args(["library", "model", "inputs", "output"].map(path_arg))
        .arg(
            Arg::new("device_id")
                .long("device-id")
                .help("NVIDIA device index for CUDA or TensorRT.")
                .default_value("0")
                .value_parser(value_parser!(i32).range(0..)),
        )
        .arg(
            Arg::new("workspace_mib")
                .long("workspace-mib")
                .help("TensorRT builder workspace limit. This is not total GPU memory.")
                .default_value("256")
                .value_parser(value_parser!(u32).range(1..=65536)),
        )
        .arg(
            Arg::new("save_outputs")
                .long("save-outputs")
                .help("Save float32 output tensors for numerical comparison.")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("provider")
                .long("provider")
                .default_value("cpu")
                .value_parser(value_parser!(Provider)),
        )
        .arg(
            Arg::new("repetitions")
                .long("repetitions")
                .default_value("10")
                .value_parser(value_parser!(u32).range(1..=10000)),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .default_value("4")
                .value_parser(value_parser!(u32).range(1..=64)),
        )
}
fn take<T: Clone + Send + Sync + 'static>(
    matches: &mut ArgMatches,
    id: &'static str,
) -> Result<T, ProbeError> {
    matches
        .remove_one(id)
        .ok_or_else(|| ProbeError::Input(format!("missing --{id}")))
}
impl Args {
    pub(crate) fn from_matches(mut matches: ArgMatches) -> Result<Self, ProbeError> {
        Ok(Self {
            library: take(&mut matches, "library")?,
            model: take(&mut matches, "model")?,
            inputs: take(&mut matches, "inputs")?,
            output: take(&mut matches, "output")?,
            device_id: take(&mut matches, "device_id")?,
            workspace_mib: take(&mut matches, "workspace_mib")?,
            save_outputs: matches.get_flag("save_outputs"),
            provider: take(&mut matches, "provider")?,
            repetitions: take(&mut matches, "repetitions")?,
            threads: take(&mut matches, "threads")?,
        })
    }
}
#[cfg(test)]
mod tests;

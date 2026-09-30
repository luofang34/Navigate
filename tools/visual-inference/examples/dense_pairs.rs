//! Run a resident dense matcher on local reference/query PNG pairs.
use clap::{Arg, ArgMatches, Command, ValueEnum, builder::PossibleValue, value_parser};
use navigate_visual::ImageMatcher;
use navigate_visual_onnx::{ExecutionConfig, LoFtrMatcher, Provider, initialize_blocking};
use serde_json::json;
use std::{io::Write, path::PathBuf, time::Instant};

#[derive(Clone, Copy)]
enum Device {
    Cpu,
    CoremlAne,
    CoremlGpu,
}
impl ValueEnum for Device {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Cpu, Self::CoremlAne, Self::CoremlGpu]
    }
    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(PossibleValue::new(match self {
            Self::Cpu => "cpu",
            Self::CoremlAne => "coreml-ane",
            Self::CoremlGpu => "coreml-gpu",
        }))
    }
}
struct Args {
    library: PathBuf,
    model: PathBuf,
    pairs: PathBuf,
    output: PathBuf,
    device: Device,
}
fn take<T: Clone + Send + Sync + 'static>(
    matches: &mut ArgMatches,
    id: &'static str,
) -> Result<T, Box<dyn std::error::Error>> {
    matches
        .remove_one(id)
        .ok_or_else(|| format!("missing --{id}").into())
}
fn command() -> Command {
    Command::new("dense_pairs")
        .args(["library", "model", "pairs", "output"].map(|id| {
            Arg::new(id)
                .long(id)
                .required(true)
                .value_parser(value_parser!(PathBuf))
        }))
        .arg(
            Arg::new("device")
                .long("device")
                .default_value("cpu")
                .value_parser(value_parser!(Device)),
        )
}
impl Args {
    fn from_matches(mut m: ArgMatches) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            library: take(&mut m, "library")?,
            model: take(&mut m, "model")?,
            pairs: take(&mut m, "pairs")?,
            output: take(&mut m, "output")?,
            device: take(&mut m, "device")?,
        })
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let args = Args::from_matches(command().try_get_matches()?)?;
    initialize_blocking(&args.library)?;
    let provider = match args.device {
        Device::Cpu => Provider::Cpu,
        Device::CoremlAne => Provider::CoreMlAne,
        Device::CoremlGpu => Provider::CoreMlGpu,
    };
    let start = Instant::now();
    let mut matcher = LoFtrMatcher::load_blocking(
        &args.model,
        &ExecutionConfig {
            provider,
            ..Default::default()
        },
    )?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut files = std::fs::read_dir(&args.pairs)?.collect::<Result<Vec<_>, _>>()?;
    files.sort_by_key(|f| f.file_name());
    let mut rows = Vec::new();
    for file in files {
        let name = file.file_name();
        let name = name.to_str().ok_or("pair name is not UTF-8")?;
        let Some(base) = name.strip_suffix("-reference.png") else {
            continue;
        };
        let reference = image::open(file.path())?.to_luma8();
        let query = image::open(args.pairs.join(format!("{base}-query.png")))?.to_luma8();
        let start = Instant::now();
        let pairs = matcher.match_images_blocking(&reference, &query)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            name,
            matches = pairs.len(),
            elapsed_ms,
            "dense correspondence result"
        );
        rows.push(json!({"name":name,"elapsed_ms":elapsed_ms,"pairs":pairs.iter().map(|p|json!({"reference":[p.reference.x,p.reference.y],"query":[p.query.x,p.query.y]})).collect::<Vec<_>>() }));
    }
    let report = json!({"scope":"local image correspondences only; no geographic acceptance or accuracy claim", "backend":matcher.identity(), "load_ms":load_ms, "cases":rows});
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    output.write_all(&serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

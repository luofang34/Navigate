//! Run a resident dense matcher on local reference/query PNG pairs.
#[path = "dense_pairs/stream.rs"]
mod stream;

use clap::{Arg, ArgAction, ArgMatches, Command, ValueEnum, builder::PossibleValue, value_parser};
use navigate_visual::ImageMatcher;
use navigate_visual_onnx::{
    ExecutionConfig, LightGlueMatcher, LoFtrMatcher, Provider, initialize_blocking,
};
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
fn take<T: Clone + Send + Sync + 'static>(
    matches: &mut ArgMatches,
    id: &'static str,
) -> Result<T, Box<dyn std::error::Error>> {
    matches
        .remove_one(id)
        .ok_or_else(|| format!("missing --{id}").into())
}
fn path(id: &'static str) -> Arg {
    Arg::new(id).long(id).value_parser(value_parser!(PathBuf))
}
fn flag(id: &'static str, long: &'static str) -> Arg {
    Arg::new(id).long(long).action(ArgAction::SetTrue)
}
fn device() -> Arg {
    Arg::new("device")
        .long("device")
        .default_value("cpu")
        .value_parser(value_parser!(Device))
}
struct Args {
    refinement_patches: bool,
    detector: Option<PathBuf>,
    keypoints: u32,
    library: PathBuf,
    model: PathBuf,
    coreml_cache: Option<PathBuf>,
    serve: bool,
    pairs: Option<PathBuf>,
    output: Option<PathBuf>,
    device: Device,
}
fn command() -> Command {
    Command::new("dense_pairs")
        .arg(flag("refinement_patches", "refinement-patches"))
        .arg(path("detector"))
        .arg(
            Arg::new("keypoints")
                .long("keypoints")
                .help("Maximum SuperPoint features for the LightGlue adapter.")
                .default_value("2048")
                .value_parser(value_parser!(u32).range(32..=4096)),
        )
        .arg(path("library").required(true))
        .arg(path("model").required(true))
        .arg(
            path("coreml_cache")
                .long("coreml-cache")
                .help("Local compiled-model cache for Core ML and embedded weights."),
        )
        .arg(flag("serve", "serve").conflicts_with_all(["pairs", "output"]))
        .arg(
            path("pairs")
                .required_unless_present("serve")
                .requires("output"),
        )
        .arg(
            path("output")
                .required_unless_present("serve")
                .requires("pairs"),
        )
        .arg(device())
}
impl Args {
    fn from_matches(mut m: ArgMatches) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            refinement_patches: m.get_flag("refinement_patches"),
            detector: m.remove_one("detector"),
            keypoints: take(&mut m, "keypoints")?,
            library: take(&mut m, "library")?,
            model: take(&mut m, "model")?,
            coreml_cache: m.remove_one("coreml_cache"),
            serve: m.get_flag("serve"),
            pairs: m.remove_one("pairs"),
            output: m.remove_one("output"),
            device: take(&mut m, "device")?,
        })
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let matches = match command().try_get_matches() {
        Ok(matches) => matches,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            error.print()?;
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let args = Args::from_matches(matches)?;
    initialize_blocking(&args.library)?;
    let provider = match args.device {
        Device::Cpu => Provider::Cpu,
        Device::CoremlAne => Provider::CoreMlAne,
        Device::CoremlGpu => Provider::CoreMlGpu,
    };
    let start = Instant::now();
    let execution = ExecutionConfig {
        provider,
        coreml_cache_directory: args.coreml_cache.clone(),
        ..Default::default()
    };
    let mut matcher = load_matcher_blocking(&args, execution)?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    if args.serve {
        stream::serve_blocking(
            &mut *matcher,
            std::io::stdin().lock(),
            std::io::stdout().lock(),
            load_ms,
        )?;
        return Ok(());
    }
    batch_blocking(&args, &mut *matcher, load_ms)
}

fn batch_blocking(
    args: &Args,
    matcher: &mut dyn ImageMatcher,
    load_ms: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let pairs_directory = args.pairs.as_ref().ok_or("missing pairs directory")?;
    let output_path = args.output.as_ref().ok_or("missing report path")?;
    let mut files = std::fs::read_dir(pairs_directory)?.collect::<Result<Vec<_>, _>>()?;
    files.sort_by_key(|f| f.file_name());
    let mut rows = Vec::new();
    for file in files {
        let name = file.file_name();
        let name = name.to_str().ok_or("pair name is not UTF-8")?;
        let Some(base) = name.strip_suffix("-reference.png") else {
            continue;
        };
        let reference = image::open(file.path())?.to_luma8();
        let query = image::open(pairs_directory.join(format!("{base}-query.png")))?.to_luma8();
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
        .open(output_path)?;
    output.write_all(&serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

fn load_matcher_blocking(
    args: &Args,
    execution: ExecutionConfig,
) -> Result<Box<dyn ImageMatcher>, Box<dyn std::error::Error>> {
    if args.refinement_patches && args.detector.is_some() {
        return Err("refinement patches require the LoFTR adapter".into());
    }
    Ok(match &args.detector {
        Some(detector) => Box::new(LightGlueMatcher::load_blocking(
            detector,
            &args.model,
            [640, 360],
            args.keypoints as usize,
            &execution,
        )?),
        None => {
            let matcher = LoFtrMatcher::load_blocking(&args.model, &execution)?;
            let matcher = if args.refinement_patches {
                matcher.with_refinement_patches()
            } else {
                matcher
            };
            Box::new(matcher)
        }
    })
}

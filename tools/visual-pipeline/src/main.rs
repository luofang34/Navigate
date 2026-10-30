//! Command-line interface for the pure-Rust visual pipeline.

mod corridor_cli;
mod dispatch;

use clap::{Arg, ArgAction, Command, value_parser};
use navigate_visual_pipeline::{PipelineError, util::write_stdout_line};
use std::path::PathBuf;

fn main() -> Result<(), PipelineError> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .without_time()
        .init();
    let matches = match cli().try_get_matches() {
        Ok(matches) => matches,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            return error.print().map_err(|source| PipelineError::Io {
                path: PathBuf::from("stdout"),
                source,
            });
        }
        Err(error) => return Err(PipelineError::Invalid(error.to_string())),
    };
    let (sub, args) = matches
        .subcommand()
        .ok_or_else(|| PipelineError::Invalid("missing subcommand".into()))?;
    let summary = dispatch::dispatch(sub, args)?;
    let line = serde_json::to_string_pretty(&summary).map_err(|source| PipelineError::Json {
        context: "CLI summary".into(),
        source,
    })?;
    write_stdout_line(&line)
}

fn path_arg(name: &'static str, required: bool) -> Arg {
    Arg::new(name)
        .long(name)
        .required(required)
        .value_parser(value_parser!(PathBuf))
}

fn multi_path_arg(name: &'static str) -> Arg {
    Arg::new(name)
        .long(name)
        .required(true)
        .action(ArgAction::Append)
        .value_parser(value_parser!(PathBuf))
}

fn str_arg(name: &'static str, default: Option<&'static str>) -> Arg {
    let mut arg = Arg::new(name).long(name);
    if let Some(default) = default {
        arg = arg.default_value(default);
    } else {
        arg = arg.required(true);
    }
    arg
}

fn matcher_args(cmd: Command) -> Command {
    cmd.arg(str_arg("backend", Some("xfeat")))
        .arg(path_arg("model", false))
        .arg(path_arg("lighterglue", false))
        .arg(path_arg("onnx-runtime", false))
}

fn cli() -> Command {
    Command::new("navigate-visual-pipeline")
        .about("Run photo and video preparation, terrain pose search, segmentation, and debrief export")
        .subcommand_required(true)
        .subcommand(
            Command::new("prepare")
                .about("Prepare a photo test set from a raw source directory")
                .arg(path_arg("source", true))
                .arg(path_arg("output", true))
                .arg(
                    Arg::new("long-edge")
                        .long("long-edge")
                        .default_value("1600")
                        .value_parser(value_parser!(u32)),
                ),
        )
        .subcommand(
            Command::new("priors")
                .about("Interpolate missing position priors along flights")
                .arg(path_arg("dataset", true)),
        )
        .subcommand(
            Command::new("reference")
                .about("Build a bounded reference tile catalog")
                .arg(path_arg("profile", true))
                .arg(path_arg("output", true))
                .arg(str_arg("imagery-url", None))
                .arg(str_arg("dem-url", None))
                .arg(path_arg("cache", false))
                .arg(
                    Arg::new("radius-m")
                        .long("radius-m")
                        .default_value("15000.0")
                        .value_parser(value_parser!(f64)),
                )
                .arg(
                    Arg::new("imagery-zoom")
                        .long("imagery-zoom")
                        .default_value("14")
                        .value_parser(value_parser!(u32)),
                )
                .arg(
                    Arg::new("dem-zoom")
                        .long("dem-zoom")
                        .default_value("12")
                        .value_parser(value_parser!(u32)),
                ),
        )
        .subcommand(
            Command::new("mask")
                .about("Generate SegFormer-B0 occlusion masks for a photo dataset")
                .arg(path_arg("dataset", true))
                .arg(path_arg("model", true))
                .arg(str_arg("policy", Some("airframe-occlusions")))
                .arg(
                    Arg::new("margin")
                        .long("margin")
                        .default_value("6")
                        .value_parser(value_parser!(usize)),
                ),
        )
        .subcommand(
            Command::new("video-masks")
                .about("Generate SegFormer-B0 and propeller masks for video frames")
                .arg(path_arg("frames", true))
                .arg(path_arg("model", true))
                .arg(path_arg("output", true))
                .arg(
                    Arg::new("threshold")
                        .long("threshold")
                        .default_value("0.35")
                        .value_parser(value_parser!(f32)),
                )
                .arg(
                    Arg::new("margin")
                        .long("margin")
                        .default_value("6")
                        .value_parser(value_parser!(usize)),
                ),
        )
        .subcommand(
            Command::new("segment-finetune")
                .about("Fine-tune a 4-class SegFormer-B0 checkpoint in Rust")
                .arg(path_arg("dataset", true))
                .arg(path_arg("base-model", false))
                .arg(path_arg("output", true))
                .arg(
                    Arg::new("size")
                        .long("size")
                        .default_value("512")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    Arg::new("batch-size")
                        .long("batch-size")
                        .default_value("8")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    Arg::new("epochs")
                        .long("epochs")
                        .default_value("40")
                        .value_parser(value_parser!(usize)),
                )
                .arg(str_arg("color", Some("rand-gray")))
                .arg(
                    Arg::new("seed")
                        .long("seed")
                        .default_value("0")
                        .value_parser(value_parser!(u64)),
                ),
        )
        .subcommand(
            Command::new("airframe-labels")
                .about("Compose 4-class training labels and review sheets")
                .arg(path_arg("image", true))
                .arg(path_arg("sky-mask", true))
                .arg(path_arg("airframe-mask", false))
                .arg(path_arg("propeller-mask", false))
                .arg(path_arg("output", true))
                .arg(path_arg("sheet", false)),
        )
        .subcommands(matcher_subcommands())
        .subcommands(corridor_cli::commands())
        .subcommands(export_and_video_subcommands())
}

fn matcher_subcommands() -> [Command; 7] {
    [
        matcher_args(
            Command::new("match")
                .about("Search terrain poses for a case profile")
                .arg(path_arg("dataset", true))
                .arg(path_arg("profile", true))
                .arg(path_arg("catalog", true))
                .arg(path_arg("context-catalog", false))
                .arg(path_arg("output", true))
                .arg(
                    Arg::new("attempts")
                        .long("attempts")
                        .default_value("12")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    Arg::new("steps")
                        .long("steps")
                        .default_value("2")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    Arg::new("heading-step")
                        .long("heading-step")
                        .default_value("0.0")
                        .value_parser(value_parser!(f64)),
                )
                .arg(Arg::new("only").long("only").action(ArgAction::Append)),
        ),
        Command::new("bank-build")
            .about("Precompute and save an NVKBANK1 Kashin reference feature bank")
            .arg(path_arg("dataset", true))
            .arg(path_arg("profile", true))
            .arg(path_arg("catalog", true))
            .arg(path_arg("model", true))
            .arg(path_arg("output", true))
            .arg(
                Arg::new("attempts")
                    .long("attempts")
                    .default_value("12")
                    .value_parser(value_parser!(usize)),
            )
            .arg(
                Arg::new("heading-step")
                    .long("heading-step")
                    .default_value("0.0")
                    .value_parser(value_parser!(f64)),
            )
            .arg(
                Arg::new("max-keypoints")
                    .long("max-keypoints")
                    .default_value("2048")
                    .value_parser(value_parser!(usize)),
            )
            .arg(
                Arg::new("adaptive-stride")
                    .long("adaptive-stride")
                    .default_value("4")
                    .value_parser(value_parser!(u32)),
            )
            .arg(Arg::new("only").long("only").action(ArgAction::Append)),
        Command::new("bank-match")
            .about("Match query photos in real time against an NVKBANK1 reference bank")
            .arg(path_arg("dataset", true))
            .arg(path_arg("profile", true))
            .arg(path_arg("catalog", true))
            .arg(path_arg("bank", true))
            .arg(path_arg("model", true))
            .arg(path_arg("lighterglue", false))
            .arg(path_arg("output", true))
            .arg(
                Arg::new("shortlist")
                    .long("shortlist")
                    .default_value("4")
                    .value_parser(value_parser!(usize)),
            )
            .arg(
                Arg::new("query-keypoints")
                    .long("query-keypoints")
                    .default_value("2048")
                    .value_parser(value_parser!(usize)),
            )
            .arg(Arg::new("polish").long("polish").action(ArgAction::SetTrue))
            .arg(Arg::new("only").long("only").action(ArgAction::Append)),
        matcher_args(
            Command::new("track")
                .about("Propagate poses from parent anchor runs")
                .arg(path_arg("dataset", true))
                .arg(path_arg("profile", true))
                .arg(path_arg("catalog", true))
                .arg(multi_path_arg("parents"))
                .arg(path_arg("output", true)),
        ),
        matcher_args(
            Command::new("refine")
                .about("Refine parent candidates against full-resolution renders")
                .arg(path_arg("dataset", true))
                .arg(path_arg("profile", true))
                .arg(path_arg("catalog", true))
                .arg(multi_path_arg("parents"))
                .arg(path_arg("output", true)),
        ),
        Command::new("terrain-bundle")
            .about("Run DEM-regularized multi-view bundle adjustment")
            .arg(path_arg("catalog", true))
            .arg(multi_path_arg("parents"))
            .arg(path_arg("output", true)),
        matcher_args(
            Command::new("pipeline")
                .about("Run a multi-pass pipeline.json recipe")
                .arg(path_arg("recipe", true)),
        ),
    ]
}

fn export_and_video_subcommands() -> [Command; 10] {
    [
        Command::new("preview")
            .about("Export debrief.json and replay.json")
            .arg(path_arg("dataset", true))
            .arg(path_arg("catalog", true))
            .arg(multi_path_arg("runs"))
            .arg(path_arg("output", true)),
        Command::new("freeze")
            .about("Freeze accepted candidate poses into a baseline archive")
            .arg(multi_path_arg("runs"))
            .arg(path_arg("output", true)),
        Command::new("publish")
            .about("Export publication.json with SHA-256 verification")
            .arg(path_arg("files-dir", true))
            .arg(str_arg("base-url", None))
            .arg(path_arg("output", true)),
        Command::new("reproduce")
            .about("Hydrate and verify files from publication.json")
            .arg(path_arg("publication", true))
            .arg(path_arg("local-files", false))
            .arg(path_arg("target", true))
            .arg(Arg::new("prefix").long("prefix"))
            .arg(
                Arg::new("limit")
                    .long("limit")
                    .value_parser(value_parser!(usize)),
            ),
        Command::new("live-prior")
            .about("Seed acquisition.json from a Sokoly live survey")
            .arg(path_arg("replay-dir", true))
            .arg(path_arg("video", true))
            .arg(path_arg("map-manifest", true))
            .arg(path_arg("output", true))
            .arg(
                Arg::new("offset-s")
                    .long("offset-s")
                    .default_value("0.0")
                    .value_parser(value_parser!(f64)),
            )
            .arg(
                Arg::new("terrain-m")
                    .long("terrain-m")
                    .default_value("0.0")
                    .value_parser(value_parser!(f64)),
            ),
        Command::new("import-video")
            .about("Import raw video metadata and QuickTime clock")
            .arg(path_arg("video", true))
            .arg(path_arg("output", true)),
        Command::new("video-prepare")
            .about("Extract keyframes from a video into a dataset")
            .arg(path_arg("video", true))
            .arg(path_arg("output", true))
            .arg(
                Arg::new("fps")
                    .long("fps")
                    .default_value("1.0")
                    .value_parser(value_parser!(f64)),
            )
            .arg(
                Arg::new("long-edge")
                    .long("long-edge")
                    .default_value("1280")
                    .value_parser(value_parser!(u32)),
            ),
        Command::new("video-priors")
            .about("Seed video frame priors from acquisition.json")
            .arg(path_arg("dataset", true))
            .arg(path_arg("acquisition", true)),
        Command::new("navigate-debrief")
            .about("Run navigate_recording::Processor over recorded evidence")
            .arg(path_arg("evidence", true))
            .arg(path_arg("output", true)),
        Command::new("compose-survey")
            .about("Compose multi-flight sokoly-debrief v1 streams")
            .arg(multi_path_arg("flights"))
            .arg(path_arg("output", true)),
    ]
}

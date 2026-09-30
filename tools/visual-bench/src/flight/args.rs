//! Host controls for a resident video-processing lane.
use crate::{
    BenchError,
    cli::args::{flag, path, positional, take, value},
};
use clap::{ArgMatches, Command, value_parser};
use std::path::PathBuf;

pub(crate) struct FlightArgs {
    pub package: PathBuf,
    pub video: PathBuf,
    pub prior: PathBuf,
    pub output: PathBuf,
    pub track: Option<PathBuf>,
    pub inference: super::inference::InferenceArgs,
    pub candidates: Option<PathBuf>,
    pub timeline_cache: Option<PathBuf>,
    pub width: u32,
    pub fps: f64,
    pub adaptive: bool,
    pub utilization: f64,
    pub budget_file: Option<PathBuf>,
    pub realtime: bool,
    pub warm_start: bool,
    pub start: f64,
    pub duration: Option<f64>,
    pub map_interval: f64,
    pub reanchor: bool,
    pub surface_tracks: bool,
    pub keyframe_interval: f64,
    pub dense_only: bool,
    pub fixed_tilt: bool,
}

impl FlightArgs {
    pub fn args(command: Command) -> Command {
        let command = command
            .arg(positional("package"))
            .arg(positional("video"))
            .arg(path("prior").required(true).help(
                "Camera calibration at the source video resolution and a fixed navigation prior.",
            ))
            .arg(path("output").required(true))
            .arg(
                path("track")
                    .help("New GeoJSON file with separate candidate paths and map-check points."),
            );
        super::inference::InferenceArgs::args(command)
            .arg(
                path("candidates")
                    .help("Optional estimated candidate poses, separate from navigation admission bounds."),
            )
            .arg(
                path("timeline-cache")
                    .help("Timeline cache verified against the complete source-video digest."),
            )
            .arg(
                value("width", "960", value_parser!(u32))
                    .help("Processing width. Intrinsics retain pixel-centre alignment after resizing."),
            )
            .arg(
                value("fps", "5", value_parser!(f64))
                    .help("Largest requested sample rate. Actual admitted rate can be smaller."),
            )
            .arg(flag("adaptive").help(
                "Adapt sampling from measured cost and the available processing-time fraction.",
            ))
            .arg(value("utilization", "0.7", value_parser!(f64)))
            .arg(
                path("budget-file")
                    .help("Optional file with a live fraction from zero to one. The host can change it."),
            )
            .arg(
                flag("realtime")
                    .help("Pace the file by its presentation clock and discard obsolete frames."),
            )
            .arg(flag("warm-start").requires("realtime").help(
                "Start paced playback after the first map anchor. Measures warm tracking, not live acquisition.",
            ))
            .arg(value("start", "0", value_parser!(f64)))
            .arg(
                clap::Arg::new("duration")
                    .long("duration")
                    .value_parser(value_parser!(f64)),
            )
            .arg(
                value("map-interval", "5", value_parser!(f64))
                    .help("Independent map checks are reported without overwriting the relative path."),
            )
            .arg(flag("reanchor").help(
                "Restart a path at supported map checks. Retain the prior relative hypothesis and mark a discontinuity.",
            ))
            .arg(flag("surface-tracks").help(
                "Retain surface-seeded world points across supported camera frames. Uses free attitude.",
            ))
            .arg(
                value("keyframe-interval", "0", value_parser!(f64))
                    .help("Maximum reference-frame age. Zero updates the reference at each supported frame."),
            )
            .arg(flag("dense-only").help(
                "Use the dense matcher for every pair instead of trying classical tracking first.",
            ))
            .arg(
                flag("fixed-tilt")
                    .help("Assume constant camera tilt during relative tracking. No attitude accuracy claim."),
            )
    }

    pub fn from_matches(matches: &mut ArgMatches) -> Result<Self, BenchError> {
        Ok(Self {
            package: take(matches, "package")?,
            video: take(matches, "video")?,
            prior: take(matches, "prior")?,
            output: take(matches, "output")?,
            track: matches.remove_one("track"),
            inference: super::inference::InferenceArgs::from_matches(matches)?,
            candidates: matches.remove_one("candidates"),
            timeline_cache: matches.remove_one("timeline-cache"),
            width: take(matches, "width")?,
            fps: take(matches, "fps")?,
            adaptive: matches.get_flag("adaptive"),
            utilization: take(matches, "utilization")?,
            budget_file: matches.remove_one("budget-file"),
            realtime: matches.get_flag("realtime"),
            warm_start: matches.get_flag("warm-start"),
            start: take(matches, "start")?,
            duration: matches.remove_one("duration"),
            map_interval: take(matches, "map-interval")?,
            reanchor: matches.get_flag("reanchor"),
            surface_tracks: matches.get_flag("surface-tracks"),
            keyframe_interval: take(matches, "keyframe-interval")?,
            dense_only: matches.get_flag("dense-only"),
            fixed_tilt: matches.get_flag("fixed-tilt"),
        })
    }
}

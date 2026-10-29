//! CLI command definitions and dispatch for `corridor-build` and `corridor-track`.

use clap::{Arg, ArgAction, ArgMatches, Command, value_parser};
use navigate_visual_pipeline::{
    CorridorBuildOptions, CorridorTrackOptions, PipelineError, build_corridor_bank,
    run_corridor_track,
};
use serde_json::Value;
use std::{collections::BTreeSet, path::PathBuf};

fn path_arg(name: &'static str, required: bool) -> Arg {
    Arg::new(name)
        .long(name)
        .required(required)
        .value_parser(value_parser!(PathBuf))
}

fn float_arg(name: &'static str, default: Option<&'static str>) -> Arg {
    let mut a = Arg::new(name)
        .long(name)
        .allow_hyphen_values(true)
        .value_parser(value_parser!(f64));
    if let Some(d) = default {
        a = a.default_value(d);
    }
    a
}

fn req<T: Clone + Send + Sync + 'static>(
    args: &ArgMatches,
    name: &str,
) -> Result<T, PipelineError> {
    args.get_one::<T>(name)
        .cloned()
        .ok_or_else(|| PipelineError::Invalid(format!("missing --{name}")))
}

pub(crate) fn commands() -> [Command; 2] {
    [
        Command::new("corridor-build")
            .about("Build and save an NVKTILE1 3D corridor landmark bank")
            .arg(path_arg("catalog", true))
            .arg(path_arg("profile", false))
            .arg(path_arg("model", true))
            .arg(path_arg("output", true))
            .arg(float_arg("tile-size-m", Some("600.0")))
            .arg(float_arg("radius-m", Some("1500.0")))
            .arg(float_arg("nominal-agl-m", Some("480.0")))
            .arg(
                Arg::new("max-landmarks")
                    .long("max-landmarks")
                    .default_value("16384")
                    .value_parser(value_parser!(usize)),
            )
            .arg(
                Arg::new("adaptive-stride")
                    .long("adaptive-stride")
                    .default_value("4")
                    .value_parser(value_parser!(u32)),
            )
            .arg(
                Arg::new("no-obliques")
                    .long("no-obliques")
                    .action(ArgAction::SetTrue),
            ),
        Command::new("corridor-track")
            .about("Localize and track sequential frames against an NVKTILE1 corridor bank")
            .arg(path_arg("dataset", true))
            .arg(path_arg("profile", true))
            .arg(path_arg("catalog", true))
            .arg(path_arg("corridor-bank", true))
            .arg(path_arg("model", true))
            .arg(path_arg("lighterglue", false))
            .arg(path_arg("seg-model", false))
            .arg(
                Arg::new("mask-policy")
                    .long("mask-policy")
                    .default_value("airframe-occlusions"),
            )
            .arg(
                Arg::new("mask-interval")
                    .long("mask-interval")
                    .default_value("10")
                    .value_parser(value_parser!(usize)),
            )
            .arg(path_arg("output", true))
            .arg(
                Arg::new("anchor-interval")
                    .long("anchor-interval")
                    .default_value("30")
                    .value_parser(value_parser!(usize)),
            )
            .arg(
                Arg::new("min-track-inliers")
                    .long("min-track-inliers")
                    .default_value("32")
                    .value_parser(value_parser!(usize)),
            )
            .arg(float_arg("map-cell-m", Some("500.0")))
            .arg(
                Arg::new("attempts")
                    .long("attempts")
                    .default_value("12")
                    .value_parser(value_parser!(usize)),
            )
            .arg(float_arg("heading-step", Some("0.0")))
            .arg(
                Arg::new("height-factors")
                    .long("height-factors")
                    .default_value("1.0"),
            )
            .arg(
                Arg::new("query-keypoints")
                    .long("query-keypoints")
                    .default_value("2048")
                    .value_parser(value_parser!(usize)),
            )
            .arg(Arg::new("polish").long("polish").action(ArgAction::SetTrue))
            .arg(float_arg("prior-lat", None))
            .arg(float_arg("prior-lon", None))
            .arg(float_arg("prior-alt", None))
            .arg(float_arg("prior-heading", None))
            .arg(float_arg("focal-35mm", None))
            .arg(Arg::new("only").long("only").action(ArgAction::Append)),
    ]
}

pub(crate) fn dispatch(sub: &str, args: &ArgMatches) -> Result<Value, PipelineError> {
    match sub {
        "corridor-build" => build_corridor_bank(&CorridorBuildOptions {
            catalog_path: req(args, "catalog")?,
            profile_path: args.get_one::<PathBuf>("profile").cloned(),
            xfeat_weights_path: req(args, "model")?,
            output_bank_path: req(args, "output")?,
            tile_size_m: req(args, "tile-size-m")?,
            corridor_radius_m: req(args, "radius-m")?,
            nominal_agl_m: req(args, "nominal-agl-m")?,
            max_landmarks_per_tile: req(args, "max-landmarks")?,
            adaptive_stride: req(args, "adaptive-stride")?,
            include_obliques: !args.get_flag("no-obliques"),
        }),
        "corridor-track" => {
            let only_cases = args
                .try_get_many::<String>("only")
                .ok()
                .flatten()
                .map(|it| it.cloned().collect::<BTreeSet<_>>());
            let hf_str: String = req(args, "height-factors")?;
            let height_factors: Vec<f64> = hf_str
                .split(',')
                .filter_map(|s| s.trim().parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v > 0.1)
                .collect();
            let prior_override = match (
                args.get_one::<f64>("prior-lat").copied(),
                args.get_one::<f64>("prior-lon").copied(),
            ) {
                (Some(lat), Some(lon)) => {
                    let alt = args.get_one::<f64>("prior-alt").copied().unwrap_or(950.0);
                    let hdg = args.get_one::<f64>("prior-heading").copied();
                    Some(([lat, lon, alt], hdg))
                }
                _ => None,
            };
            run_corridor_track(&CorridorTrackOptions {
                dataset_dir: req(args, "dataset")?,
                profile_path: req(args, "profile")?,
                catalog_path: req(args, "catalog")?,
                corridor_bank_path: req(args, "corridor-bank")?,
                xfeat_weights_path: req(args, "model")?,
                lighterglue_weights_path: args.get_one::<PathBuf>("lighterglue").cloned(),
                seg_model_dir: args.get_one::<PathBuf>("seg-model").cloned(),
                mask_policy: req(args, "mask-policy")?,
                mask_interval: req(args, "mask-interval")?,
                output_dir: req(args, "output")?,
                anchor_interval: req(args, "anchor-interval")?,
                min_track_inliers: req(args, "min-track-inliers")?,
                map_cell_m: req(args, "map-cell-m")?,
                anchor_attempts: req(args, "attempts")?,
                heading_step_deg: req(args, "heading-step")?,
                height_factors,
                query_keypoints: req(args, "query-keypoints")?,
                polish_anchors: args.get_flag("polish"),
                prior_override,
                focal_35mm_override: args.get_one::<f64>("focal-35mm").copied(),
                only_cases,
            })
        }
        other => Err(PipelineError::Invalid(format!(
            "unknown corridor subcommand `{other}`"
        ))),
    }
}

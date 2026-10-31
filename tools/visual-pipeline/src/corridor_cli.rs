//! CLI command definitions and dispatch for `corridor-build` and `corridor-track`.

use clap::{Arg, ArgAction, ArgMatches, Command, value_parser};
use navigate_visual_pipeline::{
    CorridorBuildOptions, CorridorTrackOptions, PipelineError, XFeatRuntime, XFeatRuntimeConfig,
    build_corridor_bank, corridor::VehicleDynamicsProfile, run_corridor_track,
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

fn xfeat_runtime_args(cmd: Command) -> Command {
    cmd.arg(
        Arg::new("xfeat-runtime")
            .long("xfeat-runtime")
            .default_value("burn-f32"),
    )
    .arg(path_arg("xfeat-onnx", false))
    .arg(path_arg("onnx-runtime", false))
}

pub(crate) fn parse_xfeat_runtime(args: &ArgMatches) -> Result<XFeatRuntimeConfig, PipelineError> {
    let raw = args
        .get_one::<String>("xfeat-runtime")
        .map_or("burn-f32", String::as_str);
    Ok(XFeatRuntimeConfig {
        runtime: XFeatRuntime::parse(raw)?,
        onnx_model_path: args.get_one::<PathBuf>("xfeat-onnx").cloned(),
        onnx_runtime_path: args.get_one::<PathBuf>("onnx-runtime").cloned(),
    })
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
        xfeat_runtime_args(
            Command::new("corridor-build")
                .about("Build and save an NVKTILE1 file or sharded NVKTIDX1 3D corridor tile store")
                .arg(path_arg("catalog", true))
                .arg(path_arg("profile", false))
                .arg(path_arg("model", true))
                .arg(path_arg("output", true))
                .arg(float_arg("tile-size-m", Some("600.0")))
                .arg(float_arg("radius-m", Some("3500.0")))
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
        ),
        xfeat_runtime_args(
            Command::new("corridor-track")
                .about("Localize and track sequential frames against a corridor bank or sharded tile store")
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
                .arg(float_arg("mask-interval-s", None))
                .arg(path_arg("output", true))
                .arg(
                    Arg::new("anchor-interval")
                        .long("anchor-interval")
                        .default_value("30")
                        .value_parser(value_parser!(usize)),
                )
                .arg(float_arg("anchor-interval-s", None))
                .arg(float_arg("anchor-uncertainty-m", None))
                .arg(
                    Arg::new("anchor-on-cell-boundary")
                        .long("anchor-on-cell-boundary")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("dynamics-profile")
                        .long("dynamics-profile")
                        .default_value("rotary-survey"),
                )
                .arg(float_arg("max-speed-mps", None))
                .arg(float_arg("max-vertical-speed-mps", None))
                .arg(float_arg("rms-accel-mps2", None))
                .arg(float_arg("max-angular-rate-rad-s", None))
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
                .arg(
                    Arg::new("background-refresh")
                        .long("background-refresh")
                        .help("Keep background refresh enabled (default behavior)")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("sync-refresh")
                        .long("sync-refresh")
                        .help("Run Segformer refreshes and map anchors synchronously inside the frame loop")
                        .action(ArgAction::SetTrue),
                )
                .arg(float_arg("prior-lat", None))
                .arg(float_arg("prior-lon", None))
                .arg(float_arg("prior-alt", None))
                .arg(float_arg("prior-heading", None))
                .arg(float_arg("focal-35mm", None))
                .arg(Arg::new("only").long("only").action(ArgAction::Append)),
        ),
    ]
}

pub(crate) fn dispatch(sub: &str, args: &ArgMatches) -> Result<Value, PipelineError> {
    let xfeat_runtime = parse_xfeat_runtime(args)?;
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
            xfeat_runtime,
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
            let background_refresh = !args.get_flag("sync-refresh");
            let profile_name = args
                .get_one::<String>("dynamics-profile")
                .map_or("rotary-survey", String::as_str);
            let mut dynamics_profile = VehicleDynamicsProfile::from_name(profile_name)
                .unwrap_or_else(VehicleDynamicsProfile::rotary_survey);
            if let Some(v) = args.get_one::<f64>("max-speed-mps").copied() {
                dynamics_profile.max_speed_mps = v;
            }
            if let Some(v) = args.get_one::<f64>("max-vertical-speed-mps").copied() {
                dynamics_profile.max_vertical_speed_mps = v;
            }
            if let Some(v) = args.get_one::<f64>("rms-accel-mps2").copied() {
                dynamics_profile.rms_accel_mps2 = v;
            }
            if let Some(v) = args.get_one::<f64>("max-angular-rate-rad-s").copied() {
                dynamics_profile.max_angular_rate_rad_s = v;
            }
            run_corridor_track(&CorridorTrackOptions {
                dataset_dir: req(args, "dataset")?,
                profile_path: req(args, "profile")?,
                catalog_path: req(args, "catalog")?,
                corridor_bank_path: req(args, "corridor-bank")?,
                xfeat_weights_path: req(args, "model")?,
                lighterglue_weights_path: args.get_one::<PathBuf>("lighterglue").cloned(),
                seg_model_dir: args.get_one::<PathBuf>("seg-model").cloned(),
                dynamics_profile,
                mask_policy: req(args, "mask-policy")?,
                mask_interval: req(args, "mask-interval")?,
                mask_interval_s: args.get_one::<f64>("mask-interval-s").copied(),
                output_dir: req(args, "output")?,
                anchor_interval: req(args, "anchor-interval")?,
                anchor_interval_s: args.get_one::<f64>("anchor-interval-s").copied(),
                anchor_uncertainty_m: args.get_one::<f64>("anchor-uncertainty-m").copied(),
                anchor_on_cell_boundary: args.get_flag("anchor-on-cell-boundary"),
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
                background_refresh,
                xfeat_runtime,
            })
        }
        other => Err(PipelineError::Invalid(format!(
            "unknown corridor subcommand `{other}`"
        ))),
    }
}

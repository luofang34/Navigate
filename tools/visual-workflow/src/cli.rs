//! Parse only explicit paths and bounded execution counts.
use clap::{Arg, Command, value_parser};
use std::path::PathBuf;

pub fn command() -> Command {
    Command::new("navigate-workflow")
        .arg(gpu_timing_argument())
        .subcommand_required(true)
        .subcommand(
            Command::new("prepare-surface")
                .about("Prepare image, calibration, and surface arrays without query pose truth")
                .arg(path("input"))
                .arg(path("output")),
        )
        .subcommand(
            Command::new("locate-surface")
                .about("Localize against an orthographic surface in local metric coordinates")
                .arg(path("config"))
                .arg(path("job"))
                .arg(path("output"))
                .arg(
                    Arg::new("iterations")
                        .long("iterations")
                        .default_value("1")
                        .value_parser(value_parser!(usize)),
                ),
        )
        .subcommand(
            Command::new("refine-scene")
                .about("Refine conditional poses and points with per-camera calibrations")
                .arg(path("input"))
                .arg(path("output")),
        )
        .subcommand(
            Command::new("evaluate")
                .about("Compare local pipelines with fixed spatial holdouts")
                .arg(path("config"))
                .arg(path("manifest"))
                .arg(path("output")),
        )
        .subcommand(
            Command::new("replay")
                .about("Process ordered job records while retaining models and bounded references")
                .arg(path("config"))
                .arg(path("manifest"))
                .arg(path("output")),
        )
        .subcommands(surface_evaluation_commands())
        .subcommand(acquisition_command())
        .subcommand(prepare_features_command())
        .subcommand(acquire_surface_command())
        .subcommand(acquisition_grid_command())
        .subcommand(acquisition_pyramid_command())
        .subcommand(window_command("probe-surface-windows"))
        .subcommand(window_command("probe-surface-scores"))
        .subcommand(cascade_command())
        .subcommand(motion_command())
        .subcommand(render_command())
        .subcommand(rectify_command())
        .subcommand(
            Command::new("pack")
                .about("Build one offline reference feature packet")
                .arg(path("config"))
                .arg(path("image"))
                .arg(path("output")),
        )
        .subcommand(
            Command::new("locate")
                .about("Localize one query against bounded references")
                .arg(path("config"))
                .arg(path("job"))
                .arg(path("output"))
                .arg(
                    Arg::new("iterations")
                        .long("iterations")
                        .default_value("1")
                        .value_parser(value_parser!(usize)),
                ),
        )
}
fn path(name: &'static str) -> Arg {
    Arg::new(name)
        .required(true)
        .value_parser(value_parser!(PathBuf))
}
fn render_command() -> Command {
    Command::new("render")
        .about("Render terrain references and save a reusable localization job")
        .arg(path("job"))
        .arg(path("output"))
        .arg(
            Arg::new("iterations")
                .long("iterations")
                .default_value("1")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("pixels")
                .long("pixels")
                .default_value("2000000")
                .value_parser(value_parser!(usize)),
        )
}
#[cfg(test)]
mod tests;

fn cascade_command() -> Command {
    Command::new("cascade-surface")
        .about("Test sparse-first matching with ranked pair-matcher fallback")
        .arg(path("config"))
        .arg(path("fallback"))
        .arg(path("job"))
        .arg(path("output"))
        .arg(
            Arg::new("iterations")
                .long("iterations")
                .default_value("1")
                .value_parser(value_parser!(usize)),
        )
}

fn motion_command() -> Command {
    Command::new("probe-motion")
        .about("Check frozen poses against image-only temporal correspondences")
        .arg(path("manifest"))
        .arg(path("output"))
}

fn window_command(name: &'static str) -> Command {
    Command::new(name)
        .about("Compare independent crops and shared map-block features on fixed windows")
        .arg(path("config"))
        .arg(path("job"))
        .arg(path("output"))
        .arg(
            Arg::new("iterations")
                .long("iterations")
                .default_value("1")
                .value_parser(value_parser!(usize)),
        )
}

fn acquisition_command() -> Command {
    Command::new("evaluate-acquisition")
        .arg(
            Arg::new("retained-window-recovery")
                .long("retained-window-recovery")
                .help("Check the previous map window before full recovery; needs 80 candidates")
                .action(clap::ArgAction::SetTrue),
        )
        .about("Evaluate bounded search, tracking, and frame arrival policies")
        .arg(
            Arg::new("gpu-image-reuse")
                .long("gpu-image-reuse")
                .help("Retain the previous query GPU image for exact-reference reuse")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("reference-feature-catalog")
                .long("reference-feature-catalog")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("reference-cache-mib")
                .long("reference-cache-mib")
                .default_value("0")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("reload-model-on-recovery")
                .long("reload-model-on-recovery")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("arrival-policy")
                .long("arrival-policy")
                .default_value("offline")
                .value_parser(["offline", "fifo", "latest", "catch-up"]),
        )
        .arg(
            Arg::new("queue-capacity")
                .long("queue-capacity")
                .default_value("64")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("capture-start")
                .long("capture-start")
                .default_value("ready")
                .value_parser(["ready", "setup"]),
        )
        .arg(path("config"))
        .arg(path("manifest"))
        .arg(path("output"))
}
pub fn arrival_options(
    args: &clap::ArgMatches,
) -> Result<crate::surface::journey::arrival::Options, crate::Error> {
    crate::surface::journey::arrival::Options::new(
        args.get_one::<String>("arrival-policy")
            .map_or("offline", String::as_str),
        args.get_one::<usize>("queue-capacity")
            .copied()
            .unwrap_or(64),
        args.get_one::<String>("capture-start")
            .map_or("ready", String::as_str),
    )
}

fn surface_evaluation_commands() -> [Command; 6] {
    [
        Command::new("probe-surface-accumulation")
            .about("Propagate bounded frozen map pairs before pose initialization")
            .arg(path("config"))
            .arg(path("manifest"))
            .arg(path("output"))
            .arg(
                Arg::new("history")
                    .long("history")
                    .default_value("5")
                    .value_parser(value_parser!(usize)),
            ),
        Command::new("probe-surface-holdout")
            .about("Replay frozen acquisition and test five fixed spatial holdouts")
            .arg(path("config"))
            .arg(path("manifest"))
            .arg(path("output")),
        Command::new("probe-surface-appearance")
            .about("Compare frozen poses on shared image patches outside fitted map points")
            .arg(path("config"))
            .arg(path("manifest"))
            .arg(path("output")),
        Command::new("probe-hypotheses")
            .about("Track up to four accepted map poses without truth ranking or recovery")
            .arg(path("config"))
            .arg(path("manifest"))
            .arg(path("output"))
            .arg(
                Arg::new("hypotheses")
                    .long("hypotheses")
                    .default_value("4")
                    .value_parser(value_parser!(usize)),
            ),
        Command::new("probe-surface")
            .about("Recheck frozen image matches with an alternative registered height surface")
            .arg(path("config"))
            .arg(path("manifest"))
            .arg(path("output")),
        Command::new("evaluate-surface")
            .about("Compare orthographic map matching with fixed-map temporal tracking")
            .arg(path("config"))
            .arg(path("manifest"))
            .arg(path("output")),
    ]
}

fn prepare_features_command() -> Command {
    Command::new("prepare-acquisition")
        .about("Prepare query-independent coarse and local map feature packets")
        .arg(
            Arg::new("pyramid-side")
                .long("pyramid-side")
                .requires("pyramid-levels")
                .conflicts_with("include-fine")
                .value_parser(value_parser!(u32)),
        )
        .arg(
            Arg::new("pyramid-levels")
                .long("pyramid-levels")
                .requires("pyramid-side")
                .value_parser(value_parser!(usize)),
        )
        .arg(path("config"))
        .arg(path("surface"))
        .arg(path("output"))
        .arg(
            Arg::new("include-fine")
                .long("include-fine")
                .action(clap::ArgAction::SetTrue),
        )
}

fn acquire_surface_command() -> Command {
    Command::new("acquire-surface")
        .arg(
            Arg::new("rectification")
                .long("rectification")
                .value_parser(value_parser!(PathBuf))
                .help("Bind raw query calibration and require raw pixel geometry checks"),
        )
        .about("Search coarse and adjacent map windows with one query encoding")
        .arg(path("config"))
        .arg(path("job"))
        .arg(path("output"))
        .arg(
            Arg::new("probe-fine-after-pass")
                .long("probe-fine-after-pass")
                .action(clap::ArgAction::SetTrue),
        )
}

fn gpu_timing_argument() -> Arg {
    Arg::new("gpu-timings")
        .long("gpu-timings")
        .global(true)
        .help("Log host GPU stages and logical buffer bytes; changes timing overhead")
        .action(clap::ArgAction::SetTrue)
}

fn rectify_command() -> Command {
    Command::new("rectify-image")
        .about("Prepare a calibrated pinhole image with bounded reusable pixel mapping")
        .arg(path("specification"))
        .arg(path("output"))
        .arg(
            Arg::new("iterations")
                .long("iterations")
                .default_value("1")
                .value_parser(value_parser!(usize)),
        )
}

fn acquisition_grid_command() -> Command {
    Command::new("probe-acquisition-grid")
        .about("Check every fixed overlapping map window with shared query features")
        .arg(path("config"))
        .arg(path("job"))
        .arg(path("output"))
        .arg(
            Arg::new("side")
                .long("side")
                .required(true)
                .value_parser(value_parser!(u32)),
        )
        .arg(
            Arg::new("rectification")
                .long("rectification")
                .value_parser(value_parser!(PathBuf)),
        )
}

fn acquisition_pyramid_command() -> Command {
    Command::new("probe-acquisition-pyramid")
        .about("Retain multiple map regions and refine windows with shared query features")
        .arg(
            Arg::new("reference-feature-catalog")
                .long("reference-feature-catalog")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(path("config"))
        .arg(path("job"))
        .arg(path("output"))
        .arg(
            Arg::new("side")
                .long("side")
                .required(true)
                .value_parser(value_parser!(u32)),
        )
        .arg(
            Arg::new("levels")
                .long("levels")
                .required(true)
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("beam")
                .long("beam")
                .required(true)
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("rectification")
                .long("rectification")
                .value_parser(value_parser!(PathBuf)),
        )
}

pub fn feature_coverage(
    args: &clap::ArgMatches,
) -> Result<crate::surface::acquire::Coverage, crate::Error> {
    use crate::surface::acquire::Coverage;
    if let Some(&side) = args.get_one::<u32>("pyramid-side") {
        let levels = args
            .get_one::<usize>("pyramid-levels")
            .copied()
            .ok_or_else(|| crate::Error::Input("missing pyramid levels".into()))?;
        Ok(Coverage::Pyramid { side, levels })
    } else {
        Ok(Coverage::Acquisition {
            include_fine: args.get_flag("include-fine"),
        })
    }
}

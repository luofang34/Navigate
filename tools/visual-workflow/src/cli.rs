//! Parse only explicit paths and bounded execution counts.
use clap::{Arg, Command, value_parser};
use std::path::PathBuf;

pub fn command() -> Command {
    Command::new("navigate-workflow")
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
        .subcommand(
            Command::new("acquire-surface")
                .about("Search coarse and adjacent map windows with one query encoding")
                .arg(path("config"))
                .arg(path("job"))
                .arg(path("output")),
        )
        .subcommand(window_command("probe-surface-windows"))
        .subcommand(window_command("probe-surface-scores"))
        .subcommand(cascade_command())
        .subcommand(motion_command())
        .subcommand(render_command())
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
        .about("Evaluate bounded search, tracking, and frame arrival policies")
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

fn surface_evaluation_commands() -> [Command; 2] {
    [
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

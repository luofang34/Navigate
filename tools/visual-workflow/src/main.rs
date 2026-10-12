//! Native offline feature packages and bounded visual localization.
mod anchoring;
mod assets;
mod camera_input;
mod cli;
mod config;
mod error;
mod evaluate;
mod job;
mod motion_probe;
mod pair;
mod render;
mod replay;
mod run;
mod scene;
mod surface;
mod terrain;

use error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Error> {
    let matches = match cli::command().try_get_matches() {
        Ok(matches) => matches,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            return error.print().map_err(|source| Error::Io {
                operation: "write CLI help",
                path: "stdout".into(),
                source,
            });
        }
        Err(error) => return Err(error.into()),
    };
    initialize_logging(matches.get_flag("gpu-timings"));
    dispatch_blocking(matches)
}
fn dispatch_blocking(matches: clap::ArgMatches) -> Result<(), Error> {
    let (name, args) = matches
        .subcommand()
        .ok_or_else(|| Error::Input("missing command".into()))?;
    let path = |name: &str| path_argument(args, name);
    if name == "probe-motion" {
        #[cfg(feature = "gpu")]
        return motion_probe::run_blocking(&path("manifest")?, &path("output")?);
        #[cfg(not(feature = "gpu"))]
        return Err(Error::Input(
            "motion probe requires the gpu build feature".into(),
        ));
    }
    if name == "rectify-image" {
        return camera_input::run_blocking(
            &path("specification")?,
            &path("output")?,
            count(args, "iterations")?,
        );
    }
    if name == "prepare-surface" {
        return surface::prepare::run_blocking(&path("input")?, &path("output")?);
    }
    if name == "refine-scene" {
        return scene::run_blocking(&path("input")?, &path("output")?);
    }
    if name == "render" {
        return render::run_blocking(
            &path("job")?,
            &path("output")?,
            count(args, "iterations")?,
            count(args, "pixels")?,
        );
    }
    let config_path = path("config")?;
    let config: config::Config = assets::read_json_blocking(&config_path)?;
    let root = config_path
        .parent()
        .ok_or_else(|| Error::Input("config path has no parent".into()))?;
    if name == "cascade-surface" {
        return surface::cascade::run_blocking(
            &config_path,
            &path("fallback")?,
            &path("job")?,
            &path("output")?,
            count(args, "iterations")?,
        );
    }
    configured_blocking(name, args, &config, root)
}
fn configured_blocking(
    name: &str,
    args: &clap::ArgMatches,
    config: &config::Config,
    root: &std::path::Path,
) -> Result<(), Error> {
    let path = |name: &str| path_argument(args, name);
    match name {
        "prepare-acquisition" => surface::acquire::prepare_features_blocking(
            config,
            root,
            &path("surface")?,
            &path("output")?,
            args.get_flag("include-fine"),
        ),
        "evaluate-acquisition" => surface::journey::run_blocking(
            config,
            root,
            &path("manifest")?,
            &path("output")?,
            acquisition_options(args)?,
        ),
        "probe-acquisition-pyramid" => pyramid_blocking(args, config, root),
        "probe-acquisition-grid" => surface::acquire::grid::run_blocking(
            config,
            root,
            &path("job")?,
            &path("output")?,
            args.get_one::<u32>("side")
                .copied()
                .ok_or_else(|| Error::Input("missing grid side".into()))?,
            args.get_one::<PathBuf>("rectification")
                .map(PathBuf::as_path),
        ),
        "acquire-surface" => surface::acquire::run_blocking(
            config,
            root,
            &path("job")?,
            &path("output")?,
            args.get_flag("probe-fine-after-pass"),
            args.get_one::<PathBuf>("rectification")
                .map(PathBuf::as_path),
        ),
        "probe-surface-accumulation" => surface::accumulation::run_blocking(
            config,
            &path("manifest")?,
            &path("output")?,
            count(args, "history")?,
        ),
        "probe-surface-holdout" => {
            surface::holdout::run_blocking(config, &path("manifest")?, &path("output")?)
        }
        "probe-surface-appearance" => {
            surface::appearance::run_blocking(config, &path("manifest")?, &path("output")?)
        }
        "probe-hypotheses" => surface::hypotheses::run_blocking(
            config,
            root,
            &path("manifest")?,
            &path("output")?,
            count(args, "hypotheses")?,
        ),
        "probe-surface" => {
            surface::probe::run_blocking(config, &path("manifest")?, &path("output")?)
        }
        "evaluate-surface" => {
            surface::sequence::run_blocking(config, root, &path("manifest")?, &path("output")?)
        }
        "evaluate" => evaluate::run_blocking(config, root, &path("manifest")?, &path("output")?),
        "replay" => replay::run_blocking(config, root, &path("manifest")?, &path("output")?),
        "pack" => run::pack_blocking(config, root, &path("image")?, &path("output")?),
        "locate" | "locate-surface" | "probe-surface-windows" | "probe-surface-scores" => {
            locate_blocking(name, args, config, root)
        }

        _ => Err(Error::Input(format!("unknown command {name}"))),
    }
}

fn path_argument(args: &clap::ArgMatches, name: &str) -> Result<PathBuf, Error> {
    args.get_one::<PathBuf>(name)
        .cloned()
        .ok_or_else(|| Error::Input(format!("missing {name}")))
}

fn count(args: &clap::ArgMatches, name: &str) -> Result<usize, Error> {
    args.get_one::<usize>(name)
        .copied()
        .ok_or_else(|| Error::Input(format!("missing {name}")))
}

fn acquisition_options(args: &clap::ArgMatches) -> Result<surface::journey::Options, Error> {
    Ok(surface::journey::Options {
        reload_model_on_recovery: args.get_flag("reload-model-on-recovery"),
        reference_cache_mib: count(args, "reference-cache-mib")?,
        gpu_image_reuse: args.get_flag("gpu-image-reuse"),
        arrival: cli::arrival_options(args)?,
        feature_catalog: args
            .get_one::<PathBuf>("reference-feature-catalog")
            .cloned(),
    })
}

fn initialize_logging(gpu_timings: bool) {
    if !gpu_timings {
        tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .init();
        return;
    }
    use tracing_subscriber::prelude::*;
    let filter = tracing_subscriber::filter::filter_fn(move |metadata| {
        *metadata.level() <= tracing::Level::INFO
            || (gpu_timings && metadata.target() == "navigate_visual::gpu::timings")
    })
    .with_max_level_hint(tracing_subscriber::filter::LevelFilter::DEBUG);
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_filter(filter),
        )
        .init();
}

fn locate_blocking(
    name: &str,
    args: &clap::ArgMatches,
    config: &config::Config,
    root: &std::path::Path,
) -> Result<(), Error> {
    let path = |name: &str| path_argument(args, name);
    (if name == "probe-surface-scores" {
        surface::windows::run_scored_blocking
    } else if name == "probe-surface-windows" {
        surface::windows::run_blocking
    } else if name == "locate" {
        run::locate_blocking
    } else {
        surface::run_blocking
    })(
        config,
        root,
        &path("job")?,
        &path("output")?,
        count(args, "iterations")?,
    )
}

fn pyramid_blocking(
    args: &clap::ArgMatches,
    config: &config::Config,
    root: &std::path::Path,
) -> Result<(), Error> {
    surface::acquire::pyramid::run_blocking(
        config,
        root,
        &path_argument(args, "job")?,
        &path_argument(args, "output")?,
        surface::acquire::pyramid::Options {
            side: args
                .get_one::<u32>("side")
                .copied()
                .ok_or_else(|| Error::Input("missing pyramid side".into()))?,
            levels: count(args, "levels")?,
            beam: count(args, "beam")?,
        },
        args.get_one::<PathBuf>("rectification")
            .map(PathBuf::as_path),
    )
}

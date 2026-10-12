//! Native offline feature packages and bounded visual localization.
mod anchoring;
mod assets;
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
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
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
    dispatch_blocking(matches)
}
fn dispatch_blocking(matches: clap::ArgMatches) -> Result<(), Error> {
    let (name, args) = matches
        .subcommand()
        .ok_or_else(|| Error::Input("missing command".into()))?;
    let path = |name: &str| {
        args.get_one::<PathBuf>(name)
            .cloned()
            .ok_or_else(|| Error::Input(format!("missing {name}")))
    };
    if name == "probe-motion" {
        #[cfg(feature = "gpu")]
        return motion_probe::run_blocking(&path("manifest")?, &path("output")?);
        #[cfg(not(feature = "gpu"))]
        return Err(Error::Input(
            "motion probe requires the gpu build feature".into(),
        ));
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
    match name {
        "evaluate-acquisition" => surface::journey::run_blocking(
            &config,
            root,
            &path("manifest")?,
            &path("output")?,
            args.get_flag("reload-model-on-recovery"),
            count(args, "reference-cache-mib")?,
        ),
        "acquire-surface" => {
            surface::acquire::run_blocking(&config, root, &path("job")?, &path("output")?)
        }
        "probe-surface" => {
            surface::probe::run_blocking(&config, &path("manifest")?, &path("output")?)
        }
        "evaluate-surface" => {
            surface::sequence::run_blocking(&config, root, &path("manifest")?, &path("output")?)
        }
        "evaluate" => evaluate::run_blocking(&config, root, &path("manifest")?, &path("output")?),
        "replay" => replay::run_blocking(&config, root, &path("manifest")?, &path("output")?),
        "pack" => run::pack_blocking(&config, root, &path("image")?, &path("output")?),
        "locate" | "locate-surface" | "probe-surface-windows" | "probe-surface-scores" => {
            (if name == "probe-surface-scores" {
                surface::windows::run_scored_blocking
            } else if name == "probe-surface-windows" {
                surface::windows::run_blocking
            } else if name == "locate" {
                run::locate_blocking
            } else {
                surface::run_blocking
            })(
                &config,
                root,
                &path("job")?,
                &path("output")?,
                count(args, "iterations")?,
            )
        }
        _ => Err(Error::Input(format!("unknown command {name}"))),
    }
}

fn count(args: &clap::ArgMatches, name: &str) -> Result<usize, Error> {
    args.get_one::<usize>(name)
        .copied()
        .ok_or_else(|| Error::Input(format!("missing {name}")))
}

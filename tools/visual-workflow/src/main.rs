//! Native offline feature packages and bounded visual localization.
mod assets;
mod cli;
mod config;
mod error;
mod evaluate;
mod job;
mod pair;
mod render;
mod replay;
mod run;
mod scene;
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
    let (name, args) = matches
        .subcommand()
        .ok_or_else(|| Error::Input("missing command".into()))?;
    let path = |name: &str| {
        args.get_one::<PathBuf>(name)
            .cloned()
            .ok_or_else(|| Error::Input(format!("missing {name}")))
    };
    if name == "refine-scene" {
        return scene::run_blocking(&path("input")?, &path("output")?);
    }
    if name == "render" {
        return render::run_blocking(
            &path("job")?,
            &path("output")?,
            *args
                .get_one::<usize>("iterations")
                .ok_or_else(|| Error::Input("missing iterations".into()))?,
            *args
                .get_one::<usize>("pixels")
                .ok_or_else(|| Error::Input("missing pixels".into()))?,
        );
    }
    let config_path = path("config")?;
    let config: config::Config = assets::read_json_blocking(&config_path)?;
    let root = config_path
        .parent()
        .ok_or_else(|| Error::Input("config path has no parent".into()))?;
    match name {
        "evaluate" => evaluate::run_blocking(&config, root, &path("manifest")?, &path("output")?),
        "replay" => replay::run_blocking(&config, root, &path("manifest")?, &path("output")?),
        "pack" => run::pack_blocking(&config, root, &path("image")?, &path("output")?),
        "locate" => run::locate_blocking(
            &config,
            root,
            &path("job")?,
            &path("output")?,
            *args
                .get_one::<usize>("iterations")
                .ok_or_else(|| Error::Input("missing iterations".into()))?,
        ),
        _ => Err(Error::Input(format!("unknown command {name}"))),
    }
}

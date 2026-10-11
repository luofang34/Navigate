//! Native offline feature packages and bounded visual localization.
mod assets;
mod cli;
mod config;
mod error;
mod job;
mod pair;
mod run;

use error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let matches = cli::command().try_get_matches()?;
    let (name, args) = matches
        .subcommand()
        .ok_or_else(|| Error::Input("missing command".into()))?;
    let path = |name: &str| {
        args.get_one::<PathBuf>(name)
            .cloned()
            .ok_or_else(|| Error::Input(format!("missing {name}")))
    };
    let config_path = path("config")?;
    let config: config::Config = assets::read_json_blocking(&config_path)?;
    let root = config_path
        .parent()
        .ok_or_else(|| Error::Input("config path has no parent".into()))?;
    match name {
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

//! Offline, hash-bound graph preparation without an inference runtime.
mod error;
mod graph;
mod loftr;
mod masked;
mod recipe;
mod xfeat;
use clap::{Arg, Command, value_parser};
use error::Error;
use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let args = match command().try_get_matches() {
        Ok(args) => args,
        Err(error) if error.kind() == clap::error::ErrorKind::DisplayHelp => {
            return error.print().map_err(|source| Error::Io {
                path: "stdout".into(),
                source,
            });
        }
        Err(error) => return Err(error.into()),
    };
    let input = args
        .get_one::<PathBuf>("input")
        .ok_or(Error::Invalid("input path"))?;
    let output = args
        .get_one::<PathBuf>("output")
        .ok_or(Error::Invalid("output path"))?;
    let count = args.get_one::<usize>("features").copied();
    let batch = args.get_one::<usize>("batch").copied();
    let recipe = args
        .get_one::<String>("recipe")
        .map(String::as_str)
        .unwrap_or("lighterglue");
    let compact = args
        .get_one::<String>("output-kind")
        .is_some_and(|s| s == "compact");
    prepare_blocking(input, output, count, compact, recipe, batch)
}
fn command() -> Command {
    Command::new("navigate-model")
        .about("Prepare supported feature extraction and matching graphs")
        .arg(Arg::new("recipe").long("recipe").value_parser([
            "lighterglue",
            "lighterglue-masked",
            "xfeat-dense",
            "xfeat-gray",
            "xfeat-gray-2x",
            "loftr-coarse",
            "loftr-fine",
        ]))
        .arg(
            Arg::new("input")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("features")
                .long("features")
                .required_if_eq("recipe", "lighterglue")
                .required_if_eq("recipe", "lighterglue-masked")
                .required_unless_present("recipe")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("batch")
                .long("batch")
                .required_if_eq("recipe", "loftr-fine")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("output-kind")
                .long("output-kind")
                .default_value("compact")
                .value_parser(["compact", "probability"]),
        )
}
fn prepare_blocking(
    input: &Path,
    output: &Path,
    count: Option<usize>,
    compact: bool,
    recipe: &str,
    batch: Option<usize>,
) -> Result<(), Error> {
    let mut bytes = Vec::new();
    File::open(input)
        .and_then(|f| f.take(64 * 1024 * 1024 + 1).read_to_end(&mut bytes))
        .map_err(|source| Error::Io {
            path: input.into(),
            source,
        })?;
    if batch.is_some() && recipe != "loftr-fine" {
        return Err(Error::Invalid("batch requires loftr-fine"));
    }
    let prepared = if recipe.starts_with("loftr-") {
        if count.is_some() || !compact {
            return Err(Error::Invalid(
                "LoFTR does not accept feature budgets or probability outputs",
            ));
        }
        loftr::prepare(&bytes, batch)?
    } else if recipe == "lighterglue-masked" {
        masked::prepare(
            &bytes,
            count.ok_or(Error::Invalid("feature count"))?,
            compact,
        )?
    } else if recipe != "lighterglue" {
        if count.is_some() || !compact {
            return Err(Error::Invalid(
                "XFeat does not accept features or probability outputs",
            ));
        }
        if recipe == "xfeat-gray-2x" {
            xfeat::prepare_scaled(&bytes, 1, true)?
        } else {
            xfeat::prepare_channels(&bytes, if recipe == "xfeat-gray" { 1 } else { 3 })?
        }
    } else {
        recipe::prepare(
            &bytes,
            count.ok_or(Error::Invalid("feature count"))?,
            compact,
        )?
    };
    File::options()
        .write(true)
        .create_new(true)
        .open(output)
        .and_then(|mut f| f.write_all(&prepared).and_then(|()| f.sync_all()))
        .map_err(|source| Error::Io {
            path: output.into(),
            source,
        })?;
    tracing::info!(path=%output.display(),?count,compact,recipe,bytes=prepared.len(),"Saved prepared model");
    Ok(())
}

#[cfg(test)]
mod tests;

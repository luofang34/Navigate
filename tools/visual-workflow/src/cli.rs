//! Parse only explicit paths and bounded execution counts.
use clap::{Arg, Command, value_parser};
use std::path::PathBuf;

pub fn command() -> Command {
    let path = |name: &'static str| {
        Arg::new(name)
            .required(true)
            .value_parser(value_parser!(PathBuf))
    };
    Command::new("navigate-workflow")
        .subcommand_required(true)
        .subcommand(
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
                ),
        )
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
#[cfg(test)]
mod tests;

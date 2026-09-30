use crate::error::ServiceError;
use clap::{Arg, ArgMatches, Command, value_parser};
use std::{net::SocketAddr, path::PathBuf};

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub(crate) struct Config {
    pub state: PathBuf,
    pub webapp: PathBuf,
    pub origin: String,
    pub catalog: Option<PathBuf>,
    pub bind: SocketAddr,
}

fn command() -> Command {
    Command::new("navigate-visual-service")
        .about("Serve offline coverage packages. Camera media stays in the browser.")
        .arg(
            Arg::new("state")
                .long("state")
                .env("NAVIGATE_DATA")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("webapp")
                .long("webapp")
                .default_value("tools/visual-bench/webapp")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("origin")
                .long("origin")
                .env("NAVIGATE_ORIGIN")
                .required(true),
        )
        .arg(
            Arg::new("catalog")
                .long("catalog")
                .env("NAVIGATE_CATALOG")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("bind")
                .long("bind")
                .default_value("127.0.0.1:8080")
                .value_parser(value_parser!(SocketAddr)),
        )
}

fn required<T: Clone + Send + Sync + 'static>(
    matches: &mut ArgMatches,
    id: &'static str,
) -> Result<T, ServiceError> {
    matches.remove_one(id).ok_or(ServiceError::Configuration(
        "a required argument is missing",
    ))
}

impl Config {
    /// Parses the process arguments; `--help` and invalid arguments end the process as clap does.
    pub fn parse() -> Result<Self, ServiceError> {
        Self::from_matches(command().get_matches())
    }

    fn from_matches(mut matches: ArgMatches) -> Result<Self, ServiceError> {
        Ok(Self {
            state: required(&mut matches, "state")?,
            webapp: required(&mut matches, "webapp")?,
            origin: required(&mut matches, "origin")?,
            catalog: matches.remove_one("catalog"),
            bind: required(&mut matches, "bind")?,
        })
    }

    pub fn validate(&self) -> Result<(), ServiceError> {
        let url =
            url::Url::parse(&self.origin).map_err(|source| ServiceError::Origin { source })?;
        if url.origin().ascii_serialization() != self.origin
            || !url.username().is_empty()
            || url.password().is_some()
            || (url.scheme() != "https"
                && !(url.scheme() == "http"
                    && matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"))))
        {
            return Err(ServiceError::Configuration(
                "origin must be HTTPS or HTTP on localhost",
            ));
        }
        Ok(())
    }
}

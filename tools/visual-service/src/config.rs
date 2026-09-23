use crate::error::ServiceError;
use clap::Parser;
use std::{net::SocketAddr, path::PathBuf};

#[derive(Clone, Parser)]
#[command(about = "Serve offline coverage packages. Camera media stays in the browser.")]
pub(crate) struct Config {
    #[arg(long, env = "NAVIGATE_DATA")]
    pub state: PathBuf,
    #[arg(long, default_value = "tools/visual-bench/webapp")]
    pub webapp: PathBuf,
    #[arg(long, env = "NAVIGATE_ORIGIN")]
    pub origin: String,
    #[arg(long, env = "NAVIGATE_CATALOG")]
    pub catalog: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub bind: SocketAddr,
}

impl Config {
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

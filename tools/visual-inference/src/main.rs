//! Measure native ONNX execution and record provider placement.
mod cli;
mod runtime;
use cli::{Args, Provider, command};
use thiserror::Error;

#[derive(Debug, Error)]
enum ProbeError {
    #[error("{operation}: {source}")]
    Operation {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error>,
    },
    #[error("invalid probe input: {0}")]
    Input(String),
}
fn context<E: std::error::Error + 'static>(operation: impl Into<String>, source: E) -> ProbeError {
    ProbeError::Operation {
        operation: operation.into(),
        source: Box::new(source),
    }
}
fn main() -> Result<(), ProbeError> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let matches = match command().try_get_matches() {
        Ok(matches) => matches,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            error
                .print()
                .map_err(|source| context("write command help", source))?;
            return Ok(());
        }
        Err(error) => return Err(context("parse command arguments", error)),
    };
    runtime::run_blocking(&Args::from_matches(matches)?)
}

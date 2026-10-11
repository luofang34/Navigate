use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("model file {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid model recipe input: {0}")]
    Invalid(&'static str),
    #[error("unsupported model digest {actual}; expected {expected}")]
    Digest {
        actual: String,
        expected: &'static str,
    },
    #[error("decode model protobuf: {0}")]
    Decode(#[from] prost::DecodeError),
    #[error(transparent)]
    Cli(#[from] clap::Error),
}

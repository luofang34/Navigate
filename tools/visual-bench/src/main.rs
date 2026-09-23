//! Render and evaluate visual localization with an offline map package.
//!
//! Run `visual-bench --help` for command usage. [`package`] defines the shared
//! source-data contract. [`stream`] defines streamed frame input. [`video`]
//! describes decoding and prior association. [`navigate_visual::LocalFrame`]
//! describes the coordinate model. [`track`] describes geographic track output.
//!
//! Build this tool beside the MapLibre fork with
//! `cargo build --release --manifest-path tools/visual-bench/Cargo.toml`.
//! Use `cargo doc --document-private-items` from the tool directory to read
//! its internal input contracts. The navigation library's public contracts
//! are in [`navigate_visual`]. Benchmarks produce files at run time; measured
//! results are not source documentation.

mod backend;
mod cli;
mod error;
mod fixture;
mod matches;
mod offline_pack;
mod package;
mod renderer;
mod scenario;
mod session;
mod stream;
mod track;
mod trial;
mod video;
mod worker;

/// MapLibre fork commit that renders every reference, recorded with each result.
const RENDERER_REVISION: &str = include_str!("../MAPLIBRE_REVISION");

use error::BenchError;
use std::path::Path;

fn read_blocking(path: &Path) -> Result<Vec<u8>, BenchError> {
    std::fs::read(path).map_err(|source| BenchError::Io {
        path: path.to_owned(),
        source,
    })
}

fn write_blocking(path: &Path, bytes: &[u8]) -> Result<(), BenchError> {
    std::fs::write(path, bytes).map_err(|source| BenchError::Io {
        path: path.to_owned(),
        source,
    })
}

fn directory_blocking(path: &Path) -> Result<(), BenchError> {
    std::fs::create_dir_all(path).map_err(|source| BenchError::Io {
        path: path.to_owned(),
        source,
    })
}

#[tokio::main]
async fn main() -> Result<(), BenchError> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter("warn,visual_bench=info")
        .init();
    cli::run_blocking().await
}

#[cfg(test)]
mod tests;

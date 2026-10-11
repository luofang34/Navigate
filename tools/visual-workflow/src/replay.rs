//! Ordered, bounded replay with one resident inference backend.
mod cache;
mod frames;
mod stream;

use crate::{
    Error,
    config::{Backend, Config},
};
use navigate_visual::LocalizerConfig;
use navigate_visual_workflow::FeatureLocalizer;
use std::{path::Path, time::Instant};

pub fn run_blocking(
    config: &Config,
    root: &Path,
    manifest: &Path,
    output: &Path,
) -> Result<(), Error> {
    let start = Instant::now();
    let Backend::Features(backend) = config.load_blocking(root)? else {
        return Err(Error::Input(
            "replay requires a feature-stage backend".into(),
        ));
    };
    let workflow =
        FeatureLocalizer::new(*backend, LocalizerConfig::default(), config.limits.model())?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut frames = frames::Frames::new(workflow, config);
    stream::run_blocking(manifest, output, load_ms, |record, sequence| {
        frames.process_blocking(record, sequence)
    })
}

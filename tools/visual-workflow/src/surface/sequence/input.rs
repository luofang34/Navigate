//! Time-ordered image evidence; evaluation truth is not an estimator input.
use crate::{Error, job::Image};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub job: PathBuf,
    pub frames: Vec<Sample>,
    pub refresh_interval: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Sample {
    pub image: Image,
    pub frame_id: u64,
    pub capture_time_ns: u64,
}
impl Manifest {
    pub fn validate(&self) -> Result<(), Error> {
        if self.frames.is_empty()
            || self.frames.len() > 1000
            || !(1..=1000).contains(&self.refresh_interval)
            || self.frames.windows(2).any(|p| {
                p[1].frame_id <= p[0].frame_id || p[1].capture_time_ns <= p[0].capture_time_ns
            })
        {
            return Err(Error::Input(
                "surface sequence needs ordered frame IDs/timestamps and bounded refresh interval"
                    .into(),
            ));
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests;

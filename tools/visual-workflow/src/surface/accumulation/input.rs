use super::pool::Evidence;
use crate::{Error, assets, config::Config, job::Image, surface::job::SurfaceJob};
use navigate_visual::{Frame, FrameStamp, PixelMatch};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
pub(super) struct Frozen {
    pub backend: String,
    pub job: SurfaceJob,
    pub frames: Vec<Sample>,
}
#[derive(Deserialize)]
pub(super) struct Sample {
    pub image: Image,
    pub frame_id: u64,
    pub capture_time_ns: u64,
    pub observation_sha256: String,
    pub map_candidates: Vec<Candidate>,
}
#[derive(Deserialize)]
pub(super) struct Candidate {
    pub reference_quarter_turns: u8,
    pub matches: usize,
    pub correspondences: Vec<[f64; 4]>,
}
impl Frozen {
    pub fn validate(&self) -> Result<(), Error> {
        if self.backend.is_empty()
            || self.frames.is_empty()
            || self.frames.len() > 1000
            || self.job.reference_quarter_turns != [0, 1, 2, 3]
            || self.frames.windows(2).any(|p| {
                p[1].frame_id <= p[0].frame_id || p[1].capture_time_ns <= p[0].capture_time_ns
            })
        {
            return Err(Error::Input(
                "accumulation requires ordered map-only frames and four rotations".into(),
            ));
        }
        for sample in &self.frames {
            if sample.map_candidates.len() != 4
                || sample.map_candidates.iter().enumerate().any(|(i, c)| {
                    usize::from(c.reference_quarter_turns) != i
                        || c.matches != c.correspondences.len()
                        || c.matches > 4096
                        || c.correspondences.iter().flatten().any(|v| !v.is_finite())
                })
            {
                return Err(Error::Input(format!(
                    "invalid frozen pairs at frame {}",
                    sample.frame_id
                )));
            }
        }
        Ok(())
    }
    pub fn frame_blocking(
        &self,
        sample: &Sample,
        root: &Path,
        config: &Config,
    ) -> Result<Frame, Error> {
        let frame = Frame {
            camera: self.job.camera.model(),
            stamp: FrameStamp {
                sequence: sample.frame_id,
                capture_time_ns: sample.capture_time_ns,
            },
            image: assets::image_verified_blocking(
                &root.join(&sample.image.path),
                config.limits.image_pixels,
                Some(&sample.image.sha256),
            )?,
        };
        if frame.image.dimensions() != (frame.camera.width, frame.camera.height)
            || frame.evidence_sha256() != sample.observation_sha256
        {
            return Err(Error::Input(format!(
                "frozen observation mismatch at frame {}",
                sample.frame_id
            )));
        }
        Ok(frame)
    }
}
impl Sample {
    pub fn evidence(
        &self,
        map_size: (u32, u32),
        query_size: (u32, u32),
    ) -> Result<Vec<Evidence>, Error> {
        let mut result = Vec::new();
        for c in &self.map_candidates {
            for (source_pair_index, p) in c.correspondences.iter().enumerate() {
                if [p[0], p[1], p[2], p[3]]
                    .into_iter()
                    .zip([map_size.0, map_size.1, query_size.0, query_size.1])
                    .any(|(v, limit)| v < 0.0 || v >= f64::from(limit))
                {
                    return Err(Error::Input(format!(
                        "pair outside images at frame {}",
                        self.frame_id
                    )));
                }
                result.push(Evidence {
                    pair: PixelMatch {
                        reference: [p[0], p[1]].into(),
                        query: [p[2], p[3]].into(),
                    },
                    rotation: c.reference_quarter_turns,
                    origin: self.frame_id,
                    source_pair_index,
                    age: 0,
                });
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;

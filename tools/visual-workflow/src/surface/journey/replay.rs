//! Decode only admitted records and retain their original observation timestamps.
use super::{Runner, arrival::Arrivals};
use crate::{
    Error, assets,
    config::Config,
    surface::sequence::input::{Manifest, Sample},
};
use navigate_visual::{CameraModel, Frame, FrameStamp};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
impl Runner {
    pub(super) fn frames_blocking(
        &mut self,
        spec: &Manifest,
        camera: CameraModel,
        manifest_root: &Path,
        config: &Config,
        mut arrivals: Arrivals,
    ) -> Result<(Vec<Value>, Value), Error> {
        let mut frames = Vec::new();
        while let Some(selection) = arrivals.next_blocking(
            &spec.frames,
            self.state
                .as_ref()
                .map(|s| s.tracks.previous.stamp.capture_time_ns),
        ) {
            let sample = &spec.frames[selection.index];
            let mut record = self.sample_blocking(sample, camera, manifest_root, config)?;
            let arrival = arrivals.completed(selection, &spec.frames);
            if !arrival.is_null() {
                record["arrival"] = arrival;
            }
            frames.push(record);
        }
        Ok((frames, arrivals.report(&spec.frames)))
    }
    fn sample_blocking(
        &mut self,
        sample: &Sample,
        camera: CameraModel,
        root: &Path,
        config: &Config,
    ) -> Result<Value, Error> {
        let started = Instant::now();
        let frame = Frame {
            camera,
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
        if frame.image.dimensions() != (frame.camera.width, frame.camera.height) {
            return Err(Error::Input(
                "acquisition sequence image does not match calibration".into(),
            ));
        }
        let frame_loading_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut record = self.process_blocking(frame, config)?;
        record["frame_loading_ms"] = json!(frame_loading_ms);
        record["pipeline_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
        record["image"] = json!(sample.image);
        Ok(record)
    }
}

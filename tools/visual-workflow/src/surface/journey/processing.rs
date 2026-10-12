//! Each new map anchor resets the map-pixel frame and motion continuity.
use super::{Runner, Tracking};
use crate::{
    Error,
    config::Config,
    surface::{
        acquire,
        job::window,
        sequence::tracks::{MAX_TRACK_GAP_NS, Tracks},
    },
};
use navigate_recording::Pose;
use navigate_visual::{Frame, OrthographicEstimate, OrthographicReference, PixelMatch};
use serde_json::{Value, json};
use std::time::Instant;

type Decision = (OrthographicEstimate, Vec<PixelMatch>);
struct Recovery {
    region: Option<OrthographicReference>,
    decision: Option<Decision>,
    window: Option<[u32; 4]>,
}
impl Runner {
    pub(super) fn process_blocking(
        &mut self,
        frame: Frame,
        config: &Config,
    ) -> Result<Value, Error> {
        let observation = frame.evidence_sha256();
        let mut record = json!({"frame_id":frame.stamp.sequence,"capture_time_ns":frame.stamp.capture_time_ns,
            "observation_sha256":observation,"tracking_attempted":false,"map_attempted":false,
            "tracking_ms":0.0,"tracking_geometry_ms":0.0,"map_ms":0.0});
        let mut decision = self.track_blocking(&frame, &mut record)?;
        let refreshed = decision.is_none();
        let mut region = None;
        let mut selected_window = self.state.as_ref().and_then(|s| s.window);
        if refreshed {
            record["map_reason"] = json!(self.refresh_reason(&frame));
            let recovered = self.acquire_blocking(&frame, config, &mut record)?;
            region = recovered.region;
            decision = recovered.decision;
            selected_window = recovered.window;
        }
        record["accepted"] = json!(decision.is_some());
        record["source"] = json!(if decision.is_none() {
            "rejected"
        } else if refreshed {
            "map"
        } else {
            "track"
        });
        record["motion_continuity"] = self.continuity.step(
            &frame,
            &observation,
            decision.as_ref().map(|d| &d.0),
            refreshed,
        )?;
        if let Some((estimate, pairs)) = decision {
            record["pose"] = json!(Pose::from(estimate.pose));
            record["quality"] = json!({"inliers":estimate.quality.inliers,"spatial_support":estimate.quality.spatial_support,
                "occupied_cells":estimate.quality.occupied_cells,"rms_px":estimate.quality.reprojection_rms_px});
            let reference = if refreshed {
                region.as_ref()
            } else {
                self.state.as_ref().and_then(|s| s.region.as_ref())
            }
            .unwrap_or(&self.reference);
            let tracks = Tracks::new(
                frame,
                reference,
                &estimate,
                pairs,
                self.state.as_ref().map(|s| &s.tracks),
                refreshed,
            );
            record["anchor_observation_sha256"] = json!(tracks.origin);
            record["retained_tracks"] = json!(tracks.pairs.len());
            record["reference_window"] = json!(selected_window);
            if !refreshed {
                region = self.state.take().and_then(|s| s.region);
            }
            self.state = Some(Tracking {
                region,
                tracks,
                window: selected_window,
            });
        }
        Ok(record)
    }
    fn acquire_blocking(
        &mut self,
        frame: &Frame,
        config: &Config,
        record: &mut Value,
    ) -> Result<Recovery, Error> {
        self.state = None;
        record["map_attempted"] = json!(true);
        let started = Instant::now();
        let (model_ms, query_ms) = self.models.prepare_blocking(frame, config)?;
        record["model_reload_ms"] = json!(model_ms);
        record["map_query_extraction_ms"] = json!(query_ms);
        record["adapter_tensor_io_before"] = json!(self.models.statistics_before);
        let engine = self.models.engine_mut()?;
        let acquired = acquire::search_blocking(
            frame,
            &self.reference,
            &self.prior,
            engine,
            config,
            self.record_matches,
        )?;
        record["acquisition"] = acquired.record;
        let result = if let Some(selected) = acquired.selected {
            if let Some(area) = selected.window {
                engine.prepare_raster(
                    self.reference.image.as_raw().len() * 5
                        + area[2] as usize * area[3] as usize * 5,
                    config,
                )?;
            }
            let region = selected
                .window
                .map(|area| {
                    window::crop_borrowed(&self.reference, area, config.limits.reference_bytes)
                })
                .transpose()?;
            record["reference_window"] = json!(selected.window);
            Recovery {
                region,
                decision: Some((selected.estimate, selected.pairs)),
                window: selected.window,
            }
        } else {
            Recovery {
                region: None,
                decision: None,
                window: None,
            }
        };
        record["reference_encodings"] = json!(engine.reference_encodings);
        record["reference_cache"] = engine.cache_report();
        record["reference_catalog"] = engine.catalog_report();
        record["adapter_tensor_io_after"] = json!(engine.backend.tensor_statistics());
        record["peak_reference_accounted_bytes"] = json!(engine.peak_reference_bytes);
        record["map_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
        Ok(result)
    }
    fn track_blocking(
        &mut self,
        frame: &Frame,
        record: &mut Value,
    ) -> Result<Option<Decision>, Error> {
        let Some(state) = &self.state else {
            return Ok(None);
        };
        if !state.tracks.eligible(frame, self.interval) {
            return Ok(None);
        }
        record["tracking_attempted"] = json!(true);
        let started = Instant::now();
        let pairs = state
            .tracks
            .propagate_blocking(frame, self.tracker.backend.as_mut())?;
        record["tracking_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
        let started = Instant::now();
        let decision = self.verifier.verify_orthographic(
            frame,
            state.region.as_ref().unwrap_or(&self.reference),
            &self.prior,
            &pairs,
            &self.tracker.identity,
        );
        record["tracking_geometry_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
        match decision {
            Ok(estimate) => Ok(Some((estimate, pairs))),
            Err(error) => {
                record["tracking_rejection"] = json!(error.to_string());
                Ok(None)
            }
        }
    }
    fn refresh_reason(&self, frame: &Frame) -> &'static str {
        let Some(state) = &self.state else {
            return "uninitialized_or_lost";
        };
        if frame.stamp.capture_time_ns - state.tracks.previous.stamp.capture_time_ns
            > MAX_TRACK_GAP_NS
        {
            "time_gap"
        } else if state.tracks.age >= self.interval {
            "scheduled"
        } else {
            "tracking_geometry_rejected"
        }
    }
}

//! Original-video tracking with bounded admission and explicit timing scopes.
mod args;
pub(crate) mod export;
pub(crate) mod inference;
mod record;
mod scheduling;
mod setup;
mod tracker;
use crate::{
    BenchError,
    package::MapPackage,
    renderer::ReferenceRenderer,
    stream::write_record_blocking,
    trial::{config_blocking, writer_blocking},
    video::{decoder::Decoder, probe},
};
pub(crate) use args::FlightArgs;
use navigate_visual::{CameraModel, Frame, FrameStamp};
use record::PoseRecord;
use scheduling::Schedule;
use serde_json::json;
use std::time::{Duration, Instant};

pub(crate) async fn run_blocking(
    args: &FlightArgs,
    backend: crate::backend::BackendKind,
) -> Result<(), BenchError> {
    scheduling::validate(args)?;
    let initialized = Instant::now();
    let setup::Prepared {
        camera,
        timestamps,
        metadata,
        tracker,
    } = setup::prepare_blocking(args, backend).await?;
    let mut run = FlightRun {
        tracker,
        metadata,
        schedule: Schedule::new(args)?,
        counts: Counts::default(),
        writer: writer_blocking(&args.output)?,
    };
    let initialization_ms = initialized.elapsed().as_secs_f64() * 1000.0;
    let overall = Instant::now();
    run.replay_blocking(args, camera, &timestamps)?;
    let report = json!({"context":run.metadata,"initialization_ms":initialization_ms,"wall_ms":overall.elapsed().as_secs_f64()*1000.0,
        "decoded_frames":run.counts.decoded,"processed_frames":run.counts.processed,"supported_frames":run.counts.supported,
        "decode_ms":run.counts.decode_ms,"processing_ms":run.counts.costs,"scope":"measured host run; supported geometry is not absolute accuracy"});
    let summary = args.output.with_extension("summary.json");
    write_record_blocking(&mut writer_blocking(&summary)?, &summary, &report)?;
    if let Some(track) = &args.track {
        export::write_blocking(&args.output, track)?;
    }
    Ok(())
}
struct FlightRun {
    tracker: tracker::Tracker,
    metadata: serde_json::Value,
    schedule: Schedule,
    counts: Counts,
    writer: std::io::BufWriter<std::fs::File>,
}
impl FlightRun {
    fn replay_blocking(
        &mut self,
        args: &FlightArgs,
        camera: CameraModel,
        timestamps: &[u64],
    ) -> Result<(), BenchError> {
        let mut decoder = Decoder::resized_blocking(&args.video, camera)?;
        let mut wall = None;
        let mut complete_video = true;
        for (index, &timestamp) in timestamps.iter().enumerate() {
            if args
                .duration
                .is_some_and(|d| timestamp as f64 / 1e9 >= args.start + d)
            {
                complete_video = false;
                break;
            }
            let decode = Instant::now();
            let image = decoder.next_blocking()?;
            let decode_ms = decode.elapsed().as_secs_f64() * 1000.0;
            self.counts.decode_ms += decode_ms;
            self.counts.decoded = self.counts.decoded.wrapping_add(1);
            if (timestamp as f64 / 1e9) < args.start {
                continue;
            }
            let frame = Frame {
                camera,
                image,
                stamp: FrameStamp {
                    sequence: index as u64,
                    capture_time_ns: timestamp,
                },
            };
            if args.warm_start && wall.is_none() {
                self.process_blocking(frame, decode_ms, args, Instant::now(), true)?;
                if self.counts.supported == 0 {
                    return Err(BenchError::Record {
                        reason: "warm-start frame has no accepted map anchor".into(),
                    });
                }
                wall = Some(Instant::now());
                continue;
            }
            let clock = *wall.get_or_insert_with(Instant::now);
            if !self.schedule.admit_blocking(
                timestamp,
                timestamps.get(index + 1).copied(),
                clock,
                args,
            )? {
                continue;
            }
            self.process_blocking(frame, decode_ms, args, clock, false)?;
        }
        if complete_video {
            decoder.finish_blocking()?;
        }
        Ok(())
    }
    fn process_blocking(
        &mut self,
        frame: Frame,
        decode_ms: f64,
        args: &FlightArgs,
        clock: Instant,
        warmup: bool,
    ) -> Result<(), BenchError> {
        let started = Instant::now();
        let (mut report, timing) = self.tracker.observe_blocking(&frame)?;
        let cost = started.elapsed() + Duration::from_secs_f64(decode_ms / 1000.0);
        let status = self.schedule.finish_work(cost, warmup)?;
        self.counts.record(&report, cost.as_secs_f64() * 1000.0);
        report["sequence"] = frame.stamp.sequence.into();
        report["capture_time_ns"] = frame.stamp.capture_time_ns.into();
        report["observation_sha256"] = frame.evidence_sha256().into();
        report["context"] = self.metadata.clone();
        report["timing"] = json!({"decode_ms":decode_ms,"render_ms":timing.render_ms,"matching_ms":timing.matching_ms,
            "geometry_ms":timing.geometry_ms,"processing_ms":cost.as_secs_f64()*1000.0,"dense_runs":timing.dense_runs,
            "fast_runs":timing.fast_runs,"interval_ms":status.interval.as_secs_f64()*1000.0,
            "deadline_unattainable":status.deadline_unattainable,"utilization":status.utilization,"warmup":warmup,
            "age_at_completion_ms":if args.realtime && !warmup {Some((clock.elapsed().as_secs_f64()+args.start-frame.stamp.capture_time_ns as f64/1e9)*1000.0)}else{None}});
        write_record_blocking(&mut self.writer, &args.output, &report)?;
        tracing::info!(
            sequence = frame.stamp.sequence,
            pts_s = frame.stamp.capture_time_ns as f64 / 1e9,
            processing_ms = cost.as_secs_f64() * 1000.0,
            supported = self.counts.supported,
            processed = self.counts.processed,
            "flight frame"
        );
        Ok(())
    }
}
#[derive(Default)]
struct Counts {
    decoded: u64,
    processed: u64,
    supported: u64,
    decode_ms: f64,
    costs: Vec<f64>,
}
impl Counts {
    fn record(&mut self, report: &serde_json::Value, cost: f64) {
        self.processed = self.processed.wrapping_add(1);
        self.costs.push(cost);
        if report["candidate_hypotheses"]
            .as_array()
            .is_some_and(|values| {
                values
                    .iter()
                    .any(|v| v["accepted"] == true || v["tracking_supported"] == true)
            })
        {
            self.supported = self.supported.wrapping_add(1);
        }
    }
}
pub(crate) fn scaled_camera(source: CameraModel, width: u32) -> Result<CameraModel, BenchError> {
    let height = (f64::from(source.height) * f64::from(width) / f64::from(source.width) / 8.0)
        .round() as u32
        * 8;
    let x = f64::from(width) / f64::from(source.width);
    let y = f64::from(height) / f64::from(source.height);
    let camera = CameraModel {
        width,
        height,
        fx: source.fx * x,
        fy: source.fy * y,
        cx: (source.cx + 0.5) * x - 0.5,
        cy: (source.cy + 0.5) * y - 0.5,
    };
    camera.validate()?;
    Ok(camera)
}

#[cfg(test)]
mod tests;

//! Resident rendering and candidate refinement for one observation at a time.
mod observation;
mod record;
use crate::{
    BenchError,
    package::MapPackage,
    renderer::ReferenceRenderer,
    stream::{PriorRecord, write_record_blocking},
    trial::{config_blocking, export_reference_blocking},
};
use navigate_visual::{CameraModel, FrameStamp, LocalFrame, ReferenceView, VisualError};
use observation::{Observation, Refinement};
use record::CandidateRecord;
use serde::Deserialize;
use std::{
    io::{BufRead, BufReader, BufWriter},
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Begin {
        query: PathBuf,
        prior: PriorRecord,
        sequence: u64,
        capture_time_ns: u64,
    },
    Render {
        candidate_id: u64,
        candidate: CandidateRecord,
        output: PathBuf,
    },
    Refine {
        candidate_id: u64,
        matches: PathBuf,
        output: PathBuf,
    },
    Select,
}
struct Worker {
    renderer: ReferenceRenderer,
    reference: Option<(u64, ReferenceView)>,
    camera: CameraModel,
    frame: LocalFrame,
    map_context: serde_json::Value,
    observation: Option<Observation>,
}
impl Worker {
    async fn new(package: &Path, prior: &Path) -> Result<Self, BenchError> {
        let camera = config_blocking(prior)?.camera.model();
        let package = MapPackage::open_blocking(package)?;
        let frame = package.frame;
        let map_context = serde_json::json!({"map_release":package.revision.release_id,"map_manifest_sha256":package.revision.manifest_sha256,"anchor_lat_lon":package.manifest.anchor_lat_lon,"elevation_datum":package.manifest.elevation_datum,"coordinate_model":"local-mercator","reference_geometry":"rendered_world_model; not independently verified scene geometry"});
        Ok(Self {
            renderer: ReferenceRenderer::new(package, camera).await?,
            reference: None,
            camera,
            frame,
            map_context,
            observation: None,
        })
    }
    fn begin_blocking(
        &mut self,
        query: &Path,
        prior: PriorRecord,
        stamp: FrameStamp,
    ) -> Result<serde_json::Value, BenchError> {
        let observation =
            Observation::new_blocking(query, stamp, self.camera, prior.prior(self.frame)?)?;
        if let Some(previous) = &self.observation {
            if stamp.capture_time_ns <= previous.frame.stamp.capture_time_ns {
                return Err(VisualError::FrameOrder {
                    previous_ns: previous.frame.stamp.capture_time_ns,
                    received_ns: stamp.capture_time_ns,
                }
                .into());
            }
            let advance = stamp.sequence.wrapping_sub(previous.frame.stamp.sequence);
            if advance == 0 || advance >= (1_u64 << 63) {
                return Err(VisualError::Invalid {
                    field: "observation sequence",
                }
                .into());
            }
        }
        let identity = observation.frame.evidence_sha256();
        self.observation = Some(observation);
        self.reference = None;
        Ok(serde_json::json!({"ok":true,"observation_sha256":identity}))
    }
    fn handle_blocking(&mut self, request: Request) -> Result<serde_json::Value, BenchError> {
        match request {
            Request::Begin {
                query,
                prior,
                sequence,
                capture_time_ns,
            } => self.begin_blocking(
                &query,
                prior,
                FrameStamp {
                    sequence,
                    capture_time_ns,
                },
            ),
            Request::Render {
                candidate_id,
                candidate,
                output,
            } => {
                self.reference = None;
                self.observation
                    .as_mut()
                    .ok_or_else(|| protocol("begin an observation before rendering"))?
                    .invalidate(candidate_id)?;
                let started = Instant::now();
                let reference = self.renderer.render_blocking(candidate.pose()?)?;
                export_reference_blocking(&reference, self.camera, &output)?;
                self.reference = Some((candidate_id, reference));
                Ok(
                    serde_json::json!({"ok":true,"render_ms":started.elapsed().as_secs_f64()*1000.0}),
                )
            }
            Request::Refine {
                candidate_id,
                matches,
                output,
            } => {
                let (id, reference) = self
                    .reference
                    .as_ref()
                    .ok_or_else(|| protocol("render a candidate before refinement"))?;
                if *id != candidate_id {
                    return Err(protocol(
                        "refinement candidate does not match the rendered reference",
                    ));
                }
                let observation = self
                    .observation
                    .as_mut()
                    .ok_or_else(|| protocol("begin an observation before refinement"))?;
                let estimate = observation.refine_blocking(Refinement {
                    id: candidate_id,
                    reference,
                    matches: &matches,
                    output: &output,
                    map_context: &self.map_context,
                    map_frame: self.frame,
                })?;
                Ok(serde_json::json!({"ok":true,"estimate":estimate}))
            }
            Request::Select => {
                let observation = self
                    .observation
                    .as_ref()
                    .ok_or_else(|| protocol("begin an observation before selection"))?;
                Ok(serde_json::json!({"ok":true,"observation":observation.select()}))
            }
        }
    }
}
fn protocol(reason: &str) -> BenchError {
    BenchError::Record {
        reason: reason.into(),
    }
}
pub(crate) async fn run_blocking(package: &Path, prior: &Path) -> Result<(), BenchError> {
    let mut worker = Worker::new(package, prior).await?;
    let input = BufReader::new(std::io::stdin());
    let mut output = BufWriter::new(std::io::stdout());
    for line in input.lines() {
        let line = line.map_err(|source| BenchError::Io {
            path: "<worker stdin>".into(),
            source,
        })?;
        let request = serde_json::from_str(&line).map_err(|source| BenchError::Json {
            path: "<worker request>".into(),
            source,
        });
        let report = match request.and_then(|request| worker.handle_blocking(request)) {
            Ok(report) => report,
            Err(error) => {
                worker.reference = None;
                serde_json::json!({"ok":false,"error":error.to_string()})
            }
        };
        write_record_blocking(&mut output, Path::new("<worker stdout>"), &report)?;
    }
    Ok(())
}

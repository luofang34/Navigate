use crate::{
    BenchError, package::MapPackage, renderer::ReferenceRenderer, worker::observation::Observation,
};
use navigate_visual::{CameraModel, CameraPose, Frame, ImageMatcher, PosePrior, RefinementSeed};
use serde_json::json;
use std::time::Instant;

pub(super) struct Search {
    pub camera: CameraModel,
    prior: PosePrior,
    renderer: ReferenceRenderer,
    context: serde_json::Value,
}
impl Search {
    pub async fn new(
        package: MapPackage,
        camera: CameraModel,
        prior: PosePrior,
    ) -> Result<Self, BenchError> {
        let context = json!({"map_release":package.revision.release_id,"map_manifest_sha256":package.revision.manifest_sha256,
            "anchor_lat_lon":package.manifest.anchor_lat_lon,"elevation_datum":package.manifest.elevation_datum,
            "coordinate_model":"local-mercator","renderer_revision":crate::RENDERER_REVISION.trim()});
        Ok(Self {
            camera,
            prior,
            context,
            renderer: ReferenceRenderer::new(package, camera).await?,
        })
    }
    pub fn evaluate_blocking(
        &mut self,
        frame: Frame,
        candidates: &[CameraPose],
        matcher: &mut dyn ImageMatcher,
    ) -> Result<serde_json::Value, BenchError> {
        let mut observation = Observation::from_frame(frame, self.prior)?;
        let mut proposals = Vec::new();
        let mut work = Work::default();
        for (id, &pose) in candidates.iter().enumerate() {
            if let Some(seed) = self.evaluate_candidate_blocking(
                &mut observation,
                id as u64,
                pose,
                matcher,
                &mut work,
            )? {
                proposals.push((id as u64, seed));
            }
        }
        proposals.sort_by(|(aid, a), (bid, b)| {
            b.spatial_support.cmp(&a.spatial_support).then(aid.cmp(bid))
        });
        for (id, mut seed) in proposals.into_iter().take(12) {
            for _ in 0..2 {
                let Some(next) = self.evaluate_candidate_blocking(
                    &mut observation,
                    id,
                    seed.pose,
                    matcher,
                    &mut work,
                )?
                else {
                    break;
                };
                seed = next;
            }
        }
        let mut report = observation.select();
        report["camera"] = json!({"width":self.camera.width,"height":self.camera.height,
            "fx":self.camera.fx,"fy":self.camera.fy,"cx":self.camera.cx,"cy":self.camera.cy});
        report["navigation_prior"] = json!({"position_enu_m":self.prior.pose.position.as_slice(),
            "eye_to_enu_xyzw":self.prior.pose.orientation.coords.as_slice(),
            "position_radius_m":self.prior.position_radius_m,"attitude_radius_rad":self.prior.attitude_radius_rad});
        report["verification_work"] = json!({"attempts":work.attempts,"render_ms":work.render_ms,"matching_ms":work.matching_ms,"geometry_ms":work.geometry_ms,"scope":"wall-clock time; includes device waits"});
        Ok(report)
    }
    fn evaluate_candidate_blocking(
        &mut self,
        observation: &mut Observation,
        id: u64,
        pose: CameraPose,
        matcher: &mut dyn ImageMatcher,
        work: &mut Work,
    ) -> Result<Option<RefinementSeed>, BenchError> {
        observation.invalidate(id)?;
        let started = Instant::now();
        let reference = self.renderer.render_blocking(pose)?;
        work.render_ms += started.elapsed().as_secs_f64() * 1000.0;
        let (report, seed) =
            observation.match_candidate_blocking(id, &reference, matcher, &self.context)?;
        work.attempts = work.attempts.wrapping_add(1);
        work.matching_ms += report["matching_ms"].as_f64().unwrap_or(0.0);
        work.geometry_ms += report["geometry_ms"].as_f64().unwrap_or(0.0);
        tracing::debug!(candidate=id,accepted=?report["accepted"],"candidate evaluation");
        Ok(seed)
    }
}
#[derive(Default)]
struct Work {
    attempts: u64,
    render_ms: f64,
    matching_ms: f64,
    geometry_ms: f64,
}

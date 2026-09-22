//! Shared observation path for images, decoded video, and frame streams.

use crate::{
    BenchError,
    backend::{Backend, BackendKind},
    coordinates::MapFrame,
    package::MapPackage,
    renderer::ReferenceRenderer,
};
use navigate_visual::{CameraModel, Frame, Localizer, LocalizerConfig, PosePrior};

pub(crate) struct Session {
    renderer: ReferenceRenderer,
    localizer: Localizer<Backend>,
    pub camera: CameraModel,
    pub frame: MapFrame,
    map_context: serde_json::Value,
}

impl Session {
    pub async fn new(
        package: MapPackage,
        camera: CameraModel,
        backend: BackendKind,
    ) -> Result<Self, BenchError> {
        let frame = MapFrame::new(package.manifest.anchor_lat_lon);
        let map_context = serde_json::json!({"map_release":package.revision.release_id,
            "map_manifest_sha256":package.revision.manifest_sha256,
            "anchor_lat_lon":package.manifest.anchor_lat_lon,"elevation_datum":package.manifest.elevation_datum,
            "coordinate_model":"local-mercator"});
        Ok(Self {
            frame,
            camera,
            map_context,
            renderer: ReferenceRenderer::new(package, camera).await?,
            localizer: Localizer::new(Backend::new(backend).await?, LocalizerConfig::default())?,
        })
    }

    pub fn observe_blocking(
        &mut self,
        frame: &Frame,
        prior: &PosePrior,
    ) -> Result<serde_json::Value, BenchError> {
        if frame.camera != self.camera {
            return Err(BenchError::ChangedCamera);
        }
        let started = std::time::Instant::now();
        let reference = self.renderer.render_blocking(prior.pose)?;
        let estimate = self.localizer.estimate_blocking(frame, &reference, prior);
        let mut report = self.map_context.clone();
        report["sequence"] = frame.stamp.sequence.into();
        report["capture_time_ns"] = frame.stamp.capture_time_ns.into();
        report["total_ms"] = (started.elapsed().as_secs_f64() * 1000.0).into();
        match estimate {
            Ok(estimate) => {
                let [lon, lat, alt] = self
                    .frame
                    .longitude_latitude_altitude(estimate.pose.position);
                report["accepted"] = true.into();
                report["longitude_deg"] = lon.into();
                report["latitude_deg"] = lat.into();
                report["altitude_m"] = alt.into();
                report["position_enu_m"] = serde_json::json!(estimate.pose.position.as_slice());
                report["eye_to_enu_xyzw"] =
                    serde_json::json!(estimate.pose.orientation.coords.as_slice());
                report["backend"] = estimate.backend.into();
                report["inliers"] = estimate.quality.inliers.into();
                report["reprojection_rms_px"] = estimate.quality.reprojection_rms_px.into();
                report["occupied_cells"] = estimate.quality.occupied_cells.into();
                report["condition_number"] = estimate.quality.condition_number.into();
                let covariance: Vec<Vec<f64>> = (0..6)
                    .map(|r| {
                        (0..6)
                            .map(|c| estimate.geometry_covariance[(r, c)])
                            .collect()
                    })
                    .collect();
                report["geometry_covariance"] = serde_json::json!(covariance);
            }
            Err(error) => {
                report["accepted"] = false.into();
                report["reason"] = error.to_string().into();
            }
        }
        Ok(report)
    }
}

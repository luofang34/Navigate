//! Camera-to-camera tracking remains separate from accepted map candidates.
use super::*;
use navigate_visual::TrackingReference;

#[derive(Deserialize)]
struct Previous {
    candidate_id: u32,
    sequence: u32,
    capture_time_ns: f64,
}
impl Session {
    pub(super) fn track(
        &self,
        id: u32,
        previous: &Frame,
        reference: &ReferenceView,
        pairs: &[PixelMatch],
        backend: &str,
    ) -> Result<Value, PreviewError> {
        if self.active != Some(id) {
            return Err(PreviewError::Input {
                reason: "tracking reference candidate identity does not match".into(),
            });
        }
        let result = PoseVerifier::new(LocalizerConfig::default())?.track(
            &self.frame,
            TrackingReference {
                observation: previous,
                surface: reference,
            },
            &self.prior,
            pairs,
            backend,
        );
        let mut report = match result {
            Ok(e) => {
                let p = e.pose.position;
                let q = e.pose.orientation.quaternion();
                let [latitude, longitude, _] = e.frame.geodetic(p);
                json!({"candidate_id":id,"accepted":false,"tracking_supported":true,"acceptance_stage":"conditional_relative_geometry","position_enu_m":[p.x,p.y,p.z],"eye_to_enu_xyzw":[q.i,q.j,q.k,q.w],"latitude_deg":latitude,"longitude_deg":longitude,"altitude_m":p.z,"inliers":e.quality.inliers,"reprojection_rms_px":e.quality.reprojection_rms_px,"occupied_cells":e.quality.occupied_cells,"condition_number":e.quality.condition_number,"backend":e.backend,"observation_sha256":e.observation_sha256,"reference_observation_sha256":e.reference_observation_sha256,"map_release_id":e.map.release_id,"map_manifest_sha256":e.map.manifest_sha256,"uncertainty":"unknown; conditional on estimated anchor pose, camera calibration and rendered depth","evidence_correlation":"unknown; inherits the reference pose and map evidence"})
            }
            Err(e) => {
                json!({"candidate_id":id,"accepted":false,"tracking_supported":false,"reason":e.to_string()})
            }
        };
        reference_provenance(&mut report, reference);
        Ok(report)
    }
}

#[wasm_bindgen]
impl Preview {
    /// Fit relative camera matches without creating an accepted map measurement.
    pub fn track(
        &self,
        previous_pixels: Vec<u8>,
        previous_json: String,
        pairs_json: String,
        backend: String,
    ) -> Result<String, JsValue> {
        let previous: Previous =
            serde_json::from_str(&previous_json).map_err(PreviewError::from)?;
        let session = self
            .camera_session
            .as_ref()
            .ok_or_else(|| PreviewError::Input {
                reason: "no active observation".into(),
            })?;
        if !previous.capture_time_ns.is_finite()
            || !(0.0..=9_007_199_254_740_991.0).contains(&previous.capture_time_ns)
        {
            return Err(PreviewError::Input {
                reason: "invalid previous capture time".into(),
            }
            .into());
        }
        let frame = Frame {
            camera: self.camera.model(),
            stamp: FrameStamp {
                sequence: u64::from(previous.sequence),
                capture_time_ns: previous.capture_time_ns as u64,
            },
            image: GrayImage::from_raw(self.camera.width, self.camera.height, previous_pixels)
                .ok_or_else(|| PreviewError::Input {
                    reason: "previous image dimensions do not match calibration".into(),
                })?,
        };
        let pairs: Vec<Pair> = serde_json::from_str(&pairs_json).map_err(PreviewError::from)?;
        if pairs.len() > 4096 {
            return Err(PreviewError::Input {
                reason: "too many pixel correspondences".into(),
            }
            .into());
        }
        let pairs: Vec<_> = pairs
            .into_iter()
            .map(|p| PixelMatch {
                reference: Vector2::from(p.reference),
                query: Vector2::from(p.query),
            })
            .collect();
        let reference = self.reference.as_ref().ok_or_else(|| PreviewError::Input {
            reason: "no rendered tracking reference".into(),
        })?;
        Ok(session
            .track(previous.candidate_id, &frame, reference, &pairs, &backend)?
            .to_string())
    }
}

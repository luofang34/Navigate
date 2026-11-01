//! Session bridge types, coordinate conversion, and filter telemetry snapshots.

use super::VisualCorridorSession;
use crate::corridor::plausibility::compute_fisher_geometry_covariance;
use nalgebra::{Matrix3, Matrix6, Translation3, UnitQuaternion, Vector3};
use navigate_contract::{
    FaultDetection, IntegrityAssessment, MonotonicNanos, Redundancy, SolutionQuality,
};
use navigate_fusion::IngestOutcome;
use navigate_visual::{
    CameraModel, CameraPose, Estimate, EstimateQuality, FrameStamp, LocalFrame, MapRevision,
    holdout_pnp::PoseCandidate,
};
use navigate_visual_fusion::EvidenceIndependence;
use navigate_visual_session::{
    AnchorDecision, AnchorObservation, AnchorPolicy, Calibration, CameraMount, FrameKey,
    FusionEligibility, LensModel, MapCell, MapReliability, Pose, RegionChange, ScaleSource,
    SessionConfig, SessionEvent, SurfaceModel, VerticalReference, VisualSession, from_camera,
};
use sha2::{Digest, Sha256};

/// Summary of map-anchor admission and filter ingestion for one frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionAnchorReport {
    /// Fusion eligibility label for telemetry.
    pub fusion_eligibility: &'static str,
    /// True when an independent fix entered the navigation filter.
    pub independent_ekf_fix: bool,
    /// Map cell coordinates `[east, north]` when tracked.
    pub map_cell: Option<[i64; 2]>,
    /// Earlier frame identifier when the map cell was already visited.
    pub earlier_anchor_id: Option<String>,
}

/// Snapshot of navigation filter integrity and trajectory revision counts.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterIntegritySnapshot {
    /// Solution quality label (`"good"`, `"degraded"`, `"unusable"`, or `"none"`).
    pub quality: &'static str,
    /// Redundancy label (`"none"`, `"same_class"`, `"independent_classes"`).
    pub redundancy: &'static str,
    /// Fault-detection label (`"unavailable"`, `"monitoring"`).
    pub fault_detection: &'static str,
    /// Horizontal one-sigma uncertainty in meters when a filter solution exists.
    pub horizontal_1sigma_m: Option<f64>,
    /// Vertical one-sigma uncertainty in meters when a filter solution exists.
    pub vertical_1sigma_m: Option<f64>,
    /// Cumulative count of `TrajectoryRevised` events emitted by `VisualSession`.
    pub trajectory_revisions: usize,
    /// Current `VisualSession` revision number.
    pub session_revision: u64,
}

impl Default for FilterIntegritySnapshot {
    fn default() -> Self {
        Self {
            quality: "none",
            redundancy: "none",
            fault_detection: "unavailable",
            horizontal_1sigma_m: None,
            vertical_1sigma_m: None,
            trajectory_revisions: 0,
            session_revision: 0,
        }
    }
}

impl FilterIntegritySnapshot {
    /// Build a snapshot from an optional [`IntegrityAssessment`] and revision counters.
    #[must_use]
    pub fn from_assessment(
        assessment: Option<IntegrityAssessment>,
        trajectory_revisions: usize,
        session_revision: u64,
    ) -> Self {
        let Some(a) = assessment else {
            return Self {
                trajectory_revisions,
                session_revision,
                ..Self::default()
            };
        };
        let quality = match a.quality {
            SolutionQuality::Good => "good",
            SolutionQuality::Degraded => "degraded",
            SolutionQuality::Unusable => "unusable",
            _ => "unusable",
        };
        let redundancy = match a.redundancy {
            Redundancy::None => "none",
            Redundancy::SameClass => "same_class",
            Redundancy::IndependentClasses => "independent_classes",
            _ => "independent_classes",
        };
        let fault_detection = match a.fault_detection {
            FaultDetection::Unavailable => "unavailable",
            FaultDetection::Monitoring => "monitoring",
            _ => "monitoring",
        };
        Self {
            quality,
            redundancy,
            fault_detection,
            horizontal_1sigma_m: Some(a.horizontal_1sigma_m),
            vertical_1sigma_m: Some(a.vertical_1sigma_m),
            trajectory_revisions,
            session_revision,
        }
    }
}

/// Convert OpenCV world-to-camera pose (`+X` right, `+Y` down, `+Z` forward) into [`CameraPose`].
#[must_use]
pub fn cv_world_to_cam_to_camera_pose(
    eye_enu: [f64; 3],
    rot_world_to_cam: &Matrix3<f64>,
) -> CameraPose {
    let flip = Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0));
    let r_world_from_cam = rot_world_to_cam.transpose() * flip;
    let orientation =
        UnitQuaternion::new_normalize(UnitQuaternion::from_matrix(&r_world_from_cam).into_inner());
    CameraPose {
        position: Vector3::from(eye_enu),
        orientation,
    }
}

/// Convert OpenCV world-to-camera pose into [`navigate_visual_session::Pose`].
#[must_use]
pub fn cv_world_to_cam_to_session_pose(eye_enu: [f64; 3], rot_world_to_cam: &Matrix3<f64>) -> Pose {
    let cam = cv_world_to_cam_to_camera_pose(eye_enu, rot_world_to_cam);
    Pose::from_parts(Translation3::from(cam.position), cam.orientation)
}

/// Transform 6-DoF Fisher covariance from OpenCV camera axes to [`CameraPose`] camera axes.
#[must_use]
pub fn cv_fisher_to_camera_covariance(cand: &PoseCandidate) -> Matrix6<f64> {
    let cov = compute_fisher_geometry_covariance(cand);
    let flip = Matrix6::from_diagonal(&nalgebra::Vector6::new(1.0, 1.0, 1.0, 1.0, -1.0, -1.0));
    let sym = flip * cov * flip;
    (sym + sym.transpose()) * 0.5
}

fn normalize_manifest_sha256(raw: &str) -> String {
    let valid_hex = raw.len() == 64
        && raw
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase());
    if valid_hex {
        return raw.to_owned();
    }
    let digest: [u8; 32] = Sha256::digest(raw.as_bytes()).into();
    hex::encode(digest)
}

/// Build a [`navigate_visual::Estimate`] from an accepted [`PoseCandidate`].
#[must_use]
pub fn build_estimate_from_candidate(
    cand: &PoseCandidate,
    local_frame: LocalFrame,
    catalog_sha256: &str,
    observation_sha256: &str,
    sequence: u64,
    capture_time_ns: u64,
    backend: &str,
) -> Estimate {
    let pose = cv_world_to_cam_to_camera_pose(cand.eye, &cand.rotation);
    let geometry_covariance = cv_fisher_to_camera_covariance(cand);
    Estimate {
        stamp: FrameStamp {
            sequence,
            capture_time_ns,
        },
        observation_sha256: observation_sha256.to_owned(),
        map: MapRevision {
            release_id: "corridor-catalog".into(),
            manifest_sha256: normalize_manifest_sha256(catalog_sha256),
        },
        frame: local_frame,
        pose,
        quality: EstimateQuality {
            depth_matches: cand.score.pairs,
            inliers: cand.score.inliers,
            spatial_support: cand.score.inliers,
            reprojection_rms_px: cand.score.train_rms_px,
            occupied_cells: 6,
            condition_number: 10.0,
        },
        geometry_covariance,
        backend: backend.to_owned(),
    }
}

impl VisualCorridorSession {
    /// Reconfigure calibration intrinsics and shared map-cell size before frame processing.
    ///
    /// # Errors
    /// Returns an error string when calibration or anchor policy validation fails.
    pub fn configure_corridor(
        &mut self,
        intrinsics: CameraModel,
        shared_error_cell_m: f64,
    ) -> Result<(), String> {
        let calib = Calibration {
            intrinsics,
            lens: LensModel::Undistorted {
                residual_rms_px: 0.5,
            },
            mount: CameraMount::Fixed {
                body_from_camera: Pose::translation(0.0, 0.0, 0.0),
            },
            map_vertical: VerticalReference::Wgs84Ellipsoid,
        };
        let mut cfg = SessionConfig::standard();
        cfg.anchors = AnchorPolicy {
            shared_error_cell_m: shared_error_cell_m.clamp(50.0, 20_000.0),
            ..AnchorPolicy::default()
        };
        self.session = VisualSession::new(cfg, calib).map_err(|e| format!("{e:?}"))?;
        self.next_seq = 0;
        self.current_epoch = navigate_visual_session::ContinuityEpoch(0);
        self.prev_registered_key = None;
        self.last_registered_key = None;
        self.prev_registered_sha256 = None;
        self.last_registered_sha256 = None;
        self.last_odom_frame = None;
        self.seen_digests.clear();
        self.frame_case_ids.clear();
        self.trajectory_revisions = 0;
        self.last_capture_ns = 0;
        Ok(())
    }

    /// Derive a unique 64-hex evidence digest for one processed frame.
    #[must_use]
    pub fn unique_frame_digest(&mut self, raw_sha256: &str, case_id: &str, seq: u64) -> String {
        let valid_hex = raw_sha256.len() == 64
            && raw_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase());
        if valid_hex && !self.seen_digests.contains(raw_sha256) {
            let out = raw_sha256.to_owned();
            self.seen_digests.insert(out.clone());
            return out;
        }
        let mut hasher = Sha256::new();
        hasher.update(raw_sha256.as_bytes());
        hasher.update(b":");
        hasher.update(case_id.as_bytes());
        hasher.update(b":");
        hasher.update(seq.to_le_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        let mut out = String::with_capacity(64);
        for byte in digest {
            let hi = char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0');
            let lo = char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('0');
            out.push(hi);
            out.push(lo);
        }
        self.seen_digests.insert(out.clone());
        out
    }

    /// Drain pending `VisualSession` events and update trajectory revision counters.
    pub fn drain_session_events(&mut self) -> usize {
        let batch = self.session.drain_events();
        let added = batch
            .events
            .iter()
            .filter(|ev| matches!(ev, SessionEvent::TrajectoryRevised(_)))
            .count();
        self.trajectory_revisions += added;
        added
    }

    /// Tick the navigation filter at `t_ns` and refresh Path A search-prior state.
    pub fn refresh_filter_snapshot(
        &mut self,
        t_ns: u64,
        local_frame: Option<LocalFrame>,
    ) -> FilterIntegritySnapshot {
        let _ = self.drain_session_events();
        let sol = self.filter.tick(MonotonicNanos::from_nanos(t_ns));
        if let (Some(solution), Some(frame)) = (sol, local_frame) {
            self.dead_reckoner
                .set_filter_solution(solution, frame, t_ns);
        }
        FilterIntegritySnapshot::from_assessment(
            sol.map(|s| s.integrity),
            self.trajectory_revisions,
            self.session.revision(),
        )
    }

    /// Record a warm-tracked metric step in `VisualSession` from `PoseCandidate`.
    pub fn record_warm_track_step(
        &mut self,
        eye_enu: [f64; 3],
        rot_world_to_cam: &Matrix3<f64>,
        catalog_sha256: &str,
    ) {
        let cur_pose = cv_world_to_cam_to_session_pose(eye_enu, rot_world_to_cam);
        let (Some(to_key), Some(to_sha)) = (
            self.last_registered_key,
            self.last_registered_sha256.clone(),
        ) else {
            return;
        };
        let Some((from_key, from_sha, from_pose)) = self.last_odom_frame.clone() else {
            return;
        };
        if from_key.continuity != to_key.continuity || to_key.index <= from_key.index {
            return;
        }
        let from_to = from_pose.inv_mul(&cur_pose);
        let scale = ScaleSource::SeededMapPoints {
            map: MapRevision {
                release_id: "corridor-catalog".into(),
                manifest_sha256: normalize_manifest_sha256(catalog_sha256),
            },
        };
        let motion = navigate_visual_session::RelativeMotion {
            from: from_key,
            to: to_key,
            from_to,
            scale,
            from_observation_sha256: from_sha,
            to_observation_sha256: to_sha.clone(),
        };
        if self.session.apply_motion(motion).is_ok() {
            self.last_odom_frame = Some((to_key, to_sha, cur_pose));
        } else {
            self.advance_epoch();
        }
        let _ = self.drain_session_events();
    }

    /// Submit a solved [`PoseCandidate`] anchor to `VisualSession`, `VisualFixSource`, and `NavigationFilter`.
    ///
    /// # Errors
    /// Returns an error string when anchor submission fails.
    #[allow(clippy::too_many_arguments)]
    pub fn record_anchor_candidate(
        &mut self,
        cand: &PoseCandidate,
        local_frame: LocalFrame,
        catalog_sha256: &str,
        cell_m: f64,
        case_id: &str,
        backend: &str,
    ) -> Result<SessionAnchorReport, String> {
        let need_auto_frame = match self.last_registered_key {
            None => true,
            Some(k) => self.frame_case_ids.contains_key(&k),
        };
        if need_auto_frame {
            let next_ns = self
                .last_capture_ns
                .max(1_000_000_000)
                .saturating_add(33_333_333);
            let seq = self.next_seq;
            let auto_sha = self.unique_frame_digest("", case_id, seq);
            self.register_frame_with_digest(next_ns, &auto_sha)?;
        }
        let key = self
            .last_registered_key
            .ok_or_else(|| "no registered frame for anchor".to_owned())?;
        let sha = self
            .last_registered_sha256
            .clone()
            .ok_or_else(|| "no registered digest for anchor".to_owned())?;
        let t_ns = self.last_capture_ns.max(1);
        self.frame_case_ids.insert(key, case_id.to_owned());

        let estimate = build_estimate_from_candidate(
            cand,
            local_frame,
            catalog_sha256,
            &sha,
            key.index,
            t_ns,
            backend,
        );
        let reliability = MapReliability {
            horizontal_m: 1.5,
            vertical_m: 2.0,
            imagery_age_years: None,
            age_growth_m_per_year: 0.1,
            unknown_age_years: 2.0,
            surface: SurfaceModel::BareEarth {
                max_object_height_m: 15.0,
            },
            change: RegionChange::Unknown,
        };
        let anchor = AnchorObservation::from_estimate(key, &estimate, reliability);
        let cur_pose = from_camera(&estimate.pose);
        let decision = self
            .session
            .submit_anchor(anchor)
            .map_err(|e| format!("{e:?}"))?;
        let _ = self.drain_session_events();

        let cell = MapCell {
            map_manifest_sha256: catalog_sha256.to_owned(),
            east: (cand.eye[0] / cell_m).floor() as i64,
            north: (cand.eye[1] / cell_m).floor() as i64,
        };
        let coords = Some([cell.east, cell.north]);

        match decision {
            AnchorDecision::Accepted { eligibility } => {
                self.last_odom_frame = Some((key, sha, cur_pose));
                match eligibility {
                    FusionEligibility::Independent => {
                        let ind =
                            EvidenceIndependence::ValidatedIndependent.restricted_by(&eligibility);
                        let ingested = self.fix_source.convert(&estimate, ind).is_ok_and(|fix| {
                            self.filter
                                .ingest(&fix.observation, MonotonicNanos::from_nanos(t_ns))
                                == IngestOutcome::Accepted
                        });
                        Ok(SessionAnchorReport {
                            fusion_eligibility: "independent",
                            independent_ekf_fix: ingested,
                            map_cell: coords,
                            earlier_anchor_id: None,
                        })
                    }
                    FusionEligibility::SharedMapError { earlier, .. } => {
                        let earlier_id = self.frame_case_ids.get(&earlier).cloned();
                        Ok(SessionAnchorReport {
                            fusion_eligibility: "shared_map_cell",
                            independent_ekf_fix: false,
                            map_cell: coords,
                            earlier_anchor_id: earlier_id,
                        })
                    }
                    FusionEligibility::Untracked => Ok(SessionAnchorReport {
                        fusion_eligibility: "untracked",
                        independent_ekf_fix: false,
                        map_cell: coords,
                        earlier_anchor_id: None,
                    }),
                }
            }
            AnchorDecision::Relocated { .. } => {
                self.last_odom_frame = Some((key, sha, cur_pose));
                self.handle_relocation(cand.eye, cand.rotation, 0.0);
                let ingested = self
                    .fix_source
                    .convert(&estimate, EvidenceIndependence::ValidatedIndependent)
                    .is_ok_and(|fix| {
                        self.filter
                            .ingest(&fix.observation, MonotonicNanos::from_nanos(t_ns))
                            == IngestOutcome::Accepted
                    });
                Ok(SessionAnchorReport {
                    fusion_eligibility: "relocated",
                    independent_ekf_fix: ingested,
                    map_cell: coords,
                    earlier_anchor_id: None,
                })
            }
            AnchorDecision::SameViewpoint => {
                self.last_odom_frame = Some((key, sha, cur_pose));
                let earlier_id = self.find_earlier_case_id_other_than(key);
                Ok(SessionAnchorReport {
                    fusion_eligibility: "shared_map_cell",
                    independent_ekf_fix: false,
                    map_cell: coords,
                    earlier_anchor_id: earlier_id,
                })
            }
            AnchorDecision::Rejected(_) => Ok(SessionAnchorReport {
                fusion_eligibility: "awaiting_consensus",
                independent_ekf_fix: false,
                map_cell: coords,
                earlier_anchor_id: None,
            }),
        }
    }

    fn find_earlier_case_id_other_than(&self, current: FrameKey) -> Option<String> {
        self.frame_case_ids
            .iter()
            .rev()
            .find(|(k, _)| **k != current)
            .map(|(_, id)| id.clone())
    }
}

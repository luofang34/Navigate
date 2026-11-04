//! Map-anchor admission and delayed visual-odometry propagation bridge for [`super::VisualCorridorSession`].

use super::{
    VisualCorridorSession,
    bridge::{SessionAnchorReport, build_estimate_from_candidate},
};
use nalgebra::Matrix3;
use navigate_contract::MonotonicNanos;
use navigate_fusion::IngestOutcome;
use navigate_visual::{
    LocalFrame,
    fusion_bridge::{EvidenceIndependence, VisualFix, position_fix_observation_propagated},
    holdout_pnp::PoseCandidate,
    session::{
        AnchorDecision, AnchorObservation, FrameKey, FusionEligibility, MapCell, MapReliability,
        RegionChange, SurfaceModel, from_camera,
    },
};

impl VisualCorridorSession {
    /// Submit a zero-lag [`PoseCandidate`] anchor on the most recently registered frame.
    ///
    /// # Errors
    /// Returns an error string when no frame has been registered or anchor submission fails.
    pub fn record_anchor_candidate(
        &mut self,
        cand: &PoseCandidate,
        local_frame: LocalFrame,
        catalog_sha256: &str,
        cell_m: f64,
        case_id: &str,
        backend: &str,
    ) -> Result<SessionAnchorReport, String> {
        let t_ns = self.last_capture_ns;
        self.record_propagated_anchor_candidate(
            cand,
            local_frame,
            (catalog_sha256, cell_m, case_id, backend),
            (t_ns, t_ns),
            &Matrix3::zeros(),
        )
    }

    /// Submit a [`PoseCandidate`] anchor captured at `capture_ns` and forward-propagated to `arrival_ns`.
    ///
    /// # Errors
    /// Returns an error string when no frame is registered, timestamps are zero or inverted,
    /// or session submission fails.
    pub fn record_propagated_anchor_candidate(
        &mut self,
        cand: &PoseCandidate,
        local_frame: LocalFrame,
        (catalog_sha256, cell_m, case_id, backend): (&str, f64, &str, &str),
        (capture_ns, arrival_ns): (u64, u64),
        q_vo_enu_m2: &Matrix3<f64>,
    ) -> Result<SessionAnchorReport, String> {
        if capture_ns == 0 || arrival_ns == 0 || arrival_ns < capture_ns {
            return Err(format!(
                "invalid anchor monotonic timestamps capture={capture_ns} arrival={arrival_ns}"
            ));
        }
        let key = self
            .last_registered_key
            .ok_or_else(|| "no registered frame for anchor".to_owned())?;
        let sha = self
            .last_registered_sha256
            .clone()
            .ok_or_else(|| "no registered digest for anchor".to_owned())?;
        self.frame_case_ids.insert(key, case_id.to_owned());

        let session_estimate = build_estimate_from_candidate(
            cand,
            local_frame,
            catalog_sha256,
            &sha,
            key.index,
            self.last_capture_ns,
            backend,
        );
        let fusion_estimate = if capture_ns == self.last_capture_ns {
            session_estimate.clone()
        } else {
            let prop_sha = self.unique_frame_digest(&sha, case_id, key.index ^ 0x5052_4F50);
            build_estimate_from_candidate(
                cand,
                local_frame,
                catalog_sha256,
                &prop_sha,
                key.index,
                capture_ns,
                backend,
            )
        };
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
        let anchor = AnchorObservation::from_estimate(key, &session_estimate, reliability);
        let cur_pose = from_camera(&session_estimate.pose);
        let decision = self
            .session
            .submit_anchor(anchor)
            .map_err(|e| format!("{e:?}"))?;
        let _ = self.drain_session_events();
        self.ensure_ellipsoid(local_frame.reference_ellipsoid());

        let cell = MapCell {
            map_manifest_sha256: catalog_sha256.to_owned(),
            east: (cand.eye[0] / cell_m).floor() as i64,
            north: (cand.eye[1] / cell_m).floor() as i64,
        };
        let coords = Some([cell.east, cell.north]);
        let lag_ns = arrival_ns.saturating_sub(capture_ns);

        match decision {
            AnchorDecision::Accepted { eligibility } => {
                self.last_odom_frame = Some((key, sha, cur_pose));
                match eligibility {
                    FusionEligibility::Independent => {
                        let ind =
                            EvidenceIndependence::ValidatedIndependent.restricted_by(&eligibility);
                        let (ingested, visual_fix) = self.ingest_propagated_fix(
                            &fusion_estimate,
                            ind,
                            arrival_ns,
                            key.index,
                            q_vo_enu_m2,
                        );
                        Ok(SessionAnchorReport {
                            fusion_eligibility: "independent",
                            independent_ekf_fix: ingested,
                            map_cell: coords,
                            earlier_anchor_id: None,
                            visual_fix,
                            capture_lag_ns: lag_ns,
                        })
                    }
                    FusionEligibility::SharedMapError { earlier, .. } => {
                        let earlier_id = self.frame_case_ids.get(&earlier).cloned();
                        Ok(SessionAnchorReport {
                            fusion_eligibility: "shared_map_cell",
                            independent_ekf_fix: false,
                            map_cell: coords,
                            earlier_anchor_id: earlier_id,
                            visual_fix: None,
                            capture_lag_ns: lag_ns,
                        })
                    }
                    FusionEligibility::Untracked => Ok(SessionAnchorReport {
                        fusion_eligibility: "untracked",
                        independent_ekf_fix: false,
                        map_cell: coords,
                        earlier_anchor_id: None,
                        visual_fix: None,
                        capture_lag_ns: lag_ns,
                    }),
                }
            }
            AnchorDecision::Relocated { .. } => {
                self.last_odom_frame = Some((key, sha, cur_pose));
                self.handle_relocation(cand.eye, cand.rotation, 0.0);
                let (ingested, visual_fix) = self.ingest_propagated_fix(
                    &fusion_estimate,
                    EvidenceIndependence::ValidatedIndependent,
                    arrival_ns,
                    key.index,
                    q_vo_enu_m2,
                );
                Ok(SessionAnchorReport {
                    fusion_eligibility: "relocated",
                    independent_ekf_fix: ingested,
                    map_cell: coords,
                    earlier_anchor_id: None,
                    visual_fix,
                    capture_lag_ns: lag_ns,
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
                    visual_fix: None,
                    capture_lag_ns: lag_ns,
                })
            }
            AnchorDecision::Rejected(_) => Ok(SessionAnchorReport {
                fusion_eligibility: "awaiting_consensus",
                independent_ekf_fix: false,
                map_cell: coords,
                earlier_anchor_id: None,
                visual_fix: None,
                capture_lag_ns: lag_ns,
            }),
        }
    }

    fn ingest_propagated_fix(
        &mut self,
        estimate: &navigate_visual::Estimate,
        independence: EvidenceIndependence,
        arrival_ns: u64,
        sequence: u64,
        q_vo_enu_m2: &Matrix3<f64>,
    ) -> (bool, Option<VisualFix>) {
        if let Some(ref att_budget) = self.attitude_budget {
            let q_vo_att_rad2 = if arrival_ns > estimate.stamp.capture_time_ns {
                let dt_s = ((arrival_ns - estimate.stamp.capture_time_ns) as f64) * 1e-9;
                let sigma_rad = (0.005 + 0.04 * dt_s.sqrt()).min(0.35);
                Matrix3::identity() * (sigma_rad * sigma_rad)
            } else {
                Matrix3::zeros()
            };
            let Ok(pose_fix) = self.fix_source.convert_pose_propagated(
                estimate,
                independence,
                att_budget,
                arrival_ns,
                sequence,
                q_vo_enu_m2,
                &q_vo_att_rad2,
            ) else {
                return (false, None);
            };
            let Ok(pos_obs) = position_fix_observation_propagated(
                estimate,
                &self.identity,
                &self.error_budget,
                arrival_ns,
                sequence,
                q_vo_enu_m2,
            ) else {
                return (false, None);
            };
            let accepted = self
                .filter
                .ingest(&pos_obs, MonotonicNanos::from_nanos(arrival_ns))
                == IngestOutcome::Accepted;
            return (accepted, accepted.then_some(pose_fix));
        }
        let Ok(fix) = self.fix_source.convert_propagated(
            estimate,
            independence,
            arrival_ns,
            sequence,
            q_vo_enu_m2,
        ) else {
            return (false, None);
        };
        let accepted = self
            .filter
            .ingest(&fix.observation, MonotonicNanos::from_nanos(arrival_ns))
            == IngestOutcome::Accepted;
        (accepted, accepted.then_some(fix))
    }

    fn find_earlier_case_id_other_than(&self, current: FrameKey) -> Option<String> {
        self.frame_case_ids
            .iter()
            .rev()
            .find(|(k, _)| **k != current)
            .map(|(_, id)| id.clone())
    }
}

//! Per-frame execution step (`process_frame_inner`) for [`super::FlightVisualEngine`].

use super::{
    FlightVisualEngine,
    prep::{
        attitude_override_deg, build_anchor_prior, extract_dead_reckon_step, prior_odometry_hint,
    },
    solve::{FrameSolveInput, solve_frame_pose},
};
use crate::{
    anchor_arrival::AnchorApplyCtx,
    anchor_worker::BackgroundAnchor,
    diagnostics::FrameDiagnostics,
    error::FlightVisualError,
    pixels::{prepare_tracking_buffers, tracking_render_size},
    types::{ExecutionPolicy, FlightFrameOutcome, FrameTelemetry, LiveFrameInput},
};
use navigate_match::PreparedKashinQuery;
use std::time::Instant;

impl FlightVisualEngine {
    pub(super) fn process_frame_inner(
        &mut self,
        frame: &LiveFrameInput<'_>,
        frame_label: &str,
        collect_diag: bool,
    ) -> Result<(FlightFrameOutcome, Option<FrameDiagnostics>), FlightVisualError> {
        let frame_t0 = Instant::now();
        let cur_ns = frame.stamp.acquired_at.as_nanos();
        if cur_ns == 0 {
            return Err(FlightVisualError::NonMonotonicTimestamp(
                "monotonic frame timestamp must be non-zero".into(),
            ));
        }
        if let Some(prev_ns) = self.state.last_monotonic_ns
            && frame.stamp.acquired_at <= prev_ns
        {
            return Err(FlightVisualError::NonMonotonicTimestamp(format!(
                "received {} <= previous {}",
                cur_ns,
                prev_ns.as_nanos()
            )));
        }

        let virtual_now_ms = cur_ns / 1_000_000;
        let polled_outcome = if let Some(worker) = self.background_anchor.as_mut() {
            let out = worker.poll_with_policy(self.config.policy, virtual_now_ms)?;
            if let Some(ref done) = out
                && let Some(locked) = done.locked_focal_px
            {
                self.focal_estimator.locked_focal_px = Some(locked);
            }
            out
        } else {
            None
        };

        let anchor_prior = build_anchor_prior(frame);
        let cell_m = self.config.map_cell_m.clamp(50.0, 20_000.0);
        let (tile_build_ms, sync_built_this_frame) =
            self.step_lookahead_tiles(anchor_prior, virtual_now_ms, collect_diag)?;

        let render_size = tracking_render_size(frame.pixels.dimensions());
        let anchor_busy = polled_outcome.is_none()
            && self
                .background_anchor
                .as_ref()
                .is_some_and(BackgroundAnchor::is_in_flight);
        let est_rot_for_mask = self
            .state
            .prev_rot
            .or(self.state.vo_rot)
            .map(|r| self.state.prev_rot_step * r);
        let is_coreml = self.resources.xfeat.runtime_name() == "coreml";
        let throttle_warm_seg = matches!(self.config.policy, ExecutionPolicy::LiveFlight)
            && !self.state.active_tracks.is_empty()
            && (anchor_busy
                || self.masker.should_throttle_live_warm_refresh_for_runtime(
                    render_size,
                    est_rot_for_mask.as_ref(),
                    is_coreml,
                ));
        let allow_seg = !throttle_warm_seg
            && !(self.masker.has_usable_key(render_size) && anchor_busy)
            && crate::masking::runs_semantic_mask(frame.band);
        let need_rgb = self.masker.needs_rgb_for_step(
            frame.mount.has_airframe_occlusion,
            render_size,
            allow_seg,
        ) || (self.resources.search_tier_matcher.is_some()
            && self.state.active_tracks.is_empty());
        let (qry_gray, small_rgb) = prepare_tracking_buffers(frame.pixels, render_size, need_rgb)?;

        let eff_intr = self.effective_intrinsics(frame.intrinsics);
        let k_render = eff_intr.scaled_to(render_size).to_matrix3();
        self.ensure_session_configured(render_size, &k_render, cell_m)?;
        let skip_prop = is_coreml && !allow_seg && !self.state.active_tracks.is_empty();
        let (mask_small, mask_tel, luma_p50) = self.masker.mask_frame_flight_fast(
            frame.mount.has_airframe_occlusion,
            small_rgb.as_ref(),
            &qry_gray,
            frame.static_valid_mask,
            est_rot_for_mask.as_ref(),
            &k_render,
            allow_seg,
            skip_prop,
            frame.band,
        )?;

        let (qry_feat, query_xfeat_ms, skip_cold_xfeat) = self.extract_query_features(
            &qry_gray,
            &mask_small,
            mask_tel.valid_fraction,
            anchor_busy,
        )?;
        let prepared = PreparedKashinQuery::from_features(&self.resources.projector, &qry_feat)?;

        let (dt_s, discontinuous) =
            self.advance_timing_and_continuity(frame, anchor_prior, frame_label)?;
        if discontinuous || !self.state.active_tracks.is_empty() {
            self.vo_accum_dt_s = 0.0;
        }
        let attitude_override = attitude_override_deg(frame.imu_attitude);
        let (prior_rot_hint, mut nominal_agl_m) = prior_odometry_hint(
            &self.resources.surface,
            &self.state,
            anchor_prior,
            attitude_override,
            &self.masker,
            &k_render,
        );
        self.sync_uncalibrated_regime(
            frame.intrinsics,
            nominal_agl_m,
            attitude_override,
            &k_render,
        );
        self.init_cold_prior_if_needed(frame, anchor_prior, nominal_agl_m);

        if skip_cold_xfeat {
            self.vo_accum_dt_s += dt_s.max(0.0);
            self.state.record_skipped_odometry_step(dt_s);
        } else {
            let vo_dt_s = dt_s + std::mem::take(&mut self.vo_accum_dt_s);
            self.state.record_odometry_step(
                &prepared,
                &k_render,
                prior_rot_hint,
                nominal_agl_m,
                vo_dt_s,
                &self.config.dynamics_profile,
            );
        }
        let dead_reckon = extract_dead_reckon_step(&self.state, &k_render, nominal_agl_m);

        let mut anchor_arrivals = Vec::new();
        let arrived_anchor = polled_outcome.and_then(|outcome| {
            let arrived_at =
                (!discontinuous).then_some((self.state.frame_seq, cur_ns, frame_label));
            let mut ctx = AnchorApplyCtx {
                state: &mut self.state,
                active_status: &mut self.active_status,
                recent_backup: &mut self.recent_backup,
                independent_fixes: &mut self.independent_fixes,
                focal_estimator: &mut self.focal_estimator,
                catalog_sha256: &self.resources.surface.catalog_sha256,
                local_frame: self.resources.local_frame,
                cell_m,
                dynamics_profile: self.config.dynamics_profile,
            };
            let (staged, rec) = ctx.apply(outcome, arrived_at);
            if collect_diag {
                anchor_arrivals.push(rec);
            }
            if discontinuous {
                self.state.reset_discontinuous_track();
            }
            staged
        });

        let had_arrived_anchor = arrived_anchor.is_some();
        let arrived_bg_fix = arrived_anchor
            .as_ref()
            .and_then(|a| a.admission.visual_fix.clone());
        let eff_after = self.effective_intrinsics(frame.intrinsics);
        let k_full_after = eff_after.to_matrix3();
        let k_render_after = eff_after.scaled_to(render_size).to_matrix3();
        if had_arrived_anchor {
            (_, nominal_agl_m) = prior_odometry_hint(
                &self.resources.surface,
                &self.state,
                anchor_prior,
                attitude_override,
                &self.masker,
                &k_render_after,
            );
        }

        let solved = solve_frame_pose(
            self,
            FrameSolveInput {
                frame,
                frame_label,
                virtual_now_ms,
                anchor_prior,
                attitude_override,
                qry_feat,
                prepared: &prepared,
                small_rgb,
                mask_small,
                luma_p50,
                k_full: k_full_after,
                k_render: k_render_after,
                render_size,
                nominal_agl_m,
                dt_s,
                cell_m,
                arrived_anchor,
                collect_diag,
            },
        )?;

        if let Some((_, _, ref cand)) = solved.best
            && cand.score.geometry_accepted
        {
            self.masker.note_solved_rotation(cand.rotation);
            if solved.used_arrived_anchor {
                self.state.prev_prepared = Some(prepared);
            } else if had_arrived_anchor {
                self.state.snap_pose_preserving_velocity(cand, prepared);
            } else {
                self.state
                    .update_motion_state(cand, prepared, dt_s, &self.config.dynamics_profile);
            }
        } else if !skip_cold_xfeat {
            self.state
                .advance_unsolved_prepared(prepared, dt_s, &self.config.dynamics_profile);
        }
        if self.config.background_workers
            && matches!(self.config.policy, ExecutionPolicy::LiveFlight)
            && self.state.frame_seq == 1
        {
            self.masker.enable_background_refresh()?;
        }

        let filter_snapshot = self.state.filter_snapshot(Some(self.resources.local_frame));
        let is_accepted = solved
            .best
            .as_ref()
            .is_some_and(|(_, _, c)| c.score.geometry_accepted);
        let anchor_confidence = is_accepted
            .then(|| self.active_status.map(|s| s.confidence))
            .flatten();
        let visual_fix = solved.admission.visual_fix.or(arrived_bg_fix);
        let total_frame_ms = frame_t0.elapsed().as_secs_f64() * 1000.0;

        let telemetry = FrameTelemetry {
            accepted: is_accepted,
            tracking_mode: solved.admission.tracking_mode,
            fusion_eligibility: solved.admission.fusion_eligibility,
            independent_ekf_fix: solved.admission.independent_ekf_fix,
            map_cell: solved.admission.map_cell,
            active_tracks: self.state.active_tracks.len() as u32,
            cached_tiles: self.resources.store.cached_tile_count() as u32,
            resident_window_bytes: self.resources.store.resident_window_bytes(),
            query_keypoints: solved.qry_feat_len as u32,
            lighterglue_layers: solved.lg_layers.min(u16::MAX as usize) as u16,
            polish_rays_cast: solved.polish_rays as u32,
            segformer_refreshed: mask_tel.segformer_refreshed,
            segformer_submitted: mask_tel.segformer_submitted,
            segformer_ms: mask_tel.segformer_ms,
            valid_mask_fraction: mask_tel.valid_fraction as f32,
            propeller_pixels: mask_tel.propeller_pixels as u32,
            luma_p50,
            mask_ms: mask_tel.mask_ms,
            query_xfeat_ms,
            solve_ms: solved.solve_ms,
            total_frame_ms,
            capture_lag_ns: solved.admission.capture_lag_ns,
            worker_health: self.worker_health(),
        };
        let outcome = FlightFrameOutcome {
            visual_fix,
            anchor_confidence,
            dead_reckon,
            telemetry,
        };
        let diag = collect_diag.then_some(FrameDiagnostics {
            candidate: solved.best,
            render_size,
            earlier_anchor_id: solved.admission.earlier_anchor_id,
            locked_focal_px: self.focal_estimator.locked_focal_px,
            filter_snapshot,
            attempts: solved.attempts,
            anchor_arrivals,
            tile_build_ms,
            sync_tiles_built: sync_built_this_frame,
        });
        Ok((outcome, diag))
    }
}

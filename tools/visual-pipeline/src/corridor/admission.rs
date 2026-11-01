//! Track seeding, [`MapCell`] EKF independence gating, and report emission for `corridor-track`.

use super::{
    CorridorTrackOptions,
    masking::AmortizedMasker,
    motion::TrackState,
    session::{FilterIntegritySnapshot, SessionAnchorReport},
};
use crate::{
    PipelineError,
    metadata::PositionClaim,
    pose::{PoseCandidate, save_candidate_npz},
    realtime::helpers::rot_to_row_major,
    surface::Surface,
    testset::PhotoCase,
    util::write_json,
};
use nalgebra::Matrix3;
use navigate_match::{
    ActiveTrackSet, CorridorTileStore, HadamardProjector, PreparedKashinQuery, SparseFeatures,
    VirtualCorridorView, VirtualViewCamera,
};
use navigate_visual_session::MapCell;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

pub(super) const TOP_UP_FRUSTUM_MARGIN_PX: f64 = 48.0;
const TOP_UP_DEDUP_METRES: f64 = 1.5;

#[derive(Clone, Debug)]
pub(super) struct FusionAdmission {
    pub tracking_mode: &'static str,
    pub evidence_kind: &'static str,
    pub independent_ekf_fix: bool,
    pub fusion_eligibility: &'static str,
    pub map_cell: Option<[i64; 2]>,
    pub earlier_anchor_id: Option<String>,
}

impl FusionAdmission {
    pub(super) fn warm_track() -> Self {
        Self {
            tracking_mode: "warm_track",
            evidence_kind: "surface_track_odometry",
            independent_ekf_fix: false,
            fusion_eligibility: "correlated_track",
            map_cell: None,
            earlier_anchor_id: None,
        }
    }

    pub(super) fn acquiring_anchor() -> Self {
        Self {
            tracking_mode: "acquiring_anchor",
            evidence_kind: "visual_odometry",
            independent_ekf_fix: false,
            fusion_eligibility: "acquiring_anchor",
            map_cell: None,
            earlier_anchor_id: None,
        }
    }

    pub(super) fn unresolved() -> Self {
        Self {
            tracking_mode: "unresolved",
            evidence_kind: "corridor_map_anchor",
            independent_ekf_fix: false,
            fusion_eligibility: "rejected",
            map_cell: None,
            earlier_anchor_id: None,
        }
    }

    pub(super) fn from_session_report(report: SessionAnchorReport) -> Self {
        Self {
            tracking_mode: "corridor_anchor",
            evidence_kind: "corridor_map_anchor",
            independent_ekf_fix: report.independent_ekf_fix,
            fusion_eligibility: report.fusion_eligibility,
            map_cell: report.map_cell,
            earlier_anchor_id: report.earlier_anchor_id,
        }
    }
}

pub(super) fn estimate_lookahead_enu(
    surface: &Surface,
    state: &TrackState,
    case: &PhotoCase,
) -> Option<[f64; 2]> {
    if let Some(eye) = state.prev_eye {
        let mut fwd = [state.prev_vel[0] * 2.0, state.prev_vel[1] * 2.0];
        if let Some(rot) = state.prev_rot {
            let look = (eye[2].abs() * 1.4).clamp(200.0, 2_000.0);
            fwd[0] += rot[(2, 0)] * look;
            fwd[1] += rot[(2, 1)] * look;
        }
        return Some([eye[0] + fwd[0], eye[1] + fwd[1]]);
    }
    let claims = case.prior.as_ref()?;
    let pos = claims.position.as_ref()?;
    let enu = surface.local(pos.latitude, pos.longitude, 0.0);
    if let Some(hdg_deg) = claims.direction_deg {
        let dem = surface
            .sample_dem(pos.latitude, pos.longitude)
            .unwrap_or(0.0);
        let agl = pos
            .altitude_m
            .map_or(480.0, |alt| (alt - dem).clamp(25.0, 4_000.0));
        let stand_off = (agl * 1.4).clamp(150.0, 2_000.0);
        let rad = hdg_deg.to_radians();
        return Some([
            enu[0] + stand_off * rad.sin(),
            enu[1] + stand_off * rad.cos(),
        ]);
    }
    Some([enu[0], enu[1]])
}

pub(super) fn prior_odometry_hint(
    surface: &Surface,
    state: &TrackState,
    case: &PhotoCase,
    masker: &AmortizedMasker,
    k_render: &Matrix3<f64>,
) -> (Option<Matrix3<f64>>, f64) {
    let solved_agl = state.prev_eye.map(|eye| {
        let geo = surface.geodetic(eye);
        let dem = surface.sample_dem(geo[0], geo[1]).unwrap_or(0.0);
        (geo[2] - dem).clamp(20.0, 6_000.0)
    });
    let Some(claims) = case.prior.as_ref() else {
        return (None, solved_agl.unwrap_or(480.0));
    };
    let Some(pos) = claims.position.as_ref() else {
        return (None, solved_agl.unwrap_or(480.0));
    };
    let dem = surface
        .sample_dem(pos.latitude, pos.longitude)
        .unwrap_or(0.0);
    let agl = solved_agl.unwrap_or_else(|| {
        pos.altitude_m
            .map_or(480.0, |alt| (alt - dem).clamp(25.0, 6_000.0))
    });
    let (down_deg, roll_deg) = masker.horizon_attitude(k_render).unwrap_or((55.0, 0.0));
    let rot = claims
        .direction_deg
        .map(|hdg| surface.camera_rotation(pos.latitude, pos.longitude, hdg, down_deg, roll_deg));
    (rot, agl)
}

pub(super) fn apply_case_overrides(case: &PhotoCase, opts: &CorridorTrackOptions) -> PhotoCase {
    let mut out = case.clone();
    if opts.prior_override.is_none() && opts.focal_35mm_override.is_none() {
        return out;
    }
    let mut claims = out.prior.unwrap_or_default();
    if let Some(([lat, lon, alt], heading)) = opts.prior_override {
        claims.position = Some(PositionClaim {
            latitude: lat,
            longitude: lon,
            reported_horizontal_error_m: None,
            altitude_m: Some(alt),
            unsigned_altitude_m: Some(alt.abs()),
            altitude_reference_text: None,
            altitude_datum: "override_unverified".into(),
        });
        if let Some(hdg) = heading {
            claims.direction_deg = Some(hdg.rem_euclid(360.0));
        }
    }
    if let Some(f35) = opts.focal_35mm_override {
        claims.focal_35mm_mm = Some(f35);
    }
    out.prior = Some(claims);
    out
}

pub(super) fn seed_tracks(
    surface: &Surface,
    prepared: &PreparedKashinQuery,
    qry_feat: &SparseFeatures,
    cand: &PoseCandidate,
    virt_opt: &Option<VirtualCorridorView>,
) -> ActiveTrackSet {
    let seed_cap = qry_feat.pixels.len().min(1024);
    let hits = surface.raycast_pixels(
        cand.eye,
        &cand.rotation,
        &cand.k,
        &qry_feat.pixels[..seed_cap],
    );
    let world_opts: Vec<Option<[f64; 3]>> =
        hits.into_iter().map(|h| h.map(|x| x.world_enu)).collect();
    let mut set = ActiveTrackSet::from_prepared_query(prepared, &world_opts, 1024);
    if let Some(virt) = virt_opt {
        set.extend_capped(
            &virt.world_points_enu,
            &virt.sign_bits,
            &virt.kashin_i8,
            2048,
        );
    }
    set
}

pub(super) fn top_up_tracks(
    tracks: &mut ActiveTrackSet,
    surface: &Surface,
    prepared: &PreparedKashinQuery,
    qry_feat: &SparseFeatures,
    cand: &PoseCandidate,
    render_size: [u32; 2],
) {
    let intrinsics = [
        cand.k[(0, 0)],
        cand.k[(1, 1)],
        cand.k[(0, 2)],
        cand.k[(1, 2)],
    ];
    tracks.retain_in_frustum(
        cand.eye,
        &rot_to_row_major(&cand.rotation),
        intrinsics,
        render_size,
        TOP_UP_FRUSTUM_MARGIN_PX,
    );
    let sin_down = -cand.rotation[(2, 2)];
    let ray_cap = if sin_down < 0.08 {
        qry_feat.pixels.len().min(128)
    } else {
        qry_feat.pixels.len().min(384)
    };
    let hits = surface.raycast_pixels(
        cand.eye,
        &cand.rotation,
        &cand.k,
        &qry_feat.pixels[..ray_cap],
    );
    let world_opts: Vec<Option<[f64; 3]>> =
        hits.into_iter().map(|h| h.map(|x| x.world_enu)).collect();
    let fresh = ActiveTrackSet::from_prepared_query(prepared, &world_opts, 384);
    extend_dedup_enu(tracks, &fresh, TOP_UP_DEDUP_METRES, 2048);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn top_up_tracks_with_bank(
    tracks: &mut ActiveTrackSet,
    surface: &Surface,
    store: &CorridorTileStore,
    projector: &HadamardProjector,
    prepared: &PreparedKashinQuery,
    qry_feat: &SparseFeatures,
    cand: &PoseCandidate,
    render_size: [u32; 2],
) {
    top_up_tracks(tracks, surface, prepared, qry_feat, cand, render_size);
    let sin_down = -cand.rotation[(2, 2)];
    let steep_nadir = sin_down > 0.85;
    let intrinsics = [
        cand.k[(0, 0)],
        cand.k[(1, 1)],
        cand.k[(0, 2)],
        cand.k[(1, 2)],
    ];
    let _ = projector;
    if let Ok(virt) = store.project_virtual_kashin(&VirtualViewCamera {
        eye_enu: cand.eye,
        world_to_cam: rot_to_row_major(&cand.rotation),
        intrinsics,
        image_size: render_size,
        max_landmarks: 512,
        min_cos_incidence: if steep_nadir {
            0.45
        } else if sin_down < 0.25 {
            0.72
        } else {
            0.84
        },
        min_cos_orientation: if sin_down > 0.95 { 0.94 } else { 0.80 },
    }) && !virt.world_points_enu.is_empty()
    {
        let bank_fresh = ActiveTrackSet {
            world_points_enu: virt.world_points_enu,
            sign_bits: virt.sign_bits,
            kashin_i8: virt.kashin_i8,
        };
        extend_dedup_enu(tracks, &bank_fresh, TOP_UP_DEDUP_METRES, 2048);
    }
}

fn extend_dedup_enu(
    tracks: &mut ActiveTrackSet,
    fresh: &ActiveTrackSet,
    min_spacing_m: f64,
    max_tracks: usize,
) {
    let cell = min_spacing_m.max(0.1);
    let min_sq = min_spacing_m * min_spacing_m;
    let voxel = |p: [f64; 3]| {
        (
            (p[0] / cell).floor() as i64,
            (p[1] / cell).floor() as i64,
            (p[2] / cell).floor() as i64,
        )
    };
    let mut occupied: HashMap<(i64, i64, i64), Vec<[f64; 3]>> =
        HashMap::with_capacity(tracks.len() + fresh.len());
    for &p in &tracks.world_points_enu {
        if p.iter().all(|v| v.is_finite()) {
            occupied.entry(voxel(p)).or_default().push(p);
        }
    }

    let mut add_world = Vec::with_capacity(fresh.len());
    let mut add_signs = Vec::with_capacity(fresh.len());
    let mut add_codes = Vec::with_capacity(fresh.len());
    let count = fresh
        .world_points_enu
        .len()
        .min(fresh.sign_bits.len())
        .min(fresh.kashin_i8.len());

    for i in 0..count {
        if tracks.len() + add_world.len() >= max_tracks {
            break;
        }
        let p = fresh.world_points_enu[i];
        if !p.iter().all(|v| v.is_finite()) {
            continue;
        }
        let (vx, vy, vz) = voxel(p);
        let mut near = false;
        'neighbors: for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(bucket) = occupied.get(&(vx + dx, vy + dy, vz + dz))
                        && bucket.iter().any(|q| {
                            let de = p[0] - q[0];
                            let dn = p[1] - q[1];
                            let du = p[2] - q[2];
                            de * de + dn * dn + du * du < min_sq
                        })
                    {
                        near = true;
                        break 'neighbors;
                    }
                }
            }
        }
        if !near {
            occupied.entry((vx, vy, vz)).or_default().push(p);
            add_world.push(p);
            add_signs.push(fresh.sign_bits[i]);
            add_codes.push(fresh.kashin_i8[i]);
        }
    }
    tracks.extend_capped(&add_world, &add_signs, &add_codes, max_tracks);
}

pub(super) fn classify_anchor_cell(
    visited: &mut BTreeMap<MapCell, String>,
    independent_fixes: &mut usize,
    catalog_sha256: &str,
    eye: [f64; 3],
    cell_m: f64,
    case_id: &str,
) -> FusionAdmission {
    let cell = MapCell {
        map_manifest_sha256: catalog_sha256.to_owned(),
        east: (eye[0] / cell_m).floor() as i64,
        north: (eye[1] / cell_m).floor() as i64,
    };
    let coords = Some([cell.east, cell.north]);
    let (independent_ekf_fix, fusion_eligibility, earlier_anchor_id) =
        if let Some(earlier) = visited.get(&cell).cloned() {
            (false, "shared_map_cell", Some(earlier))
        } else {
            visited.insert(cell, case_id.to_owned());
            *independent_fixes += 1;
            (true, "independent", None)
        };
    FusionAdmission {
        tracking_mode: "corridor_anchor",
        evidence_kind: "corridor_map_anchor",
        independent_ekf_fix,
        fusion_eligibility,
        map_cell: coords,
        earlier_anchor_id,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_corridor_report(
    output_dir: &Path,
    surface: &Surface,
    case_id: &str,
    adm: &FusionAdmission,
    filter_snap: &FilterIntegritySnapshot,
    attempts: Vec<Value>,
    best: Option<(usize, usize, PoseCandidate)>,
    accepted_count: &mut usize,
) -> Result<(), PipelineError> {
    let filter_integrity = json!({
        "quality": filter_snap.quality,
        "redundancy": filter_snap.redundancy,
        "fault_detection": filter_snap.fault_detection,
        "horizontal_1sigma_m": filter_snap.horizontal_1sigma_m,
        "vertical_1sigma_m": filter_snap.vertical_1sigma_m,
    });
    let report = if let Some((seed, step, cand)) = best
        && cand.score.geometry_accepted
    {
        *accepted_count += 1;
        let case_dir = output_dir.join(case_id);
        fs::create_dir_all(&case_dir).map_err(|source| PipelineError::Io {
            path: case_dir.clone(),
            source,
        })?;
        save_candidate_npz(&case_dir.join("candidate.npz"), &cand)?;
        let [lat, lon, alt] = surface.geodetic(cand.eye);
        json!({
            "id": case_id,
            "status": "candidate_requires_review",
            "attempts": attempts,
            "filter_integrity": filter_integrity,
            "session_revision": filter_snap.session_revision,
            "trajectory_revisions": filter_snap.trajectory_revisions,
            "candidate": {
                "latitude": lat,
                "longitude": lon,
                "altitude_m": alt,
                "evidence_kind": adm.evidence_kind,
                "tracking_mode": adm.tracking_mode,
                "independent_ekf_fix": adm.independent_ekf_fix,
                "fusion_eligibility": adm.fusion_eligibility,
                "map_cell": adm.map_cell,
                "earlier_anchor_id": adm.earlier_anchor_id,
                "seed": seed,
                "step": step,
                "score": cand.score,
            },
        })
    } else {
        let status = if adm.tracking_mode == "acquiring_anchor" {
            "acquiring_anchor"
        } else {
            "unresolved"
        };
        json!({
            "id": case_id,
            "status": status,
            "tracking_mode": adm.tracking_mode,
            "evidence_kind": adm.evidence_kind,
            "fusion_eligibility": adm.fusion_eligibility,
            "filter_integrity": filter_integrity,
            "session_revision": filter_snap.session_revision,
            "trajectory_revisions": filter_snap.trajectory_revisions,
            "attempts": attempts,
            "candidate": null,
        })
    };
    write_json(&output_dir.join(format!("{case_id}.json")), &report)
}

#[cfg(test)]
mod tests;

//! Track seeding, [`MapCell`] EKF independence gating, and report emission for `corridor-track`.

use crate::{
    PipelineError,
    pose::{PoseCandidate, save_candidate_npz},
    surface::Surface,
    util::write_json,
};
use navigate_match::{ActiveTrackSet, PreparedKashinQuery, SparseFeatures, VirtualCorridorView};
use navigate_visual_session::MapCell;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};

pub(super) struct FusionAdmission {
    pub tracking_mode: &'static str,
    pub evidence_kind: &'static str,
    pub independent_ekf_fix: bool,
    pub fusion_eligibility: &'static str,
    pub map_cell: Option<[i64; 2]>,
    pub earlier_anchor_id: Option<String>,
}

pub(super) fn seed_tracks(
    surface: &Surface,
    prepared: &PreparedKashinQuery,
    qry_feat: &SparseFeatures,
    cand: &PoseCandidate,
    virt_opt: &Option<VirtualCorridorView>,
) -> ActiveTrackSet {
    let hits = surface.raycast_pixels(cand.eye, &cand.rotation, &cand.k, &qry_feat.pixels);
    let world_opts: Vec<Option<[f64; 3]>> =
        hits.into_iter().map(|h| h.map(|x| x.world_enu)).collect();
    let mut set = ActiveTrackSet::from_prepared_query(prepared, &world_opts, 1536);
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
) {
    let hits = surface.raycast_pixels(cand.eye, &cand.rotation, &cand.k, &qry_feat.pixels);
    let world_opts: Vec<Option<[f64; 3]>> =
        hits.into_iter().map(|h| h.map(|x| x.world_enu)).collect();
    let fresh = ActiveTrackSet::from_prepared_query(prepared, &world_opts, 768);
    tracks.extend_capped(
        &fresh.world_points_enu,
        &fresh.sign_bits,
        &fresh.kashin_i8,
        2048,
    );
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

pub(super) fn write_corridor_report(
    output_dir: &Path,
    surface: &Surface,
    case_id: &str,
    adm: &FusionAdmission,
    attempts: Vec<Value>,
    best: Option<(usize, usize, PoseCandidate)>,
    accepted_count: &mut usize,
) -> Result<(), PipelineError> {
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
        json!({
            "id": case_id,
            "status": "unresolved",
            "attempts": attempts,
            "candidate": null,
        })
    };
    write_json(&output_dir.join(format!("{case_id}.json")), &report)
}

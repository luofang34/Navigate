//! Binary `NVKTILE1` encoder, decoder, and frustum geometry helpers for [`CorridorLandmarkBank`].

use super::{
    CORRIDOR_MAGIC, CORRIDOR_VERSION, CorridorLandmarkBank, CorridorTileEntry, MAX_TILE_LANDMARKS,
    MAX_TILES, VirtualViewCamera,
};
use crate::{MatchError, ViewSketch};
use sha2::{Digest, Sha256};

const HEADER_BYTES: usize = 92;
const CHECKSUM_BYTES: usize = 32;
const TILE_HEADER_BYTES: usize = 4 + 4 + 24 + 12 + 12 + ViewSketch::SERIALIZED_BYTES + 4;
const LANDMARK_BYTES: usize = 12 + 3 + 3 + 2 + 1 + 8 + 64;

pub(super) fn ensure(ok: bool, msg: &'static str) -> Result<(), MatchError> {
    ok.then_some(()).ok_or(MatchError::Output(msg))
}

pub(super) fn encode_corridor_bank(bank: &CorridorLandmarkBank) -> Vec<u8> {
    let tile_count = bank.tiles.len().min(MAX_TILES);
    let lm_total: usize = bank.tiles.iter().take(tile_count).map(|t| t.len()).sum();
    let cap =
        HEADER_BYTES + tile_count * TILE_HEADER_BYTES + lm_total * LANDMARK_BYTES + CHECKSUM_BYTES;
    let mut out = Vec::with_capacity(cap);
    out.extend_from_slice(CORRIDOR_MAGIC);
    out.extend_from_slice(&CORRIDOR_VERSION.to_le_bytes());
    out.extend_from_slice(&bank.catalog_sha256);
    out.extend_from_slice(&bank.encoder_contract);
    out.extend_from_slice(&bank.seed.to_le_bytes());
    out.extend_from_slice(&bank.tile_size_m.to_le_bytes());
    out.extend_from_slice(&(tile_count as u32).to_le_bytes());

    for tile in bank.tiles.iter().take(tile_count) {
        write_tile(&mut out, tile);
    }

    let digest: [u8; 32] = Sha256::digest(&out).into();
    out.extend_from_slice(&digest);
    out
}

pub(super) fn write_tile(out: &mut Vec<u8>, tile: &CorridorTileEntry) {
    let n = tile.len().min(MAX_TILE_LANDMARKS);
    out.extend_from_slice(&tile.tile_x.to_le_bytes());
    out.extend_from_slice(&tile.tile_y.to_le_bytes());
    for &c in &tile.anchor_enu {
        out.extend_from_slice(&c.to_le_bytes());
    }
    for &c in tile.bounds_min_rel.iter().chain(&tile.bounds_max_rel) {
        out.extend_from_slice(&c.to_le_bytes());
    }
    out.extend_from_slice(&tile.sketch.to_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());

    for &c in tile.rel_enu.iter().take(n).flatten() {
        out.extend_from_slice(&c.to_le_bytes());
    }
    for &d_xyz in tile.obs_dirs_i8.iter().take(n) {
        out.extend_from_slice(&d_xyz.map(|v| v as u8));
    }
    for &x_xyz in tile.cam_x_i8.iter().take(n) {
        out.extend_from_slice(&x_xyz.map(|v| v as u8));
    }
    for &dm in tile.ref_dist_m.iter().take(n) {
        out.extend_from_slice(&dm.to_le_bytes());
    }
    out.extend_from_slice(&tile.scores_u8[..n]);
    for &s in tile.sign_bits.iter().take(n) {
        out.extend_from_slice(&s.to_le_bytes());
    }
    for code in tile.kashin_i8.iter().take(n) {
        out.extend_from_slice(code);
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn encode_tile_shard(magic: &[u8; 8], tile: &CorridorTileEntry) -> Vec<u8> {
    let n = tile.len().min(MAX_TILE_LANDMARKS);
    let mut out = Vec::with_capacity(12 + TILE_HEADER_BYTES + n * LANDMARK_BYTES + CHECKSUM_BYTES);
    out.extend_from_slice(magic);
    out.extend_from_slice(&CORRIDOR_VERSION.to_le_bytes());
    write_tile(&mut out, tile);
    let digest: [u8; 32] = Sha256::digest(&out).into();
    out.extend_from_slice(&digest);
    out
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn decode_tile_shard(
    magic: &[u8; 8],
    bytes: &[u8],
) -> Result<CorridorTileEntry, MatchError> {
    ensure(bytes.len() >= 12 + CHECKSUM_BYTES, "truncated tile shard")?;
    let (payload, trailer) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);
    let mut cur = payload;
    ensure(&take::<8>(&mut cur)? == magic, "tile shard magic mismatch")?;
    ensure(u32_le(&mut cur)? == CORRIDOR_VERSION, "bad shard version")?;
    let digest: [u8; 32] = Sha256::digest(payload).into();
    ensure(digest.as_slice() == trailer, "tile shard checksum mismatch")?;
    let tile = read_tile(&mut cur)?;
    ensure(cur.is_empty(), "tile shard has trailing bytes")?;
    Ok(tile)
}

pub(super) fn decode_corridor_bank(bytes: &[u8]) -> Result<CorridorLandmarkBank, MatchError> {
    ensure(bytes.len() >= HEADER_BYTES + CHECKSUM_BYTES, "truncated")?;
    let (payload, trailer) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);
    let mut cur = payload;
    ensure(
        &take::<8>(&mut cur)? == CORRIDOR_MAGIC,
        "corridor magic mismatch",
    )?;
    ensure(
        u32_le(&mut cur)? == CORRIDOR_VERSION,
        "bad corridor version",
    )?;
    let digest: [u8; 32] = Sha256::digest(payload).into();
    ensure(digest.as_slice() == trailer, "corridor checksum mismatch")?;

    let catalog_sha256 = take(&mut cur)?;
    let encoder_contract = take(&mut cur)?;
    let seed = u64::from_le_bytes(take(&mut cur)?);
    let tile_size_m = f32::from_le_bytes(take(&mut cur)?);
    let tile_count = u32_le(&mut cur)? as usize;
    let fits = tile_count <= MAX_TILES && tile_count.saturating_mul(TILE_HEADER_BYTES) <= cur.len();
    ensure(fits, "corridor tile count exceeds buffer")?;
    let tiles = many(tile_count, || read_tile(&mut cur))?;
    ensure(cur.is_empty(), "corridor bank has trailing bytes")?;
    CorridorLandmarkBank::new(catalog_sha256, encoder_contract, seed, tile_size_m, tiles)
}

fn take<const N: usize>(cur: &mut &[u8]) -> Result<[u8; N], MatchError> {
    let (&chunk, rest) = cur
        .split_first_chunk::<N>()
        .ok_or(MatchError::Output("truncated"))?;
    *cur = rest;
    Ok(chunk)
}

fn u32_le(cur: &mut &[u8]) -> Result<u32, MatchError> {
    Ok(u32::from_le_bytes(take(cur)?))
}

fn i32_le(cur: &mut &[u8]) -> Result<i32, MatchError> {
    Ok(i32::from_le_bytes(take(cur)?))
}

fn f32_le(cur: &mut &[u8]) -> Result<f32, MatchError> {
    Ok(f32::from_le_bytes(take(cur)?))
}

fn f64_le(cur: &mut &[u8]) -> Result<f64, MatchError> {
    Ok(f64::from_le_bytes(take(cur)?))
}

fn many<T>(n: usize, mut f: impl FnMut() -> Result<T, MatchError>) -> Result<Vec<T>, MatchError> {
    (0..n).map(|_| f()).collect()
}

fn read_tile(cur: &mut &[u8]) -> Result<CorridorTileEntry, MatchError> {
    let tile_x = i32_le(cur)?;
    let tile_y = i32_le(cur)?;
    let anchor_enu = [f64_le(cur)?, f64_le(cur)?, f64_le(cur)?];
    let bounds_min_rel = [f32_le(cur)?, f32_le(cur)?, f32_le(cur)?];
    let bounds_max_rel = [f32_le(cur)?, f32_le(cur)?, f32_le(cur)?];
    let sketch = ViewSketch::from_bytes(take(cur)?);
    let n = u32_le(cur)? as usize;
    ensure(n <= MAX_TILE_LANDMARKS, "tile landmark limit")?;
    ensure(n.saturating_mul(LANDMARK_BYTES) <= cur.len(), "truncated")?;

    let rel_enu = many(n, || Ok([f32_le(cur)?, f32_le(cur)?, f32_le(cur)?]))?;
    let obs_dirs_i8 = many(n, || Ok(take::<3>(cur)?.map(|v| v as i8)))?;
    let cam_x_i8 = many(n, || Ok(take::<3>(cur)?.map(|v| v as i8)))?;
    let ref_dist_m = many(n, || Ok(u16::from_le_bytes(take(cur)?)))?;
    let scores_u8 = {
        let (slice, rest) = cur.split_at(n);
        *cur = rest;
        slice.to_vec()
    };
    let sign_bits = many(n, || Ok(u64::from_le_bytes(take(cur)?)))?;
    let kashin_i8 = many(n, || take(cur))?;

    let entry = CorridorTileEntry {
        tile_x,
        tile_y,
        anchor_enu,
        bounds_min_rel,
        bounds_max_rel,
        sketch,
        rel_enu,
        obs_dirs_i8,
        cam_x_i8,
        ref_dist_m,
        scores_u8,
        sign_bits,
        kashin_i8,
    };
    entry.validate().map(|()| entry)
}

pub(super) fn quantize_unit_i8(n: [f32; 3], fallback: [i8; 3]) -> [i8; 3] {
    let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if !norm.is_finite() || norm <= 1e-6 {
        return fallback;
    }
    n.map(|c| ((c / norm) * 127.0).round().clamp(-127.0, 127.0) as i8)
}

pub(super) fn unit_cosine_i8(a_i8: [i8; 3], b: [f64; 3]) -> f32 {
    let a = a_i8.map(f64::from);
    let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if na <= 1e-6 || nb <= 1e-6 {
        return 1.0;
    }
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    ((dot / (na * nb)) as f32).clamp(-1.0, 1.0)
}

pub(super) fn incidence_cosine(obs_i8: [i8; 3], eye_to_pt: [f64; 3]) -> f32 {
    let n = obs_i8.map(f64::from);
    let n_norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let v_norm =
        (eye_to_pt[0] * eye_to_pt[0] + eye_to_pt[1] * eye_to_pt[1] + eye_to_pt[2] * eye_to_pt[2])
            .sqrt();
    if n_norm <= 1e-6 || v_norm <= 1e-6 {
        return 1.0;
    }
    let dot = -(n[0] * eye_to_pt[0] + n[1] * eye_to_pt[1] + n[2] * eye_to_pt[2]);
    ((dot / (n_norm * v_norm)) as f32).clamp(-1.0, 1.0)
}

pub(super) fn scale_similarity(
    ref_dist_m: u16,
    eye_to_pt: [f64; 3],
    fx: f64,
    _is_nadir: bool,
) -> Option<f32> {
    if ref_dist_m == 0 {
        return Some(1.0);
    }
    let d_live =
        (eye_to_pt[0] * eye_to_pt[0] + eye_to_pt[1] * eye_to_pt[1] + eye_to_pt[2] * eye_to_pt[2])
            .sqrt();
    let d_ref = (f64::from(ref_dist_m) * (fx / 600.0).max(0.1)).max(1.0);
    let ratio = d_live / d_ref;
    let bounds = 0.22..=5.50;
    if !bounds.contains(&ratio) {
        return None;
    }
    let ls = ratio.ln() as f32;
    Some((-0.9 * ls * ls).exp())
}

pub(super) fn has_nearby_pixel(
    grid: &[Option<[f64; 2]>],
    gw: usize,
    gh: usize,
    px: [f64; 2],
    min_dist_sq: f64,
) -> bool {
    let cx = ((px[0] as u32) >> 2) as usize;
    let cy = ((px[1] as u32) >> 2) as usize;
    for ny in cy.saturating_sub(1)..=(cy + 1).min(gh.saturating_sub(1)) {
        for nx in cx.saturating_sub(1)..=(cx + 1).min(gw.saturating_sub(1)) {
            if let Some(prev) = grid[ny * gw + nx] {
                let dx = prev[0] - px[0];
                let dy = prev[1] - px[1];
                if dx * dx + dy * dy < min_dist_sq {
                    return true;
                }
            }
        }
    }
    false
}

pub(super) fn tile_overlaps_frustum(tile: &CorridorTileEntry, cam: &VirtualViewCamera) -> bool {
    let [fx, fy, cx, cy] = cam.intrinsics;
    let s_left = -cx / fx;
    let s_right = (f64::from(cam.image_size[0]) - cx) / fx;
    let s_top = -cy / fy;
    let s_bottom = (f64::from(cam.image_size[1]) - cy) / fy;

    let xs = [tile.bounds_min_rel[0], tile.bounds_max_rel[0]];
    let ys = [tile.bounds_min_rel[1], tile.bounds_max_rel[1]];
    let zs = [tile.bounds_min_rel[2], tile.bounds_max_rel[2]];
    let mut corners = [[0.0_f64; 3]; 8];
    let mut idx = 0_usize;
    for &dx in &xs {
        for &dy in &ys {
            for &dz in &zs {
                let rel_eye = [
                    tile.anchor_enu[0] + f64::from(dx) - cam.eye_enu[0],
                    tile.anchor_enu[1] + f64::from(dy) - cam.eye_enu[1],
                    tile.anchor_enu[2] + f64::from(dz) - cam.eye_enu[2],
                ];
                corners[idx] = mat3_mul(&cam.world_to_cam, rel_eye);
                idx += 1;
            }
        }
    }
    if corners.iter().all(|c| c[2] <= 0.5) || corners.iter().all(|c| c[2] > 60_000.0) {
        return false;
    }
    if corners.iter().all(|c| c[0] - s_left * c[2] < 0.0)
        || corners.iter().all(|c| s_right * c[2] - c[0] < 0.0)
        || corners.iter().all(|c| c[1] - s_top * c[2] < 0.0)
        || corners.iter().all(|c| s_bottom * c[2] - c[1] < 0.0)
    {
        return false;
    }
    true
}

pub(super) fn mat3_mul(m: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

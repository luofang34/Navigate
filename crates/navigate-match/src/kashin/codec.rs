//! Binary `NVKBANK1` encoder and decoder for [`PrecomputedReferenceBank`].

use super::{
    KashinDescriptor, ViewSketch,
    bank::{
        BANK_MAGIC, BANK_VERSION, MAX_KPS, MAX_VIEWS, PrecomputedReferenceBank,
        PrecomputedViewEntry,
    },
    unpack_kashin_i8,
};
use crate::MatchError;
use sha2::{Digest, Sha256};

const HEADER_BYTES: usize = 88;
const CHECKSUM_BYTES: usize = 32;
const MIN_ENTRY_BYTES: usize = 4 + 24 + 72 + 8 + ViewSketch::SERIALIZED_BYTES + 4 + 1;

const FLAG_HAS_WORLD: u8 = 1 << 0;
const FLAG_HAS_I8: u8 = 1 << 1;
const FLAG_COMPACT_DESC: u8 = 1 << 2;
const FLAG_REL_WORLD_F32: u8 = 1 << 3;

pub(super) fn encode_rel_world_f32(eye_enu: [f64; 3], world_enu: [f64; 3]) -> Option<[f32; 3]> {
    let rel = [
        (world_enu[0] - eye_enu[0]) as f32,
        (world_enu[1] - eye_enu[1]) as f32,
        (world_enu[2] - eye_enu[2]) as f32,
    ];
    rel.iter().all(|v| v.is_finite()).then_some(rel)
}

pub(super) fn decode_rel_world_f32(eye_enu: [f64; 3], rel: [f32; 3]) -> [f64; 3] {
    [
        eye_enu[0] + f64::from(rel[0]),
        eye_enu[1] + f64::from(rel[1]),
        eye_enu[2] + f64::from(rel[2]),
    ]
}

pub(super) fn descriptor_from_i8_and_sign(sign_bits: u64, code: &[u8; 64]) -> KashinDescriptor {
    let mut desc = KashinDescriptor::from_rotated(&unpack_kashin_i8(code));
    desc.sign_bits = sign_bits;
    desc
}

pub(super) fn encode_bank(bank: &PrecomputedReferenceBank) -> Vec<u8> {
    let count = bank.entries.len().min(MAX_VIEWS);
    let mut out = Vec::with_capacity(HEADER_BYTES + count * MIN_ENTRY_BYTES + CHECKSUM_BYTES);
    out.extend_from_slice(BANK_MAGIC);
    out.extend_from_slice(&BANK_VERSION.to_le_bytes());
    out.extend_from_slice(&bank.catalog_sha256);
    out.extend_from_slice(&bank.encoder_contract);
    out.extend_from_slice(&bank.seed.to_le_bytes());
    out.extend_from_slice(&(count as u32).to_le_bytes());

    for e in bank.entries.iter().take(count) {
        let n = e
            .pixels
            .len()
            .min(e.descriptors.len())
            .min(e.sign_bits.len())
            .min(MAX_KPS);
        let has_w = n > 0 && e.world_points_enu.len() >= n;
        let has_i8 = n > 0 && e.kashin_i8.len() >= n;
        let rel_world_ok = has_w
            && e.world_points_enu.iter().take(n).all(|&w| {
                encode_rel_world_f32(e.eye_enu, w)
                    .is_some_and(|rel| decode_rel_world_f32(e.eye_enu, rel) == w)
            });
        let compact_desc_ok = has_i8
            && (0..n).all(|i| {
                descriptor_from_i8_and_sign(e.sign_bits[i], &e.kashin_i8[i]) == e.descriptors[i]
            });

        let flags = (if has_w { FLAG_HAS_WORLD } else { 0 })
            | (if has_i8 { FLAG_HAS_I8 } else { 0 })
            | (if compact_desc_ok {
                FLAG_COMPACT_DESC
            } else {
                0
            })
            | (if rel_world_ok { FLAG_REL_WORLD_F32 } else { 0 });

        out.extend_from_slice(&e.id.to_le_bytes());
        for &c in e.eye_enu.iter().chain(&e.world_to_cam) {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for &d in &e.image_size {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&e.sketch.to_bytes());
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.push(flags);

        for &c in e.pixels.iter().take(n).flatten() {
            out.extend_from_slice(&c.to_le_bytes());
        }
        if has_w {
            if rel_world_ok {
                for &w in e.world_points_enu.iter().take(n) {
                    let rel = encode_rel_world_f32(e.eye_enu, w).unwrap_or([0.0; 3]);
                    for c in rel {
                        out.extend_from_slice(&c.to_le_bytes());
                    }
                }
            } else {
                for &c in e.world_points_enu.iter().take(n).flatten() {
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        if compact_desc_ok {
            for &sign in e.sign_bits.iter().take(n) {
                out.extend_from_slice(&sign.to_le_bytes());
            }
        } else {
            for desc in e.descriptors.iter().take(n) {
                out.extend_from_slice(&desc.to_bytes());
            }
        }
        if has_i8 {
            for code in e.kashin_i8.iter().take(n) {
                out.extend_from_slice(code);
            }
        }
    }

    let digest: [u8; 32] = Sha256::digest(&out).into();
    out.extend_from_slice(&digest);
    out
}

pub(super) fn decode_bank(bytes: &[u8]) -> Result<PrecomputedReferenceBank, MatchError> {
    ensure(bytes.len() >= HEADER_BYTES + CHECKSUM_BYTES, "truncated")?;
    let (payload, trailer) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);
    let mut cur = payload;
    ensure(&take::<8>(&mut cur)? == BANK_MAGIC, "bank magic mismatch")?;
    ensure(u32_le(&mut cur)? == BANK_VERSION, "bad bank version")?;
    let digest: [u8; 32] = Sha256::digest(payload).into();
    ensure(digest.as_slice() == trailer, "bank checksum mismatch")?;
    let (catalog_sha256, encoder_contract) = (take(&mut cur)?, take(&mut cur)?);
    let seed = u64::from_le_bytes(take(&mut cur)?);
    let count = u32_le(&mut cur)? as usize;
    let fits = count <= MAX_VIEWS && count.saturating_mul(MIN_ENTRY_BYTES) <= cur.len();
    ensure(fits, "bank entry count exceeds buffer")?;
    let entries = many(count, || read_entry(&mut cur))?;
    ensure(cur.is_empty(), "kashin bank has trailing bytes")?;
    Ok(PrecomputedReferenceBank {
        catalog_sha256,
        encoder_contract,
        seed,
        entries,
    })
}

pub(super) fn ensure(ok: bool, msg: &'static str) -> Result<(), MatchError> {
    ok.then_some(()).ok_or(MatchError::Output(msg))
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

fn read_f64s<const N: usize>(cur: &mut &[u8]) -> Result<[f64; N], MatchError> {
    let mut out = [0.0_f64; N];
    for dst in &mut out {
        *dst = f64::from_le_bytes(take(cur)?);
    }
    Ok(out)
}

fn many<T>(n: usize, mut f: impl FnMut() -> Result<T, MatchError>) -> Result<Vec<T>, MatchError> {
    (0..n).map(|_| f()).collect()
}

fn read_entry(cur: &mut &[u8]) -> Result<PrecomputedViewEntry, MatchError> {
    let (id, eye_enu, world_to_cam) = (u32_le(cur)?, read_f64s::<3>(cur)?, read_f64s::<9>(cur)?);
    let image_size = [u32_le(cur)?, u32_le(cur)?];
    let sketch = ViewSketch::from_bytes(take(cur)?);
    let (n, flags) = (u32_le(cur)? as usize, take::<1>(cur)?[0]);
    let has_w = flags & FLAG_HAS_WORLD != 0;
    let has_i8 = flags & FLAG_HAS_I8 != 0;
    let compact_desc = flags & FLAG_COMPACT_DESC != 0;
    let rel_world = flags & FLAG_REL_WORLD_F32 != 0;
    let flags_ok = flags <= 0x0F
        && (!compact_desc || has_i8)
        && (!rel_world || has_w)
        && (n > 0 || flags == 0);
    ensure(n <= MAX_KPS && flags_ok, "bad header")?;

    let world_bytes = if !has_w {
        0
    } else if rel_world {
        12
    } else {
        24
    };
    let desc_bytes = if compact_desc {
        8
    } else {
        KashinDescriptor::SERIALIZED_BYTES
    };
    let per_kp = 8 + world_bytes + desc_bytes + if has_i8 { 64 } else { 0 };
    ensure(n.saturating_mul(per_kp) <= cur.len(), "truncated")?;

    let f32_le = |c: &mut &[u8]| take(c).map(f32::from_le_bytes);
    let pixels = many(n, || Ok([f32_le(cur)?, f32_le(cur)?]))?;
    let world_points_enu = if !has_w {
        Vec::new()
    } else if rel_world {
        many(n, || {
            let rel = [f32_le(cur)?, f32_le(cur)?, f32_le(cur)?];
            Ok(decode_rel_world_f32(eye_enu, rel))
        })?
    } else {
        many(n, || read_f64s::<3>(cur))?
    };

    let (raw_signs, full_descs) = if compact_desc {
        let signs = many(n, || Ok(u64::from_le_bytes(take(cur)?)))?;
        (signs, Vec::new())
    } else {
        let descs = many(n, || KashinDescriptor::from_bytes(take(cur)?))?;
        let signs = descs.iter().map(|d| d.sign_bits).collect();
        (signs, descs)
    };
    let kashin_i8 = many(if has_i8 { n } else { 0 }, || take(cur))?;
    let descriptors = if compact_desc {
        raw_signs
            .iter()
            .zip(&kashin_i8)
            .map(|(&s, code)| descriptor_from_i8_and_sign(s, code))
            .collect()
    } else {
        full_descs
    };

    let entry = PrecomputedViewEntry {
        id,
        eye_enu,
        world_to_cam,
        image_size,
        sketch,
        pixels,
        world_points_enu,
        sign_bits: raw_signs,
        descriptors,
        kashin_i8,
    };
    entry.validate().map(|()| entry)
}

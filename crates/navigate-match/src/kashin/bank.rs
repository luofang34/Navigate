//! Offline reference feature bank and fast Kashin 3D-2D matcher.

use super::{
    HadamardProjector, KashinDescriptor, ViewSketch, pack_kashin_i8, reconstruct_sparse_features,
    sign_bits64, try_normalize64,
};
use crate::{MatchError, SparseFeatures};
use navigate_visual::PixelMatch;
use sha2::{Digest, Sha256};

/// Eight-byte magic header of a serialized [`PrecomputedReferenceBank`].
pub const BANK_MAGIC: &[u8; 8] = b"NVKBANK1";
/// Binary format version of [`PrecomputedReferenceBank`].
pub const BANK_VERSION: u32 = 1;

const HEADER_BYTES: usize = 88;
const CHECKSUM_BYTES: usize = 32;
const MIN_ENTRY_BYTES: usize = 4 + 24 + 72 + 8 + ViewSketch::SERIALIZED_BYTES + 4 + 1;
const MAX_VIEWS: usize = 65_536;
const MAX_KPS: usize = 16_384;

/// Correspondence between a [`PrecomputedViewEntry`] keypoint and a query keypoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrecomputedMatch {
    /// Keypoint pixel coordinates `[x, y]` in the reference view.
    pub reference_pixel: [f64; 2],
    /// Keypoint pixel coordinates `[x, y]` in the query image.
    pub query_pixel: [f64; 2],
    /// Optional 3D terrain point `[east, north, up]` in local East-North-Up (ENU) meters.
    pub world_point_enu: Option<[f64; 3]>,
    /// Asymmetric 2-bit Kashin similarity score.
    pub score: f32,
}

impl PrecomputedMatch {
    /// Convert this match into a [`PixelMatch`].
    #[must_use]
    pub fn to_pixel_match(&self) -> PixelMatch {
        let (reference, query) = (self.reference_pixel.into(), self.query_pixel.into());
        PixelMatch { reference, query }
    }

    /// Convert a slice of matches into [`PixelMatch`] pairs.
    #[must_use]
    pub fn to_pixel_matches(matches: &[Self]) -> Vec<PixelMatch> {
        matches.iter().map(Self::to_pixel_match).collect()
    }
}

impl From<PrecomputedMatch> for PixelMatch {
    fn from(value: PrecomputedMatch) -> Self {
        value.to_pixel_match()
    }
}

/// Precomputed keypoints, optional 3D terrain points, and Kashin descriptors for one view.
#[derive(Clone, Debug, PartialEq)]
pub struct PrecomputedViewEntry {
    /// Reference view identifier in the catalog.
    pub id: u32,
    /// Camera center `[east, north, up]` in ENU meters.
    pub eye_enu: [f64; 3],
    /// Row-major `3 x 3` rotation matrix from ENU world coordinates to camera coordinates.
    pub world_to_cam: [f64; 9],
    /// Reference image dimensions `[width, height]` in pixels.
    pub image_size: [u32; 2],
    /// Global 160-byte [`ViewSketch`] for candidate pose pre-screening.
    pub sketch: ViewSketch,
    /// Keypoint coordinates `[x, y]` in reference image pixels.
    pub pixels: Vec<[f32; 2]>,
    /// Optional 3D terrain points `[east, north, up]` in ENU meters.
    pub world_points_enu: Vec<[f64; 3]>,
    /// Compressed 28-byte [`KashinDescriptor`] list (`pixels.len()` elements).
    pub descriptors: Vec<KashinDescriptor>,
    /// Optional bounded 64-byte rotated Kashin `i8` codes (`4x` compressed vs `f32`).
    pub kashin_i8: Vec<[u8; 64]>,
}

impl PrecomputedViewEntry {
    /// Encode one [`SparseFeatures`] view into a [`PrecomputedViewEntry`].
    pub fn from_features(
        id: u32,
        eye_enu: [f64; 3],
        world_to_cam: [f64; 9],
        image_size: [u32; 2],
        projector: &HadamardProjector,
        features: &SparseFeatures,
        world_points_enu: Vec<[f64; 3]>,
    ) -> Result<Self, MatchError> {
        let n = features.pixels.len();
        let ok = n <= MAX_KPS
            && features.descriptors.len() == n * 64
            && (features.scores.is_empty() || features.scores.len() == n);
        ensure(ok, "invalid sparse features")?;
        let px = |p: &[f64; 2]| p.map(|v| v as f32);
        let pixels = features.pixels.iter().map(px).collect();
        let mut descriptors = Vec::with_capacity(n);
        let mut kashin_i8 = Vec::with_capacity(n);
        for chunk in features.descriptors.as_chunks::<64>().0 {
            let u = try_normalize64(chunk).ok_or(MatchError::Output("invalid descriptor"))?;
            let rotated = projector.project(&u);
            descriptors.push(KashinDescriptor::from_rotated(&rotated));
            kashin_i8.push(pack_kashin_i8(&rotated));
        }
        let sketch = ViewSketch::from_features(projector, features);
        let entry = Self {
            id,
            eye_enu,
            world_to_cam,
            image_size,
            sketch,
            pixels,
            world_points_enu,
            descriptors,
            kashin_i8,
        };
        entry.validate().map(|()| entry)
    }

    /// Reconstruct [`SparseFeatures`] from Kashin codes for [`LighterGlue`](crate::LighterGlue).
    #[must_use]
    pub fn to_sparse_features(&self, projector: &HadamardProjector) -> SparseFeatures {
        reconstruct_sparse_features(
            self.image_size,
            &self.pixels,
            &self.descriptors,
            &self.kashin_i8,
            projector,
        )
    }

    fn validate(&self) -> Result<(), MatchError> {
        let n = self.pixels.len();
        let pose_ok = self
            .eye_enu
            .iter()
            .chain(&self.world_to_cam)
            .all(|v| v.is_finite());
        let world_ok = self.world_points_enu.is_empty() || self.world_points_enu.len() == n;
        let i8_ok = self.kashin_i8.is_empty() || self.kashin_i8.len() == n;
        let pix_ok = self.pixels.iter().flatten().all(|v| v.is_finite());
        let v_ok = self
            .descriptors
            .iter()
            .flat_map(|d| &d.polytope_vertices)
            .all(|&v| v < 32);
        let geom_ok = self.image_size[0] > 0 && self.image_size[1] > 0 && pose_ok && pix_ok;
        ensure(
            geom_ok && n <= MAX_KPS && self.descriptors.len() == n && world_ok && i8_ok && v_ok,
            "bad entry",
        )
    }
}

/// Precomputed bank of [`PrecomputedViewEntry`] records for one encoder contract.
#[derive(Clone, Debug, PartialEq)]
pub struct PrecomputedReferenceBank {
    /// SHA-256 digest of the source terrain and imagery catalog.
    pub catalog_sha256: [u8; 32],
    /// SHA-256 digest of the [`XFeat`](crate::XFeat) encoder contract.
    pub encoder_contract: [u8; 32],
    /// Seed of the [`HadamardProjector`] for all entries in this bank.
    pub seed: u64,
    /// Precomputed reference view entries.
    pub entries: Vec<PrecomputedViewEntry>,
}

impl PrecomputedReferenceBank {
    /// Create and validate a [`PrecomputedReferenceBank`].
    pub fn new(
        catalog_sha256: [u8; 32],
        encoder_contract: [u8; 32],
        seed: u64,
        entries: Vec<PrecomputedViewEntry>,
    ) -> Result<Self, MatchError> {
        let bank = Self {
            catalog_sha256,
            encoder_contract,
            seed,
            entries,
        };
        bank.check().map(|()| bank)
    }

    fn check(&self) -> Result<(), MatchError> {
        ensure(self.entries.len() <= MAX_VIEWS, "bank entry limit")?;
        self.entries.iter().try_for_each(|e| e.validate())
    }

    /// Serialize this bank into deterministic `NVKBANK1` bytes with a 32-byte SHA-256 checksum.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let count = self.entries.len().min(MAX_VIEWS);
        let mut out = Vec::with_capacity(HEADER_BYTES + count * MIN_ENTRY_BYTES + CHECKSUM_BYTES);
        out.extend_from_slice(BANK_MAGIC);
        out.extend_from_slice(&BANK_VERSION.to_le_bytes());
        out.extend_from_slice(&self.catalog_sha256);
        out.extend_from_slice(&self.encoder_contract);
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&(count as u32).to_le_bytes());
        for e in self.entries.iter().take(count) {
            let n = e.pixels.len().min(e.descriptors.len()).min(MAX_KPS);
            let has_w = n > 0 && e.world_points_enu.len() >= n;
            let has_i8 = n > 0 && e.kashin_i8.len() >= n;
            let flags = u8::from(has_w) | (u8::from(has_i8) << 1);
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
                for &c in e.world_points_enu.iter().take(n).flatten() {
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
            for desc in e.descriptors.iter().take(n) {
                out.extend_from_slice(&desc.to_bytes());
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

    /// Decode and verify a `NVKBANK1` byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MatchError> {
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
        Ok(Self {
            catalog_sha256,
            encoder_contract,
            seed,
            entries,
        })
    }

    /// Write this bank to `path` on native targets.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_file(&self, path: &std::path::Path) -> Result<(), MatchError> {
        self.check()?;
        std::fs::write(path, self.to_bytes()).map_err(|e| MatchError::Device {
            stage: "kashin bank write",
            message: e.to_string(),
        })
    }

    /// Read a [`PrecomputedReferenceBank`] from `path` on native targets.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_file(path: &std::path::Path) -> Result<Self, MatchError> {
        let bytes = std::fs::read(path).map_err(|e| MatchError::Device {
            stage: "kashin bank read",
            message: e.to_string(),
        })?;
        Self::from_bytes(&bytes)
    }

    /// Return at most `top_k` `(index, score)` entry pairs ranked by [`ViewSketch::similarity`].
    #[must_use]
    pub fn shortlist(&self, query: &SparseFeatures, top_k: usize) -> Vec<(usize, f32)> {
        if top_k == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let q_sketch = ViewSketch::from_features(&HadamardProjector::new(self.seed), query);
        let mut ranked: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| (idx, entry.sketch.similarity(&q_sketch)))
            .collect();
        let cmp =
            |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
        if top_k < ranked.len() {
            ranked.select_nth_unstable_by(top_k - 1, cmp);
            ranked.truncate(top_k);
        }
        ranked.sort_by(cmp);
        ranked
    }

    /// Match `query` features against `entry_index` with Hamming, Kashin, mutual-nearest, and ratio gates.
    pub fn match_entry(
        &self,
        entry_index: usize,
        query: &SparseFeatures,
        hamming_gate: u32,
        min_dot: f32,
        ratio_limit: f32,
    ) -> Result<Vec<PrecomputedMatch>, MatchError> {
        let entry = self
            .entries
            .get(entry_index)
            .ok_or(MatchError::Output("bad index"))?;
        entry.validate()?;
        ensure(
            min_dot.is_finite() && ratio_limit.is_finite() && ratio_limit > 0.0,
            "gate",
        )?;
        let q_count = query.pixels.len();
        let q_ok = query.descriptors.len() == q_count * 64
            && query.pixels.iter().flatten().all(|v| v.is_finite());
        ensure(q_ok, "query feature buffer is invalid")?;
        if entry.pixels.is_empty() || q_count == 0 {
            return Ok(Vec::new());
        }

        let proj = HadamardProjector::new(self.seed);
        let q_chunks = &query.descriptors.as_chunks::<64>().0[..q_count];
        let queries: Vec<_> = q_chunks
            .iter()
            .map(|c| {
                try_normalize64(c)
                    .map(|u| proj.project(&u))
                    .map(|r| (sign_bits64(&r), r))
            })
            .collect();

        let mut ref_best = vec![Nearest::EMPTY; entry.pixels.len()];
        let mut qry_best = vec![Nearest::EMPTY; q_count];
        for (i, ref_desc) in entry.descriptors.iter().enumerate() {
            for (j, slot) in queries.iter().enumerate() {
                if let Some((q_sign, q_rot)) = slot {
                    let hd = (ref_desc.sign_bits ^ *q_sign).count_ones();
                    if hd > hamming_gate {
                        ref_best[i].3 = ref_best[i].3.min(hd);
                        qry_best[j].3 = qry_best[j].3.min(hd);
                    } else {
                        let score = ref_desc.approx_dot_query(q_rot);
                        ref_best[i].update(j, score);
                        qry_best[j].update(i, score);
                    }
                }
            }
        }

        let r2 = ratio_limit * ratio_limit;
        let mut out = Vec::new();
        for (i, fwd) in ref_best.iter().enumerate() {
            let ok = qry_best.get(fwd.0).is_some_and(|rev| {
                rev.0 == i && fwd.accepts(min_dot, r2) && rev.accepts(min_dot, r2)
            });
            if ok {
                let world_point_enu = entry
                    .world_points_enu
                    .get(i)
                    .copied()
                    .filter(|pt| pt.iter().all(|v| v.is_finite()));
                out.push(PrecomputedMatch {
                    reference_pixel: entry.pixels[i].map(f64::from),
                    query_pixel: query.pixels[fwd.0],
                    world_point_enu,
                    score: fwd.1,
                });
            }
        }
        Ok(out)
    }
}

fn ensure(ok: bool, msg: &'static str) -> Result<(), MatchError> {
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
    ensure(
        n <= MAX_KPS && flags <= 3 && (n > 0 || flags == 0),
        "bad header",
    )?;
    let (has_w, has_i8) = (flags & 1 != 0, flags & 2 != 0);
    let per_kp = 8
        + KashinDescriptor::SERIALIZED_BYTES
        + if has_w { 24 } else { 0 }
        + if has_i8 { 64 } else { 0 };
    ensure(n.saturating_mul(per_kp) <= cur.len(), "truncated")?;
    let f32_le = |c: &mut &[u8]| take(c).map(f32::from_le_bytes);
    let pixels = many(n, || Ok([f32_le(cur)?, f32_le(cur)?]))?;
    let world_points_enu = many(if has_w { n } else { 0 }, || read_f64s::<3>(cur))?;
    let descriptors = many(n, || KashinDescriptor::from_bytes(take(cur)?))?;
    let kashin_i8 = many(if has_i8 { n } else { 0 }, || take(cur))?;
    let entry = PrecomputedViewEntry {
        id,
        eye_enu,
        world_to_cam,
        image_size,
        sketch,
        pixels,
        world_points_enu,
        descriptors,
        kashin_i8,
    };
    entry.validate().map(|()| entry)
}

#[derive(Clone, Copy)]
struct Nearest(usize, f32, f32, u32);

impl Nearest {
    const EMPTY: Self = Self(usize::MAX, -2.0, -2.0, u32::MAX);

    fn update(&mut self, index: usize, score: f32) {
        if score > self.1 {
            (self.2, self.1, self.0) = (self.1, score, index);
        } else if score > self.2 {
            self.2 = score;
        }
    }

    fn accepts(self, min_dot: f32, ratio_sq: f32) -> bool {
        if self.0 == usize::MAX || self.1 < min_dot {
            return false;
        }
        let runner = if self.3 <= 64 {
            self.2
                .max((std::f32::consts::PI * (self.3 as f32) / 64.0).cos())
        } else {
            self.2
        };
        runner <= -1.5 || (1.0 - self.1).max(0.0) <= ratio_sq * (1.0 - runner).max(0.0)
    }
}

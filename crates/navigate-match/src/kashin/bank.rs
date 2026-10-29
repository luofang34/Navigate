//! Offline reference feature bank and fast Kashin 3D-2D matcher.

use super::{
    HadamardProjector, KashinDescriptor, PreparedKashinQuery, ViewSketch,
    codec::{
        decode_bank, decode_rel_world_f32, descriptor_from_i8_and_sign, encode_bank,
        encode_rel_world_f32, ensure,
    },
    pack_kashin_i8, reconstruct_sparse_features, sign_bits64, try_normalize64,
};
use crate::{MatchError, SparseFeatures};
use navigate_visual::PixelMatch;

/// Eight-byte magic header of a serialized [`PrecomputedReferenceBank`].
pub const BANK_MAGIC: &[u8; 8] = b"NVKBANK1";
/// Binary format version of [`PrecomputedReferenceBank`].
pub const BANK_VERSION: u32 = 1;

pub(super) const MAX_VIEWS: usize = 65_536;
pub(super) const MAX_KPS: usize = 16_384;

/// Correspondence between a [`PrecomputedViewEntry`] keypoint and a query keypoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrecomputedMatch {
    /// Keypoint pixel coordinates `[x, y]` in the reference view.
    pub reference_pixel: [f64; 2],
    /// Keypoint pixel coordinates `[x, y]` in the query image.
    pub query_pixel: [f64; 2],
    /// Optional 3D terrain point `[east, north, up]` in local East-North-Up (ENU) meters.
    pub world_point_enu: Option<[f64; 3]>,
    /// Kashin similarity score (`[-1.0, 1.0]`).
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
    /// Contiguous 64-bit sign sketches (`pixels.len()` elements) for Hamming gating.
    pub sign_bits: Vec<u64>,
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
        let pixels = features
            .pixels
            .iter()
            .map(|p| p.map(|v| v as f32))
            .collect();
        let mut sign_bits = Vec::with_capacity(n);
        let mut descriptors = Vec::with_capacity(n);
        let mut kashin_i8 = Vec::with_capacity(n);
        for chunk in features.descriptors.as_chunks::<64>().0 {
            let u = try_normalize64(chunk).ok_or(MatchError::Output("invalid descriptor"))?;
            let rotated = projector.project(&u);
            let sign = sign_bits64(&rotated);
            let code = pack_kashin_i8(&rotated);
            sign_bits.push(sign);
            descriptors.push(descriptor_from_i8_and_sign(sign, &code));
            kashin_i8.push(code);
        }
        let world_points_enu = world_points_enu
            .into_iter()
            .map(|w| {
                encode_rel_world_f32(eye_enu, w).map_or(w, |rel| decode_rel_world_f32(eye_enu, rel))
            })
            .collect();
        let sketch = ViewSketch::from_features(projector, features);
        let entry = Self {
            id,
            eye_enu,
            world_to_cam,
            image_size,
            sketch,
            pixels,
            world_points_enu,
            sign_bits,
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

    pub(super) fn validate(&self) -> Result<(), MatchError> {
        let n = self.pixels.len();
        let pose_ok = self
            .eye_enu
            .iter()
            .chain(&self.world_to_cam)
            .all(|v| v.is_finite());
        let world_ok = (self.world_points_enu.is_empty() || self.world_points_enu.len() == n)
            && self
                .world_points_enu
                .iter()
                .flatten()
                .all(|v| v.is_finite());
        let i8_ok = self.kashin_i8.is_empty() || self.kashin_i8.len() == n;
        let pix_ok = self.pixels.iter().flatten().all(|v| v.is_finite());
        let sign_ok = self.sign_bits.len() == n
            && self
                .sign_bits
                .iter()
                .zip(&self.descriptors)
                .all(|(&s, d)| s == d.sign_bits);
        let v_ok = self
            .descriptors
            .iter()
            .flat_map(|d| &d.polytope_vertices)
            .all(|&v| v < 32);
        let geom_ok = self.image_size[0] > 0 && self.image_size[1] > 0 && pose_ok && pix_ok;
        ensure(
            geom_ok
                && n <= MAX_KPS
                && self.descriptors.len() == n
                && sign_ok
                && world_ok
                && i8_ok
                && v_ok,
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

    pub(super) fn check(&self) -> Result<(), MatchError> {
        ensure(self.entries.len() <= MAX_VIEWS, "bank entry limit")?;
        self.entries.iter().try_for_each(|e| e.validate())
    }

    /// Serialize this bank into deterministic `NVKBANK1` bytes with a 32-byte SHA-256 checksum.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        encode_bank(self)
    }

    /// Decode and verify a `NVKBANK1` byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MatchError> {
        decode_bank(bytes)
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
        self.shortlist_sketch(&q_sketch, top_k)
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
        let prepared =
            PreparedKashinQuery::from_features(&HadamardProjector::new(self.seed), query)?;
        self.match_prepared(entry_index, &prepared, hamming_gate, min_dot, ratio_limit)
    }
}

//! Fast Walsh-Hadamard Kashin descriptor sketches and precomputed feature banks.
//!
//! [`HadamardProjector`] rotates 64-D [`SparseFeatures`] descriptors on the stack.
//! [`KashinDescriptor`] compresses each unit descriptor into 28 bytes.
//! [`ViewSketch`] summarizes one feature set in 160 bytes.
//! [`PrecomputedReferenceBank`] stores offline reference views and 3D terrain points.

mod bank;
mod codec;
pub(crate) mod query;

pub use bank::{
    BANK_MAGIC, BANK_VERSION, PrecomputedMatch, PrecomputedReferenceBank, PrecomputedViewEntry,
};
pub use query::PreparedKashinQuery;

use crate::{MatchError, SparseFeatures};

/// Default seed for [`HadamardProjector`] and [`PrecomputedReferenceBank`].
pub const DEFAULT_KASHIN_SEED: u64 = 0x4E56_4B41_5348_494E;
/// Coordinate standard deviation (`1 / sqrt(64)`) of a rotated unit 64-D vector.
pub const KASHIN_SIGMA: f32 = 0.125;

const RAW_2BIT_LEVELS: [f32; 4] = [-3.0, -1.0, 1.0, 3.0];
const SCALED_2BIT_LEVELS: [f32; 4] = [-0.1875, -0.0625, 0.0625, 0.1875];
const KASHIN_I8_CLIP: f32 = 3.5 * KASHIN_SIGMA;
const EVEN_BITS_U128: u128 = 0x5555_5555_5555_5555_5555_5555_5555_5555;

/// Apply one normalized 64-point Fast Walsh-Hadamard Transform in place (`scale = 0.125`).
pub fn fwht64(values: &mut [f32; 64]) {
    let mut step = 1usize;
    while step < 64 {
        for base in (0..64).step_by(step << 1) {
            for offset in 0..step {
                let (a, b) = (values[base + offset], values[base + offset + step]);
                values[base + offset] = a + b;
                values[base + offset + step] = a - b;
            }
        }
        step <<= 1;
    }
    for value in values.iter_mut() {
        *value *= KASHIN_SIGMA;
    }
}

/// Deterministic `64 x 64` orthogonal structured rotation on stack arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HadamardProjector {
    masks: [u64; 3],
    seed: u64,
}

impl Default for HadamardProjector {
    fn default() -> Self {
        Self::new(DEFAULT_KASHIN_SEED)
    }
}

impl HadamardProjector {
    /// Create a deterministic projector from `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        let (s1, m0) = splitmix64(seed);
        let (s2, m1) = splitmix64(s1);
        let (_, m2) = splitmix64(s2);
        Self {
            masks: [m0, m1, m2],
            seed,
        }
    }

    /// Return the seed that generates the Rademacher sign masks.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Return the three 64-bit Rademacher sign masks.
    #[must_use]
    pub const fn masks(&self) -> [u64; 3] {
        self.masks
    }

    /// Rotate `values` in place with three Rademacher sign masks and [`fwht64`] rounds.
    pub fn project_in_place(&self, values: &mut [f32; 64]) {
        for &mask in &self.masks {
            apply_rademacher64(values, mask);
            fwht64(values);
        }
    }

    /// Return a rotated copy of `descriptor`.
    #[must_use]
    pub fn project(&self, descriptor: &[f32; 64]) -> [f32; 64] {
        let mut out = *descriptor;
        self.project_in_place(&mut out);
        out
    }

    /// Apply the inverse rotation to `rotated`.
    #[must_use]
    pub fn unproject(&self, rotated: &[f32; 64]) -> [f32; 64] {
        let mut out = *rotated;
        for &mask in self.masks.iter().rev() {
            fwht64(&mut out);
            apply_rademacher64(&mut out, mask);
        }
        out
    }
}

/// Compact 28-byte Kashin sketch of one unit 64-D descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KashinDescriptor {
    /// Sign sketch where bit `i` is `1` when `(Q u)[i] >= 0.0`.
    pub sign_bits: u64,
    /// Symmetric 2-bit Kashin codes (4 coordinates per byte, 64 coordinates total).
    pub packed_2bit: [u8; 16],
    /// Cross-Polytope vertex index (`0..32`) for each of the four 16-D blocks.
    pub polytope_vertices: [u8; 4],
}

impl KashinDescriptor {
    /// Serialized size of one [`KashinDescriptor`] in bytes (`8 + 16 + 4`).
    pub const SERIALIZED_BYTES: usize = 28;

    /// Encode one 64-D `descriptor` into 28 bytes with `projector`.
    #[must_use]
    pub fn encode(projector: &HadamardProjector, descriptor: &[f32; 64]) -> Self {
        let unit = try_normalize64(descriptor).unwrap_or([0.0_f32; 64]);
        Self::from_rotated(&projector.project(&unit))
    }

    /// Quantize a rotated 64-D vector into 28 bytes.
    #[must_use]
    pub fn from_rotated(rotated: &[f32; 64]) -> Self {
        let mut packed_2bit = [0u8; 16];
        for (byte_idx, slot) in packed_2bit.iter_mut().enumerate() {
            let base = byte_idx << 2;
            let mut packed = 0u8;
            for sub in 0..4 {
                let v = rotated[base + sub];
                let c =
                    u8::from(v >= -KASHIN_SIGMA) + u8::from(v >= 0.0) + u8::from(v >= KASHIN_SIGMA);
                packed |= c << (sub << 1);
            }
            *slot = packed;
        }
        let polytope_vertices =
            [0, 16, 32, 48].map(|s| octahedron_vertex(&rotated[s..s + 16]).0 as u8);
        Self {
            sign_bits: sign_bits64(rotated),
            packed_2bit,
            polytope_vertices,
        }
    }

    /// Return the Hamming distance (`0..=64`) between two sign sketches.
    #[must_use]
    pub const fn hamming_distance(&self, other: &Self) -> u32 {
        (self.sign_bits ^ other.sign_bits).count_ones()
    }

    /// Compute the cosine similarity estimate `cos(pi * d / 64)` from `sign_bits`.
    #[must_use]
    pub fn hamming_similarity(&self, other: &Self) -> f32 {
        (std::f32::consts::PI * (self.hamming_distance(other) as f32) / 64.0).cos()
    }

    /// Unpack the 64 coordinates to `{-1.5*sigma, -0.5*sigma, +0.5*sigma, +1.5*sigma}`.
    #[must_use]
    pub fn unpack_levels(&self) -> [f32; 64] {
        let mut out = [0.0_f32; 64];
        for (byte_idx, &packed) in self.packed_2bit.iter().enumerate() {
            let base = byte_idx << 2;
            for sub in 0..4 {
                out[base + sub] = SCALED_2BIT_LEVELS[usize::from((packed >> (sub << 1)) & 0b11)];
            }
        }
        out
    }

    /// Compute the symmetric cosine similarity between two 2-bit Kashin descriptors.
    #[must_use]
    pub fn approx_dot(&self, other: &Self) -> f32 {
        let (wa, wb) = (
            u128::from_le_bytes(self.packed_2bit),
            u128::from_le_bytes(other.packed_2bit),
        );
        let (sa, sb) = ((wa >> 1) & EVEN_BITS_U128, (wb >> 1) & EVEN_BITS_U128);
        let (ma, mb) = (
            (!((wa >> 1) ^ wa)) & EVEN_BITS_U128,
            (!((wb >> 1) ^ wb)) & EVEN_BITS_U128,
        );
        let neg = sa ^ sb;
        let pos = (!neg) & EVEN_BITS_U128;
        let sgn = |m: u128| ((m & pos).count_ones() as i32) - ((m & neg).count_ones() as i32);
        let raw =
            (64 - 2 * (neg.count_ones() as i32) + 2 * sgn(ma) + 2 * sgn(mb) + 4 * sgn(ma & mb))
                as f32;
        let denom =
            (((64 + 8 * ma.count_ones()) as f32) * ((64 + 8 * mb.count_ones()) as f32)).sqrt();
        (raw / denom).clamp(-1.0, 1.0)
    }

    /// Compute the asymmetric inner product between this descriptor and `rotated_query`.
    #[must_use]
    pub fn approx_dot_query(&self, rotated_query: &[f32; 64]) -> f32 {
        let mut sum = 0.0_f32;
        for (byte_idx, &packed) in self.packed_2bit.iter().enumerate() {
            let b = byte_idx << 2;
            sum += RAW_2BIT_LEVELS[usize::from(packed & 3)] * rotated_query[b]
                + RAW_2BIT_LEVELS[usize::from((packed >> 2) & 3)] * rotated_query[b + 1]
                + RAW_2BIT_LEVELS[usize::from((packed >> 4) & 3)] * rotated_query[b + 2]
                + RAW_2BIT_LEVELS[usize::from((packed >> 6) & 3)] * rotated_query[b + 3];
        }
        let w = u128::from_le_bytes(self.packed_2bit);
        let outer = ((!((w >> 1) ^ w)) & EVEN_BITS_U128).count_ones();
        sum / ((64 + 8 * outer) as f32).sqrt()
    }

    /// Encode this descriptor into 28 little-endian bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_BYTES] {
        let mut out = [0u8; Self::SERIALIZED_BYTES];
        out[0..8].copy_from_slice(&self.sign_bits.to_le_bytes());
        out[8..24].copy_from_slice(&self.packed_2bit);
        out[24..28].copy_from_slice(&self.polytope_vertices);
        out
    }

    /// Decode 28 bytes into a [`KashinDescriptor`].
    pub fn from_bytes(bytes: [u8; Self::SERIALIZED_BYTES]) -> Result<Self, MatchError> {
        let sign_bits = u64::from_le_bytes(std::array::from_fn(|i| bytes[i]));
        let packed_2bit = std::array::from_fn(|i| bytes[8 + i]);
        let polytope_vertices = [bytes[24], bytes[25], bytes[26], bytes[27]];
        if polytope_vertices.iter().any(|&v| v >= 32) {
            return Err(MatchError::Output("kashin polytope vertex out of range"));
        }
        Ok(Self {
            sign_bits,
            packed_2bit,
            polytope_vertices,
        })
    }
}

pub(crate) fn pack_kashin_i8(rotated: &[f32; 64]) -> [u8; 64] {
    std::array::from_fn(|i| {
        (((rotated[i] / KASHIN_I8_CLIP) * 127.0)
            .round()
            .clamp(-127.0, 127.0) as i8) as u8
    })
}

pub(crate) fn unpack_kashin_i8(bytes: &[u8; 64]) -> [f32; 64] {
    std::array::from_fn(|i| f32::from(bytes[i] as i8) * (KASHIN_I8_CLIP / 127.0))
}

pub(crate) fn reconstruct_sparse_features(
    image_size: [u32; 2],
    pixels_f32: &[[f32; 2]],
    descs: &[KashinDescriptor],
    kashin_i8: &[[u8; 64]],
    projector: &HadamardProjector,
) -> SparseFeatures {
    let n = pixels_f32.len().min(descs.len());
    let (iw, ih) = (
        f64::from(image_size[0].max(1)),
        f64::from(image_size[1].max(1)),
    );
    let scale = (800.0 / iw).min(600.0 / ih);
    let ox = (800.0 - iw * scale) * 0.5 + (scale - 1.0) * 0.5;
    let oy = (600.0 - ih * scale) * 0.5 + (scale - 1.0) * 0.5;
    let mut out = SparseFeatures {
        pixels: Vec::with_capacity(n),
        model_pixels: Vec::with_capacity(n),
        scores: Vec::with_capacity(n),
        descriptors: Vec::with_capacity(n * 64),
    };
    let denom = (n.max(1)) as f32;
    for i in 0..n {
        let p = [f64::from(pixels_f32[i][0]), f64::from(pixels_f32[i][1])];
        out.pixels.push(p);
        out.model_pixels
            .push([(p[0] * scale + ox) as f32, (p[1] * scale + oy) as f32]);
        out.scores.push(1.0 - (i as f32) / denom);
        let rot = kashin_i8
            .get(i)
            .map_or_else(|| descs[i].unpack_levels(), unpack_kashin_i8);
        let unit = try_normalize64(&rot).unwrap_or([0.0_f32; 64]);
        out.descriptors
            .extend_from_slice(&projector.unproject(&unit));
    }
    out
}

/// Compact 160-byte summary of one [`SparseFeatures`] set for pose pre-screening.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewSketch {
    /// Unit-normalized signed mass across the 128 Cross-Polytope vertices.
    pub vertex_bins: [i8; 128],
    /// Hadamard sign signature (256 bits) of partitioned descriptor first moments.
    pub sign_signature: [u64; 4],
}

impl ViewSketch {
    /// Serialized size of one [`ViewSketch`] in bytes (`128 + 32`).
    pub const SERIALIZED_BYTES: usize = 160;

    /// Create an empty zero sketch.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            vertex_bins: [0i8; 128],
            sign_signature: [0u64; 4],
        }
    }

    /// Return `true` when all bins and signature bits are zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertex_bins == [0i8; 128] && self.sign_signature == [0u64; 4]
    }

    /// Build a 160-byte [`ViewSketch`] from `features` with `projector`.
    #[must_use]
    pub fn from_features(projector: &HadamardProjector, features: &SparseFeatures) -> Self {
        let chunks = features.descriptors.as_chunks::<64>().0;
        let count = chunks.len().min(features.pixels.len());
        let mut raw_bins = [0.0_f32; 128];
        let mut moments = [[0.0_f32; 64]; 4];
        let mut valid = 0usize;

        for (idx, desc) in chunks.iter().take(count).enumerate() {
            let score = features.scores.get(idx).copied().unwrap_or(1.0);
            let Some(unit) = (score.is_finite() && score >= 0.0)
                .then(|| try_normalize64(desc))
                .flatten()
            else {
                continue;
            };
            let weight = if score > 1e-6 { score } else { 1e-3 };
            let rotated = projector.project(&unit);
            let (primary_v, primary_peak) = octahedron_vertex(&rotated);
            let mass = weight * (1.0 + primary_peak);
            raw_bins[primary_v] += mass;
            raw_bins[primary_v ^ 1] -= 0.5 * mass;
            for block in 0..4 {
                let start = block << 4;
                let (local_v, peak) = octahedron_vertex(&rotated[start..start + 16]);
                let (sub_v, sub_mass) = ((block << 5) + local_v, 0.35 * weight * (1.0 + peak));
                raw_bins[sub_v] += sub_mass;
                raw_bins[sub_v ^ 1] -= 0.5 * sub_mass;
            }
            for (dst, &val) in moments[(primary_v >> 5) & 3].iter_mut().zip(&rotated) {
                *dst += weight * val;
            }
            valid += 1;
        }
        if valid == 0 {
            return Self::empty();
        }
        let mean = raw_bins.iter().sum::<f32>() / 128.0;
        for bin in &mut raw_bins {
            *bin -= mean;
        }
        let norm = raw_bins.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm <= 1e-12 {
            return Self::empty();
        }
        let vertex_bins = raw_bins.map(|v| (v / norm * 127.0).round().clamp(-127.0, 127.0) as i8);
        Self {
            vertex_bins,
            sign_signature: hadamard_moment_signature(projector.seed(), &mut moments),
        }
    }

    /// Compute the similarity score (`[-1.0, 1.0]`) between two view sketches.
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f32 {
        let (mut dot, mut norm_a, mut norm_b) = (0i32, 0i32, 0i32);
        for (&a, &b) in self.vertex_bins.iter().zip(&other.vertex_bins) {
            let (ai, bi) = (i32::from(a), i32::from(b));
            dot += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }
        if norm_a == 0 || norm_b == 0 {
            return 0.0;
        }
        let hist_cos = (dot as f32) / ((norm_a as f32) * (norm_b as f32)).sqrt();
        let hd: u32 = (0..4)
            .map(|g| (self.sign_signature[g] ^ other.sign_signature[g]).count_ones())
            .sum();
        (0.65 * hist_cos + 0.35 * (std::f32::consts::PI * (hd as f32) / 256.0).cos())
            .clamp(-1.0, 1.0)
    }

    /// Encode this view sketch into 160 little-endian bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_BYTES] {
        let mut out = [0u8; Self::SERIALIZED_BYTES];
        for (idx, &bin) in self.vertex_bins.iter().enumerate() {
            out[idx] = bin as u8;
        }
        for (idx, &word) in self.sign_signature.iter().enumerate() {
            out[128 + (idx << 3)..136 + (idx << 3)].copy_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// Decode 160 little-endian bytes into a [`ViewSketch`].
    #[must_use]
    pub fn from_bytes(bytes: [u8; Self::SERIALIZED_BYTES]) -> Self {
        let vertex_bins = std::array::from_fn(|i| bytes[i] as i8);
        let sign_signature = std::array::from_fn(|g| {
            u64::from_le_bytes(std::array::from_fn(|b| bytes[128 + (g << 3) + b]))
        });
        Self {
            vertex_bins,
            sign_signature,
        }
    }
}

const fn splitmix64(mut state: u64) -> (u64, u64) {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (state, z ^ (z >> 31))
}

fn apply_rademacher64(values: &mut [f32; 64], mask: u64) {
    for (idx, value) in values.iter_mut().enumerate() {
        if ((mask >> idx) & 1) == 1 {
            *value = -*value;
        }
    }
}

pub(crate) fn sign_bits64(rotated: &[f32; 64]) -> u64 {
    let mut bits = 0u64;
    for (idx, &val) in rotated.iter().enumerate() {
        if val >= 0.0 {
            bits |= 1u64 << idx;
        }
    }
    bits
}

pub(crate) fn try_normalize64(descriptor: &[f32; 64]) -> Option<[f32; 64]> {
    if descriptor.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let norm_sq: f32 = descriptor.iter().map(|v| v * v).sum();
    (norm_sq > 1e-24).then(|| descriptor.map(|v| v / norm_sq.sqrt()))
}

pub(crate) fn octahedron_vertex(slice: &[f32]) -> (usize, f32) {
    let (mut best_k, mut best_abs, mut neg) = (0usize, -1.0_f32, 0usize);
    for (k, &val) in slice.iter().enumerate() {
        if val.abs() > best_abs {
            best_abs = val.abs();
            best_k = k;
            neg = usize::from(val < 0.0);
        }
    }
    ((best_k << 1) | neg, best_abs)
}

pub(crate) fn hadamard_moment_signature(seed: u64, moments: &mut [[f32; 64]; 4]) -> [u64; 4] {
    let mut state = seed ^ 0x534B_4554_4348_3235;
    for block in moments.iter_mut() {
        let (next, mask) = splitmix64(state);
        state = next;
        apply_rademacher64(block, mask);
        fwht64(block);
    }
    let [m0, m1, m2, m3] = moments;
    for (((x0, x1), x2), x3) in m0.iter_mut().zip(m1).zip(m2).zip(m3) {
        let (a0, a1, a2, a3) = (*x0 + *x1, *x0 - *x1, *x2 + *x3, *x2 - *x3);
        (*x0, *x1) = (0.5 * (a0 + a2), 0.5 * (a1 + a3));
        (*x2, *x3) = (0.5 * (a0 - a2), 0.5 * (a1 - a3));
    }
    moments.each_ref().map(sign_bits64)
}

#[cfg(test)]
mod tests;

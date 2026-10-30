use super::*;
use navigate_visual::PixelMatch;
use sha2::{Digest, Sha256};

struct Rng64(u64);

impl Rng64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32_signed(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 23) as f32) - 1.0
    }

    fn next_gaussian(&mut self) -> f32 {
        let mut sum = 0.0_f32;
        for _ in 0..6 {
            sum += self.next_f32_signed();
        }
        sum * 0.70710677
    }

    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn unit_vec64(&mut self) -> [f32; 64] {
        let mut raw = [0.0_f32; 64];
        for v in &mut raw {
            *v = self.next_gaussian();
        }
        try_normalize64(&raw).expect("non-zero random vector")
    }
}

fn dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm64(a: &[f32; 64]) -> f32 {
    dot64(a, a).sqrt()
}

fn synthetic_features(rng: &mut Rng64, count: usize) -> SparseFeatures {
    let mut pixels = Vec::with_capacity(count);
    let mut model_pixels = Vec::with_capacity(count);
    let mut scores = Vec::with_capacity(count);
    let mut descriptors = Vec::with_capacity(count * 64);
    for idx in 0..count {
        let px = ((idx * 17 + 11) % 640) as f64 + 0.5;
        let py = ((idx * 31 + 7) % 480) as f64 + 0.5;
        pixels.push([px, py]);
        model_pixels.push([px as f32, py as f32]);
        scores.push(0.4 + 0.5 * (rng.next_unit() as f32));
        descriptors.extend_from_slice(&rng.unit_vec64());
    }
    SparseFeatures {
        pixels,
        model_pixels,
        scores,
        descriptors,
    }
}

#[test]
fn hadamard_projector_isometry_and_kashin_spike_spreading() {
    for axis in 0..64 {
        let mut spike = [0.0_f32; 64];
        spike[axis] = 1.0;
        fwht64(&mut spike);
        for &coord in &spike {
            assert!(
                (coord.abs() - 0.125).abs() < 1e-7,
                "single FWHT must map 1-sparse spike to +-1/8, got {coord}"
            );
        }
    }

    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);
    let mut spike = [0.0_f32; 64];
    spike[0] = 1.0;
    let rotated_spike = projector.project(&spike);
    assert!((norm64(&rotated_spike) - 1.0).abs() < 1e-6);
    let max_coord = rotated_spike.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    let active_axes = rotated_spike.iter().filter(|v| v.abs() > 0.0).count();
    assert!(
        max_coord <= 0.45 && active_axes >= 50,
        "3-round Hadamard rotation must spread 1-sparse spike (max={max_coord}, active={active_axes})"
    );

    let mut rng = Rng64(0x1234_5678_9ABC_DEF0);
    for _ in 0..256 {
        let scale_u = 0.25 + 3.5 * (rng.next_unit() as f32);
        let scale_v = 0.25 + 3.5 * (rng.next_unit() as f32);
        let mut u = rng.unit_vec64();
        let mut v = rng.unit_vec64();
        for x in &mut u {
            *x *= scale_u;
        }
        for y in &mut v {
            *y *= scale_v;
        }
        let qu = projector.project(&u);
        let qv = projector.project(&v);
        assert!((norm64(&qu) - norm64(&u)).abs() < 1e-5);
        assert!((dot64(&qu, &qv) - dot64(&u, &v)).abs() < 1e-5);

        let recovered = projector.unproject(&qu);
        for (orig, rec) in u.iter().zip(&recovered) {
            assert!((orig - rec).abs() < 1e-5);
        }
    }
}

#[test]
fn kashin_descriptor_asymmetric_dot_mae_and_hamming_gate() {
    assert_eq!(KashinDescriptor::SERIALIZED_BYTES, 28);
    let projector = HadamardProjector::default();
    let mut rng = Rng64(0xCAFE_BABE_2026_1029);

    let mut abs_err_random = 0.0_f32;
    let pairs = 1_000usize;
    let mut orthogonal_rejected = 0usize;
    for _ in 0..pairs {
        let u = rng.unit_vec64();
        let v = rng.unit_vec64();
        let true_cos = dot64(&u, &v);
        let kd_u = KashinDescriptor::encode(&projector, &u);
        let kd_v = KashinDescriptor::encode(&projector, &v);
        let rot_v = projector.project(&v);

        let approx = kd_u.approx_dot_query(&rot_v);
        abs_err_random += (approx - true_cos).abs();

        assert!((kd_u.approx_dot(&kd_u) - 1.0).abs() < 1e-6);
        let unpacked = kd_u.unpack_levels();
        for &lv in &unpacked {
            assert!(
                [-0.1875_f32, -0.0625, 0.0625, 0.1875]
                    .iter()
                    .any(|&target| (lv - target).abs() < 1e-7)
            );
        }
        if true_cos.abs() < 0.15 && kd_u.hamming_distance(&kd_v) > 20 {
            orthogonal_rejected += 1;
        }
    }
    let mae_random = abs_err_random / (pairs as f32);
    assert!(
        mae_random < 0.06,
        "2-bit asymmetric MAE on random unit vectors must be < 0.06, got {mae_random}"
    );
    assert!(
        orthogonal_rejected > 700,
        "Hamming gate must reject the vast majority of near-orthogonal pairs ({orthogonal_rejected})"
    );

    let mut abs_err_correlated = 0.0_f32;
    let mut correlated_kept = 0usize;
    let mut opposite_rejected = 0usize;
    for step in 0..400 {
        let u = rng.unit_vec64();
        let z = rng.unit_vec64();
        let z_dot = dot64(&u, &z);
        let mut ortho = [0.0_f32; 64];
        for i in 0..64 {
            ortho[i] = z[i] - z_dot * u[i];
        }
        let ortho = try_normalize64(&ortho).expect("orthogonal component");
        let rho = -0.95 + 1.90 * ((step as f32) / 399.0);
        let s = (1.0 - rho * rho).sqrt();
        let mut v = [0.0_f32; 64];
        for i in 0..64 {
            v[i] = rho * u[i] + s * ortho[i];
        }
        let kd_u = KashinDescriptor::encode(&projector, &u);
        let kd_v = KashinDescriptor::encode(&projector, &v);
        let approx = kd_u.approx_dot_query(&projector.project(&v));
        abs_err_correlated += (approx - rho).abs();

        let hd = kd_u.hamming_distance(&kd_v);
        if rho >= 0.85 && hd <= 20 {
            correlated_kept += 1;
        }
        if rho <= -0.80 && hd > 44 {
            opposite_rejected += 1;
        }
        let roundtrip = KashinDescriptor::from_bytes(kd_u.to_bytes()).expect("valid bytes");
        assert_eq!(kd_u, roundtrip);
    }
    let mae_corr = abs_err_correlated / 400.0;
    assert!(
        mae_corr < 0.06,
        "2-bit asymmetric MAE across [-0.95, 0.95] must be < 0.06, got {mae_corr}"
    );
    assert!(
        correlated_kept >= 20,
        "correlated pairs must pass Hamming gate"
    );
    assert!(
        opposite_rejected >= 30,
        "opposite pairs must fail Hamming gate"
    );

    let mut bad = [0u8; 28];
    bad[27] = 32;
    assert!(KashinDescriptor::from_bytes(bad).is_err());
}

#[test]
fn view_sketch_size_and_matching_view_ranking() {
    assert_eq!(std::mem::size_of::<ViewSketch>(), 160);
    let projector = HadamardProjector::default();
    let mut rng = Rng64(0x9988_7766_5544_3322);

    let reference = synthetic_features(&mut rng, 96);
    let ref_sketch = ViewSketch::from_features(&projector, &reference);
    assert_eq!(ref_sketch, ViewSketch::from_bytes(ref_sketch.to_bytes()));
    assert!((ref_sketch.similarity(&ref_sketch) - 1.0).abs() < 1e-5);

    let mut query = SparseFeatures::empty();
    let ref_chunks = reference.descriptors.as_chunks::<64>().0;
    for (idx, &chunk) in ref_chunks.iter().enumerate().take(64) {
        let mut noisy = chunk;
        let jitter = rng.unit_vec64();
        for k in 0..64 {
            noisy[k] += 0.18 * jitter[k];
        }
        let noisy = try_normalize64(&noisy).expect("normalized noisy descriptor");
        query.pixels.push([
            reference.pixels[idx][0] + 1.5,
            reference.pixels[idx][1] - 1.0,
        ]);
        query.model_pixels.push([0.0, 0.0]);
        query.scores.push(0.85);
        query.descriptors.extend_from_slice(&noisy);
    }
    let outliers = synthetic_features(&mut rng, 32);
    query.pixels.extend_from_slice(&outliers.pixels);
    query.model_pixels.extend_from_slice(&outliers.model_pixels);
    query.scores.extend_from_slice(&outliers.scores);
    query.descriptors.extend_from_slice(&outliers.descriptors);

    let query_sketch = ViewSketch::from_features(&projector, &query);
    let match_sim = ref_sketch.similarity(&query_sketch);
    assert!(
        match_sim > 0.45,
        "matching view similarity too low: {match_sim}"
    );

    for _ in 0..24 {
        let unrelated = synthetic_features(&mut rng, 96);
        let unrelated_sketch = ViewSketch::from_features(&projector, &unrelated);
        let distractor_sim = unrelated_sketch.similarity(&query_sketch);
        assert!(
            match_sim > distractor_sim + 0.25,
            "matching view ({match_sim}) must rank strictly above unrelated ({distractor_sim})"
        );
    }
}

#[test]
fn precomputed_bank_roundtrip_corruption_rejection_and_3d_matching() {
    let mut rng = Rng64(0xA1B2_C3D4_E5F6_0718);
    let seed = DEFAULT_KASHIN_SEED;
    let projector = HadamardProjector::new(seed);
    let true_index = 19usize;
    let mut entries = Vec::with_capacity(33);
    let mut true_features = SparseFeatures::empty();

    for idx in 0..33 {
        let feats = synthetic_features(&mut rng, 96);
        if idx == true_index {
            true_features = feats.clone();
        }
        let world_points = if idx % 2 == 1 {
            (0..feats.len())
                .map(|k| {
                    [
                        100.0 + k as f64 * 2.5,
                        -50.0 + idx as f64,
                        420.0 + k as f64 * 0.25,
                    ]
                })
                .collect()
        } else {
            Vec::new()
        };
        let entry = PrecomputedViewEntry::from_features(
            idx as u32 + 1000,
            [idx as f64 * 25.0, -10.0, 950.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [640, 480],
            &projector,
            &feats,
            world_points,
        )
        .expect("valid precomputed view entry");
        entries.push(entry);
    }

    let bank = PrecomputedReferenceBank::new([0xAA; 32], [0xBB; 32], seed, entries)
        .expect("valid reference bank");

    let encoded = bank.to_bytes();
    assert_eq!(&encoded[0..8], BANK_MAGIC);
    let decoded = PrecomputedReferenceBank::from_bytes(&encoded).expect("decode valid bank");
    assert_eq!(bank, decoded);
    assert_eq!(encoded, decoded.to_bytes());

    #[cfg(not(target_arch = "wasm32"))]
    {
        let path = std::env::temp_dir().join(format!(
            "navigate_kashin_bank_test_{}.bin",
            std::process::id()
        ));
        bank.save_file(&path).expect("save bank file");
        let loaded = PrecomputedReferenceBank::load_file(&path).expect("load bank file");
        std::fs::remove_file(&path).expect("remove temp bank file");
        assert_eq!(bank, loaded);
    }

    assert!(PrecomputedReferenceBank::from_bytes(&encoded[..50]).is_err());

    let mut bad_magic = encoded.clone();
    bad_magic[7] = b'0';
    assert!(PrecomputedReferenceBank::from_bytes(&bad_magic).is_err());

    let mut bad_version = encoded.clone();
    bad_version[8..12].copy_from_slice(&2u32.to_le_bytes());
    let payload_end = bad_version.len() - 32;
    let recomputed: [u8; 32] = Sha256::digest(&bad_version[..payload_end]).into();
    bad_version[payload_end..].copy_from_slice(&recomputed);
    assert!(PrecomputedReferenceBank::from_bytes(&bad_version).is_err());

    let mut bad_payload = encoded.clone();
    bad_payload[120] ^= 0x01;
    assert!(PrecomputedReferenceBank::from_bytes(&bad_payload).is_err());

    let mut bad_checksum = encoded.clone();
    let last = bad_checksum.len() - 1;
    bad_checksum[last] ^= 0x80;
    assert!(PrecomputedReferenceBank::from_bytes(&bad_checksum).is_err());

    let mut forged_count = encoded[..88 + 32].to_vec();
    forged_count[84..88].copy_from_slice(&u32::MAX.to_le_bytes());
    let digest: [u8; 32] = Sha256::digest(&forged_count[..88]).into();
    forged_count[88..120].copy_from_slice(&digest);
    assert!(PrecomputedReferenceBank::from_bytes(&forged_count).is_err());

    let mut query = SparseFeatures::empty();
    let true_chunks = true_features.descriptors.as_chunks::<64>().0;
    let inlier_count = 72usize;
    for k in (0..inlier_count).rev() {
        let mut desc = true_chunks[k];
        let noise = rng.unit_vec64();
        for d in 0..64 {
            desc[d] += 0.14 * noise[d];
        }
        let desc = try_normalize64(&desc).expect("unit query descriptor");
        let qpx = [
            true_features.pixels[k][0] + 4.25,
            true_features.pixels[k][1] - 2.75,
        ];
        query.pixels.push(qpx);
        query.model_pixels.push([qpx[0] as f32, qpx[1] as f32]);
        query.scores.push(0.9);
        query.descriptors.extend_from_slice(&desc);
    }
    let distractors = synthetic_features(&mut rng, 28);
    query.pixels.extend_from_slice(&distractors.pixels);
    query
        .model_pixels
        .extend_from_slice(&distractors.model_pixels);
    query.scores.extend_from_slice(&distractors.scores);
    query
        .descriptors
        .extend_from_slice(&distractors.descriptors);

    let shortlist = decoded.shortlist(&query, 5);
    assert_eq!(shortlist.len(), 5);
    assert_eq!(
        shortlist[0].0, true_index,
        "true view must rank #1 among 32 distractors: {shortlist:?}"
    );
    assert!(shortlist[0].1 > shortlist[1].1 + 0.25);

    let matches = decoded
        .match_entry(true_index, &query, 20, 0.72, 0.80)
        .expect("match true entry");
    assert!(
        matches.len() >= 55,
        "expected >= 55 recovered correspondences, got {}",
        matches.len()
    );
    let true_entry = &decoded.entries[true_index];
    for m in &matches {
        let ref_idx = true_entry
            .pixels
            .iter()
            .position(|p| {
                (f64::from(p[0]) - m.reference_pixel[0]).abs() < 1e-4
                    && (f64::from(p[1]) - m.reference_pixel[1]).abs() < 1e-4
            })
            .expect("matched reference pixel exists in entry");
        assert!(ref_idx < inlier_count, "outlier matched at {ref_idx}");
        assert!((m.query_pixel[0] - (m.reference_pixel[0] + 4.25)).abs() < 1e-3);
        assert!((m.query_pixel[1] - (m.reference_pixel[1] - 2.75)).abs() < 1e-3);
        assert_eq!(
            m.world_point_enu,
            Some(true_entry.world_points_enu[ref_idx])
        );
    }

    let pixel_matches = PrecomputedMatch::to_pixel_matches(&matches);
    assert_eq!(pixel_matches.len(), matches.len());
    let single: PixelMatch = matches[0].into();
    assert_eq!(single.reference, pixel_matches[0].reference);
    assert_eq!(single.query, pixel_matches[0].query);

    let prepared = PreparedKashinQuery::from_features(&projector, &query).expect("prepared query");
    assert_eq!(
        *prepared.sketch(),
        ViewSketch::from_features(&projector, &query)
    );
    assert_eq!(decoded.shortlist_prepared(&prepared, 5), shortlist);
    let prepared_matches = decoded
        .match_prepared(true_index, &prepared, 20, 0.72, 0.80)
        .expect("prepared match");
    assert_eq!(prepared_matches, matches);

    let prefix_query = query.truncate_top(60);
    let prefix_prepared = decoded
        .match_prepared_limit(true_index, &prepared, 60, 20, 0.72, 0.80)
        .expect("prefix prepared match");
    let prefix_direct = decoded
        .match_entry(true_index, &prefix_query, 20, 0.72, 0.80)
        .expect("prefix direct match");
    assert_eq!(prefix_prepared, prefix_direct);

    // Compact 92 B/kp check: 16 odd entries with 3D (92 B/kp) + 17 even entries without 3D (80 B/kp).
    let expected_bytes = 88 + 32 + 33 * 273 + 16 * 96 * 92 + 17 * 96 * 80;
    assert_eq!(encoded.len(), expected_bytes);
}

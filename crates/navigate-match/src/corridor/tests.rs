use super::*;
use crate::{DEFAULT_KASHIN_SEED, PreparedKashinQuery};
use sha2::{Digest, Sha256};

fn unit_desc(seed: usize) -> [f32; 64] {
    let mut raw = [0.0_f32; 64];
    for (k, slot) in raw.iter_mut().enumerate() {
        let bits = ((seed.wrapping_mul(6364136223846793005) ^ k.wrapping_mul(1442695040888963407))
            >> 32) as u32;
        *slot = (bits as f32) / (u32::MAX as f32) * 2.0 - 1.0;
    }
    try_normalize64(&raw).expect("non-zero synthetic descriptor")
}

fn build_tile_landmarks(
    tx: i32,
    ty: i32,
    tile_size_m: f64,
    count: usize,
    projector: &HadamardProjector,
) -> CorridorTileEntry {
    let anchor = [
        (f64::from(tx) + 0.5) * tile_size_m,
        (f64::from(ty) + 0.5) * tile_size_m,
        120.0,
    ];
    let mut features = SparseFeatures::empty();
    let mut world = Vec::with_capacity(count);
    let mut obs_dirs = Vec::with_capacity(count);
    let mut cam_xs = Vec::with_capacity(count);
    let side = (count as f64).sqrt().ceil().max(1.0);
    for i in 0..count {
        let gx = (i as f64 % side) / side - 0.45;
        let gy = ((i as f64 / side).floor()) / side - 0.45;
        let w = [
            anchor[0] + gx * tile_size_m * 0.9,
            anchor[1] + gy * tile_size_m * 0.9,
            anchor[2] + ((i % 7) as f64) * 3.0,
        ];
        world.push(w);
        obs_dirs.push([0.0_f32, 0.0, 1.0]);
        cam_xs.push([0.0_f32, 1.0, 0.0]);
        features
            .pixels
            .push([(i % 64) as f64 * 10.0, (i / 64) as f64 * 10.0]);
        features.model_pixels.push([0.0, 0.0]);
        features.scores.push(0.95 - (i as f32) * 0.002);
        let seed_id = ((tx + 10) as usize) * 10_000 + ((ty + 10) as usize) * 500 + i + 1;
        features.descriptors.extend_from_slice(&unit_desc(seed_id));
    }
    CorridorTileEntry::from_landmarks(
        CorridorTileKey {
            tile_x: tx,
            tile_y: ty,
        },
        anchor,
        projector,
        &world,
        (&obs_dirs, &cam_xs, &[]),
        &features,
    )
    .expect("valid tile")
}

#[test]
fn corridor_bank_roundtrips_nvktile1_and_projects_arbitrary_shifted_pose() {
    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);
    let tile_size_m = 2_000.0_f32;
    let mut tiles = Vec::new();
    for ty in -1..=1 {
        for tx in -1..=1 {
            tiles.push(build_tile_landmarks(
                tx,
                ty,
                f64::from(tile_size_m),
                64,
                &projector,
            ));
        }
    }
    let bank = CorridorLandmarkBank::new(
        [0x11; 32],
        [0x22; 32],
        DEFAULT_KASHIN_SEED,
        tile_size_m,
        tiles,
    )
    .expect("valid corridor bank");
    assert_eq!(bank.total_landmarks(), 9 * 64);

    let encoded = bank.to_bytes();
    assert_eq!(&encoded[..8], CORRIDOR_MAGIC);
    // 92 header + 9 * 220 tile headers + 576 * 93 landmarks + 32 checksum
    assert_eq!(encoded.len(), 92 + 9 * 220 + 576 * 93 + 32);

    let decoded = CorridorLandmarkBank::from_bytes(&encoded).expect("decode corridor bank");
    assert_eq!(bank, decoded);
    assert_eq!(encoded, decoded.to_bytes());

    let mut bad_magic = encoded.clone();
    bad_magic[7] = b'9';
    assert!(CorridorLandmarkBank::from_bytes(&bad_magic).is_err());

    let mut bad_payload = encoded.clone();
    bad_payload[150] ^= 0x04;
    assert!(CorridorLandmarkBank::from_bytes(&bad_payload).is_err());

    let mut bad_ver = encoded.clone();
    bad_ver[8..12].copy_from_slice(&7u32.to_le_bytes());
    let pay_end = bad_ver.len() - 32;
    let digest: [u8; 32] = Sha256::digest(&bad_ver[..pay_end]).into();
    bad_ver[pay_end..].copy_from_slice(&digest);
    assert!(CorridorLandmarkBank::from_bytes(&bad_ver).is_err());

    // Arbitrary aircraft pose: +1,450 m East, -820 m North, 1,950 m altitude, nadir-oblique camera!
    // Camera optical axis (+Z) looks downward-West: X_cam = +North, Y_cam = +East, Z_cam = -Up.
    let w2c = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0];
    let cam = VirtualViewCamera {
        eye_enu: [1_450.0, -820.0, 1_950.0],
        world_to_cam: w2c,
        intrinsics: [600.0, 600.0, 400.0, 300.0],
        image_size: [800, 600],
        max_landmarks: 256,
        min_cos_incidence: 0.15,
        min_cos_orientation: 0.80,
    };
    let virt = decoded.project_virtual_view(&cam, &projector);
    // Upside-down camera (+X_cam = -North instead of +North) must reject inverted landmarks!
    let mut inverted_cam = cam;
    inverted_cam.world_to_cam = [0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0];
    assert!(
        decoded
            .project_virtual_view(&inverted_cam, &projector)
            .features
            .is_empty()
    );
    assert!(
        virt.visible_tiles >= 2 && virt.visible_tiles < 9,
        "frustum culling must keep overlapping tiles and cull distant ones, got {}",
        virt.visible_tiles
    );
    assert!(
        virt.features.len() >= 40,
        "expected >= 40 projected landmarks at arbitrary shifted pose, got {}",
        virt.features.len()
    );
    assert_eq!(virt.world_points_enu.len(), virt.features.len());
    assert_eq!(virt.sign_bits.len(), virt.features.len());
    assert_eq!(virt.kashin_i8.len(), virt.features.len());

    // Verify exact pinhole projection of every synthesized landmark!
    for (px, world) in virt.features.pixels.iter().zip(&virt.world_points_enu) {
        let rel = [
            world[0] - cam.eye_enu[0],
            world[1] - cam.eye_enu[1],
            world[2] - cam.eye_enu[2],
        ];
        let pc = mat3_mul(&w2c, rel);
        assert!(pc[2] > 1.0);
        let u = 600.0 * (pc[0] / pc[2]) + 400.0;
        let v = 600.0 * (pc[1] / pc[2]) + 300.0;
        assert!((px[0] - u).abs() < 1e-6);
        assert!((px[1] - v).abs() < 1e-6);
    }
}

#[test]
fn active_track_radius_matcher_recovers_shifted_3d_tracks_and_rejects_far_clutter() {
    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);
    let w2c = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0];
    let k = [600.0, 600.0, 400.0, 300.0];
    let eye_t0 = [0.0, 0.0, 1_120.0];
    let eye_t1 = [18.0, -12.0, 1_118.0];

    let mut anchor_feat = SparseFeatures::empty();
    let mut hits = Vec::new();
    for i in 0..80 {
        let east = -350.0 + (i % 10) as f64 * 75.0;
        let north = -450.0 + (i / 10) as f64 * 110.0;
        let up = 120.0;
        let pc0 = mat3_mul(&w2c, [east - eye_t0[0], north - eye_t0[1], up - eye_t0[2]]);
        let u0 = k[0] * (pc0[0] / pc0[2]) + k[2];
        let v0 = k[1] * (pc0[1] / pc0[2]) + k[3];
        anchor_feat.pixels.push([u0, v0]);
        anchor_feat.model_pixels.push([u0 as f32, v0 as f32]);
        anchor_feat.scores.push(0.9);
        anchor_feat
            .descriptors
            .extend_from_slice(&unit_desc(500 + i));
        hits.push(Some([east, north, up]));
    }
    let prep0 = PreparedKashinQuery::from_features(&projector, &anchor_feat).expect("prep0");
    let tracks = ActiveTrackSet::from_prepared_query(&prep0, &hits, 128);
    assert_eq!(tracks.len(), 80);

    // Frame t1: true pose is eye_t1, plus an identical twin descriptor placed 200 px away
    // to prove spatial radius gating rejects far repetitive texture!
    let mut qry_t1 = SparseFeatures::empty();
    for (i, hit_opt) in hits.iter().enumerate().take(80) {
        let w = hit_opt.expect("world hit");
        let pc1 = mat3_mul(&w2c, [w[0] - eye_t1[0], w[1] - eye_t1[1], w[2] - eye_t1[2]]);
        let u1 = k[0] * (pc1[0] / pc1[2]) + k[2];
        let v1 = k[1] * (pc1[1] / pc1[2]) + k[3];
        let mut noisy = unit_desc(500 + i);
        let pert = unit_desc(9_000 + i);
        for c in 0..64 {
            noisy[c] += 0.10 * pert[c];
        }
        let noisy = try_normalize64(&noisy).expect("unit noisy");
        qry_t1.pixels.push([u1, v1]);
        qry_t1.model_pixels.push([u1 as f32, v1 as f32]);
        qry_t1.scores.push(0.9);
        qry_t1.descriptors.extend_from_slice(&noisy);
    }
    // Far duplicate twin of track 0 at +220 px (outside 32 px search radius!)
    let far_px = [
        (qry_t1.pixels[0][0] + 220.0).min(790.0),
        (qry_t1.pixels[0][1] + 180.0).min(590.0),
    ];
    qry_t1.pixels.push(far_px);
    qry_t1
        .model_pixels
        .push([far_px[0] as f32, far_px[1] as f32]);
    qry_t1.scores.push(0.99);
    qry_t1.descriptors.extend_from_slice(&unit_desc(500));

    let prep1 = PreparedKashinQuery::from_features(&projector, &qry_t1).expect("prep1");
    // Predict using unshifted eye_t0 (true motion causes ~7-11 px optical flow, well inside 32 px!)
    let matched = tracks
        .match_radius(
            &prep1,
            eye_t0,
            &w2c,
            k,
            [800, 600],
            &TrackMatchConfig::default(),
        )
        .expect("radius match");
    assert!(
        matched.len() >= 70,
        "expected >= 70 tracked 3D landmarks, got {}",
        matched.len()
    );
    for m in &matched {
        assert_eq!(m.track_index, m.query_index);
    }
}

#[test]
fn retain_in_frustum_evicts_outside_and_behind_landmarks_and_nadir_accepts_any_heading()
-> Result<(), MatchError> {
    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);
    let tile = build_tile_landmarks(0, 0, 600.0, 48, &projector);
    let bank = CorridorLandmarkBank::new([1; 32], [2; 32], DEFAULT_KASHIN_SEED, 600.0, vec![tile])?;

    // South-looking camera rotated 180 deg in yaw on a nadir tile at low 85 m AGL (205 m up vs 120 m ground):
    let w2c_south = [0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0];
    let nadir_cam = VirtualViewCamera {
        eye_enu: [300.0, 300.0, 225.0],
        world_to_cam: w2c_south,
        intrinsics: [600.0, 600.0, 400.0, 300.0],
        image_size: [800, 600],
        max_landmarks: 64,
        min_cos_incidence: 0.80,
        min_cos_orientation: -1.0,
    };
    let virt = bank.project_virtual_view(&nadir_cam, &projector);
    assert!(
        !virt.features.is_empty(),
        "nadir camera with relaxed orientation must project landmarks at arbitrary heading"
    );

    let mut set = ActiveTrackSet {
        world_points_enu: vec![
            [300.0, 300.0, 120.0],   // dead center in front
            [300.0, 300.0, 400.0],   // behind camera (pc[2] < 0)
            [5_000.0, 300.0, 120.0], // far outside horizontal FOV
        ],
        sign_bits: vec![11, 22, 33],
        kashin_i8: vec![[1; 64], [2; 64], [3; 64]],
    };
    let kept = set.retain_in_frustum(
        [300.0, 300.0, 225.0],
        &w2c_south,
        [600.0, 600.0, 400.0, 300.0],
        [800, 600],
        16.0,
    );
    assert_eq!(kept, 1);
    assert_eq!(set.len(), 1);
    assert_eq!(set.sign_bits, vec![11]);
    assert_eq!(set.kashin_i8, vec![[1; 64]]);
    Ok(())
}

#[test]
fn corridor_tile_store_bounds_lru_window_and_supports_live_tile_insertion() -> Result<(), MatchError>
{
    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);
    let mut tiles = Vec::new();
    for ty in -3..=3 {
        for tx in -3..=3 {
            tiles.push(build_tile_landmarks(tx, ty, 600.0, 64, &projector));
        }
    }
    let bank = CorridorLandmarkBank::new([7; 32], [8; 32], DEFAULT_KASHIN_SEED, 600.0, tiles)?;
    let store = super::CorridorTileStore::from_bank(bank.clone(), 9);
    assert_eq!(store.indexed_tile_count(), 49);
    assert_eq!(store.cached_tile_count(), 0);

    let w2c = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0];
    let cam = VirtualViewCamera {
        eye_enu: [300.0, 300.0, 1_120.0],
        world_to_cam: w2c,
        intrinsics: [600.0, 600.0, 400.0, 300.0],
        image_size: [800, 600],
        max_landmarks: 512,
        min_cos_incidence: 0.70,
        min_cos_orientation: 0.50,
    };
    let from_store = store.project_virtual_view(&cam, &projector)?;
    let from_bank = bank.project_virtual_view(&cam, &projector);
    assert_eq!(from_store.features.pixels, from_bank.features.pixels);
    assert!(store.cached_tile_count() <= 9);
    assert!(store.resident_window_bytes() < 10 * 1024 * 1024);

    let extra = build_tile_landmarks(12, 12, 600.0, 32, &projector);
    let extra_key = super::CorridorTileKey {
        tile_x: 12,
        tile_y: 12,
    };
    assert!(!store.has_tile(extra_key));
    store.insert_tile(extra, false)?;
    assert!(store.has_tile(extra_key));
    assert!(store.cached_tile_count() <= 9);
    Ok(())
}

#[test]
fn corridor_tile_store_directory_culls_frustum_without_lru_self_eviction() -> Result<(), MatchError>
{
    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);
    let mut tiles = Vec::new();
    for ty in -2..=2 {
        for tx in -2..=2 {
            tiles.push(build_tile_landmarks(tx, ty, 150.0, 5_000, &projector));
        }
    }
    let bank = CorridorLandmarkBank::new([9; 32], [10; 32], DEFAULT_KASHIN_SEED, 150.0, tiles)?;
    let tmp =
        std::env::temp_dir().join(format!("navigate_store_frustum_lru_{}", std::process::id()));
    std::fs::remove_dir_all(&tmp).ok();
    drop(super::CorridorTileStore::write_bank_dir(&bank, &tmp, 25)?);
    let reopened = super::CorridorTileStore::open(&tmp, 25)?;
    let cam = VirtualViewCamera {
        eye_enu: [75.0, 75.0, 220.0],
        world_to_cam: [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        intrinsics: [700.0, 700.0, 400.0, 300.0],
        image_size: [800, 600],
        max_landmarks: 512,
        min_cos_incidence: 0.45,
        min_cos_orientation: 0.50,
    };
    let expected = bank.project_virtual_view(&cam, &projector);
    let first = reopened.project_virtual_view(&cam, &projector)?;
    let cached_after_first = reopened.cached_tile_count();
    let second = reopened.project_virtual_view(&cam, &projector)?;
    std::fs::remove_dir_all(&tmp).ok();

    assert_eq!(first.features.pixels, expected.features.pixels);
    assert_eq!(second.features.pixels, expected.features.pixels);
    assert!((1..=9).contains(&cached_after_first));
    assert_eq!(reopened.cached_tile_count(), cached_after_first);
    assert!(reopened.resident_window_bytes() < 10 * 1024 * 1024);
    Ok(())
}

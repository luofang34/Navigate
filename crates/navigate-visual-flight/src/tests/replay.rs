//! Deterministic replay, zero-copy pixel downsampling, night bypass, and clock guard tests.

use crate::{
    AmortizedMasker, CalibrationMode, CameraIntrinsics, CameraMountProfile, ExecutionPolicy,
    FlightEngineConfig, FlightEngineResources, FlightVisualEngine, FlightVisualError, FramePixels,
    LiveFrameInput, LoadedXFeat, SearchPrior, Surface, TileSynthParams, TrackingMode,
    nadir::{rotate_image_gray, rotate_image_gray_bilinear, unrotate_pixel},
    pixels::{median_luma_u8, prepare_tracking_buffers},
    synthesize_corridor_tiles,
};
use burn::tensor::Device;
use image::{GrayImage, Luma, Rgb, RgbImage};
use nalgebra::Matrix3;
use navigate_contract::{
    ClockDomainId, GeodeticPosition, MonotonicNanos, ObservationStamp, SourceEpoch, SourceId,
    WrappingSequence,
};
use navigate_match::{
    CorridorTileKey, CorridorTileStore, DEFAULT_KASHIN_SEED, HadamardProjector, MatchError,
    SparseFeatureExtractor, SparseFeatures,
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

fn test_stamp(seq: u32, nanos: u64) -> ObservationStamp {
    ObservationStamp::new(
        SourceId::new(1),
        SourceEpoch::new(1),
        WrappingSequence::new(seq),
        MonotonicNanos::from_nanos(nanos),
        ClockDomainId::new(1),
    )
}

pub(super) struct PatchGridExtractor {
    pub(super) contract: [u8; 32],
    pub(super) jitter_on_worker: Arc<AtomicBool>,
}

impl SparseFeatureExtractor for PatchGridExtractor {
    fn encoder_contract(&self) -> [u8; 32] {
        self.contract
    }

    fn runtime_name(&self) -> &str {
        "patch-grid-test"
    }

    fn extract_masked_blocking(
        &self,
        image: &GrayImage,
        valid_mask: Option<&GrayImage>,
        limit: usize,
    ) -> Result<SparseFeatures, MatchError> {
        if self.jitter_on_worker.load(Ordering::Relaxed)
            && std::thread::current()
                .name()
                .is_some_and(|n| n.starts_with("corridor-"))
        {
            std::thread::sleep(Duration::from_millis(12));
        }
        let (w, h) = image.dimensions();
        if w < 24 || h < 24 || limit == 0 {
            return Ok(SparseFeatures::empty());
        }
        let raw = image.as_raw();
        let w_u = w as usize;
        let mut items: Vec<(f32, [f32; 2], [f32; 64])> = Vec::new();
        let mut y = 12_usize;
        while y + 12 < h as usize {
            let mut x = 12_usize;
            while x + 12 < w_u {
                if let Some(m) = valid_mask
                    && m.get_pixel(x as u32, y as u32).0[0] < 128
                {
                    x += 16;
                    continue;
                }
                let mut desc = [0.0_f32; 64];
                let mut mean = 0.0_f32;
                for py in 0..8 {
                    for px in 0..8 {
                        let v = f32::from(raw[(y + py * 2 - 7) * w_u + (x + px * 2 - 7)]);
                        desc[py * 8 + px] = v;
                        mean += v;
                    }
                }
                mean /= 64.0;
                let mut norm_sq = 0.0_f32;
                for d in &mut desc {
                    *d -= mean;
                    norm_sq += *d * *d;
                }
                if norm_sq > 64.0 {
                    let inv = norm_sq.sqrt().recip();
                    for d in &mut desc {
                        *d *= inv;
                    }
                    items.push((norm_sq, [x as f32 + 0.5, y as f32 + 0.5], desc));
                }
                x += 16;
            }
            y += 16;
        }
        items.sort_by(|a, b| b.0.total_cmp(&a.0));
        items.truncate(limit);
        let mut pixels = Vec::with_capacity(items.len());
        let mut model_pixels = Vec::with_capacity(items.len());
        let mut scores = Vec::with_capacity(items.len());
        let mut descriptors = Vec::with_capacity(items.len() * 64);
        for (sc, px, desc) in items {
            pixels.push([f64::from(px[0]), f64::from(px[1])]);
            model_pixels.push(px);
            scores.push((sc / 10_000.0).min(1.0));
            descriptors.extend_from_slice(&desc);
        }
        Ok(SparseFeatures {
            pixels,
            model_pixels,
            scores,
            descriptors,
        })
    }

    fn extract_batch_masked_blocking(
        &self,
        items: &[(&GrayImage, Option<&GrayImage>)],
        limit: usize,
    ) -> Result<Vec<SparseFeatures>, MatchError> {
        items
            .iter()
            .map(|(img, mask)| self.extract_masked_blocking(img, *mask, limit))
            .collect()
    }
}

pub(super) fn build_textured_tile() -> RgbImage {
    let mut img = RgbImage::new(256, 256);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let h1 = (x.wrapping_mul(73) ^ y.wrapping_mul(179)).wrapping_mul(31);
        let h2 = ((x / 5).wrapping_mul(97) ^ (y / 5).wrapping_mul(131)).wrapping_mul(53);
        let v = (((h1 >> 3) ^ (h2 >> 2)) & 0xff) as u8;
        *p = Rgb([v, v.wrapping_add(37), v.wrapping_mul(3)]);
    }
    img
}

type ReplayFrameTrace = (bool, TrackingMode, Option<[i64; 2]>, u64, Option<[u64; 3]>);

/// Regression Test (e): `ExecutionPolicy::DeterministicReplay` produces bit-identical outcomes
/// across runs even when background workers experience variable OS scheduling latency.
#[test]
fn deterministic_replay_is_bit_identical_under_worker_jitter() -> Result<(), String> {
    let tile_rgb = build_textured_tile();
    let surface = Arc::new(Surface::synthetic(39.0, -75.5, 10.0, tile_rgb));
    let contract = [7_u8; 32];

    let run_once = |enable_jitter: bool| -> Result<Vec<ReplayFrameTrace>, String> {
        let jitter_flag = Arc::new(AtomicBool::new(false));
        let extractor = Arc::new(PatchGridExtractor {
            contract,
            jitter_on_worker: Arc::clone(&jitter_flag),
        });
        let xfeat = LoadedXFeat::from_extractor(extractor, Device::flex(), contract);
        let store = Arc::new(
            CorridorTileStore::in_memory([0_u8; 32], contract, DEFAULT_KASHIN_SEED, 250.0, 9)
                .map_err(|e| e.to_string())?,
        );
        let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);
        let entries = synthesize_corridor_tiles(
            &surface,
            None,
            &xfeat,
            &projector,
            CorridorTileKey {
                tile_x: 0,
                tile_y: 0,
            },
            TileSynthParams {
                tile_m: 250.0,
                agl_m: 100.0,
                max_per_tile: 1024,
                stride: 4,
                include_obliques: false,
            },
        )
        .map_err(|e| e.to_string())?;
        for e in entries {
            store.insert_tile(e, true).map_err(|err| err.to_string())?;
        }

        jitter_flag.store(enable_jitter, Ordering::Relaxed);
        let resources = Arc::new(
            FlightEngineResources::from_parts(
                Arc::clone(&surface),
                store,
                xfeat,
                None,
                None,
                None,
                true,
            )
            .map_err(|e| e.to_string())?,
        );
        let mut cfg = FlightEngineConfig {
            policy: ExecutionPolicy::DeterministicReplay {
                anchor_budget_ms: 100,
                tile_budget_ms: 150,
            },
            background_workers: true,
            enable_live_tile_synth: true,
            ..FlightEngineConfig::default()
        };
        cfg.search_opts.polish_anchors = false;
        cfg.search_opts.query_keypoints = 512;
        let mut engine = FlightVisualEngine::new(resources, cfg).map_err(|e| e.to_string())?;

        let k = Matrix3::new(300.0, 0.0, 199.5, 0.0, 300.0, 149.5, 0.0, 0.0, 1.0);
        let intr = CameraIntrinsics {
            width: 400,
            height: 300,
            fx: 300.0,
            fy: 300.0,
            cx: 199.5,
            cy: 149.5,
            calibration_confidence: CalibrationMode::Calibrated,
        };
        let mut trace = Vec::with_capacity(8);
        for i in 0..8_u32 {
            let eye = [125.0 + f64::from(i) * 1.5, 125.0, 110.0];
            let geo = surface.geodetic(eye);
            let rot = surface.camera_rotation(geo[0], geo[1], 0.0, 88.0, 0.0);
            let (rgb, _, _) = surface.render(eye, &rot, &k, [400, 300]);
            let t_ns = u64::from(i + 1) * 33_333_333;
            let inp = LiveFrameInput {
                stamp: test_stamp(i + 1, t_ns),
                pixels: FramePixels::Rgb8 {
                    data: rgb.as_raw(),
                    width: 400,
                    height: 300,
                    stride: 400 * 3,
                },
                static_valid_mask: None,
                intrinsics: intr,
                mount: CameraMountProfile::rotary_gimbal_nadir(),
                imu_attitude: None,
                baro_alt_msl_m: Some((110.0, 5.0)),
                filter_prior: Some(SearchPrior {
                    center_geodetic: GeodeticPosition::new(
                        geo[0].to_radians(),
                        geo[1].to_radians(),
                        110.0,
                    ),
                    horizontal_sigma_m: 25.0,
                    velocity_ned_mps: None,
                }),
            };
            let (out, diag) = engine
                .process_frame_with_diagnostics(&inp, &format!("f-{i:04}"))
                .map_err(|e| e.to_string())?;
            let eye_bits = diag
                .candidate
                .map(|(_, _, c)| [c.eye[0].to_bits(), c.eye[1].to_bits(), c.eye[2].to_bits()]);
            trace.push((
                out.telemetry.accepted,
                out.telemetry.tracking_mode,
                out.telemetry.map_cell,
                out.telemetry.capture_lag_ns,
                eye_bits,
            ));
        }
        Ok(trace)
    };

    let run_fast = run_once(false)?;
    let run_jittered = run_once(true)?;
    assert_eq!(run_fast, run_jittered);
    Ok(())
}

#[test]
fn strided_nv12_downsampling_night_bypass_quadrant_rotation_and_clock_guards() -> Result<(), String>
{
    let mut padded = vec![0_u8; 64 * 96];
    for y in 0..64_usize {
        for x in 0..80_usize {
            padded[y * 96 + x] = ((x + y) & 0xff) as u8;
        }
    }
    let (gray, rgb_opt) = prepare_tracking_buffers(
        FramePixels::Nv12YPlane {
            y_plane: &padded,
            width: 80,
            height: 64,
            y_stride: 96,
        },
        [80, 64],
        false,
    )
    .map_err(|e| e.to_string())?;
    assert!(rgb_opt.is_none());
    assert_eq!(gray.get_pixel(30, 20).0[0], 50);
    assert!(
        prepare_tracking_buffers(
            FramePixels::Nv12YPlane {
                y_plane: &padded,
                width: 80,
                height: 64,
                y_stride: 72,
            },
            [80, 64],
            false,
        )
        .is_err()
    );

    let dark = GrayImage::from_pixel(64, 48, Luma([18]));
    assert_eq!(median_luma_u8(&dark), 18);
    let mut masker = AmortizedMasker::new(None, "airframe-occlusions", 6, 10, &Device::flex())
        .map_err(|e| e.to_string())?;
    let k = Matrix3::identity();
    let (_, tel, p50) = masker
        .mask_frame_flight(true, None, &dark, None, None, &k, true)
        .map_err(|e| e.to_string())?;
    assert_eq!(p50, 18);
    assert!((tel.valid_fraction - 1.0).abs() < 1e-6);

    for angle in [90.0, 180.0, 270.0] {
        let fast = rotate_image_gray(&gray, angle);
        let bilin = rotate_image_gray_bilinear(&gray, angle);
        assert_eq!(fast.as_raw(), bilin.as_raw());
        let [ux, uy] = unrotate_pixel([25.0, 20.0], angle, 80, 64);
        assert!((0.0..=80.0).contains(&ux) && (0.0..=64.0).contains(&uy));
    }

    let surface = Arc::new(Surface::synthetic(39.0, -75.5, 0.0, RgbImage::new(64, 64)));
    let c = [3_u8; 32];
    let ext = Arc::new(PatchGridExtractor {
        contract: c,
        jitter_on_worker: Arc::new(AtomicBool::new(false)),
    });
    let xfeat = LoadedXFeat::from_extractor(ext, Device::flex(), c);
    let store = Arc::new(
        CorridorTileStore::in_memory([0_u8; 32], c, DEFAULT_KASHIN_SEED, 250.0, 9)
            .map_err(|e| e.to_string())?,
    );
    let res = Arc::new(
        FlightEngineResources::from_parts(surface, store, xfeat, None, None, None, true)
            .map_err(|e| e.to_string())?,
    );
    let mut cfg = FlightEngineConfig {
        background_workers: false,
        enable_live_tile_synth: false,
        ..FlightEngineConfig::default()
    };
    cfg.search_opts.polish_anchors = false;
    let mut engine = FlightVisualEngine::new(res, cfg).map_err(|e| e.to_string())?;
    let make_inp = |seq: u32, ts: u64| LiveFrameInput {
        stamp: test_stamp(seq, ts),
        pixels: FramePixels::Gray8 {
            data: dark.as_raw(),
            width: 64,
            height: 48,
            stride: 64,
        },
        static_valid_mask: None,
        intrinsics: CameraIntrinsics {
            width: 64,
            height: 48,
            fx: 50.0,
            fy: 50.0,
            cx: 31.5,
            cy: 23.5,
            calibration_confidence: CalibrationMode::Calibrated,
        },
        mount: CameraMountProfile::rotary_gimbal_nadir(),
        imu_attitude: None,
        baro_alt_msl_m: None,
        filter_prior: None,
    };
    assert!(matches!(
        engine.process_frame(&make_inp(1, 0)),
        Err(FlightVisualError::NonMonotonicTimestamp(_))
    ));
    let _ = engine
        .process_frame(&make_inp(1, 100_000))
        .map_err(|e| e.to_string())?;
    assert!(matches!(
        engine.process_frame(&make_inp(2, 100_000)),
        Err(FlightVisualError::NonMonotonicTimestamp(_))
    ));
    Ok(())
}

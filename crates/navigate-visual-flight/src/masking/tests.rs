use super::*;
use navigate_match::segment::{PreprocessorConfig, SegformerConfig};

/// A masker with an untrained `Segformer`, so the test sees each call of
/// the model without a checkpoint on disk.
fn masker_with_untrained_segformer() -> AmortizedMasker {
    let device = Device::flex();
    let cfg = SegformerConfig::mit_b0(&["ground", "airplane", "sky"]);
    let model = Segformer::new(&cfg, &device);
    let policy = MaskPolicy::parse("airframe-occlusions").unwrap();
    let mut engine = SegEngine::from_loaded(
        model,
        &cfg,
        PreprocessorConfig::default(),
        policy,
        2,
        device,
    );
    engine.clamp_input_side(64);
    AmortizedMasker {
        seg: Some(Refresher::Inline(Box::new(engine))),
        refresh_interval: 1,
        propeller: PropellerDetector::default(),
        key: KeyLayers::default(),
    }
}

fn run(band: SpectralBand) -> MaskFrameTelemetry {
    let mut masker = masker_with_untrained_segformer();
    let rgb = RgbImage::from_pixel(64, 48, image::Rgb([140, 150, 120]));
    let luma = grayscale(&rgb);
    let k = Matrix3::new(50.0, 0.0, 31.5, 0.0, 50.0, 23.5, 0.0, 0.0, 1.0);
    let (_, telemetry, luma_p50) = masker
        .mask_frame_flight_fast(true, Some(&rgb), &luma, None, None, &k, true, false, band)
        .unwrap();
    assert!(
        luma_p50 >= NIGHT_BYPASS_LUMA_P50,
        "frame must be a day frame"
    );
    telemetry
}

#[test]
fn visible_day_frame_runs_segformer() {
    let telemetry = run(SpectralBand::Visible);
    assert!(telemetry.segformer_ms.is_some());
}

#[test]
fn long_wave_infrared_frame_skips_segformer() {
    let telemetry = run(SpectralBand::LongWaveInfrared);
    assert!(telemetry.segformer_ms.is_none());
    assert!(!telemetry.segformer_refreshed);
    assert!(!telemetry.segformer_submitted);
    assert!(telemetry.valid_fraction > 0.0);
}

#[test]
fn only_the_visible_band_runs_the_semantic_mask() {
    assert!(runs_semantic_mask(SpectralBand::Visible));
    assert!(!runs_semantic_mask(SpectralBand::NearInfrared));
    assert!(!runs_semantic_mask(SpectralBand::LongWaveInfrared));
}

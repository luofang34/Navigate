//! Integration and parity tests for `navigate-segment`.

use burn::tensor::Device;
use image::{GrayImage, Luma, Rgb, RgbImage};
use navigate_segment::{
    AIRFRAME, CLASSES, ColorMode, DatasetSample, GROUND, MaskPolicy, NpyArray, PROPELLER,
    PreprocessorConfig, PropellerDetector, SKY, Score, Segformer, SegformerConfig, TrainPolicy,
    class_masks, compose_labels, drop_specks, finetune, frame_masks, parse_object, read_npy,
    read_npz, split_name, surface_mask, write_npy, write_npz,
};
use std::{collections::BTreeMap, path::Path};

#[test]
fn object_labels_and_speck_filtering_match_airframe_labels_contract() {
    assert_eq!(parse_object("sky").expect("sky").label_id(), SKY);
    assert_eq!(
        parse_object("airframe_strut").expect("airframe").label_id(),
        AIRFRAME
    );
    assert_eq!(
        parse_object("propeller-left")
            .expect("propeller")
            .label_id(),
        PROPELLER
    );
    assert!(parse_object("ground").is_err());

    let mut mask = vec![false; 100 * 100];
    // 4-pixel speck (< 10 pixels = 0.001 * 10000) should be removed.
    mask[10 * 100 + 10] = true;
    mask[10 * 100 + 11] = true;
    mask[11 * 100 + 10] = true;
    mask[11 * 100 + 11] = true;
    // 25-pixel component (>= 10 pixels) should survive.
    for y in 40..45 {
        for x in 40..45 {
            mask[y * 100 + x] = true;
        }
    }
    let filtered = drop_specks(&mask, 100, 100, 0.001);
    assert!(!filtered[10 * 100 + 10]);
    assert_eq!(filtered.iter().filter(|&&b| b).count(), 25);

    let sky = vec![true, true, false, false];
    let air = vec![false, true, true, false];
    let prop = vec![false, false, true, false];
    let composed = compose_labels(&sky, &air, &prop).expect("compose");
    assert_eq!(composed, vec![SKY, AIRFRAME, PROPELLER, GROUND]);

    let s1 = split_name("flight-alpha", 20);
    assert!(s1 == "train" || s1 == "val");
    assert_eq!(s1, split_name("flight-alpha", 20));
}

#[test]
fn score_metrics_compute_iou_and_exclusion_rates() {
    let mut score = Score::new();
    let targets = vec![SKY, GROUND, GROUND, AIRFRAME, PROPELLER, 255];
    let preds = vec![SKY, GROUND, SKY, AIRFRAME, SKY, GROUND];
    score.update(&preds, &targets);

    // Both AIRFRAME -> AIRFRAME and PROPELLER -> SKY are excluded (!= GROUND), so recall is 1.0!
    assert_eq!(score.airframe_excluded, Some(1.0));
    // One of two GROUND pixels was predicted SKY, so ground_wrongly_excluded == 0.5.
    assert_eq!(score.ground_wrongly_excluded, Some(0.5));
    assert!(score.mean_iou.expect("miou") > 0.3);
}

#[test]
fn propeller_detector_waits_for_three_history_frames_and_rejects_static_struts() {
    let mut det = PropellerDetector::default();
    let w = 240u32;
    let h = 180u32;

    let make_frame = |bg: u8, blade_box: Option<(u32, u32, u32, u32)>| -> RgbImage {
        let mut img = RgbImage::from_pixel(w, h, Rgb([bg, bg, bg]));
        // Static dark strut present in every frame: must NOT be flagged as propeller.
        for y in 0..h {
            for x in 5..15 {
                img.put_pixel(x, y, Rgb([20, 20, 20]));
            }
        }
        if let Some((x0, y0, x1, y1)) = blade_box {
            for y in y0..y1 {
                for x in x0..x1 {
                    img.put_pixel(x, y, Rgb([30, 30, 30]));
                }
            }
        }
        img
    };

    // First 3 frames have < 3 history frames -> always all false even with rotating blades!
    for (bg, b) in [
        (140u8, Some((30, 10, 80, 50))),
        (150, Some((160, 10, 210, 50))),
        (160, Some((30, 125, 80, 165))),
    ] {
        let m = det.mask(&make_frame(bg, b));
        assert!(!m.iter().any(|&v| v));
    }

    // 4th frame with transient dark blade in centre -> detected, static strut at x=8 stays false!
    let m4 = det.mask(&make_frame(155, Some((90, 60, 150, 120))));
    assert!(m4[90 * (w as usize) + 120]);
    assert!(!m4[90 * (w as usize) + 8]);
}

#[test]
fn four_class_and_ade20k_video_frame_masks_follow_policy() {
    let rgb = RgbImage::from_pixel(64, 64, Rgb([120, 140, 100]));
    let n = 64 * 64;
    let mut labels = vec![GROUND; n];
    for x in 0..64 {
        for y in 0..16 {
            labels[y * 64 + x] = SKY;
        }
    }
    let conf = vec![0.9_f32; n];
    let cfg4 = SegformerConfig::mit_b0(&CLASSES);
    let (air, prop, sky) =
        class_masks(&rgb, &labels, &conf, &cfg4, 0.35, None).expect("4-class masks");
    assert_eq!(sky.iter().filter(|&&b| b).count(), 16 * 64);
    assert!(!air.iter().any(|&b| b));
    assert!(!prop.iter().any(|&b| b));

    let res = frame_masks(&rgb, &labels, &conf, &cfg4, 0.35, 2, None).expect("frame_masks");
    // Top sky rows plus 2-pixel erosion margin are excluded (`valid == false`), bottom is valid.
    assert!(!res.valid[5 * 64 + 32]);
    assert!(res.valid[50 * 64 + 32]);
}

#[test]
fn surface_mask_policies_preserve_interior_ground_and_exclude_attached_airframe() {
    let rgb = RgbImage::from_pixel(80, 80, Rgb([100, 130, 90]));
    let mut labels = vec![9u8; 80 * 80]; // 9 = grass
    let conf = vec![0.95_f32; 80 * 80];
    // Border-attached airplane component (`90`) at top-left, plus isolated interior `90` blob.
    for y in 0..12 {
        for x in 0..12 {
            labels[y * 80 + x] = 90;
        }
    }
    for y in 40..46 {
        for x in 40..46 {
            labels[y * 80 + x] = 90;
        }
    }
    let (attached_keep, summary) = surface_mask(
        &rgb,
        &labels,
        &conf,
        None,
        MaskPolicy::AttachedOcclusions,
        2,
    )
    .expect("attached-occlusions");
    assert!(!attached_keep[2 * 80 + 2]);
    // Interior detached blob is NOT attached to the border, so `AttachedOcclusions` retains it.
    assert!(attached_keep[43 * 80 + 43]);
    assert!(summary.valid_fraction > 0.8);
}

#[test]
fn npy_and_npz_round_trip_exact_shapes_and_values() {
    let dir = tempfile::tempdir().expect("tempdir");
    let npy_path = dir.path().join("arr.npy");
    let arr = NpyArray::from_f32(vec![2, 3], &[1.5, -2.0, 3.25, 4.0, 0.0, 8.5]).expect("f32");
    write_npy(&npy_path, &arr).expect("write npy");
    let back = read_npy(&npy_path).expect("read npy");
    assert_eq!(back.shape, vec![2, 3]);
    assert_eq!(back.to_f32(), vec![1.5, -2.0, 3.25, 4.0, 0.0, 8.5]);

    let npz_path = dir.path().join("bundle.npz");
    let mut map = BTreeMap::new();
    map.insert(
        "eye".into(),
        NpyArray::from_f64(vec![3], &[10.0, 20.0, 300.0]).expect("eye"),
    );
    map.insert(
        "mask".into(),
        NpyArray::from_bool(vec![2, 2], &[true, false, false, true]).expect("mask"),
    );
    write_npz(&npz_path, &map).expect("write npz");
    let loaded = read_npz(&npz_path).expect("read npz");
    assert_eq!(loaded["eye"].to_f64(), vec![10.0, 20.0, 300.0]);
    assert_eq!(loaded["mask"].to_u8().expect("u8"), vec![1, 0, 0, 1]);
}

#[test]
fn pure_rust_segformer_finetuning_decreases_loss_and_round_trips_safetensors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ds = dir.path().join("dataset");
    let out = dir.path().join("finetuned");
    for split in ["train", "val"] {
        std::fs::create_dir_all(ds.join("images").join(split)).expect("mkdir img");
        std::fs::create_dir_all(ds.join("labels").join(split)).expect("mkdir lbl");
        for idx in 0..2 {
            let mut rgb = RgbImage::new(64, 64);
            let mut lbl = GrayImage::new(64, 64);
            for y in 0..64 {
                for x in 0..64 {
                    if y < 24 {
                        rgb.put_pixel(x, y, Rgb([90, 165, 245]));
                        lbl.put_pixel(x, y, Luma([SKY]));
                    } else if x < 14 && y > 40 {
                        rgb.put_pixel(x, y, Rgb([235, 235, 235]));
                        lbl.put_pixel(x, y, Luma([AIRFRAME]));
                    } else {
                        rgb.put_pixel(x, y, Rgb([70, 125, 60]));
                        lbl.put_pixel(x, y, Luma([GROUND]));
                    }
                }
            }
            let img_p = ds.join("images").join(split).join(format!("s{idx}.png"));
            let lbl_p = ds.join("labels").join(split).join(format!("s{idx}.png"));
            rgb.save(&img_p).expect("save img");
            lbl.save(&lbl_p).expect("save lbl");
        }
    }

    let make_samples = |split: &str| -> Vec<DatasetSample> {
        (0..2)
            .map(|idx| DatasetSample {
                id: format!("s{idx}"),
                image_path: ds.join("images").join(split).join(format!("s{idx}.png")),
                label_path: ds.join("labels").join(split).join(format!("s{idx}.png")),
            })
            .collect()
    };

    let scratch_seg = Path::new(
        "/Users/claw/.gemini/antigravity-cli/brain/25d2bd9e-b8e2-456a-bd46-6725f71bea84/scratch/models/segmentation",
    );
    let device = Device::default();

    // If the real ADE20K SegFormer-B0 checkpoint is available in scratch, also verify pretrained
    // sky/grass inference before running fine-tuning!
    if scratch_seg.join("model.safetensors").is_file() {
        let (ade_model, _cfg, mut ade_prep) =
            Segformer::load_pretrained(scratch_seg, None, &device).expect("load ADE20K SegFormer");
        ade_prep.size = serde_json::json!(128);
        let mut sky_ground = RgbImage::new(128, 128);
        for y in 0..128 {
            for x in 0..128 {
                let color = if y < 56 {
                    Rgb([110, 175, 245])
                } else {
                    Rgb([65, 115, 50])
                };
                sky_ground.put_pixel(x, y, color);
            }
        }
        let pred = ade_model
            .predict_image(&sky_ground, &ade_prep, ColorMode::Color, &[2], &device)
            .expect("predict ADE20K");
        // ADE20K class 2 is `sky` — top centre should have much higher sky probability than bottom!
        let top_sky = pred.selected_probs[0][16 * 128 + 64];
        let bot_sky = pred.selected_probs[0][108 * 128 + 64];
        assert!(
            top_sky > bot_sky + 0.25,
            "expected top_sky ({top_sky}) >> bot_sky ({bot_sky})"
        );
    }

    let base = scratch_seg
        .join("model.safetensors")
        .is_file()
        .then_some(scratch_seg);
    let policy = TrainPolicy {
        size: 64,
        batch_size: 2,
        epochs: 4,
        encoder_lr: 2e-4,
        decoder_lr: 2e-3,
        weight_decay: 0.01,
        warmup_fraction: 0.25,
        color_mode: ColorMode::Color,
        seed: 7,
    };
    let report = finetune(
        base,
        &make_samples("train"),
        &make_samples("val"),
        &policy,
        &out,
        &device,
    )
    .expect("finetune");

    assert_eq!(report.history.len(), 4);
    assert!(
        report.history[3].train_loss < report.history[0].train_loss,
        "expected training loss to decrease: {:?}",
        report
            .history
            .iter()
            .map(|h| h.train_loss)
            .collect::<Vec<_>>()
    );
    assert!(out.join("model.safetensors").is_file());
    assert!(out.join("config.json").is_file());
    assert!(out.join("preprocessor_config.json").is_file());
    assert!(out.join("training.json").is_file());

    let (reloaded, reloaded_cfg, reloaded_prep) =
        Segformer::load_pretrained(&out, None, &device).expect("reload finetuned");
    assert_eq!(reloaded_cfg.num_labels(), 4);
    let sample_img = image::open(ds.join("images/val/s0.png"))
        .expect("open val")
        .to_rgb8();
    let pred: navigate_segment::SegmentationPrediction = reloaded
        .predict_image(
            &sample_img,
            &reloaded_prep,
            ColorMode::Color,
            &[0, 1],
            &device,
        )
        .expect("predict finetuned");
    assert_eq!(pred.labels.len(), 64 * 64);
    let _ = PreprocessorConfig::default();
}

//! Semantic mask generation, video propeller masking, label composition, and pure-Rust training.

use crate::{
    PipelineError,
    matcher_client::default_burn_device,
    testset::PhotoCase,
    util::{bytes_sha256, file_sha256, read_json, write_json},
};
use image::{GrayImage, Luma};
use navigate_segment::{
    ColorMode, MaskPolicy, PropellerDetector, Segformer, TrainPolicy, compose_labels,
    contact_sheet, finetune, finetune::discover_split, frame_masks, surface_mask,
};
use serde_json::{Value, json};
use std::{fs, path::Path};

/// Run pure-Rust SegFormer-B0 occlusion masking across a photo dataset (`photo_masks`).
///
/// Writes binary `255`/`0` valid masks under `masks/{tag}/{id}.png`, semantic class maps under
/// `semantic/{tag}/{id}.png`, and updates `profiles/with-prior.json` and `no-prior.json`.
///
/// # Errors
///
/// Returns [`PipelineError`] when model loading, image prediction, or profile writing fails.
pub fn run_photo_masks(
    dataset_dir: &Path,
    model_dir: &Path,
    policy_str: &str,
    margin_px: usize,
) -> Result<Value, PipelineError> {
    let policy = MaskPolicy::parse(policy_str)?;
    let (device, _) = default_burn_device();
    let (model, cfg, preproc) = Segformer::load_pretrained(model_dir, None, &device)?;
    let model_sha = file_sha256(&model_dir.join("model.safetensors"))?;
    let tag =
        &bytes_sha256(format!("{model_sha}:{}:{margin_px}", policy.as_str()).as_bytes())[..16];

    let mask_out_dir = dataset_dir.join("masks").join(tag);
    let sem_out_dir = dataset_dir.join("semantic").join(tag);
    fs::create_dir_all(&mask_out_dir).map_err(|source| PipelineError::Io {
        path: mask_out_dir.clone(),
        source,
    })?;
    fs::create_dir_all(&sem_out_dir).map_err(|source| PipelineError::Io {
        path: sem_out_dir.clone(),
        source,
    })?;

    let with_prior_path = dataset_dir.join("profiles").join("with-prior.json");
    let mut profile_doc: Value = read_json(&with_prior_path)?;
    let mut cases: Vec<PhotoCase> =
        serde_json::from_value(profile_doc["cases"].clone()).map_err(|source| {
            PipelineError::Json {
                context: "with-prior.cases".into(),
                source,
            }
        })?;

    let airplane_idx = cfg.class_index("airplane");
    let requested: Vec<usize> = airplane_idx.into_iter().collect();
    let mut summaries = Vec::with_capacity(cases.len());

    for case in &mut cases {
        let img_path = dataset_dir.join(&case.image);
        let rgb = image::open(&img_path)
            .map_err(|source| PipelineError::Image {
                path: img_path,
                source,
            })?
            .to_rgb8();
        let pred = model.predict_image(&rgb, &preproc, ColorMode::Color, &requested, &device)?;
        let air_probs = pred.selected_probs.first().map(Vec::as_slice);
        let (keep, summary) = surface_mask(
            &rgb,
            &pred.labels,
            &pred.confidence,
            air_probs,
            policy,
            margin_px,
        )?;

        let (w, h) = rgb.dimensions();
        let mut mask_img = GrayImage::new(w, h);
        let mut sem_img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                mask_img.put_pixel(x, y, Luma([if keep[idx] { 255 } else { 0 }]));
                sem_img.put_pixel(x, y, Luma([pred.labels[idx]]));
            }
        }

        let mask_rel = format!("masks/{tag}/{}.png", case.id);
        let sem_rel = format!("semantic/{tag}/{}.png", case.id);
        let mp = dataset_dir.join(&mask_rel);
        let sp = dataset_dir.join(&sem_rel);
        mask_img.save(&mp).map_err(|source| PipelineError::Image {
            path: mp.clone(),
            source,
        })?;
        sem_img.save(&sp).map_err(|source| PipelineError::Image {
            path: sp.clone(),
            source,
        })?;

        case.valid_mask = mask_rel;
        case.mask_sha256 = file_sha256(&mp)?;
        case.mask_status = "automatic_semantic_unreviewed".into();
        case.semantic_labels = Some(sem_rel);
        case.semantic_sha256 = Some(file_sha256(&sp)?);
        summaries.push(json!({
            "id": case.id,
            "valid_fraction": summary.valid_fraction,
            "sky_fraction": summary.sky_fraction,
            "obstruction_fraction": summary.obstruction_fraction,
        }));
    }

    profile_doc["cases"] = serde_json::to_value(&cases).map_err(|source| PipelineError::Json {
        context: "with-prior.cases".into(),
        source,
    })?;
    write_json(&with_prior_path, &profile_doc)?;

    let no_prior_path = dataset_dir.join("profiles").join("no-prior.json");
    if no_prior_path.is_file() {
        let stripped = crate::testset::build_cases(&cases, false);
        write_json(
            &no_prior_path,
            &json!({ "format": "photo-terrain-cases", "version": 1, "cases": stripped }),
        )?;
    }

    Ok(json!({
        "tag": tag,
        "policy": policy.as_str(),
        "masked_photos": summaries.len(),
        "photos": summaries,
    }))
}

/// Run SegFormer-B0 + temporal [`PropellerDetector`] over an ordered frame directory (`video_masks`).
///
/// # Errors
///
/// Returns [`PipelineError`] when reading frames, running inference, or writing masks fails.
pub fn run_video_masks(
    frames_dir: &Path,
    model_dir: &Path,
    output_dir: &Path,
    threshold: f32,
    margin_px: usize,
) -> Result<Value, PipelineError> {
    fs::create_dir_all(output_dir).map_err(|source| PipelineError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;
    let (device, _) = default_burn_device();
    let (model, cfg, preproc) = Segformer::load_pretrained(model_dir, None, &device)?;

    let mut frame_paths: Vec<_> = fs::read_dir(frames_dir)
        .map_err(|source| PipelineError::Io {
            path: frames_dir.to_path_buf(),
            source,
        })?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()).is_some_and(|ext| {
                matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg")
            })
        })
        .collect();
    frame_paths.sort();

    let mut detector = PropellerDetector::default();
    for fp in &frame_paths {
        let rgb = image::open(fp)
            .map_err(|source| PipelineError::Image {
                path: fp.clone(),
                source,
            })?
            .to_rgb8();
        let pred = model.predict_image(&rgb, &preproc, ColorMode::Color, &[], &device)?;
        let res = frame_masks(
            &rgb,
            &pred.labels,
            &pred.confidence,
            &cfg,
            threshold,
            margin_px,
            Some(&mut detector),
        )?;
        let (w, h) = rgb.dimensions();
        let mut out = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                out.put_pixel(x, y, Luma([if res.valid[idx] { 255 } else { 0 }]));
            }
        }
        let stem = fp.file_stem().and_then(|s| s.to_str()).unwrap_or("frame");
        let dst = output_dir.join(format!("{stem}.png"));
        out.save(&dst)
            .map_err(|source| PipelineError::Image { path: dst, source })?;
    }

    Ok(json!({
        "frames_masked": frame_paths.len(),
        "output": output_dir.display().to_string(),
    }))
}

/// Run pure-Rust 4-class SegFormer-B0 fine-tuning (`segment-finetune`) on `burn-autodiff`.
///
/// # Errors
///
/// Returns [`PipelineError`] when dataset discovery, training, or saving fails.
pub fn run_segment_finetune(
    dataset_dir: &Path,
    base_model_dir: Option<&Path>,
    output_dir: &Path,
    policy: &TrainPolicy,
) -> Result<Value, PipelineError> {
    let train_samples = discover_split(dataset_dir, "train")?;
    let val_samples = discover_split(dataset_dir, "val")?;
    let (device, _) = default_burn_device();
    let report = finetune(
        base_model_dir,
        &train_samples,
        &val_samples,
        policy,
        output_dir,
        &device,
    )?;
    serde_json::to_value(&report).map_err(|source| PipelineError::Json {
        context: "FinetuneReport".into(),
        source,
    })
}

/// Compose a 4-class training label PNG and review contact sheet (`airframe-labels`).
///
/// # Errors
///
/// Returns [`PipelineError`] when reading masks or saving label/preview PNGs fails.
pub fn run_airframe_compose(
    image_path: &Path,
    sky_mask_path: &Path,
    airframe_mask_path: Option<&Path>,
    propeller_mask_path: Option<&Path>,
    label_out: &Path,
    sheet_out: Option<&Path>,
) -> Result<Value, PipelineError> {
    let rgb = image::open(image_path)
        .map_err(|source| PipelineError::Image {
            path: image_path.to_path_buf(),
            source,
        })?
        .to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let read_bool_mask = |p: Option<&Path>| -> Result<Vec<bool>, PipelineError> {
        let Some(path) = p else {
            return Ok(vec![false; w * h]);
        };
        let g = image::open(path)
            .map_err(|source| PipelineError::Image {
                path: path.to_path_buf(),
                source,
            })?
            .to_luma8();
        Ok(g.as_raw().iter().map(|&v| v > 127).collect())
    };
    let sky = read_bool_mask(Some(sky_mask_path))?;
    let air = read_bool_mask(airframe_mask_path)?;
    let prop = read_bool_mask(propeller_mask_path)?;
    let labels = compose_labels(&sky, &air, &prop)?;

    let mut lbl_img = GrayImage::new(rgb.width(), rgb.height());
    for (dst, &v) in lbl_img.as_mut().iter_mut().zip(&labels) {
        *dst = v;
    }
    if let Some(parent) = label_out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    lbl_img
        .save(label_out)
        .map_err(|source| PipelineError::Image {
            path: label_out.to_path_buf(),
            source,
        })?;

    if let Some(sp) = sheet_out {
        let sheet = contact_sheet(&[(rgb, labels.clone())], 320, 1);
        sheet.save(sp).map_err(|source| PipelineError::Image {
            path: sp.to_path_buf(),
            source,
        })?;
    }
    Ok(json!({ "label_out": label_out.display().to_string(), "pixels": labels.len() }))
}

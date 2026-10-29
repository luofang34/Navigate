//! Pure-Rust 4-class SegFormer-B0 fine-tuning, evaluation metrics, and provenance writer
//! (`segment_finetune`).

pub mod augment;
pub mod score;

use crate::{
    SegmentError,
    config::{CLASSES, ColorMode, PreprocessorConfig, SegformerConfig},
    model::Segformer,
};
use augment::{Rng64, augment_pair, class_weights, one_cycle_lr};
use burn::{
    module::Module,
    optim::{AdamWConfig, GradientsParams},
    tensor::{Device, Tensor, TensorData, activation},
};
use image::{GrayImage, RgbImage};
pub use score::{EpochRecord, FinetuneReport, Score};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

/// One paired RGB image and 4-class label PNG on disk.
#[derive(Clone, Debug)]
pub struct DatasetSample {
    /// Sample stem identifier.
    pub id: String,
    /// Path to RGB frame (`images/{split}/{id}.png` or `.jpg`).
    pub image_path: PathBuf,
    /// Path to single-channel class index PNG (`labels/{split}/{id}.png`).
    pub label_path: PathBuf,
}

/// Hyperparameters and reproducibility policy for 4-class SegFormer-B0 fine-tuning.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainPolicy {
    /// Training and evaluation square resolution (`512` default).
    pub size: usize,
    /// Mini-batch size (`8` default).
    pub batch_size: usize,
    /// Number of training epochs (`40` default).
    pub epochs: usize,
    /// Peak encoder learning rate (`6e-5` default).
    pub encoder_lr: f64,
    /// Peak decode-head learning rate (`6e-4` default).
    pub decoder_lr: f64,
    /// Decoupled AdamW weight decay (`0.01` default).
    pub weight_decay: f32,
    /// Fraction of total steps spent in OneCycle warmup (`0.1` default).
    pub warmup_fraction: f64,
    /// Color augmentation mode (`RandGray` default).
    pub color_mode: ColorMode,
    /// Deterministic PRNG seed (`0` default).
    pub seed: u64,
}

impl Default for TrainPolicy {
    fn default() -> Self {
        Self {
            size: 512,
            batch_size: 8,
            epochs: 40,
            encoder_lr: 6e-5,
            decoder_lr: 6e-4,
            weight_decay: 0.01,
            warmup_fraction: 0.1,
            color_mode: ColorMode::RandGray,
            seed: 0,
        }
    }
}

/// Discover `images/{split}` and `labels/{split}` pairs inside `dataset_dir`.
///
/// # Errors
///
/// Returns [`SegmentError`] when directories are missing or label files do not match images.
pub fn discover_split(dataset_dir: &Path, split: &str) -> Result<Vec<DatasetSample>, SegmentError> {
    let img_dir = dataset_dir.join("images").join(split);
    let lbl_dir = dataset_dir.join("labels").join(split);
    let entries = fs::read_dir(&img_dir).map_err(|source| SegmentError::Io {
        path: img_dir.clone(),
        source,
    })?;
    let mut samples = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| SegmentError::Io {
            path: img_dir.clone(),
            source,
        })?;
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_ascii_lowercase);
        if !matches!(ext.as_deref(), Some("png" | "jpg" | "jpeg")) {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let label_path = lbl_dir.join(format!("{stem}.png"));
        if !label_path.is_file() {
            return Err(SegmentError::Invalid(format!(
                "missing label PNG {} for image {}",
                label_path.display(),
                path.display()
            )));
        }
        samples.push(DatasetSample {
            id: stem.to_owned(),
            image_path: path,
            label_path,
        });
    }
    samples.sort_by(|a, b| a.id.cmp(&b.id));
    if samples.is_empty() {
        return Err(SegmentError::Invalid(format!(
            "split '{split}' under {} contains zero samples",
            dataset_dir.display()
        )));
    }
    Ok(samples)
}

struct LoadedSample {
    rgb: RgbImage,
    labels: Vec<u8>,
}

fn load_samples(samples: &[DatasetSample]) -> Result<Vec<LoadedSample>, SegmentError> {
    let mut out = Vec::with_capacity(samples.len());
    for s in samples {
        let rgb = image::open(&s.image_path)
            .map_err(|source| SegmentError::Image {
                path: s.image_path.clone(),
                source,
            })?
            .to_rgb8();
        let lbl_img: GrayImage = image::open(&s.label_path)
            .map_err(|source| SegmentError::Image {
                path: s.label_path.clone(),
                source,
            })?
            .to_luma8();
        if rgb.dimensions() != lbl_img.dimensions() {
            return Err(SegmentError::Shape {
                name: s.id.clone(),
                expected: vec![rgb.height() as usize, rgb.width() as usize],
                actual: vec![lbl_img.height() as usize, lbl_img.width() as usize],
            });
        }
        out.push(LoadedSample {
            rgb,
            labels: lbl_img.into_raw(),
        });
    }
    Ok(out)
}

/// Compute differentiable class-weighted cross-entropy loss on `[B, 4, H, W]` upsampled logits
/// while ignoring target pixels `>= 4`.
pub fn weighted_cross_entropy_loss(
    logits: Tensor<4>,
    targets: &[u8],
    class_weights: &[f32],
) -> Tensor<1> {
    let [b, c, h, w] = logits.dims();
    let device = logits.device();
    let n = b * h * w;
    let flat_logits = logits.permute([0, 2, 3, 1]).reshape([n, c]);
    let log_probs = activation::log_softmax(flat_logits, 1);

    let mut selector = vec![0.0_f32; n * c];
    let mut valid_weight_sum = 0.0_f32;
    for (pix, &t) in targets.iter().take(n).enumerate() {
        let cls = usize::from(t);
        if cls < c {
            let w_cls = class_weights.get(cls).copied().unwrap_or(1.0);
            selector[pix * c + cls] = -w_cls;
            valid_weight_sum += w_cls;
        }
    }
    let denom = valid_weight_sum.max(1.0);
    let sel_tensor = Tensor::<2>::from_data(TensorData::new(selector, [n, c]), &device);
    (log_probs * sel_tensor)
        .sum()
        .div_scalar(denom)
        .reshape([1])
}

/// Evaluate `model` on `samples` without gradient tracking.
///
/// # Errors
///
/// Returns [`SegmentError`] when sample decoding or prediction readback fails.
pub fn evaluate(
    model: &Segformer,
    samples: &[DatasetSample],
    preprocessor: &PreprocessorConfig,
    color_mode: ColorMode,
    device: &Device,
) -> Result<Score, SegmentError> {
    let loaded = load_samples(samples)?;
    evaluate_loaded(model, &loaded, preprocessor, color_mode, device)
}

fn evaluate_loaded(
    model: &Segformer,
    loaded: &[LoadedSample],
    preprocessor: &PreprocessorConfig,
    color_mode: ColorMode,
    device: &Device,
) -> Result<Score, SegmentError> {
    let eval_mode = if color_mode == ColorMode::Gray {
        ColorMode::Gray
    } else {
        ColorMode::Color
    };
    let mut score = Score::new();
    for item in loaded {
        let pred = model.predict_image(&item.rgb, preprocessor, eval_mode, &[], device)?;
        score.update(&pred.labels, &item.labels);
    }
    Ok(score)
}

/// Fine-tune SegFormer-B0 on `(train_samples, val_samples)` and write HuggingFace artifacts plus
/// `training.json` into `output_dir`.
///
/// Pass `Some(base_dir)` to initialize the encoder and fusion layers from an existing HuggingFace
/// checkpoint, or `None` for fresh initialization (used in unit tests).
///
/// # Errors
///
/// Returns [`SegmentError`] when dataset loading, checkpoint loading, or output writing fails.
pub fn finetune(
    base_dir: Option<&Path>,
    train_samples: &[DatasetSample],
    val_samples: &[DatasetSample],
    policy: &TrainPolicy,
    output_dir: &Path,
    device: &Device,
) -> Result<FinetuneReport, SegmentError> {
    if train_samples.is_empty() || val_samples.is_empty() {
        return Err(SegmentError::Invalid(
            "both train and val splits must be non-empty".into(),
        ));
    }
    if policy.batch_size == 0 || policy.epochs == 0 || policy.size < 32 {
        return Err(SegmentError::Invalid(
            "invalid TrainPolicy (size >= 32, batch_size >= 1, epochs >= 1 required)".into(),
        ));
    }

    let train_loaded = load_samples(train_samples)?;
    let val_loaded = load_samples(val_samples)?;
    let label_refs: Vec<&[u8]> = train_loaded.iter().map(|s| s.labels.as_slice()).collect();
    let weights = class_weights(&label_refs, CLASSES.len());

    let train_device = device.clone().autodiff();
    let (mut model, mut config, mut preprocessor, base_sha) = match base_dir {
        Some(dir) => {
            let sha = file_sha256(&dir.join("model.safetensors"))?;
            let (m, cfg, prep) = Segformer::load_pretrained(dir, Some(&CLASSES), &train_device)?;
            (m, cfg, prep, Some(sha))
        }
        None => {
            let cfg = SegformerConfig::mit_b0(&CLASSES);
            let prep = PreprocessorConfig::default();
            let m = Segformer::new(&cfg, &train_device);
            (m, cfg, prep, None)
        }
    };
    config.set_classes(&CLASSES);
    preprocessor.reduce_labels = false;
    preprocessor.size = serde_json::json!(policy.size);

    let mut encoder_opt = AdamWConfig::new()
        .with_weight_decay(policy.weight_decay)
        .init();
    let mut decoder_opt = AdamWConfig::new()
        .with_weight_decay(policy.weight_decay)
        .init();

    let steps_per_epoch = train_loaded.len().div_ceil(policy.batch_size);
    let total_steps = steps_per_epoch * policy.epochs;
    let mut step_idx = 0usize;
    let mut rng = Rng64::new(policy.seed);
    let mut order: Vec<usize> = (0..train_loaded.len()).collect();

    let mut history = Vec::with_capacity(policy.epochs);
    let mut best_epoch = 1usize;
    let mut best_miou = -1.0_f64;
    let mut best_val_score = Score::new();
    let mut best_model: Option<Segformer> = None;

    for epoch in 1..=policy.epochs {
        rng.shuffle(&mut order);
        let mut loss_sum = 0.0_f64;
        let mut batch_count = 0usize;

        for chunk in order.chunks(policy.batch_size) {
            let b = chunk.len();
            let plane = policy.size * policy.size;
            let mut batch_rgb = Vec::with_capacity(b * 3 * plane);
            let mut batch_lbl = Vec::with_capacity(b * plane);
            for &idx in chunk {
                let (t_rgb, t_lbl) = augment_pair(
                    &train_loaded[idx].rgb,
                    &train_loaded[idx].labels,
                    policy.size,
                    policy.color_mode,
                    &preprocessor,
                    true,
                    &mut rng,
                );
                batch_rgb.extend_from_slice(&t_rgb);
                batch_lbl.extend_from_slice(&t_lbl);
            }

            let x = Tensor::<4>::from_data(
                TensorData::new(batch_rgb, [b, 3, policy.size, policy.size]),
                &train_device,
            );
            let logits = model.forward_upsampled(x, policy.size, policy.size);
            let loss = weighted_cross_entropy_loss(logits, &batch_lbl, &weights);
            let loss_scalar = loss
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| SegmentError::Safetensors(format!("{e:?}")))?[0];
            loss_sum += f64::from(loss_scalar);
            batch_count += 1;

            let enc_lr = one_cycle_lr(
                step_idx,
                total_steps,
                policy.encoder_lr,
                policy.warmup_fraction,
            );
            let dec_lr = one_cycle_lr(
                step_idx,
                total_steps,
                policy.decoder_lr,
                policy.warmup_fraction,
            );
            step_idx += 1;

            let mut raw_grads = loss.backward();
            let enc_grads = GradientsParams::from_module(&mut raw_grads, &model.encoder);
            let dec_grads = GradientsParams::from_module(&mut raw_grads, &model.decode_head);
            model.encoder = encoder_opt.step(enc_lr, model.encoder, enc_grads);
            model.decode_head = decoder_opt.step(dec_lr, model.decode_head, dec_grads);
        }

        let eval_model = model.clone().to_device(device);
        let val_score = evaluate_loaded(
            &eval_model,
            &val_loaded,
            &preprocessor,
            policy.color_mode,
            device,
        )?;
        let miou = val_score.mean_iou.unwrap_or(0.0);
        if miou >= best_miou {
            best_miou = miou;
            best_epoch = epoch;
            best_val_score = val_score.clone();
            best_model = Some(eval_model);
        }
        history.push(EpochRecord {
            epoch,
            train_loss: loss_sum / (batch_count.max(1) as f64),
            val_score,
        });
    }

    let saved = match best_model {
        Some(m) => m,
        None => model.to_device(device),
    };
    saved.save_pretrained(output_dir, &config, &preprocessor)?;

    let mut dataset_sha256 = BTreeMap::new();
    for s in train_samples.iter().chain(val_samples.iter()) {
        dataset_sha256.insert(
            s.image_path.display().to_string(),
            file_sha256(&s.image_path)?,
        );
        dataset_sha256.insert(
            s.label_path.display().to_string(),
            file_sha256(&s.label_path)?,
        );
    }

    let report = FinetuneReport {
        base_model: base_dir
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "scratch-mit-b0".into()),
        base_model_sha256: base_sha,
        classes: CLASSES.iter().map(|&s| s.to_owned()).collect(),
        class_weights: weights,
        policy: policy.clone(),
        dataset_sha256,
        history,
        best_epoch,
        best_val_score,
    };
    let mut json = serde_json::to_string_pretty(&report).map_err(|source| SegmentError::Json {
        context: "training.json".into(),
        source,
    })?;
    json.push('\n');
    let report_path = output_dir.join("training.json");
    fs::write(&report_path, json).map_err(|source| SegmentError::Io {
        path: report_path,
        source,
    })?;
    Ok(report)
}

fn file_sha256(path: &Path) -> Result<String, SegmentError> {
    let bytes = fs::read(path).map_err(|source| SegmentError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

//! Confusion matrix, IoU, safety exclusion metrics, and `training.json` schema.

use super::TrainPolicy;
use crate::config::{AIRFRAME, CLASSES, GROUND, PROPELLER};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Confusion matrix, per-class IoU, and safety exclusion metrics for 4-class segmentation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Score {
    /// `4 × 4` confusion matrix `confusion[true_class][pred_class]`.
    pub confusion: [[u64; 4]; 4],
    /// Per-class intersection-over-union (`sky`, `ground`, `airframe`, `propeller`).
    pub class_iou: BTreeMap<String, Option<f64>>,
    /// Mean IoU across classes with nonzero union.
    pub mean_iou: Option<f64>,
    /// Recall of `(airframe | propeller)` pixels excluded from terrain matching (`pred != GROUND`).
    pub airframe_excluded: Option<f64>,
    /// False-exclusion rate of `GROUND` pixels (`pred != GROUND` when `true == GROUND`).
    pub ground_wrongly_excluded: Option<f64>,
}

impl Default for Score {
    fn default() -> Self {
        Self::new()
    }
}

impl Score {
    /// Create an empty 4-class accumulator.
    pub fn new() -> Self {
        Self {
            confusion: [[0; 4]; 4],
            class_iou: BTreeMap::new(),
            mean_iou: None,
            airframe_excluded: None,
            ground_wrongly_excluded: None,
        }
    }

    /// Accumulate pixel predictions `predicted` against `targets` (ignoring `target >= 4`).
    pub fn update(&mut self, predicted: &[u8], targets: &[u8]) {
        for (&p, &t) in predicted.iter().zip(targets) {
            if t < 4 && p < 4 {
                self.confusion[usize::from(t)][usize::from(p)] += 1;
            }
        }
        self.recompute();
    }

    fn recompute(&mut self) {
        self.class_iou.clear();
        let mut iou_sum = 0.0_f64;
        let mut iou_count = 0usize;
        for (c, &name) in CLASSES.iter().enumerate() {
            let tp = self.confusion[c][c];
            let row_sum: u64 = self.confusion[c].iter().sum();
            let col_sum: u64 = (0..4).map(|r| self.confusion[r][c]).sum();
            let union = row_sum + col_sum - tp;
            let val = if union > 0 {
                let v = (tp as f64) / (union as f64);
                iou_sum += v;
                iou_count += 1;
                Some(v)
            } else {
                None
            };
            self.class_iou.insert(name.to_owned(), val);
        }
        self.mean_iou = (iou_count > 0).then(|| iou_sum / (iou_count as f64));

        let air = usize::from(AIRFRAME);
        let prop = usize::from(PROPELLER);
        let gnd = usize::from(GROUND);
        let air_total: u64 =
            self.confusion[air].iter().sum::<u64>() + self.confusion[prop].iter().sum::<u64>();
        let air_missed = self.confusion[air][gnd] + self.confusion[prop][gnd];
        self.airframe_excluded =
            (air_total > 0).then(|| ((air_total - air_missed) as f64) / (air_total as f64));

        let gnd_total: u64 = self.confusion[gnd].iter().sum();
        let gnd_kept = self.confusion[gnd][gnd];
        self.ground_wrongly_excluded =
            (gnd_total > 0).then(|| ((gnd_total - gnd_kept) as f64) / (gnd_total as f64));
    }
}

/// Per-epoch training log entry written into `training.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochRecord {
    /// 1-based epoch index.
    pub epoch: usize,
    /// Mean weighted cross-entropy training loss.
    pub train_loss: f64,
    /// Validation [`Score`] at the end of the epoch.
    pub val_score: Score,
}

/// Complete `training.json` provenance and metric report produced by [`super::finetune`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinetuneReport {
    /// Base checkpoint directory path.
    pub base_model: String,
    /// SHA-256 of base `model.safetensors` when loaded from disk.
    pub base_model_sha256: Option<String>,
    /// Class names (`["sky", "ground", "airframe", "propeller"]`).
    pub classes: Vec<String>,
    /// Inverse-sqrt class weights used during training.
    pub class_weights: Vec<f32>,
    /// Training policy and hyperparameters.
    pub policy: TrainPolicy,
    /// SHA-256 digests of every training and validation sample file.
    pub dataset_sha256: BTreeMap<String, String>,
    /// Epoch-by-epoch loss and validation history.
    pub history: Vec<EpochRecord>,
    /// Best validation epoch (`1..=epochs`).
    pub best_epoch: usize,
    /// Validation [`Score`] of the saved best checkpoint.
    pub best_val_score: Score,
}

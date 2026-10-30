//! Per-stage latency samples and nearest-rank percentiles for `corridor-track`.

use serde_json::{Map, Value};

/// Nearest-rank percentile `p` (`0.0..=100.0`) of `samples`, or `None` when empty.
pub(super) fn percentile(samples: &[f64], p: f64) -> Option<f64> {
    let mut sorted: Vec<f64> = samples.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    let rank = (p.clamp(0.0, 100.0) / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted.get(rank.clamp(1, sorted.len()) - 1).copied()
}

/// Latency samples in milliseconds, one list for each tracker stage.
#[derive(Debug, Default)]
pub(super) struct StageLatencies {
    /// Whole-frame time of each frame that the warm tracker solved.
    pub(super) warm_track_ms: Vec<f64>,
    /// Map-anchor acquisition time, measured on the thread that ran it.
    pub(super) anchor_ms: Vec<f64>,
    /// `Segformer` evaluation time of each refresh, measured on the thread that ran it.
    pub(super) seg_refresh_ms: Vec<f64>,
}

impl StageLatencies {
    /// Add the summary keys to `summary`; a key is `null` when its stage has no samples.
    pub(super) fn write_summary(&self, summary: &mut Map<String, Value>) {
        let entries = [
            ("warm_track_p50_ms", &self.warm_track_ms, 50.0),
            ("warm_track_p95_ms", &self.warm_track_ms, 95.0),
            ("warm_track_p99_ms", &self.warm_track_ms, 99.0),
            ("anchor_p50_ms", &self.anchor_ms, 50.0),
            ("seg_refresh_p50_ms", &self.seg_refresh_ms, 50.0),
        ];
        for (key, samples, p) in entries {
            summary.insert(key.to_owned(), percentile(samples, p).into());
        }
    }
}

#[cfg(test)]
mod tests;

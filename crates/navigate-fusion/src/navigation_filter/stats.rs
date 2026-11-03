//! Innovation statistics for each measurement kind (ADR-0012).

use std::collections::HashMap;

use crate::observation::MeasurementKind;

/// Innovation statistics of one measurement kind in one filter epoch.
///
/// The NIS is the chi-square statistic `νᵀ S⁻¹ ν` of an innovation. The
/// statistics include each admitted update that has an innovation. This
/// includes a delayed observation at its acquisition time. They do not
/// include the position fix that initializes the filter, because that
/// fix has no innovation. They do not include a replay of an update that
/// the filter admitted before.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InnovationStats {
    /// Number of admitted updates. The counter wraps at `u64::MAX`.
    pub admitted_count: u64,
    /// Mean NIS of the admitted updates. Zero before the first update.
    pub mean_nis: f64,
    /// NIS of the most recent admitted update. Zero before the first
    /// update.
    pub last_nis: f64,
    /// Number of innovation-gate refusals since the most recent admitted
    /// update. An admitted update sets it to zero. The counter wraps at
    /// `u64::MAX`.
    pub consecutive_gate_rejections: u64,
}

/// The statistics of every measurement kind that has data.
#[derive(Debug, Clone, Default)]
pub(super) struct StatsTable {
    by_kind: HashMap<MeasurementKind, InnovationStats>,
}

impl StatsTable {
    pub(super) fn get(&self, kind: MeasurementKind) -> Option<InnovationStats> {
        self.by_kind.get(&kind).copied()
    }

    pub(super) fn clear(&mut self) {
        self.by_kind.clear();
    }

    /// Adds one admitted update with the statistic `nis`.
    pub(super) fn record_admitted(&mut self, kind: MeasurementKind, nis: f64) {
        let stats = self.by_kind.entry(kind).or_default();
        stats.admitted_count = stats.admitted_count.wrapping_add(1);
        // A wrapped count restarts the mean at the new sample.
        let count = stats.admitted_count.max(1) as f64;
        stats.mean_nis += (nis - stats.mean_nis) / count;
        stats.last_nis = nis;
        stats.consecutive_gate_rejections = 0;
    }

    /// Adds one innovation-gate refusal.
    pub(super) fn record_gate_rejection(&mut self, kind: MeasurementKind) {
        let stats = self.by_kind.entry(kind).or_default();
        stats.consecutive_gate_rejections = stats.consecutive_gate_rejections.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{InnovationStats, StatsTable};
    use crate::observation::MeasurementKind;

    #[test]
    fn mean_last_and_streak_follow_the_records() {
        let mut table = StatsTable::default();
        assert_eq!(table.get(MeasurementKind::Range), None);
        table.record_gate_rejection(MeasurementKind::Range);
        table.record_gate_rejection(MeasurementKind::Range);
        assert_eq!(
            table.get(MeasurementKind::Range),
            Some(InnovationStats {
                consecutive_gate_rejections: 2,
                ..InnovationStats::default()
            })
        );
        table.record_admitted(MeasurementKind::Range, 1.0);
        table.record_admitted(MeasurementKind::Range, 3.0);
        let stats = table.get(MeasurementKind::Range).expect("recorded");
        assert_eq!(stats.admitted_count, 2);
        assert!((stats.mean_nis - 2.0).abs() < 1e-12);
        assert!((stats.last_nis - 3.0).abs() < 1e-12);
        assert_eq!(stats.consecutive_gate_rejections, 0);
        assert_eq!(table.get(MeasurementKind::Altitude), None);
        table.clear();
        assert_eq!(table.get(MeasurementKind::Range), None);
    }
}

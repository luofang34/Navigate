//! Innovation statistics for each measurement kind (ADR-0012).
//!
//! The table has one inline slot for each measurement kind
//! (ADR-0016). [`MAX_MEASUREMENT_KINDS`] is larger than the number of
//! [`MeasurementKind`] variants, so every kind always has a slot.

use crate::observation::MeasurementKind;

/// Number of slots in the statistics table.
pub(crate) const MAX_MEASUREMENT_KINDS: usize = 20;

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
#[derive(Debug, Clone, Copy)]
pub(crate) struct StatsTable {
    slots: [(Option<MeasurementKind>, InnovationStats); MAX_MEASUREMENT_KINDS],
}

impl Default for StatsTable {
    fn default() -> Self {
        Self {
            slots: [(None, InnovationStats::default()); MAX_MEASUREMENT_KINDS],
        }
    }
}

impl StatsTable {
    pub(crate) fn get(&self, kind: MeasurementKind) -> Option<InnovationStats> {
        self.slots
            .iter()
            .find_map(|(slot_kind, stats)| (*slot_kind == Some(kind)).then_some(*stats))
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    /// The statistics of `kind`. A kind without data takes the first
    /// free slot. The result is `None` only when every slot holds
    /// another kind, which the slot count prevents.
    fn entry(&mut self, kind: MeasurementKind) -> Option<&mut InnovationStats> {
        let index = self
            .slots
            .iter()
            .position(|(slot_kind, _)| *slot_kind == Some(kind))
            .or_else(|| {
                self.slots
                    .iter()
                    .position(|(slot_kind, _)| slot_kind.is_none())
            })?;
        let (slot_kind, stats) = self.slots.get_mut(index)?;
        *slot_kind = Some(kind);
        Some(stats)
    }

    /// Adds one admitted update with the statistic `nis`.
    pub(crate) fn record_admitted(&mut self, kind: MeasurementKind, nis: f64) {
        let Some(stats) = self.entry(kind) else {
            return;
        };
        stats.admitted_count = stats.admitted_count.wrapping_add(1);
        // A wrapped count restarts the mean at the new sample.
        let count = stats.admitted_count.max(1) as f64;
        stats.mean_nis += (nis - stats.mean_nis) / count;
        stats.last_nis = nis;
        stats.consecutive_gate_rejections = 0;
    }

    /// Adds one innovation-gate refusal.
    pub(crate) fn record_gate_rejection(&mut self, kind: MeasurementKind) {
        let Some(stats) = self.entry(kind) else {
            return;
        };
        stats.consecutive_gate_rejections = stats.consecutive_gate_rejections.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{InnovationStats, MAX_MEASUREMENT_KINDS, StatsTable};
    use crate::observation::MeasurementKind;

    /// Every measurement kind. The match in [`listed`] fails to compile
    /// when a kind is added, so this list stays complete.
    const EVERY_KIND: [MeasurementKind; 16] = [
        MeasurementKind::PositionFix,
        MeasurementKind::VelocityFix,
        MeasurementKind::Range,
        MeasurementKind::Altitude,
        MeasurementKind::MagneticAnomaly,
        MeasurementKind::Pseudorange,
        MeasurementKind::VisualPose,
        MeasurementKind::TerrainClearance,
        MeasurementKind::AttitudeFix,
        MeasurementKind::CartesianPositionFix,
        MeasurementKind::CartesianVelocityFix,
        MeasurementKind::CartesianRange,
        MeasurementKind::PseudorangeRate,
        MeasurementKind::TwoWayRange,
        MeasurementKind::TwoWayRangeRate,
        MeasurementKind::LineOfSight,
    ];

    const fn listed(kind: MeasurementKind) -> bool {
        match kind {
            MeasurementKind::PositionFix
            | MeasurementKind::VelocityFix
            | MeasurementKind::Range
            | MeasurementKind::Altitude
            | MeasurementKind::MagneticAnomaly
            | MeasurementKind::Pseudorange
            | MeasurementKind::VisualPose
            | MeasurementKind::TerrainClearance
            | MeasurementKind::AttitudeFix
            | MeasurementKind::CartesianPositionFix
            | MeasurementKind::CartesianVelocityFix
            | MeasurementKind::CartesianRange
            | MeasurementKind::PseudorangeRate
            | MeasurementKind::TwoWayRange
            | MeasurementKind::TwoWayRangeRate
            | MeasurementKind::LineOfSight => true,
        }
    }

    #[test]
    fn every_measurement_kind_has_its_own_slot() {
        assert!(EVERY_KIND.len() < MAX_MEASUREMENT_KINDS);
        let mut table = StatsTable::default();
        for (index, kind) in EVERY_KIND.into_iter().enumerate() {
            assert!(listed(kind));
            table.record_admitted(kind, index as f64);
        }
        for (index, kind) in EVERY_KIND.into_iter().enumerate() {
            let stats = table.get(kind).expect("a slot for each kind");
            assert_eq!(stats.admitted_count, 1);
            assert!((stats.last_nis - index as f64).abs() < 1e-12);
        }
    }

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

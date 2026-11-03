//! Solution integrity derivation (ADR-0004): redundancy judged over the
//! distinct sources that contributed within the assessment window and
//! how their declared compositions overlap, quality derived mechanically
//! from uncertainty, observation silence, and a storm of innovation-gate
//! refusals — never asserted, never overstated.
//!
//! A surface solution gets an [`IntegrityAssessment`]. A space solution
//! gets a [`SpaceIntegrityAssessment`] with its uncertainty in orbital
//! axes (ADR-0015).

use navigate_contract::{
    FaultDetection, IntegrityAssessment, MonotonicNanos, Redundancy, SolutionQuality,
    SourceComposition, SourceId, SpaceIntegrityAssessment, SpaceUncertainty,
};

use crate::config::FusionConfig;
use crate::provenance::{AdmittedContribution, ProvenanceLedger};

mod space;

#[cfg(test)]
mod tests;

pub(crate) use space::space_uncertainty;

/// What the assessment window supports, apart from the uncertainty.
struct Support {
    contributing: SourceComposition,
    redundancy: Redundancy,
    fault_detection: FaultDetection,
}

/// Derives the surface assessment for one publication at `now`. The
/// ledger supplies the admissions inside the assessment window, the
/// observation silence, and the innovation-gate refusal streak.
/// Protection levels stay absent until an implementation earns them.
pub(crate) fn derive(
    ledger: &ProvenanceLedger,
    now: MonotonicNanos,
    horizontal_1sigma_m: f64,
    vertical_1sigma_m: f64,
    config: &FusionConfig,
) -> IntegrityAssessment {
    let support = support_of(ledger.window());
    let quality = quality_of(ledger, now, horizontal_1sigma_m, vertical_1sigma_m, config);
    IntegrityAssessment::new(
        quality,
        support.contributing,
        support.redundancy,
        horizontal_1sigma_m,
        vertical_1sigma_m,
        support.fault_detection,
    )
}

/// Derives the space assessment for one publication at `now`. A
/// Cartesian frame has no vertical, so the quality judges the largest
/// principal 1-sigma against both the horizontal and the vertical
/// bounds.
pub(crate) fn derive_space(
    ledger: &ProvenanceLedger,
    now: MonotonicNanos,
    uncertainty: SpaceUncertainty,
    config: &FusionConfig,
) -> SpaceIntegrityAssessment {
    let support = support_of(ledger.window());
    let largest_m = uncertainty.max_axis_1sigma_m();
    let quality = quality_of(ledger, now, largest_m, largest_m, config);
    SpaceIntegrityAssessment::new(
        quality,
        support.contributing,
        support.redundancy,
        uncertainty,
        support.fault_detection,
    )
}

fn support_of(records: &[AdmittedContribution]) -> Support {
    let contributing = records
        .iter()
        .fold(SourceComposition::empty(), |acc, record| {
            acc.union(record.composition)
        });
    let redundancy = redundancy_of(records);
    let fault_detection = match redundancy {
        // A single source agreeing with itself proves nothing.
        Redundancy::None => FaultDetection::Unavailable,
        _ => FaultDetection::Monitoring,
    };
    Support {
        contributing,
        redundancy,
        fault_detection,
    }
}

/// Redundancy over the distinct-source set: one source is never
/// redundant with itself no matter how many classes it declares; a pair
/// of sources with disjoint compositions cross-checks across classes;
/// sources whose declarations all pairwise overlap corroborate only
/// within a shared class.
fn redundancy_of(records: &[AdmittedContribution]) -> Redundancy {
    let sources = per_source_compositions(records);
    if sources.len() < 2 {
        return Redundancy::None;
    }
    let any_disjoint = sources.iter().enumerate().any(|(i, (_, left))| {
        sources
            .iter()
            .skip(i.wrapping_add(1))
            .any(|(_, right)| !left.overlaps(*right))
    });
    if any_disjoint {
        Redundancy::IndependentClasses
    } else {
        Redundancy::SameClass
    }
}

/// Each distinct source paired with the union of every composition it
/// declared inside the window.
fn per_source_compositions(records: &[AdmittedContribution]) -> Vec<(SourceId, SourceComposition)> {
    let mut sources: Vec<(SourceId, SourceComposition)> = Vec::with_capacity(records.len());
    for record in records {
        match sources.iter_mut().find(|(id, _)| *id == record.source) {
            Some((_, composition)) => *composition = composition.union(record.composition),
            None => sources.push((record.source, record.composition)),
        }
    }
    sources
}

/// The quality from the uncertainty, then the silence, then the
/// innovation-gate refusal streak. The silence and the streak only make
/// the quality worse.
fn quality_of(
    ledger: &ProvenanceLedger,
    now: MonotonicNanos,
    horizontal_1sigma_m: f64,
    vertical_1sigma_m: f64,
    config: &FusionConfig,
) -> SolutionQuality {
    let horizontal = axis_quality(
        horizontal_1sigma_m,
        config.good_horizontal_1sigma_m,
        config.degraded_horizontal_1sigma_m,
    );
    let vertical = axis_quality(
        vertical_1sigma_m,
        config.good_vertical_1sigma_m,
        config.degraded_vertical_1sigma_m,
    );
    let mut quality = worse_quality(horizontal, vertical);
    if let Some(silence) = ledger.silence(now) {
        if silence >= config.quality_silence_unusable {
            quality = SolutionQuality::Unusable;
        } else if silence >= config.quality_silence_degraded {
            quality = worse_quality(quality, SolutionQuality::Degraded);
        }
    }
    // A run of refused updates means the state and the sources disagree.
    // The covariance does not show it, because a refused update leaves
    // the covariance unchanged.
    let limit = config.max_consecutive_gate_rejections;
    if limit > 0 && ledger.gate_rejection_streak() >= limit {
        quality = worse_quality(quality, SolutionQuality::Degraded);
    }
    quality
}

fn axis_quality(sigma_m: f64, good_m: f64, degraded_m: f64) -> SolutionQuality {
    if sigma_m <= good_m {
        SolutionQuality::Good
    } else if sigma_m <= degraded_m {
        SolutionQuality::Degraded
    } else {
        SolutionQuality::Unusable
    }
}

const fn worse_quality(a: SolutionQuality, b: SolutionQuality) -> SolutionQuality {
    match (a, b) {
        (SolutionQuality::Good, SolutionQuality::Good) => SolutionQuality::Good,
        (SolutionQuality::Unusable, _) | (_, SolutionQuality::Unusable) => {
            SolutionQuality::Unusable
        }
        _ => SolutionQuality::Degraded,
    }
}

//! Solution integrity derivation (ADR-0004): redundancy judged over the
//! distinct sources that contributed within the assessment window and
//! how their declared compositions overlap, quality derived mechanically
//! from uncertainty and observation silence — never asserted, never
//! overstated.

use navigate_contract::{
    DurationNanos, FaultDetection, IntegrityAssessment, Redundancy, SolutionQuality,
    SourceComposition, SourceId,
};

use super::AdmittedContribution;
use crate::config::FusionConfig;

/// Derives the assessment for one publication. `records` are the
/// admissions still inside the assessment window; `silence` is the time
/// since the last admitted observation, `None` when the reference clock
/// has not passed that admission. Protection levels stay absent until an
/// implementation earns them.
pub(crate) fn derive(
    records: &[AdmittedContribution],
    horizontal_1sigma_m: f64,
    vertical_1sigma_m: f64,
    silence: Option<DurationNanos>,
    config: &FusionConfig,
) -> IntegrityAssessment {
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
    let quality = quality_of(horizontal_1sigma_m, silence, config);
    IntegrityAssessment::new(
        quality,
        contributing,
        redundancy,
        horizontal_1sigma_m,
        vertical_1sigma_m,
        fault_detection,
    )
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

fn quality_of(
    horizontal_1sigma_m: f64,
    silence: Option<DurationNanos>,
    config: &FusionConfig,
) -> SolutionQuality {
    let mut quality = if horizontal_1sigma_m <= config.good_horizontal_1sigma_m {
        SolutionQuality::Good
    } else if horizontal_1sigma_m <= config.degraded_horizontal_1sigma_m {
        SolutionQuality::Degraded
    } else {
        SolutionQuality::Unusable
    };
    if let Some(silence) = silence {
        if silence >= config.quality_silence_unusable {
            quality = SolutionQuality::Unusable;
        } else if silence >= config.quality_silence_degraded
            && matches!(quality, SolutionQuality::Good)
        {
            quality = SolutionQuality::Degraded;
        }
    }
    quality
}

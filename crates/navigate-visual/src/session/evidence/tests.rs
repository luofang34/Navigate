use super::*;
use crate::session::{ContinuityEpoch, StreamId};

#[cfg(test)]
fn frame(index: u64) -> FrameKey {
    FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index,
    }
}

#[cfg(test)]
fn cell(manifest: &str) -> MapCell {
    MapCell {
        map_manifest_sha256: manifest.into(),
        east: 3,
        north: 4,
    }
}

#[test]
fn a_second_anchor_in_one_cell_shares_the_map_error() {
    let mut ledger = Ledger::default();
    assert_eq!(
        ledger.label(cell("a"), frame(1), 16),
        FusionEligibility::Independent
    );
    assert!(matches!(
        ledger.label(cell("a"), frame(2), 16),
        FusionEligibility::SharedMapError { .. }
    ));
}

/// Tracked work (ADR-0022 §7): the map error of a cell belongs to the
/// source of the imagery, not to the package that carries it. Two packages
/// cut from the same imagery have different manifest digests, so a ledger
/// keyed by the manifest digest counts their fixes in one cell as
/// independent. The visual means of `navigate-onboard` must key `MapCell`
/// by `navigate_data::CorrelationKey` (source, revision, cell at the layer
/// correlation zoom); it then makes this test pass and removes the ignore.
#[test]
#[ignore = "tracked: ADR-0022 §7 requires MapCell keyed by the source correlation key, not the manifest digest"]
fn two_packages_of_one_source_share_the_map_error_of_a_cell() {
    let mut ledger = Ledger::default();
    let first_package = cell("manifest-of-package-1");
    let second_package = cell("manifest-of-package-2");
    assert_eq!(
        ledger.label(first_package, frame(1), 16),
        FusionEligibility::Independent
    );
    assert!(matches!(
        ledger.label(second_package, frame(2), 16),
        FusionEligibility::SharedMapError { .. }
    ));
}

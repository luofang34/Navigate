//! The names that the alignment gives a host: what the records of its
//! inputs say, and, through [`crate::AlignmentNeed::name`], what an
//! alignment waits for. A host shows and counts them, so a name never
//! changes; a new case gets a new name.

/// Record reason: the input came before the first state, and the
/// alignment took what it can use of it.
pub(crate) const ALIGNMENT: &str = "alignment";
/// Record reason: the navigator uses this sensor only to align.
pub(crate) const ALIGNMENT_ONLY: &str = "alignment_only";
/// Measurement name of a magnetometer record.
pub(crate) const MAGNETIC_FIELD: &str = "magnetic_field";
/// Measurement name of a surveyed-point record.
pub(crate) const SURVEYED_POSITION: &str = "surveyed_position";
/// Measurement name of a declared-start record.
pub(crate) const DECLARED_START: &str = "declared_start";
/// Measurement name of a known-position record.
pub(crate) const KNOWN_POSITION: &str = "known_position";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_alignment_names_do_not_change() {
        use crate::AlignmentNeed as Need;
        assert_eq!(
            [
                Need::InertialIncrements,
                Need::RestWindow,
                Need::RestSamples,
                Need::Rest,
                Need::RestEvidence,
                Need::HeadingSource,
                Need::FieldStrength,
                Need::HeadingUncertain,
                Need::PositionSource,
                Need::VelocityFixes,
                Need::VelocityAccuracy,
                Need::InertialGap,
                Need::ForceDisagreement,
                Need::CourseSpeed,
                Need::SteadyMotion,
                Need::HeightDatum,
            ]
            .map(Need::name),
            [
                "inertial_increments",
                "rest_window",
                "rest_samples",
                "rest",
                "rest_evidence",
                "heading_source",
                "field_strength",
                "heading_uncertain",
                "position_source",
                "velocity_fixes",
                "velocity_accuracy",
                "inertial_gap",
                "force_disagreement",
                "course_speed",
                "steady_motion",
                "height_datum",
            ]
        );
        assert_eq!([ALIGNMENT, ALIGNMENT_ONLY], ["alignment", "alignment_only"]);
        assert_eq!(
            [
                MAGNETIC_FIELD,
                SURVEYED_POSITION,
                KNOWN_POSITION,
                DECLARED_START
            ],
            [
                "magnetic_field",
                "surveyed_position",
                "known_position",
                "declared_start"
            ]
        );
    }
}

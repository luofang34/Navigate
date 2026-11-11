//! The names that the alignment gives a host: what an alignment waits for,
//! and what the records of its inputs say. A host shows and counts them,
//! so a name never changes; a new case gets a new name.

/// No inertial increment has come.
pub(crate) const INERTIAL_INCREMENTS: &str = "inertial_increments";
/// The rest window is not complete.
pub(crate) const REST_WINDOW: &str = "rest_window";
/// The window holds too few inertial increments to judge rest by their
/// scatter.
pub(crate) const REST_SAMPLES: &str = "rest_samples";
/// The specific force, the body rate, or the GNSS speed of the window is
/// not that of a vehicle at rest.
pub(crate) const REST: &str = "rest";
/// No sensor says that the vehicle is at rest: a GNSS fix with a velocity
/// or a surveyed point.
pub(crate) const REST_EVIDENCE: &str = "rest_evidence";
/// No magnetometer is declared, or it gave no usable field.
pub(crate) const HEADING_SOURCE: &str = "heading_source";
/// The heading of the magnetometer is less certain than the alignment
/// accepts.
pub(crate) const HEADING_UNCERTAIN: &str = "heading_uncertain";
/// No GNSS fix with a covariance and no surveyed point has come.
pub(crate) const POSITION_SOURCE: &str = "position_source";

/// Record reason: the input came before the first state, and the
/// alignment took what it can use of it.
pub(crate) const ALIGNMENT: &str = "alignment";
/// Record reason: the navigator uses this sensor only to align.
pub(crate) const ALIGNMENT_ONLY: &str = "alignment_only";
/// Measurement name of a magnetometer record.
pub(crate) const MAGNETIC_FIELD: &str = "magnetic_field";
/// Measurement name of a surveyed-point record.
pub(crate) const SURVEYED_POSITION: &str = "surveyed_position";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_alignment_names_do_not_change() {
        assert_eq!(
            [
                INERTIAL_INCREMENTS,
                REST_WINDOW,
                REST_SAMPLES,
                REST,
                REST_EVIDENCE,
                HEADING_SOURCE,
                HEADING_UNCERTAIN,
                POSITION_SOURCE,
            ],
            [
                "inertial_increments",
                "rest_window",
                "rest_samples",
                "rest",
                "rest_evidence",
                "heading_source",
                "heading_uncertain",
                "position_source",
            ]
        );
        assert_eq!([ALIGNMENT, ALIGNMENT_ONLY], ["alignment", "alignment_only"]);
        assert_eq!(
            [MAGNETIC_FIELD, SURVEYED_POSITION],
            ["magnetic_field", "surveyed_position"]
        );
    }
}

use super::*;

#[test]
fn naif_frame_ids_are_not_silently_used_as_anise_orientation_ids() {
    let almanac = Almanac::default();
    let provider = AniseEphemeris::new(&almanac, &[]);
    let state = State {
        position_m: [6_378_137.0, 0.0, 0.0],
        velocity_m_s: [0.0; 3],
        epoch: Epoch::from_tdb_seconds(0.0),
        frame: FrameId {
            center: 399,
            orientation: 13000,
        },
    };
    assert!(matches!(
        provider.barycentric_state(state),
        Err(AniseError::UnknownOrientation(13000))
    ));
    let configured = AniseEphemeris::new(&almanac, &[(13000, 3000)]);
    assert!(matches!(
        configured.barycentric_state(state),
        Err(AniseError::Evaluation(_))
    ));
}

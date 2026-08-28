//! Leg-vocabulary tests: resolution, activation refusal, and the
//! sequencing each flown terminator earns (`docs/leg-requirements.md`).

use navigate_contract::CourseReference;

use super::*;

/// Due east, the course the equatorial fixtures fly.
const EASTBOUND_RAD: f64 = core::f64::consts::FRAC_PI_2;

/// Every reserved terminator, so the refusal test walks the whole
/// reserved set rather than a sample of it.
const RESERVED: [LegPath; 8] = [
    LegPath::RadiusToFix,
    LegPath::HoldToAltitude,
    LegPath::HoldToFix,
    LegPath::HoldManual,
    LegPath::VerticalTakeoff,
    LegPath::Hover,
    LegPath::Transition,
    LegPath::VerticalLanding,
];

fn eastbound_course(reference: CourseReference) -> LegPath {
    LegPath::CourseToFix {
        course_rad: EASTBOUND_RAD,
        reference,
    }
}

/// The resolved terminator of the leg the execution is flying.
fn active_path(exec: &PlanExecution) -> LegPath {
    exec.active_leg().expect("an active leg").path
}

#[test]
fn nav_lg_002_an_undeclared_plan_flies_direct_then_tracks() {
    let mut exec = execution(
        vec![wp("W0", 0.0, 0.0), wp("W1", 0.0, 0.5), wp("W2", 0.5, 0.5)],
        ExecutionConfig::default(),
    );
    assert_eq!(
        active_path(&exec),
        LegPath::DirectToFix,
        "the first leg runs from the present position"
    );
    capture_w0(&mut exec);
    assert_eq!(
        active_path(&exec),
        LegPath::TrackToFix,
        "every later leg is the implicit great-circle track"
    );
}

#[test]
fn nav_lg_002_a_declared_terminator_overrides_the_position_rule() {
    let exec = execution(
        vec![
            wp("W0", 0.0, 0.0).with_path(LegPath::InitialFix),
            wp("W1", 0.0, 0.5),
        ],
        ExecutionConfig::default(),
    );
    assert_eq!(active_path(&exec), LegPath::InitialFix);
}

#[test]
fn nav_lg_005_every_reserved_terminator_is_refused_at_activation_by_name() {
    for path in RESERVED {
        let plan = FlightPlan::new(
            "reserved".into(),
            PlanRole::Mission,
            vec![wp("W0", 0.0, 0.0), wp("W1", 0.0, 0.5).with_path(path)],
        );
        plan.validate()
            .expect("the plan itself is structurally valid");
        let refused = PlanExecution::new(plan, ExecutionConfig::default())
            .expect_err("this build has no sequencer for it");
        assert!(
            matches!(
                &refused,
                PlanActivationError::UnsupportedLegKind { index: 1, ident, path: refused_path, .. }
                    if ident == "W1" && *refused_path == path
            ),
            "{path} must be refused by name: {refused:?}"
        );
        assert!(
            refused.to_string().contains(path.code()),
            "the message must name {path}: {refused}"
        );
    }
}

/// NAV-LG-005 and NAV-LG-012: a reserved leg is refused by its kind
/// before any geometry guard reads it. A hover holds one position, so
/// the minimum-leg-length guard would refuse it for the wrong reason and
/// hide which leg this build cannot fly.
#[test]
fn nav_lg_012_a_reserved_leg_is_refused_by_kind_not_by_a_geometry_guard() {
    let plan = FlightPlan::new(
        "hover".into(),
        PlanRole::Mission,
        vec![
            wp("W0", 0.0, 0.0),
            wp("W1", 0.0, 0.0).with_path(LegPath::Hover),
        ],
    );
    assert!(matches!(
        PlanExecution::new(plan, ExecutionConfig::default()),
        Err(PlanActivationError::UnsupportedLegKind {
            path: LegPath::Hover,
            ..
        })
    ));
}

#[test]
fn nav_lg_005_a_magnetic_course_is_refused_and_the_true_one_activates() {
    let plan = |reference| {
        FlightPlan::new(
            "course".into(),
            PlanRole::Mission,
            vec![
                wp("W0", 0.0, 0.0),
                wp("W1", 0.0, 0.5).with_path(eastbound_course(reference)),
            ],
        )
    };
    let refused = PlanExecution::new(plan(CourseReference::Magnetic), ExecutionConfig::default())
        .expect_err("no magnetic variation model exists");
    assert!(matches!(
        refused,
        PlanActivationError::MagneticCourseUnsupported { index: 1, ident, .. } if ident == "W1"
    ));
    // The same geometry against true north is flyable, so the refusal
    // is about the reference and not about the course.
    assert!(PlanExecution::new(plan(CourseReference::True), ExecutionConfig::default()).is_ok());
}

#[test]
fn nav_lg_008_an_initial_fix_flies_from_the_present_position() {
    let mut exec = execution(
        vec![
            wp("W0", 0.0, 0.5).with_path(LegPath::InitialFix),
            wp("W1", 0.0, 0.0),
        ],
        ExecutionConfig::default(),
    );
    let leg = exec.active_leg().expect("an active leg");
    assert_eq!(leg.lateral_reference(), LateralReference::PresentPosition);
    assert_eq!(
        exec.advance(&pos(0.0, 0.05), 60.0),
        SequenceEvent::None,
        "50 km out must not capture"
    );
}

#[test]
fn nav_lg_009_a_course_to_fix_leg_sequences_inside_the_capture_radius_only() {
    let mut exec = execution(
        vec![
            wp("W0", 0.0, 0.0),
            wp("W1", 0.0, 0.5).with_path(eastbound_course(CourseReference::True)),
            wp("W2", 0.5, 0.5),
        ],
        ExecutionConfig::default(),
    );
    capture_w0(&mut exec);
    // The 90° corner at W1 would earn 1129.8 m of anticipation on a
    // track leg; a course leg has no fixed inbound track to anticipate
    // from, so nothing sequences until the capture radius.
    assert_eq!(
        exec.advance(&short_of_w1(1120.0), 60.0),
        SequenceEvent::None
    );
    assert_eq!(
        exec.advance(&short_of_w1(99.0), 60.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::Overflown,
        }
    );
}

#[test]
fn nav_lg_010_an_outbound_leg_that_is_not_a_track_stops_anticipation() {
    let plan_with = |w2_path: Option<LegPath>| {
        let mut w2 = wp("W2", 0.5, 0.5);
        if let Some(path) = w2_path {
            w2 = w2.with_path(path);
        }
        vec![wp("W0", 0.0, 0.0), wp("W1", 0.0, 0.5), w2]
    };
    // Both legs are tracks: the fly-by corner anticipates.
    let mut tracks = execution(plan_with(None), ExecutionConfig::default());
    capture_w0(&mut tracks);
    assert!(matches!(
        tracks.advance(&short_of_w1(1120.0), 60.0),
        SequenceEvent::LegAdvanced {
            reason: SequenceReason::Anticipated,
            ..
        }
    ));
    // The outbound leg leaves the corner on a course of its own, so the
    // bearing to W2 would mis-size the turn: no anticipation.
    for outbound in [
        LegPath::DirectToFix,
        eastbound_course(CourseReference::True),
    ] {
        let mut exec = execution(plan_with(Some(outbound)), ExecutionConfig::default());
        capture_w0(&mut exec);
        assert_eq!(
            exec.advance(&short_of_w1(1120.0), 60.0),
            SequenceEvent::None,
            "{outbound} outbound must not anticipate"
        );
        assert_eq!(
            exec.advance(&short_of_w1(99.0), 60.0),
            SequenceEvent::LegAdvanced {
                to_index: 2,
                turn: TurnType::FlyBy,
                reason: SequenceReason::Overflown,
            }
        );
    }
}

#[test]
fn nav_lg_010_an_inbound_leg_that_is_not_a_track_stops_anticipation() {
    let mut exec = execution(
        vec![
            wp("W0", 0.0, 0.0),
            wp("W1", 0.0, 0.5).with_path(LegPath::DirectToFix),
            wp("W2", 0.5, 0.5),
        ],
        ExecutionConfig::default(),
    );
    capture_w0(&mut exec);
    assert_eq!(
        exec.advance(&short_of_w1(1120.0), 60.0),
        SequenceEvent::None,
        "a direct-to inbound has no fixed track to anticipate from"
    );
    assert_eq!(
        exec.advance(&short_of_w1(99.0), 60.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::Overflown,
        }
    );
}

#[test]
fn nav_lg_014_each_flown_terminator_yields_its_own_lateral_reference() {
    let mut tracks = execution(
        vec![wp("W0", 0.0, 0.0), wp("W1", 0.0, 0.5)],
        ExecutionConfig::default(),
    );
    assert_eq!(
        tracks
            .active_leg()
            .expect("an active leg")
            .lateral_reference(),
        LateralReference::PresentPosition,
        "the first leg runs from the present position"
    );
    capture_w0(&mut tracks);
    assert_eq!(
        tracks
            .active_leg()
            .expect("an active leg")
            .lateral_reference(),
        LateralReference::track(pos(0.0, 0.0)),
        "a track leg runs from the fix before it"
    );

    let mut courses = execution(
        vec![
            wp("W0", 0.0, 0.0),
            wp("W1", 0.0, 0.5).with_path(eastbound_course(CourseReference::True)),
        ],
        ExecutionConfig::default(),
    );
    capture_w0(&mut courses);
    assert_eq!(
        courses
            .active_leg()
            .expect("an active leg")
            .lateral_reference(),
        LateralReference::course(EASTBOUND_RAD),
        "a course leg reports its published course"
    );
}

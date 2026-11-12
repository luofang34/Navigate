# Leg vocabulary and path-terminator requirements

Requirements for the leg vocabulary and per-leg sequencing in the
flight-plan, geodesy, and guidance crates (tracking issue: Navigate #3).
Tests cite these identifiers.
[ADR-0006](adr/0006-one-leg-vocabulary-for-path-terminators-and-vertical-legs.md)
owns the architecture; this document owns the behavior.

The companion document
[procedure-requirements.md](procedure-requirements.md) owns turn types
(NAV-TT), vertical and speed constraints (NAV-VC), and the honesty rules
(NAV-HN). This document adds the NAV-LG family.

## Sources and scaling

The vocabulary follows the path terminators of ARINC 424, as used in US
instrument procedure coding:

- A **path terminator** is a two-letter code. The first letter names the
  path, and the second letter names the termination. `TF` is a track
  that ends at a fix. `CF` is a course that ends at a fix. `CA` is a
  course that ends at an altitude. `HM` is a hold that ends on a manual
  command. (ARINC 424 leg types, as coded in the FAA CIFP and described
  in FAA Order 8260.19 procedure design guidance.)
- **IF** declares the start fix of a procedure. It has no path.
- **TF** is the standard RNAV leg: a great-circle track between two
  fixes. Procedure design assumes it wherever a fix pair is published.
- **DF** flies from wherever the vehicle is to a fix. It has no published
  inbound track, so no procedure protection assumes one.
- **CF** flies a published course to a fix. The course is published, and
  the origin is the present position. A CF leg follows a leg that has no
  fixed end, such as a heading leg, a `CA`, or a `DF`.
- **CA** flies a published course until the vehicle achieves the
  declared altitude constraint.
- **RF** is a constant-radius arc between two fixes about a published
  center. RNP AR procedures use it (FAA AC 90-101).
- **HA**, **HF**, and **HM** are the three holding terminators: to an
  altitude, to a fix after one circuit, and to a manual command.
- A leg or hold MAY declare a mandatory **turn direction** (`Left` or
  `Right`) when airspace or obstacle protection forbids turning along
  the shorter arc.

Vertical procedure legs have no ARINC 424 code. A vertical-takeoff
vehicle flies a vertical takeoff, a hover, a transition between
rotor-borne and wing-borne flight, and a vertical landing. These legs sit
in the same leg order as the horizontal legs, and this document gives
them the same treatment.

Scaling note: the requirements below hold for every vehicle class. A
multirotor and a fixed-wing aircraft fly the same leg types with
different numbers. Only the vertical legs are specific to a vehicle that
can hover.

## The vocabulary

- **NAV-LG-001** Every leg carries a path terminator from one vocabulary.
  The vocabulary names `InitialFix`, `TrackToFix`, `DirectToFix`,
  `CourseToFix`, `CourseToAltitude`, `HoldToAltitude`, `HoldToFix`, and
  `HoldManual`, which this build flies. It reserves `RadiusToFix`,
  `VerticalTakeoff`, `Hover`, `Transition`, and `VerticalLanding`. A
  reserved terminator is constructible, because a plan source must be
  able to state the leg it holds. A hold carries its racetrack
  (NAV-LG-015). Every reserved terminator carries no parameters, because
  the parameters follow from the sequencer that flies it.
- **NAV-LG-002** A waypoint MAY state the path terminator of the leg that
  ends at it. A waypoint that states none gets one by position in the fly
  order: the first waypoint resolves to `DirectToFix`, and every later
  waypoint resolves to `TrackToFix`. This rule maps a waypoint-sequence
  plan onto the vocabulary without a change to how that plan flies. The
  first waypoint resolves to `DirectToFix` and not to `InitialFix`,
  because a waypoint sequence does not claim a published procedure start.
- **NAV-LG-003** The vocabulary is additive. A plan built without it
  validates, activates, and flies as before, and its sequencing numbers
  do not change. `CONTRACT_VERSION` stays at 1. No wire encoding exists
  yet (ADR-0005), so there is no encoded form to break.

## Validation and refusal

- **NAV-LG-004** Plan validation refuses a structural defect in the
  vocabulary. A `TrackToFix` at the first waypoint has no fix before it
  and MUST be refused. An `InitialFix` after the first waypoint is not a
  start and MUST be refused. A `CourseToFix` or `CourseToAltitude` whose
  course is not a finite angle in `[0, 2π)` cannot be flown and MUST be
  refused. A `CourseToAltitude` or `HoldToAltitude` at a waypoint with
  no altitude constraint has no altitude to end on and MUST be refused.
  A hold whose inbound course is not a finite angle in `[0, 2π)`, or
  whose leg length is not finite and positive, cannot be flown and MUST
  be refused. A waypoint whose `turn_direction` conflicts with the turn
  direction of its `HoldGeometry` MUST be refused. Each refusal names
  the plan, the waypoint index, and the waypoint identifier.
- **NAV-LG-005** Activation refuses a leg this build cannot fly, and the
  refusal names the leg type. A plan that carries a reserved terminator
  is a valid plan, and it is not one this build can execute: what is
  implemented is a property of the executor, not of the exchange type.
  Activation also refuses a `CourseToFix`, `CourseToAltitude`, or hold
  whose course reference is magnetic, because Navigate has no magnetic
  variation model and MUST NOT guess a variation. Refusal is
  unconditional. A reserved leg never reaches the sequencer, so no guard
  can misapply to it.
- **NAV-LG-006** A Navigate leg is an execution vocabulary, not the
  canonical plan record. It carries what the sequencer and the guidance
  derivations read. The immutable resolved plan revision, its identity,
  its effective interval, its validation evidence, and its source ledger
  belong to Pilotage's FlightPlanning component (Pilotage ADR-0036).
  Navigate repeats validation at activation and refuses what it cannot
  fly. A source plan type never becomes the canonical type.

## Per-leg sequencing

- **NAV-LG-007** A `TrackToFix` leg runs on the great circle from the fix
  before it to its own fix. It sequences on a horizontal fix capture, and
  the capture criteria of NAV-TT-003 and NAV-TT-004 apply unchanged. A
  passed fix also ends the leg (NAV-TT-006). This is the behavior of the
  implicit model.
- **NAV-LG-008** A `DirectToFix` leg runs from the present position to
  its fix. An `InitialFix` leg declares the start fix of a procedure and
  states no path, so the executor flies it from the present position in
  the same way. Both sequence on a horizontal fix capture inside the
  capture radius only. Neither anticipates a turn, because the inbound
  geometry is the live position and not a fixed track. An `InitialFix` is
  valid at the first waypoint only.
- **NAV-LG-009** A `CourseToFix` leg runs on the great circle that its
  fix and its published course define. The course is a true course in
  `[0, 2π)`, measured at the fix, in the direction of flight. The leg
  sequences on a horizontal fix capture inside the capture radius, or
  when the vehicle passes the fix (NAV-TT-006). A `CourseToAltitude` leg
  tracks its published course without horizontal arrival slowdown and
  sequences when the vehicle achieves the waypoint's altitude constraint
  (`SequenceReason::AltitudeReached`). Neither course leg anticipates a
  turn. Guidance reports the published course as the reference course,
  and it measures the cross-track deviation against the course line. A
  vehicle right of the course line reads a positive deviation, which
  matches every other lateral deviation in this workspace.
- **NAV-LG-010** Turn anticipation applies between two `TrackToFix` legs
  only. NAV-TT-003 states the rule for a fly-by fix "between two fixed
  legs", and only a `TrackToFix` leg is fixed at both ends. The leg into
  the fix MUST be a `TrackToFix`, because otherwise there is no fixed
  inbound track and no inbound length to bound the DTA against. The leg
  out of the fix MUST also be a `TrackToFix`, because otherwise the
  bearing to the next fix is not the course the vehicle will fly, and the
  corner would be mis-sized. Every other combination sequences at the
  capture radius. A plan that states no terminator resolves to all
  `TrackToFix` legs after the first waypoint, so its anticipation does
  not change.

## Holds

- **NAV-LG-015** A hold (`HoldToAltitude`, `HoldToFix`, `HoldManual`)
  carries its racetrack (`HoldGeometry`), and the waypoint that carries
  the hold is the hold fix. The racetrack is an inbound course, its
  north reference, the direction of the turns (`Left` or `Right`), and
  the length of the leg (`Time` or `Distance`). The inbound course is
  measured at the fix, in the direction of flight, in `[0, 2π)`. That
  length is geometry, and it never ends the hold (NAV-LG-011). When the
  vehicle crosses the hold fix, the executor selects a standard ICAO Doc
  8168 / FAA AIM 5-3-8 entry (`Direct`, `Parallel`, or `Teardrop`) from
  the arrival course and turn direction, tracks the displaced outbound
  leg on the holding side (`LateralReference::DisplacedCourse` offset by
  `2 · r` abeam the fix) and the inbound course back to the fix, and
  counts completed circuits:
  - `HoldToAltitude` (`HA`) ends as soon as the waypoint's altitude
    constraint is achieved (`SequenceReason::AltitudeReached`),
    including on initial fix crossing when already at or past the target
    altitude.
  - `HoldToFix` (`HF`) ends on returning to the hold fix after one
    circuit (`SequenceReason::HoldCircuitComplete`).
  - `HoldManual` (`HM`) repeats circuits until external exit is armed
    via `exit_hold()`, then completes the inbound leg to the hold fix
    and ends (`SequenceReason::HoldExited`).

## Termination conditions

- **NAV-LG-011** A leg ends on a condition the vehicle achieves. A fix
  capture, an altitude reached, a vehicle state reported, and an external
  command are achieved conditions. Elapsed time is not, and a leg MUST
  NOT end on a timer: a timed leg sequences because a clock ran out, not
  because the vehicle arrived. A published hold declares a leg length in
  time or in distance. That length is hold geometry and states how large
  the racetrack is; it does not state when the hold ends.

| Leg type | Termination | Class | Sequencer input |
|---|---|---|---|
| `InitialFix` | The fix captures | Horizontal | Position |
| `TrackToFix` | The fix captures | Horizontal | Position |
| `DirectToFix` | The fix captures | Horizontal | Position |
| `CourseToFix` | The fix captures | Horizontal | Position |
| `CourseToAltitude` | The declared altitude is reached | Vertical | Position altitude |
| `RadiusToFix` | The arc end fix captures | Horizontal | Position (reserved) |
| `HoldToAltitude` | The declared altitude is reached | Vertical | Position altitude |
| `HoldToFix` | The hold fix captures after one circuit | Horizontal | Position and circuit count |
| `HoldManual` | An external command releases the hold at the fix | Commanded | `exit_hold()` and position |
| `VerticalTakeoff` | The declared altitude is reached | Vertical | Position altitude (reserved) |
| `Hover` | An external command or a declared condition releases the hover | Commanded | Command input (reserved) |
| `Transition` | The vehicle reports the target flight mode | State | Vehicle-state input (reserved) |
| `VerticalLanding` | The vehicle reports ground contact | State | Ground-contact input (reserved) |

## Guard reconciliation

Three guards assume that a leg is a horizontal track between two
separate fixes. **NAV-LG-012** states what each guard does per leg type.
A guard "applies" when its current behavior is correct for the leg. A
guard is "replaced" when a typed condition takes its place. No guard is
relaxed.

- **G1, horizontal capture.** `PlanExecution::advance` measures the
  great-circle distance to the waypoint and ignores altitude.
- **G2, minimum leg length at activation.** `PlanExecution::new` refuses
  a plan whose consecutive waypoints are not more distant than the
  capture radius, so a leg cannot sequence at the instant it begins.
- **G3, coincident-endpoint refusal in guidance.** `admit_leg` derives
  the lateral geometry from a two-point track, and the geodesy layer
  refuses a track shorter than 1 mm.

| Leg type | G1 horizontal capture | G2 minimum leg length | G3 coincident endpoints |
|---|---|---|---|
| `InitialFix` | Applies | Applies | Applies |
| `TrackToFix` | Applies | Applies | Applies |
| `DirectToFix` | Applies | Applies | Applies |
| `CourseToFix` | Applies | Applies | Replaced: a fix and a course define the course line |
| `CourseToAltitude` | Replaced: an altitude-achieved condition | Not applicable: the leg has no horizontal termination fix | Replaced: an anchor and a course define the course line |
| `RadiusToFix` | Applies | Applies | Replaced: the arc reference is the center and the radius |
| `HoldToAltitude` | Replaced: an altitude-achieved condition | Not applicable: the hold has one fix | Replaced: the hold pattern derives the reference |
| `HoldToFix` | Applies at the hold fix, after one circuit | Not applicable: the hold has one fix | Replaced: the hold pattern derives the reference |
| `HoldManual` | Replaced: `exit_hold()` armed + return to hold fix | Not applicable: the hold has one fix | Replaced: the hold pattern derives the reference |
| `VerticalTakeoff` | Replaced: an altitude-achieved condition | Replaced: a minimum vertical separation | Replaced: a position-hold reference |
| `Hover` | Replaced: a commanded-release condition | Not applicable: the leg has one position | Replaced: a position-hold reference |
| `Transition` | Replaced: a state-achieved condition | Applies | Applies |
| `VerticalLanding` | Replaced: a ground-contact condition | Replaced: a minimum vertical separation | Replaced: a position-hold reference |

Every leg whose replacement has not landed is refused at activation
(NAV-LG-005).

## Honesty requirements

- **NAV-LG-013** A refusal names what it refused. A structural refusal
  names the plan, the waypoint index, and the identifier. An activation
  refusal names the leg type. An operator reads which leg blocked the
  plan and does not re-derive the judgment.
- **NAV-LG-014** The reference course a guidance derivation reports is
  the course of the leg the vehicle actually flies. A `TrackToFix`
  reports the track bearing, a `DirectToFix` reports the live bearing to
  the fix, a `CourseToFix` or `CourseToAltitude` reports the published
  course, and an active hold reports the course of its current stage
  (inbound, teardrop, parallel, or displaced outbound). Guidance never
  reports a course from a leg geometry the leg does not have.

## Out of scope (recorded, deferred)

- Turn anticipation into or out of a `CourseToFix` leg.
- Magnetic variation inside Navigate (`CourseReference::Magnetic` is
  converted to true north by Pilotage's geomagnetic model before
  activation, and refused if passed unresolved).
- Procedure and transition context on a leg, a stable leg identity, and a
  source record locator. These belong to Pilotage's canonical resolved
  plan (NAV-LG-006).
- DME arcs (`AF`), heading legs (`VA`, `VI`, `VM`), and `RF` arcs. The
  vocabulary grows additively as each sequencer lands.

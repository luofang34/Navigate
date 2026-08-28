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
  that ends at a fix. `CF` is a course that ends at a fix. `HM` is a
  hold that ends on a manual command. (ARINC 424 leg types, as coded in
  the FAA CIFP and described in FAA Order 8260.19 procedure design
  guidance.)
- **IF** declares the start fix of a procedure. It has no path.
- **TF** is the standard RNAV leg: a great-circle track between two
  fixes. Procedure design assumes it wherever a fix pair is published.
- **DF** flies from wherever the vehicle is to a fix. It has no published
  inbound track, so no procedure protection assumes one.
- **CF** flies a published course to a fix. The course is published, and
  the origin is the present position. A CF leg follows a leg that has no
  fixed end, such as a heading leg or a DF.
- **RF** is a constant-radius arc between two fixes about a published
  center. RNP AR procedures use it (FAA AC 90-101).
- **HA**, **HF**, and **HM** are the three holding terminators: to an
  altitude, to a fix after one circuit, and to a manual command.

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
  The vocabulary names `InitialFix`, `TrackToFix`, `DirectToFix`, and
  `CourseToFix`, which this build flies. It reserves `RadiusToFix`,
  `HoldToAltitude`, `HoldToFix`, `HoldManual`, `VerticalTakeoff`,
  `Hover`, `Transition`, and `VerticalLanding`. A reserved terminator is
  constructible, because a plan source must be able to state the leg it
  holds. A reserved terminator carries no parameters, because the
  parameters follow from the sequencer that flies it.
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
  start and MUST be refused. A `CourseToFix` whose course is not a finite
  angle in `[0, 2π)` cannot be flown and MUST be refused. Each refusal
  names the plan, the waypoint index, and the waypoint identifier.
- **NAV-LG-005** Activation refuses a leg this build cannot fly, and the
  refusal names the leg type. A plan that carries a reserved terminator
  is a valid plan, and it is not one this build can execute: what is
  implemented is a property of the executor, not of the exchange type.
  Activation also refuses a `CourseToFix` whose course reference is
  magnetic, because Navigate has no magnetic variation model and MUST NOT
  guess a variation. Refusal is unconditional. A reserved leg never
  reaches the sequencer, so no guard can misapply to it.
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
  the capture criteria of NAV-TT-003 and NAV-TT-004 apply unchanged. This
  is the behavior the implicit model already has, now named.
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
  sequences on a horizontal fix capture inside the capture radius only.
  Guidance reports the published course as the reference course, and it
  measures the cross-track deviation against the course line. A vehicle
  right of the course line reads a positive deviation, which matches
  every other lateral deviation in this workspace.
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

## Termination conditions

- **NAV-LG-011** A leg ends on a condition the vehicle achieves. A fix
  capture, an altitude reached, a vehicle state reported, and an external
  command are achieved conditions. Elapsed time is not, and a leg MUST
  NOT end on a timer: a timed leg sequences because a clock ran out, not
  because the vehicle arrived. A published hold declares a leg length in
  time or in distance. That length is hold geometry and states how large
  the racetrack is; it does not state when the hold ends.

Each leg type ends on one of five condition classes. The table states the
class, and the input the sequencer needs to evaluate it. `advance`
receives a position and a groundspeed today, so the horizontal and the
vertical classes are evaluable now. The other classes need an input the
sequencer does not receive.

| Leg type | Termination | Class | Sequencer input |
|---|---|---|---|
| `InitialFix` | The fix captures | Horizontal | Position (present) |
| `TrackToFix` | The fix captures | Horizontal | Position (present) |
| `DirectToFix` | The fix captures | Horizontal | Position (present) |
| `CourseToFix` | The fix captures | Horizontal | Position (present) |
| `RadiusToFix` | The arc end fix captures | Horizontal | Position (present) |
| `HoldToAltitude` | The declared altitude is reached | Vertical | Position altitude (present, but capture ignores it) |
| `HoldToFix` | The hold fix captures after one circuit | Horizontal | Position, and a circuit count the executor holds |
| `HoldManual` | An external command releases the hold | Commanded | A command input the sequencer does not receive |
| `VerticalTakeoff` | The declared altitude is reached | Vertical | Position altitude (present, but capture ignores it) |
| `Hover` | An external command or a declared condition releases the hover | Commanded | A command input the sequencer does not receive |
| `Transition` | The vehicle reports the target flight mode | State | A vehicle-state input the sequencer does not receive |
| `VerticalLanding` | The vehicle reports ground contact | State | A ground-contact input the sequencer does not receive |

A vertical landing terminates on reported ground contact and not on a
zero altitude. An altitude estimate is not a statement that the vehicle
is on the ground, and to treat it as one would sequence the plan while
the vehicle is still in the air.

Each reserved leg needs the parameters below when its sequencer lands.
This list is a commitment to decide, not a decision:

| Leg type | Parameters it will need |
|---|---|
| `RadiusToFix` | Arc center, turn direction, and the inbound course at the arc start |
| `HoldToAltitude` | Hold fix, inbound course, turn direction, leg length, and the target altitude |
| `HoldToFix` | Hold fix, inbound course, turn direction, and leg length |
| `HoldManual` | Hold fix, inbound course, turn direction, and leg length |
| `VerticalTakeoff` | Target altitude and a climb-rate limit |
| `Hover` | Hold position, a position tolerance, and the release condition |
| `Transition` | Target flight mode and an entry-condition envelope |
| `VerticalLanding` | Landing position and a descent-rate limit |

## Guard reconciliation

Three guards in the current code assume that a leg is a horizontal track
between two separate fixes. **NAV-LG-012** states what each guard does
per leg type. A guard "applies" when its current behavior is correct for
the leg. A guard is "replaced" when a typed condition takes its place. No
guard is relaxed.

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
| `CourseToFix` | Applies | Applies | Replaced: a fix and a course define the course line, so no two-endpoint degenerate case exists |
| `RadiusToFix` | Applies | Applies | Replaced: the arc reference is the center and the radius, not a two-point track |
| `HoldToAltitude` | Replaced: an altitude-achieved condition | Not applicable: the hold has one fix | Replaced: the hold pattern derives the reference, not a track |
| `HoldToFix` | Applies at the hold fix, after the circuit count | Not applicable: the hold has one fix | Replaced: the hold pattern derives the reference |
| `HoldManual` | Replaced: a commanded-release condition | Not applicable: the hold has one fix | Replaced: the hold pattern derives the reference |
| `VerticalTakeoff` | Replaced: an altitude-achieved condition | Replaced: a minimum **vertical** separation between the two ends | Replaced: a position-hold reference at the departure point, with no course |
| `Hover` | Replaced: a commanded-release or achieved condition | Not applicable: the leg has one position | Replaced: a position-hold reference at the hover fix, with no course |
| `Transition` | Replaced: a state-achieved condition | Applies: a transition covers ground | Applies: a transition flies the lateral reference of its leg |
| `VerticalLanding` | Replaced: a ground-contact condition | Replaced: a minimum **vertical** separation between the two ends | Replaced: a position-hold reference at the landing point, with no course |

Every leg marked "Replaced" is refused at activation until its
replacement lands (NAV-LG-005). Fail closed: a leg type whose
termination condition is not implemented is refused, and it is never
flown under a guard written for a different geometry.

## Honesty requirements

- **NAV-LG-013** A refusal names what it refused. A structural refusal
  names the plan, the waypoint index, and the identifier. An activation
  refusal names the leg type. An operator reads which leg blocked the
  plan and does not re-derive the judgment.
- **NAV-LG-014** The reference course a guidance derivation reports is
  the course of the leg the vehicle actually flies. A `TrackToFix`
  reports the track bearing, a `DirectToFix` reports the live bearing to
  the fix, and a `CourseToFix` reports the published course. Guidance
  never reports a course from a leg geometry the leg does not have.

## Out of scope (recorded, deferred)

- Turn anticipation into or out of a `CourseToFix` leg. The published
  course gives the inbound course at the fix, but the leg has no fixed
  inbound length to bound the DTA against (NAV-TT-003 caps the DTA at
  half the shorter adjoining leg). A bound that uses the live distance is
  self-referential, in the same way a `DirectToFix` inbound is.
- Magnetic variation. A `CourseToFix` may declare a magnetic reference,
  and activation refuses it until a variation model lands.
- Procedure and transition context on a leg, a stable leg identity, and a
  source record locator. These belong to Pilotage's canonical resolved
  plan (NAV-LG-006).
- DME arcs, heading legs (`VA`, `VI`, `VM`, `CA`, `CI`, `FA`, `FM`), and
  the rest of the ARINC 424 leg set. The vocabulary grows additively as
  each sequencer lands.

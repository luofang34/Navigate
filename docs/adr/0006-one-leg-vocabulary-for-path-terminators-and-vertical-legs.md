# ADR-0006: One leg vocabulary carries path terminators and vertical procedure legs

- Status: Accepted
- Date: 2026-08-28

## Context

The plan model has no leg vocabulary. Every leg is an implicit
great-circle track between two waypoints. `PlanExecution` flies the first
waypoint from the present position, and every later waypoint from the
waypoint before it. The plan cannot say what path it wants.

Two needs push on the same model.

The first need is instrument procedure execution. Published procedures
code each leg with a path terminator: IF, TF, DF, CF, RF, and the three
holding terminators HA, HF, and HM. Navigate must fly these legs to
execute a published procedure.

The second need is vertical flight. A vertical-takeoff vehicle flies a
vertical takeoff, a hover, a transition to wing-borne flight, and a
vertical landing. These legs are procedure legs in the same sequence as
the horizontal legs. A vertical takeoff runs before the first TF leg of
a departure, and a vertical landing runs after the last leg of an
arrival.

Three guards in the current code reject a vertical leg:

1. Capture is horizontal only. `PlanExecution::advance` measures the
   great-circle distance to the waypoint and ignores altitude. A leg
   that terminates on an altitude cannot sequence.
2. Activation refuses a short leg. `PlanExecution::new` refuses a plan
   whose consecutive waypoints are not more distant than the capture
   radius. A vertical takeoff has the same horizontal position at both
   ends, so the leg length is zero.
3. Guidance refuses a leg with coincident ends. `admit_leg` derives the
   lateral geometry from a two-point track. The geodesy layer refuses a
   track shorter than 1 mm, and guidance reports the target as
   implausible. A hover has one position, not two.

Each guard is correct for the leg types that exist today. Each guard
also assumes that a leg is a horizontal track between two separate
fixes. That assumption is the thing the two needs break.

One more boundary applies. Pilotage's FlightPlanning component owns the
canonical resolved plan revision (ADR-0036 in Pilotage, tracked by
Pilotage issue #566). Navigate must not build a second canonical plan
record.

## Decision

### One vocabulary, one sequencer

`navigate-contract` gains one `LegPath` vocabulary. It names every leg
type from both needs. `navigate-fpl` walks one ordered list of legs with
one sequencer.

A vertical leg is not a different machine. A departure is a vertical
takeoff, then a transition, then a sequence of TF legs. If the vertical
legs had their own type and their own sequencer, then the plan would
have two orders, two activation validations, and two sets of guards.
Sequencing, capture, and refusal would then exist twice, and the two
copies would drift apart.

`LegPath` names these leg types:

| Variant | Meaning | State |
|---|---|---|
| `InitialFix` | The published start fix of a procedure | Flown |
| `TrackToFix` | A great-circle track from the fix before it to this fix | Flown |
| `DirectToFix` | A direct path from the present position to this fix | Flown |
| `CourseToFix` | A published course that terminates at this fix | Flown |
| `RadiusToFix` | A constant-radius arc that terminates at this fix | Reserved |
| `HoldToAltitude` | A hold that ends when the vehicle reaches an altitude | Reserved |
| `HoldToFix` | A hold that ends at the hold fix after one circuit | Reserved |
| `HoldManual` | A hold that ends on an external command | Reserved |
| `VerticalTakeoff` | A vertical climb from the departure point | Reserved |
| `Hover` | A held position | Reserved |
| `Transition` | A change between rotor-borne and wing-borne flight | Reserved |
| `VerticalLanding` | A vertical descent to ground contact | Reserved |

A reserved variant carries no parameters. It names its path terminator
so that a plan source can express the leg and so that Navigate can
refuse the leg by name. Each reserved variant gets its parameters when
its sequencer lands. The parameters follow from the sequencer that flies
the leg, and that sequencer does not exist. The requirements document
lists the parameters each reserved leg needs. To add them is a contract
change, and ADR-0005 records such a change as a new decision.

### The leg belongs to the waypoint it ends at

`Waypoint` gains a `path: Option<LegPath>` field. The field states the
path terminator of the leg that ends at this waypoint. The two existing
per-leg fields, `max_speed_mps` and `gradient`, already follow this
rule.

`None` states that the plan source did not declare a path terminator.
Activation resolves `None` by position in the sequence. A plan source
that declares the terminator sets `Some`.

### The migration rule keeps the current model

A waypoint sequence maps to legs by this rule (NAV-LG-002):

- The first waypoint resolves to `DirectToFix`. This is the initial
  direct-to leg the executor flies today.
- Every later waypoint resolves to `TrackToFix`. This is the implicit
  great-circle track the executor flies today.

The first waypoint resolves to `DirectToFix` and not to `InitialFix`. A
waypoint sequence does not claim to be a published procedure, so it must
not claim a published start fix.

The change is additive and byte-compatible. There is no wire encoding to
break, because ADR-0005 defers the encoding to the sidecar split.
`Waypoint::new` keeps its signature and sets `path` to `None`. A plan
that existing code builds validates, activates, and flies with the same
numbers. `CONTRACT_VERSION` stays at 1, because the contract's own rule
is that additive growth does not bump it.

### Validation happens twice, at two levels

Plan validation in `navigate-contract` refuses a structural defect. A
`TrackToFix` at the first waypoint has no fix before it. An
`InitialFix` after the first waypoint is not a start. A course that is
not a finite angle in `[0, 2π)` cannot be flown. These defects make the
plan wrong for every consumer.

Activation in `navigate-fpl` refuses a leg this build cannot fly. A
plan that carries an `RF` leg is a valid plan. It is not a plan this
build can execute. What is implemented is a property of the executor,
not of the exchange type. Activation refuses the leg by name, so the
operator reads which leg blocked the plan.

Activation also refuses a magnetic course. Navigate has no magnetic
variation model. A magnetic course would need a variation to become a
true course, and to guess the variation would fabricate a course.

### The guards are reconciled per leg type, not relaxed

No guard changes for the legs that fly today. IF, TF, DF, and CF are all
fix-capture legs, so all three guards keep their current behavior for
them. One guard is replaced for CF: a course line is defined by a fix
and a course, so the two-endpoint degenerate case cannot occur, and the
guidance refusal has nothing to refuse.

Every reserved leg is refused at activation. A refused leg never reaches
the sequencer, so no guard can misapply to it. The requirements document
records, per leg type, which guard applies, and which typed condition
replaces it when the leg lands. This is the fail-closed rule: a leg type
that has no implemented termination condition is refused, never flown
under a guard that was written for a different geometry.

### Sequencing is on an achieved condition, never on a timer

Every leg ends on a condition the vehicle achieves. A fix capture, an
altitude reached, a vehicle state reported, or an external command are
achieved conditions. Elapsed time is not. A hover that ends after 30
seconds ends because a timer expired, not because the vehicle did
anything. Such a leg would sequence while the vehicle is in the wrong
place.

A published hold declares a leg length in time or in distance. That
length is hold geometry: it says how large the racetrack is. It does not
say when the hold ends. The hold still ends on an achieved condition:
an altitude for HA, the hold fix for HF, or a command for HM.

### Navigate owns execution, Pilotage owns the record

A Navigate leg is an execution vocabulary. It is not the canonical plan
record.

Pilotage's FlightPlanning component owns the immutable resolved plan
revision, its identity, its effective interval, its validation evidence,
and its source ledger (Pilotage ADR-0036 and Pilotage issue #566). A
Pilotage `ResolvedLeg` carries a stable leg identity, procedure context,
fix references, and a source record locator. Navigate carries none of
these.

Navigate carries what the sequencer and the guidance derivations read:
the path terminator, the course a `CourseToFix` flies, and the
constraints already on the waypoint. Navigate repeats validation at
activation, and it refuses what it cannot fly. A source plan type never
becomes the canonical type.

## Consequences

- Turn anticipation narrows to what NAV-TT-003 already claims. The DTA
  applies between two `TrackToFix` legs only. A `CourseToFix` or a
  `DirectToFix` on either side of a fix does not define the fixed track
  the corner geometry needs. Plans that carry no path terminator are
  all-TF after resolution, so their sequencing does not change.
- A `CourseToFix` leg needs one new geodesy function: the signed
  cross-track distance from a course line that a fix and a true course
  define. The existing two-point helper cannot express a course line.
- Guidance changes its lateral input from an optional origin position to
  a typed `LateralReference`. The three flown leg types map to its three
  forms: a track from a fix, the present position, and a published
  course. A caller can no longer state a lateral geometry the leg does
  not have.
- A plan that carries a reserved leg is refused loudly at activation.
  This is the intended behavior until each leg's sequencer lands, and it
  is what makes the vocabulary safe to publish before the legs fly.
- Pilotage can adopt the vocabulary when it is ready. Until then,
  Navigate's resolution rule supplies the terminator, and both models
  describe the same flight.

## Alternatives considered

**A separate vertical-leg model.** A `VerticalLeg` type beside the
waypoint list, sequenced by its own machine. Rejected: a departure
interleaves vertical and horizontal legs in one order, so the plan would
need one merged order anyway. The sequencer, the activation validation,
and the three guards would exist twice.

**Path terminators only, vertical legs later.** Ship IF, TF, DF, and CF,
and design the vertical legs when a vertical-takeoff vehicle arrives.
Rejected: the three guards encode the assumption that a leg is a
horizontal track between two separate fixes. To find that assumption
later means to rewrite the sequencing contract, not to extend it. A
vocabulary that names only the horizontal legs would also let the
sequencer treat "the plan holds no path terminator" and "the plan holds
a leg I cannot fly" as the same silence.

**Full parameters on every reserved variant now.** Give `RadiusToFix` a
center and a radius, and give each hold a full specification. Rejected
for two reasons. The parameters a leg needs follow from the sequencer
that flies it, so to choose them now is speculation. The canonical hold
specification is Pilotage's (issue #566), so to define a second one here
would build the model twice, which is the outcome this record exists to
prevent.

**A required `LegPath` with a `TrackToFix` default.** Rejected: the
first waypoint of every existing plan would then declare a track from a
fix that does not exist, and plan validation would refuse every plan
that existing code builds.

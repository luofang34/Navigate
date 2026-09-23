# Navigate design overview

This document is the orientation map; the [ADRs](adr/README.md) are the
authoritative decisions. Navigate fills the navigation slot of a three-way
vehicle decomposition: a flight controller (Aviate-class) owns control-grade
estimation and stabilization; a communication component (aerocontext-class)
owns advisory aeronautical context; the host platform (Pilotage-class) owns
sessions, authority, media, and orchestration. Navigate owns the global
navigation solution, its integrity, flight-plan execution, and guidance.

## Position in the system

```text
   Host platform (sessions, authority, displays)
        │ fenced control path        │ telemetry
        ▼                            ▲
   Navigate ── setpoints ──► Flight controller (stabilization, actuation)
     │  ▲                        │
     │  └── aiding (DRAFT, ──────┘  FC validates, fuses, or rejects
     │       bounded, with covariance + integrity + composition)
     ├── consumes: GNSS, celestial, visual sources (any subset)
     └── consumes: navdata / terrain packages supplied by the host side
```

Three boundary rules govern everything:

1. **The FC is sovereign over control-grade state.** Navigate never assumes
   an aiding observation was accepted, and the FC remains fully operational
   with Navigate absent.
2. **Guidance is a commander, not a privilege.** Setpoints enter the host
   platform's fenced authority path as an automation-class principal;
   Navigate has no side door to actuators.
3. **Correlation is declared, never discovered.** Every solution and every
   aiding observation carries its source composition so a consumer can
   refuse double-counted information. Fused outputs derived from shared
   measurements are not independent aids.

## The fusion design (ADR-0003)

A discrete-time filter over a local-level NED state anchored at an
initialization origin:

- **Typed observations, not sensor drivers.** Sources produce
  `Observation`s (position fix, velocity fix; celestial bearing/elevation
  and visual odometry are reserved variants) with a stamp (source id,
  epoch, wrapping sequence, acquisition time, clock domain), a covariance,
  and a source composition. Sensor I/O lives outside the crate; the filter
  is sans-IO and deterministic.
- **Admission before update.** Wrap-aware sequence regression, staleness
  beyond a configured bound, non-finite or non-positive-semidefinite
  covariance, undeclared composition, and innovation-gate failures are all
  rejected *and counted* before they can touch the state. Rejection
  counters are first-class observability.
- **Pluggable propagation.** A `Propagator` advances the state between
  measurements; the skeleton ships constant-velocity kinematics, and IMU
  mechanization joins as a propagator when a raw inertial source lands —
  growing the state, not changing the architecture.
- **Delayed measurements** are a designed seam: the staleness bound is
  what ships; a bounded state history enabling apply-at-acquisition
  re-propagation is the designed extension the update-path seam exists
  for (ADR-0003).
- **Single-source honesty.** With one source the filter still runs; the
  integrity assessment reports the redundancy it actually has and marks
  fault detection unavailable rather than inventing confidence (ADR-0004).

## Integrity (ADR-0004)

Every published solution carries: a quality classification, the set of
contributing source classes, a redundancy level, 1-sigma horizontal and
vertical uncertainties derived from the covariance, and a fault-detection
statement. Protection levels (RAIM-class bounds) are a reserved field, not
an implemented claim. Consumers that need integrity fail closed when it is
absent; displays show quality rather than hiding it.

## Flight plans and guidance

`navigate-fpl` owns the plan model's execution: validation, active-leg
sequencing with capture criteria, terminal behavior, and procedure
selection (a loss-of-communication procedure is a plan with a different
role, not a different machine). Each leg carries a path terminator from
one vocabulary that names the RNAV leg types and the vertical procedure
legs together, per the [leg requirements](leg-requirements.md) and
ADR-0006; a leg type this build cannot fly is refused by name at
activation, never flown under a guard written for another geometry.
Waypoints carry turn types per the
[procedure requirements](procedure-requirements.md): a fly-by fix
sequences early by the distance of turn anticipation the bank-limit
model earns at the commanded groundspeed, while fly-over and terminal
fixes sequence only inside the capture radius. `navigate-guidance`
turns the solution plus the active leg into setpoints — cross-track and
course for lateral, altitude constraints (including between-altitude
windows) for vertical, bounded by per-leg gradient and waypoint speed
constraints, and an NED velocity vector for a flight controller that
takes velocity commands — and refuses to guide when solution integrity
is below what the maneuver requires. Both derivations admit a solution
through one helper, so their floors cannot drift apart.

## The contract (ADR-0005)

`navigate-contract` is the crate external consumers pin (the same pattern
as an FC exposing a pinned contract crate): pure types, no I/O, no
behavioral traits. It defines solutions, integrity, stamps, compositions,
plans, guidance setpoints, and a **DRAFT** aiding-observation module whose
schema is owed to a joint RFC with the FC side before any aiding flows.
In-process linking and a future sidecar process both conform to the same
vocabulary; process topology is a deployment decision, not a contract one.

## What the skeleton deliberately defers

- Measurement models for the reserved `Range`, `Pseudorange`, and
  `VisualPose` variants. The filter refuses them by name until they land
  (ADR-0008). Celestial and terrain-matching variants are not reserved yet.
- Inertial and dead-reckoning propagators, attitude and bias states, and
  bounded delayed-measurement history (ADR-0008).
- Global visual retrieval for a search prior that is too uncertain to
  narrow the search (ADR-0009).
- Batched multi-view rendering, source-driven tile detail, and a completion
  signal in the MapLibre implementation of the reference port (ADR-0010).
- Coupling admission-gate storm rates into the quality derivation
  (quality currently degrades on covariance bounds and source silence).
- Position setpoint generation (the contract vocabulary mirrors the FC
  command surface; guidance derives deviation-tracking and velocity
  setpoints).
- RAIM-class fault detection and protection levels.
- Terrain database binding for EGPWS (the seam returns typed
  `Unavailable`).
- The aiding-observation wire schema (joint RFC with the FC side).
- Any transport: no sockets, no shared memory, no MAVLink — integration
  binds those in the host platform's adapters.

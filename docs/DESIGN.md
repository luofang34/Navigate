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
  measurements. `NavigationFilter` uses constant-velocity kinematics.
  `InertialNavigationFilter` is a strapdown error-state filter with
  seventeen states: position, velocity, attitude, accelerometer and
  gyroscope biases, a barometric bias, and a magnetic map bias
  (ADR-0014). Its state is always in one body-centered Cartesian frame.
  On a body surface this frame is the rotating body-fixed frame, and a
  reference ellipsoid gives the geodetic models and the geodetic
  solution at each latitude, the poles included. `change_frame` moves
  the state between rotating and inertial frames without loss, so one
  filter operates from a launch site to the surface of another body
  (ADR-0017).
- **Delayed measurements.** Each filter keeps a bounded history of its
  states before each update. The filter applies a delayed observation
  at its acquisition time and then replays the later updates
  (ADR-0012). An observation older than the history is refused with
  `OutsideHistory`.
- **Single-source honesty.** With one source the filter still runs; the
  integrity assessment reports the redundancy it actually has and marks
  fault detection unavailable rather than inventing confidence (ADR-0004).

## Integrity (ADR-0004)

Every published solution carries: a quality classification, the set of
contributing source classes, a redundancy level, 1-sigma horizontal and
vertical uncertainties derived from the covariance, and a fault-detection
statement. A run of innovation-gate refusals limits the quality to
`Degraded` until an aiding observation passes admission, because a
refused update does not change the covariance (ADR-0015). A space
solution gives its uncertainty in radial, along-track, and cross-track
axes. Protection levels (RAIM-class bounds) are a reserved field, not
an implemented claim. Consumers that need integrity fail closed when it is
absent; displays show quality rather than hiding it.

## Flight plans and guidance

`navigate-route` executes the plan model on any body. It validates the
plan, sequences the active leg with capture criteria, applies the
terminal behavior, and selects the procedure. A loss-of-communication
procedure is a plan with a different role, not a different machine. The
turn model uses the surface gravity of the body of the plan, not the
gravity of the Earth. Each leg carries a path terminator from
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

## Space trajectory and mission phases (ADR-0019)

`navigate-trajectory` holds the trajectory laws of space flight. The
burn law steers on the velocity to go. The rendezvous law solves the
Clohessy–Wiltshire transfer in the frame of the target and checks an
approach corridor. The entry law tracks a reference drag against the
specific energy and reverses the bank at the edge of a heading
deadband. The powered descent law is polynomial E-guidance. Each law
is `#![no_std]` and does not allocate.

`navigate-guidance` admits a Cartesian solution before a space law
reads it. A solution in the wrong frame, or with a position uncertainty
over the configured bound, gives a typed refusal. The space setpoints
are a thrust vector, an entry bank angle, and a Cartesian velocity.
`step_mission` sequences the phases of one mission. A reentry goes from
an orbital coast through a deorbit burn and an entry corridor to the
legs of a route. A landing goes from an orbital coast through a powered
descent to touchdown. The mission state is `Copy` and holds the bank
state of the entry inline.

## Core crate topology (ADR-0016)

The core crates compile with `#![no_std]` and do not allocate. A
microcontroller on a vehicle and a ground tool link the same code.
Each crate that calls a float function has the features `std` (the
default) and `libm`.

| Crate | Responsibility |
|---|---|
| `navigate-contract` | Boundary types: solutions, integrity, stamps, plans, and physical time |
| `navigate-geodesy` | Geodesy on an explicit reference ellipsoid of any body, vertical datums, and surface shapes |
| `navigate-astrodynamics` | Planetary bodies, gravity fields, coordinate time scales, and frame rotations |
| `navigate-fusion` | The kinematic, inertial, and particle filters, with admission and integrity |
| `navigate-magnav` | Tolles–Lawson calibration, borrowed grid maps, and Cramér–Rao bounds for magnetic anomaly navigation |
| `navigate-route` | Route validation and leg sequencing |
| `navigate-trajectory` | Orbital burn, rendezvous, atmospheric entry, and powered descent laws |
| `navigate-guidance` | Surface and space guidance setpoints, and mission-phase sequencing |
| `navigate-terrain-safety` | Terrain clearance with vertical-datum adapters |

Every table, history, and gravity model has a fixed inline capacity.
A filter refuses a new source when its source table is full, so that
the ordering checks of each source always run. The particle filter
uses two particle slices that the caller owns, and it swaps them after
each checked update.

Every function that needs a body takes it explicitly: an ellipsoid, a
`PlanetaryBody`, a `SpaceEnvironment`, or a surface gravity. A caller
cannot apply the WGS84 ellipsoid or the gravity of the Earth to the
Moon, Mars, or a small body by mistake (ADR-0013, ADR-0015). Terrain
clearance compares two ellipsoidal heights. `OrthometricTerrain` adds
the undulation of the geoid, the selenoid, or the areoid to an
orthometric terrain height.

## The contract (ADR-0005)

`navigate-contract` is the crate external consumers pin (the same pattern
as an FC exposing a pinned contract crate): pure types, no I/O, no
behavioral traits. It defines solutions, integrity, stamps, compositions,
plans, guidance setpoints, and a **DRAFT** aiding-observation module whose
schema is owed to a joint RFC with the FC side before any aiding flows.
In-process linking and a future sidecar process both conform to the same
vocabulary; process topology is a deployment decision, not a contract one.

## What the skeleton deliberately defers

- Measurement models for the reserved `Pseudorange` and `VisualPose`
  variants. The filter refuses them by name until they land (ADR-0008). The
  `Range` model is supported. Celestial and terrain-matching variants are not
  reserved yet.
- Global visual retrieval for a search prior that is too uncertain to
  narrow the search (ADR-0009).
- Batched multi-view rendering, source-driven tile detail, and a completion
  signal in the MapLibre implementation of the reference port (ADR-0010).
- Position setpoint generation. The contract vocabulary mirrors the FC
  command surface. Guidance derives deviation-tracking, NED velocity,
  thrust-vector, entry-bank, and Cartesian-velocity setpoints.
- Range control in the entry law. The caller selects the reference drag
  knots for the range of the mission.
- RAIM-class fault detection and protection levels.
- A terrain source for `navigate-terrain-safety`. Without a terrain
  source, the assessment refuses with `NoTerrain`.
- The aiding-observation wire schema (joint RFC with the FC side).
- Any transport: no sockets, no shared memory, no MAVLink — integration
  binds those in the host platform's adapters.

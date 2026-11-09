# Chapter 1: The Navigation Problem

Navigation answers three questions for a vehicle at a given instant:

1. **Where is the vehicle, and how is it moving?** (State estimation)
2. **How uncertain is that estimate?** (Covariance propagation)
3. **Can the mission rely on the estimate right now?** (Integrity assessment)

A classical estimator outputs only a state vector $\hat{\mathbf{x}}$ and a covariance matrix $\mathbf{P}$. In safety-critical flight and spaceflight, a small covariance matrix does not guarantee a safe solution. A filter with an unmodeled sensor bias, a spoofed radio range, or a double-counted aiding source computes a small covariance matrix around a wrong position. `Navigate` therefore treats state estimation, covariance propagation, and integrity honesty as one inseparable problem.

## Technical names

- **Ownship state** is the physical kinematic state of the vehicle: position, velocity, attitude, and clock offset.
- **Dead reckoning** is the integration of motion measurements (such as velocity or specific force and angular rate) from a known initial state.
- **Aiding** is the correction of a dead-reckoning state with external position, velocity, attitude, or field measurements.
- **Sovereignty** is the architectural rule that the inner-loop flight controller owns its control-grade state and never surrenders actuator authority to an outer-loop component.

## Dead reckoning versus external aiding

Every navigation sensor falls into one of two physical categories:

| Category | Physical principle | Error behavior over time | Examples in `Navigate` |
|---|---|---|---|
| Dead reckoning / propagation | Integrates internal motion or known force laws | Smooth at high frequency; error grows without bound | IMU strapdown mechanization (`PropagationMode::SensedImu`), orbital force model (`PropagationMode::ForceModel`), visual odometry (`navigate_visual::session`) |
| External aiding / fixing | Measures geometry relative to external references or maps | Bounded error; subject to outages, latency, outliers, and ambiguity | GNSS (`Pseudorange`, `PositionFix`), star sightings (`LineOfSight`), terrain-relative camera fixes (`VisualPose`), magnetic anomaly maps (`MagneticAnomaly`), radar/laser altimeters (`TerrainClearance`), deep-space tracking (`TwoWayRange`) |

Integrated navigation combines both categories. The propagation source carries the state smoothly across high-rate control steps and rejects short-term measurement noise. The external aiding sources bound the long-term drift of the propagation source and calibrate its internal biases.

## The three-layer vehicle architecture

`Navigate` operates inside a strict three-way system decomposition alongside an inner-loop flight controller (`Aviate`) and a host mission platform (`Pilotage`):

```text
   Host platform: Pilotage (sessions, operator authority, displays)
        │ fenced control path        │ telemetry
        ▼                            ▲
   Navigate ── setpoints ──► Flight controller: Aviate (1 kHz stabilization, actuators)
     │  ▲                        │
     │  └── aiding (DRAFT, ──────┘  FC validates, fuses, or rejects
     │       bounded, with covariance + integrity + composition)
     ├── consumes: GNSS, IMU, celestial, visual, magnetic, terrain, radiometric sources
     └── consumes: ephemeris, star catalogs, magnetic grids, and terrain tiles from host
```

Three boundary rules govern this decomposition:

1. **The flight controller is sovereign over control-grade state.** `Aviate` runs 1 kHz `#![no_std]` stabilization and actuator-allocation loops in `f32`. It remains operational when `Navigate` is absent. `Navigate` never assumes that the flight controller accepted an aiding observation.
2. **Guidance is a commander, not a privilege.** `Navigate` computes outer-loop trajectory setpoints (`GuidanceSetpoint`) in `f64`. These setpoints enter the host platform's fenced authority path as proposals. `Navigate` has no direct path to actuators.
3. **Correlation is declared, never discovered.** Every observation and every published solution carries a `SourceComposition` bitmask. A consumer uses this bitmask to reject measurements that share upstream sensor errors.

## The four responsibilities of `Navigate`

`Navigate` performs four tasks and refuses all others:

1. **Global and relative state estimation (`navigate-fusion`)**: Fuses multi-sensor observations into a surface `NavigationSolution` or a space `CartesianSolution`.
2. **Integrity assessment (`navigate-contract`, `navigate-fusion`)**: Attaches an `IntegrityAssessment` or a `SpaceIntegrityAssessment` to every solution. The assessment reports the solution quality, the active sensor classes, the redundancy level, the $1\sigma$ uncertainties, and the fault-detection status.
3. **Flight-plan validation and leg sequencing (`navigate-route`)**: Validates waypoint plans and sequences active legs using body-specific surface gravity and capture geometry.
4. **Integrity-gated guidance (`navigate-guidance`, `navigate-trajectory`)**: Evaluates surface guidance laws (lateral `L1`, path-tracking, vertical glideslope) and space trajectory laws (orbital burns, Clohessy–Wiltshire rendezvous, atmospheric entry bank control, and polynomial E-guidance powered descent). If the navigation solution does not meet the integrity floor of the maneuver, guidance refuses to issue a setpoint and returns a typed `GuidanceRefusal`.

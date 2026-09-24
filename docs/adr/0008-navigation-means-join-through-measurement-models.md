# ADR-0008: Navigation means join through measurement models and one propagator seam

- Status: Proposed
- Date: 2026-09-23

## Context

Navigate must fuse many navigation means. GNSS and visual positioning exist.
Dead reckoning, inertial navigation (INS), DME/DME, terrain matching, and
celestial fixes follow. A vehicle can have any subset of them.

The filter (ADR-0003) has these limits:

1. The state is position and velocity only. The propagator assumes constant
   velocity. There is no inertial or dead-reckoning propagation.
2. `ObservationValue` has two variants, and the filter core matches on them.
   Each new means is an edit to the filter core.
3. The filter accepts derived fixes only. A DME/DME fix or a GNSS position
   hides the geometry and the shared errors of its ranges.
4. The filter has no attitude state, so a visual pose loses its attitude.
5. A measurement that arrives late is refused when it is older than the
   staleness bound. A camera frame can arrive after other measurements
   that are newer.
6. Visual fixes from one map release share map error (ADR-0007).

## Decision

### One variant, one measurement model

Each `ObservationValue` variant is one measurement model. A model supplies
these items:

- the predicted measurement from the state,
- the linearization of that prediction,
- the measurement noise,
- the admission rule (the innovation gate and its degrees of freedom),
- the composition rule (ADR-0003) and the independence rule (ADR-0007).

The filter core applies a model through one update path. It does not know
the physics of a means. The crate-private `MeasurementModel` trait has a fixed
measurement dimension, so an update does not allocate. The position and
velocity blocks and the range model implement it. A new means is a new
module. The filter core does not change.

### Supported and reserved models

| Variant | Means | State in this build |
|---|---|---|
| `Range` | DME and other ranging transmitters | Supported. Gate with 1 degree of freedom (`range_gate_chi2`). Source class `RadioNavigation`. |
| `Pseudorange` | Raw GNSS | Reserved. Needs the pseudorange model and receiver clock states. |
| `VisualPose` | Visual positioning with attitude | Reserved. Needs attitude in the state. |

The filter refuses a reserved variant with
`RejectionReason::UnsupportedMeasurement { kind }` and counts it in
`RejectionCounters::unsupported_measurement`. It never drops a measurement
silently. `MeasurementKind` names each model. Terrain matching and celestial
sights get reserved variants when their sources are designed.

### Raw measurements before derived fixes

When a source can give raw measurements, the adapter sends them: DME ranges,
not a DME/DME fix; GNSS pseudoranges, not a position, when the receiver gives
them. The filter then sees the geometry and the shared errors. Derived fixes
stay valid input for sources that give nothing else.

### One propagator seam

The state grows through the propagator, not through the measurement models:

1. `ConstantVelocity` is the current propagator.
2. `Inertial` propagates an error state with IMU specific force and angular
   rate. It adds attitude and IMU bias states. `VisualPose` becomes
   supported with it.
3. `DeadReckoning` propagates with air data and heading when there is no
   IMU. It adds wind states.

Attitude in Navigate is for measurement models and integrity. Control-grade
attitude stays with the flight controller.

### Delayed measurements

The filter keeps a bounded history of states and covariances, keyed by time.
A late measurement is applied at its acquisition time, and the filter
propagates forward again. A measurement older than the history is refused.
The history depth is a configuration value. Until the history lands, the
staleness bound (ADR-0003) is the limit.

### Correlation

A model declares whether its errors are independent of the state
(ADR-0007). Shared errors, such as the error of one map release, become bias
states when the filter supports them. Until then, the adapter refuses
evidence whose independence is not validated.

### Performance

- Each model uses fixed-size matrices. An update does not allocate.
- The state dimension is fixed for each propagator, so the filter can use
  fixed-size storage.
- The cost of a late measurement is proportional to the history depth. The
  depth is bounded, so the worst-case update time is bounded.

## Consequences

- A new means is a new variant and a new model module. Consumers of the
  existing variants do not change, because `ObservationValue` and
  `MeasurementKind` are `#[non_exhaustive]`.
- Pilotage can offer reserved measurements now. The filter reports them as
  unsupported, so an integration test can see which means are not active.
- The inertial propagator is the next state change. It is a larger change
  than a new measurement model, and it gets its own record.
- Delayed-measurement history, the range model, the pseudorange model, and
  the inertial and dead-reckoning propagators are tracked as separate work.

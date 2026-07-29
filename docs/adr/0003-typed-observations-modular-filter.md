# ADR-0003: The navigation filter fuses typed, admission-gated observations with pluggable propagation

- Status: Accepted
- Date: 2026-07-29

## Context

Navigate must produce one navigation-grade solution from heterogeneous
sources — GNSS now; celestial and visual navigation later — and remain
operable with any subset, including a single source. Sources arrive at
different rates, with different latencies, different failure modes, and
sometimes shared underlying measurements. A filter that hard-codes sensor
drivers, trusts timestamps, or accepts every measurement would fail
unpredictably exactly when navigation matters most.

## Decision

- **The filter consumes typed `Observation`s, never sensors.** An
  observation is a value (position fix, velocity fix; celestial
  bearing/elevation and visual odometry are reserved `#[non_exhaustive]`
  variants), a covariance, a stamp (source identity, epoch, wrapping
  sequence, monotonic acquisition time, clock domain), and a declared
  source composition. Producing observations from real hardware is outside
  the fusion crate.
- **Admission gates run before any state update**, each with its own
  rejection counter: wrap-aware sequence regression or duplicate per
  source; epoch regression; staleness beyond a configured bound;
  non-finite values; covariance that is not finite and positive
  semidefinite; undeclared (empty) source composition; and a chi-square
  innovation gate. A rejected observation changes nothing, and the
  counters are part of the public state.
- **Correlation is enforced at admission.** An observation whose declared
  composition marks it as derived from the filter's own output class (or
  from FC-exported state) is refused; composition is how double-counting
  is prevented, so an empty composition is inadmissible rather than
  presumed independent.
- **Propagation is pluggable.** A `Propagator` advances state and
  covariance between measurement times; the skeleton ships
  constant-velocity kinematics with configurable process noise. IMU
  mechanization joins as another propagator when a raw inertial source
  lands — it grows the state vector, not the architecture.
- **The state is local-level NED** anchored at an initialization origin
  supplied with the first admitted position fix; geodetic output is a
  conversion at the boundary, so filter math never mixes ellipsoidal and
  planar arithmetic.
- **Delayed measurements are a designed seam.** The staleness bound is the
  skeleton's answer; a bounded state history enabling apply-at-acquisition
  re-propagation is the documented extension, chosen so the seam exists in
  the update path rather than being retrofitted.

## Consequences

- Every gate is a table-driven unit test with synthetic observations;
  filter behavior under duplicates, reordering, outliers, and silence is
  pinned by tests rather than discovered in flight.
- A new source class is a new observation variant plus a measurement
  model — consumers of existing variants never change.
- Single-source operation is the same code path as full fusion; what
  changes is the integrity assessment (ADR-0004), never a special mode.
- The innovation gate depends on honest covariances from sources; a source
  that lies about its noise defeats it. Cross-source consistency checking
  belongs to the integrity layer as redundancy allows.

## Alternatives considered

- **Per-sensor filter cascade (one filter per source, then blend):**
  rejected; blending hides cross-covariances and makes integrity
  accounting opaque.
- **Trusting source timestamps without admission:** rejected; replayed and
  reordered measurements are ordinary events on real links, and the
  sibling systems' ingress disciplines exist precisely because of them.

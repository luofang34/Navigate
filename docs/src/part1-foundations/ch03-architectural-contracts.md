# Chapter 3: Architectural Contracts: Correlation, Integrity, and Determinism

Software bugs in navigation filters rarely cause immediate crashes. Instead, they corrupt covariance matrices, hide sensor outages, or introduce non-deterministic timing drift. `Navigate` prevents these failures through three architectural contracts enforced by the Rust type system (ADR-0002, ADR-0004, ADR-0005, ADR-0015, ADR-0016).

## Technical names

- **Source composition** (`SourceComposition`) is a bitmask of `SensorClass` flags that identifies every physical sensor family that contributed to a measurement or a fused solution.
- **Incestuous amplification** is the false reduction of covariance that occurs when a filter fuses an external estimate that already contains the filter's own prior state or shared sensor measurements.
- **Fail-closed integrity** is the design rule that any missing, stale, or unverified integrity input causes the consumer to refuse the operation rather than assume nominal quality.
- **Zero-heap execution** is the guarantee that a crate compiles with `#![no_std]` and performs no dynamic memory allocation (`alloc` is not linked).

## Contract 1: Explicit correlation discipline (`SourceComposition`)

In a multi-sensor vehicle, sensors often form cascades. For example:

- A visual localizer uses a pose prior from the navigation filter to narrow its search window.
- A star tracker adapter uses the vehicle position to express an inertial star attitude as a local NED `AttitudeFix`.
- A flight controller exports a fused velocity estimate back to an outer-loop consumer.

If a Kalman filter treats a cascaded output as an independent measurement with white error $\mathbf{R}$, the filter double-counts the shared information. The covariance matrix $\mathbf{P}$ shrinks artificially, and the filter stops trusting new measurements.

`Navigate` enforces three rules to prevent incestuous amplification:

1. **Every observation declares a non-empty `SourceComposition` (ADR-0003).** If `composition.is_empty()` is true, the admission gate rejects the observation with `RejectionReason::UndeclaredComposition`.
2. **A search prior never enters a fused measurement (ADR-0009).** When the navigation estimate narrows a visual search region, the visual solver computes the final camera pose from image-to-map correspondences alone. The prior never enters the weighted least-squares objective.
3. **Coupled measurements use raw geometry (ADR-0021).** A star camera feeds `InertialNavigationFilter` with `ObservationValue::LineOfSight` in ECEF axes rather than a position-dependent local `AttitudeFix`.

## Contract 2: Integrity honesty (`IntegrityAssessment`)

A filter with one GNSS receiver can estimate position and velocity, but it cannot detect a smooth ramp fault on that receiver unless an independent sensor cross-checks it. `Navigate` computes an `IntegrityAssessment` for every surface `NavigationSolution` and a `SpaceIntegrityAssessment` for every `CartesianSolution` (ADR-0004, ADR-0015):

| Field | Type | Meaning |
|---|---|---|
| `quality` | `SolutionQuality` | `NavigationGrade`, `Degraded`, or `Unusable` |
| `contributing_classes` | `SourceComposition` | Union of sensor classes admitted inside the assessment window |
| `redundancy` | `Redundancy` | `None`, `SingleSource`, `SameClassMultiSource`, or `IndependentClasses` |
| `horizontal_1sigma_m` / `vertical_1sigma_m` | `f64` | $1\sigma$ position uncertainties from the NED covariance (or radial, along-track, cross-track `SpaceUncertainty` in orbit) |
| `fault_detection` | `FaultDetection` | `Unavailable` unless independent sources or RAIM cross-checks are active |
| `protection_levels` | ` Option<ProtectionLevels>` | Reserved for validated horizontal and vertical protection levels (HPL / VPL); `None` when unimplemented |

Two rules keep this assessment honest:

- **No fabricated redundancy:** Single-source operation caps `redundancy` at `Redundancy::SingleSource` and marks `fault_detection` as `FaultDetection::Unavailable`.
- **Gate-refusal degradation (ADR-0015):** When an observation fails the $\chi^2$ innovation gate, the filter rejects the update and leaves the covariance matrix $\mathbf{P}$ unchanged. However, a run of consecutive innovation-gate rejections indicates a disagreement between the filter state and the sensor. Once consecutive rejections reach `FusionConfig::max_consecutive_gate_rejections`, the filter caps `SolutionQuality` at `Degraded` until an aiding observation passes admission.

## Contract 3: Sans-IO determinism and `#![no_std]` zero-heap topology

Every core navigation crate in `Navigate` (`navigate-contract`, `navigate-camera`, `navigate-geodesy`, `navigate-astrodynamics`, `navigate-fusion`, `navigate-magnav`, `navigate-route`, `navigate-trajectory`, `navigate-guidance`, `navigate-terrain-safety`, `flamsteed`, and `eclipse-canon`) obeys two strict implementation rules:

1. **Sans-IO determinism:** Domain functions never call `std::time::Instant::now()`, read environment variables, open files, or query hardware RNGs. The caller passes time (`MonotonicNanos` or `PhysicalInstant`) as an argument. The particle filter uses a deterministic `SplitMix64` pseudo-random generator seeded by the caller. Identical input logs produce bit-identical outputs across x86_64, ARM Cortex-M (`thumbv8m.main-none-eabihf`), RISC-V (`riscv32imac-unknown-none-elf`), and WebAssembly (`wasm32-unknown-unknown`).
2. **Zero-heap fixed capacity:** All state vectors, covariance matrices (`nalgebra::SMatrix`), source tables (`MAX_SOURCES`), delayed-measurement history rings, spherical harmonic work tables (`MAX_INLINE_HARMONIC_DEGREE = 36`), and route leg arrays reside on the stack or in caller-owned slices (`ParticlePool`). No memory allocation can fail during flight.

# Chapter 3: Architectural Contracts: Correlation, Integrity, and Determinism

Software defects in navigation filters often corrupt covariance matrices, hide sensor outages, or introduce timing drift without causing a crash. `Navigate` addresses these failure modes through three architectural contracts enforced by the Rust type system (ADR-0002, ADR-0004, ADR-0005, ADR-0015, ADR-0016).

## Technical names

- **Source composition** (`SourceComposition`) is a bitmask of `SensorClass` flags that identifies the physical sensor families that contributed to a measurement or a fused solution.
- **Incestuous amplification** is the false reduction of covariance that occurs when a filter fuses an external estimate that already contains the filter's own prior state or shared sensor measurements.
- **Selection bias (prior conditioning)** is the statistical dependence introduced when a filter's prior state restricts the candidate search region of an upstream matcher or localizer.
- **Solution quality** (`SolutionQuality`) is a classification of estimator health based on active sensor classes, covariance bounds, and innovation-gate behavior.
- **Formal integrity assurance** is a probabilistic guarantee (such as Horizontal and Vertical Protection Levels, HPL/VPL, at a target Hazardously Misleading Information risk) that the true position error does not exceed an alert limit.

## Contract 1: Correlation discipline and its limits (`SourceComposition`)

In a multi-sensor vehicle, subsystems often form cascades:

- A visual localizer uses a pose prior from the navigation filter to narrow its candidate tile search region.
- A star tracker adapter uses the vehicle position to express an inertial star attitude as a local NED `AttitudeFix`.
- A flight controller exports a fused velocity estimate back to an outer-loop consumer.

If a Kalman filter treats a cascaded output as an independent measurement with white error $\mathbf{R}$, the filter double-counts the shared information and shrinks the covariance matrix $\mathbf{P}$ artificially.

`Navigate` enforces four mechanisms to bound correlation, while recognizing that sensor-class bitmasks alone do not guarantee full statistical independence:

1. **Mandatory `SourceComposition` declaration (ADR-0003):** Every observation declares a non-empty `SourceComposition`. If `composition.is_empty()` is true, the admission gate rejects the observation with `RejectionReason::UndeclaredComposition`.
2. **Prior exclusion from the estimator objective (ADR-0009):** When the navigation estimate narrows a visual search region, the visual PnP solver computes the camera pose from image-to-map correspondences alone; the prior is not added as a quadratic penalty in the PnP objective.
3. **Raw-geometry measurement models (ADR-0021):** A star camera feeds `InertialNavigationFilter` with `ObservationValue::LineOfSight` in ECEF axes rather than a position-dependent local `AttitudeFix`.
4. **Same-cell deduplication and conservative error budgets (`EvidenceIndependence`, ADR-0007):** Even when the prior does not enter the PnP objective, two residual correlation sources remain:
   - *Prior selection bias:* The search prior selects which candidate map basins are evaluated.
   - *Shared map and datum errors:* Consecutive visual fixes over the same `MapRevision` share systematic DEM and orthophoto registration biases.
   To bound these effects, `VisualFixSource` enforces `EvidenceIndependence` (refusing anchors from map cells that already produced a fix in the session) and `VisualErrorBudget` combines systematic error allowances via a worst-case correlated bound (Chapter 13).

## Contract 2: Distinguishing solution quality from formal integrity assurance (`IntegrityAssessment`)

In aviation and space navigation engineering, five distinct concepts describe the trustworthiness of an estimate:

| Concept | Mathematical meaning | Status in `Navigate` |
|---|---|---|
| **1. State uncertainty** | Posterior covariance $\mathbf{P}$ under the assumed linear-Gaussian model | Computed in every filter (`horizontal_1sigma_m`, `vertical_1sigma_m`, `SpaceUncertainty`) |
| **2. Estimator consistency** | Agreement between innovations $\mathbf{y}_k$ and predicted variance $\mathbf{S}_k$ (`InnovationStats`, OOSM replay) | Tracked per `MeasurementKind` and per constrained state component (`ConstrainedState`) |
| **3. Fault-detection capability** | Ability to cross-check and isolate a faulty source using independent redundant measurements | Reported in `Redundancy` and `FaultDetection` (`Unavailable` under single-source operation) |
| **4. Solution quality (`SolutionQuality`)** | Operational health tier (`NavigationGrade`, `Degraded`, `Unusable`) summarizing uncertainty, freshness, and gate rejections | Implemented in `IntegrityAssessment` and `SpaceIntegrityAssessment` (ADR-0004, ADR-0015) |
| **5. Formal integrity assurance** | Probabilistic bound $P(\|\hat{\mathbf{p}} - \mathbf{p}\| > \text{PL}) \le P_{\text{HMI}}$ under specified fault hypotheses (RAIM / ARAIM HPL, VPL, Time-to-Alert) | Reserved in `IntegrityAssessment::protection_levels: Option<ProtectionLevels>` (`None` until a RAIM/ARAIM engine is validated) |

Readers and integrators must not mistake `SolutionQuality::NavigationGrade` for a certified RAIM/ARAIM protection level. `NavigationGrade` means that fresh observations are passing admission, the $1\sigma$ covariance is within the configured nominal threshold, and no innovation-gate rejection storm is active.

### Per-component gate-refusal degradation (ADR-0015)

When an observation fails the $\chi^2$ innovation gate, the filter rejects the update and leaves the covariance matrix $\mathbf{P}$ unchanged. To prevent a small covariance from masking a sensor disagreement, `ProvenanceLedger` tracks consecutive innovation-gate rejections separately across four constrained state components (`ConstrainedState`: horizontal position, vertical position, velocity, and attitude). If consecutive gate rejections on any component reach `FusionConfig::max_consecutive_gate_rejections`, the filter caps `SolutionQuality` at `Degraded` until an admitted observation constrains that same component again (for example, an admitted barometric `Altitude` clears the vertical streak, but cannot clear a horizontal position rejection streak).

## Contract 3: Sans-IO determinism, `#![no_std]` zero-heap topology, and reproducibility bounds

Every core navigation crate in `Navigate` (`navigate-contract`, `navigate-camera`, `navigate-geodesy`, `navigate-astrodynamics`, `navigate-fusion`, `navigate-magnav`, `navigate-route`, `navigate-trajectory`, `navigate-guidance`, `navigate-terrain-safety`, `flamsteed`, and `eclipse-canon`) obeys two structural rules:

1. **Sans-IO execution:** Domain functions never call system clocks, read files, or query hardware random number generators. The caller passes time (`MonotonicNanos` or `PhysicalInstant`) as an argument, and the particle filter uses a deterministic `SplitMix64` PRNG seeded by the caller.
2. **Zero-heap fixed capacity:** All state vectors, covariance matrices (`nalgebra::SMatrix`), source tables (`MAX_SOURCES`), delayed-measurement history rings, spherical harmonic work tables (`MAX_INLINE_HARMONIC_DEGREE = 36`), and route leg arrays reside on the stack or in caller-owned slices (`ParticlePool`).

### Bitwise reproducibility versus cross-target floating-point tolerance

- **Same-target bitwise reproducibility:** On a fixed target architecture, compiler version, and math backend (`std` or `libm`), identical input sequences produce bit-for-bit identical state trajectories and rejection counters.
- **Cross-target floating-point behavior:** The Rust standard library (`std`) delegates transcendental `f64` operations (`sin`, `cos`, `atan2`, `exp`, `ln`) to the host platform `libm`, which can differ by $1\text{–}2\text{ ULPs}$ across operating systems, and hardware fused multiply-add (FMA) instructions can alter rounding order across CPU architectures. When strict cross-target bit-identity is required across bare-metal (`thumbv8m.main-none-eabihf`, `riscv32imac-unknown-none-elf`), WebAssembly (`wasm32-unknown-unknown`), and host targets, builds must compile with `--no-default-features --features libm` so that all transcendental functions execute the same pure-Rust software routines.

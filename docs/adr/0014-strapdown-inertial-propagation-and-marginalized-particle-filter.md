# ADR-0014: Strapdown inertial propagation and a marginalized particle filter

- Status: Proposed
- Date: 2026-11-03

## Context

The six-state `NavigationFilter` has a constant-velocity model. It has
no attitude state and no sensor bias state. These limits have these
effects:

1. The filter cannot use an IMU. Between two aiding observations, the
   position uncertainty grows with a fixed acceleration noise, not with
   the measured motion.
2. The filter cannot apply an attitude fix, a visual pose, or a range
   to sloped terrain along a body-fixed beam. Each of these measurements
   depends on the attitude.
3. A barometric altitude has a slowly changing bias. A magnetic anomaly
   map has a correlated error for each map release. Without bias states,
   these errors go into the position.
4. A spacecraft moves in a Cartesian frame that is centered on a body.
   The gravity of a third body is significant near the Moon. The frame
   center changes at the sphere of influence.
5. A linearized map model fails where the map gradient is zero or where
   the map has several matching positions. The `UninformativeGradient`
   gate of ADR-0012 refuses these measurements, but the filter then has
   no map information.

MagNav.jl gives a reference design for magnetic anomaly navigation: an
error-state filter with an IMU, a scalar map bias, a marginalized
particle filter, a Cramér–Rao bound, and the Tolles–Lawson aircraft
model. This decision adapts that design to the admission, frame, and
integrity rules of ADR-0003, ADR-0004, ADR-0012, and ADR-0013.

## Decision

### Error state

- `InertialNavigationFilter` has a nominal strapdown state and an error
  state with 17 components: `δp` (3), `δv` (3), `δθ` (3), `δb_a` (3),
  `δb_g` (3), `δb_alt` (1), and `δb_mag` (1).
- `δp` and `δv` are in the axes of the active frame. `δθ` is a right
  perturbation in the body axes: `C_true = C_nom·exp([δθ×])`. The bias
  errors are additive.
- Each update injects the error estimate into the nominal state. Then
  the filter sets the error state to zero and resets the covariance with
  `J = I`, except `J_θθ = I − ½·[δθ×]` (closed loop).
- Each bias has a first-order Gauss–Markov model (`FogmBias`). The
  scalar magnetic bias `b_mag` is the catch-all map error of MagNav.jl.

### Regimes

- In the body-surface regime, the state is in a tangent plane fixed to a
  rotating `PlanetaryBody`. The plane is a rigid Cartesian frame. The
  gravity uses the body-centered position `r_bf = r_origin + Rᵀ·p`, so
  the plane has no flat-earth error and needs no re-anchoring.
- `PlanetaryBody` gives the point mass, the `J2` zonal harmonic, the
  centrifugal term, and their exact gradient. The presets are
  `earth_wgs84`, `moon_me`, `moon_pa`, and `mars_iau`.
- In the Cartesian space regime, the state is in the frame of a
  `SpaceEnvironment`. Its `SpaceGravityModel` is `Zero`, `CentralBody`,
  or `MultiBody`. A third body adds the direct and indirect tidal
  acceleration `μ·(d/|d|³ − r_k/|r_k|³)` with `d = r_k − r`. A frame
  rotation adds the centrifugal term.
- `new_surface` on a body that is not on WGS84 and `new_space` set
  `FusionConfig::require_explicit_frame` to `true` (ADR-0013).

### Mechanization and discretization

- The mechanization uses the bias-corrected rates, the midpoint
  quaternion, midpoint RK2 for the velocity, and the trapezoidal rule for
  the position. The Coriolis term is `−2·ω × v`.
- With a right perturbation, the frame rotation cancels in the attitude
  block: `F_θθ = −[ω_ib×]`.
- The filter discretizes `F` and `Q_c` with the Van Loan block matrix of
  dimension 34 and a `[6/6]` Padé matrix exponential with scaling and
  squaring. `Q_c` of a bias is `2·σ²/τ`, so the discrete variance equals
  the `FogmBias` step.

### Measurement models

- The body-surface regime applies `PositionFix`, `VelocityFix`, `Range`,
  `Altitude`, `MagneticAnomaly`, `TerrainClearance`, `AttitudeFix`, and
  `VisualPose`. The Cartesian space regime applies
  `CartesianPositionFix`, `CartesianVelocityFix`, `CartesianRange`, and
  `AttitudeFix`. Each regime refuses the other kinds with
  `RejectionReason::UnsupportedMeasurement`. Both regimes refuse
  `Pseudorange`.
- An `Altitude` with `SensorClass::Barometric` in its composition
  measures `h + b_alt`. Any other altitude does not touch `b_alt`.
- `TerrainClearance` is the slant range `ρ = (h − T₀)/κ` with
  `m = [s_N, s_E, 1]` and `κ = m·(C_bl·b)`. The range depends on the
  position through the height and the slope, and on the attitude through
  the beam. It does not touch `b_alt`. A beam with `κ ≤ 0.05` or a
  vehicle at or below the terrain is refused with
  `RejectionReason::ImplausibleCovariance`.
- `MagneticAnomaly` measures `map_t_at_lin + g·(p − p_lin) + b_mag`. The
  checks of ADR-0012 apply. `b_mag` belongs to one map: the source and
  `Observation::map_revision`. When the map changes, the filter starts a
  new bias at zero with the stationary variance. A refused observation
  does not start a new bias.
- `AttitudeFix` and `VisualPose` use the residual
  `2·log(q_nom⁻¹ ⊗ q_meas)` in the hemisphere `w ≥ 0`. On a body
  surface, the reference axes are the local NED axes. `VisualPose` is one
  update of dimension six with `FusionConfig::pose_gate_chi2`. Its
  attitude covariance rotates from the local NED axes into the body axes.

### Admission and time

- The gates of `ingest` run in this order: clock domain, composition,
  regime support, value plausibility, freshness, per-source ordering,
  the frame at the acquisition time, the model checks, and the
  innovation gate. A refusal increments one counter and changes nothing.
- An IMU sample propagates the state from the state time to the end of
  the sample with the rates of the sample. An observation after the
  state time holds the rates of the most recent sample. An observation
  after the state time before the first sample is refused with
  `RejectionReason::NoInertialData`.
- An IMU sample enters the provenance. It does not enter the assessment
  window, and it does not reset the observation silence. A propagation
  sensor cannot cross-check the state, so the quality degrades during an
  outage of the aiding sources (ADR-0004).

### History

- The history keeps the state before each event: an IMU sample, an
  observation, a change of the space frame, or an environment update.
  `FusionConfig::history_capacity` counts events.
  `InertialConfig::default` sets a capacity that holds more events than
  the IMU rate times the staleness bound.
- An event before the state time rewinds to the state before the first
  later event. When the interval of the next IMU sample holds the
  acquisition time, the propagation to the acquisition time uses the
  rates of that sample. The replay of the sample then propagates the
  rest of the interval. Thus the split of the interval is exact.
- The frame gate compares the declared frame with the frame of the state
  that the event updates. Thus a delayed observation in `EARTH_J2000` is
  admitted after a change to `MOON_J2000` when it was acquired before the
  change.

### Space frames

- `recenter_space_frame` moves the state into a new frame at one time:
  `p' = R·(p − s_p)`, `v' = R·(v − s_v)`, and `q' = q_R ⊗ q`. The
  position and velocity blocks of the covariance rotate. The attitude
  and bias blocks do not change. The time must not be before the state
  time.
- `update_space_environment` sets new third-body positions in the same
  frame. A different frame is refused with
  `RejectionReason::FrameMismatch`.
- The transformation is exact when the two frames have the same rotation
  rate. The caller puts any transport term into the velocity shift.

### Publication

- `tick_surface` publishes a `NavigationSolution` in the local NED axes
  at the vehicle. The attitude is `q_plane→local ⊗ q_body→plane`. The
  attitude covariance is `C_bl·P_θ·C_blᵀ`.
- `tick_space` publishes a `CartesianSolution` in the active frame. A
  Cartesian frame has no vertical, so the integrity uses the largest
  position 1-sigma for both the horizontal and the vertical value.

### Marginalized particle filter

- `MarginalizedParticleFilter` follows Schön, Gustafsson, and Nordlund
  (2005). Each particle holds a position in the tangent plane and the
  conditional mean of `[v_n, v_e, v_d, b_alt, b_mag]`. The particles
  share one linear covariance.
- The time update uses the cross-coupling of their step 5. The sampled
  motion of each particle updates its linear mean. Thus map measurements
  of the position also correct the velocity.
- `ingest_scalar_map` samples a `ScalarFieldMap` at each particle with no
  linearization. `BilinearGridMap` is a regular latitude and longitude
  grid. The weights use the log-sum-exp form. The filter resamples
  systematically when the effective sample size is low.
- A deterministic `SplitMix64` source with the Box–Muller transform
  draws the noise. Identical seeds give identical results.
- The published quality is `Degraded` at best, because one map gives no
  second means. A cloud wider than `MpfConfig::unimodal_max_1sigma_m`,
  such as two modes, is `Unusable`.

### Analysis tools

- `CrlbTracker` runs the covariance recursion of the inertial filter at
  a reference trajectory. It gives the posterior Cramér–Rao bound of a
  map leg before a flight.
- `OnlineTlCalibrator` estimates the 18 Tolles–Lawson coefficients and a
  residual bias with a recursive Joseph-form update. Below
  `TlCalibratorConfig::min_b_dot_norm_t_per_s`, the coefficients do not
  learn. Thus a crustal anomaly in straight and level flight does not go
  into the aircraft coefficients.

### Six-state filter

- `NavigationFilter` refuses `TerrainClearance`, `AttitudeFix`,
  `VisualPose`, and the Cartesian kinds with
  `RejectionReason::UnsupportedMeasurement`.
- `FusionConfig::unmodeled_magnetic_bias_variance_t2` adds the error
  budget of the map bias to each magnetic variance. The default is 0.0.
  A negative or non-finite value fails closed.

## Consequences

- An IMU, an attitude source, a surface altimeter, and a camera pose can
  aid one state. Each refusal is counted, as in ADR-0003.
- A barometric bias and a map bias do not go into the position. Each map
  revision has its own bias.
- One filter covers the surface of the Earth, the Moon, and Mars, and a
  cislunar transfer with a change of frame center.
- Each IMU sample costs one matrix exponential of dimension 34. Each
  history entry holds one copy of the 17 × 17 covariance.
- A map with a zero gradient or with several modes needs the particle
  filter. The linearized models refuse those measurements.
- The particle filter has no admission gates. The caller must apply the
  clock, frame, and provenance rules before it calls the filter.

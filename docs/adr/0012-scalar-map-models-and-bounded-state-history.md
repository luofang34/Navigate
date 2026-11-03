# ADR-0012: Scalar map models and a bounded state history extend the filter core

- Status: Proposed
- Date: 2026-11-02

## Context

ADR-0008 sets the rule: a new navigation means is a new measurement model,
and the state grows through one propagator seam. ADR-0008 also tells the
filter to apply a delayed measurement at its acquisition time.

Magnetic anomaly navigation (MagNav) is the next means. It has these
properties:

1. The measurement is a scalar. The sensor measures the anomaly of the
   magnetic field. A map gives the anomaly at each position.
2. The map is not linear. The model is valid only near the point where
   the adapter linearized the map.
3. In a flat map cell, the gradient is almost zero. A measurement there
   gives almost no position information. If the filter admits it, the
   filter shows a cross-check that did not occur.
4. Survey values are small in SI units. A gradient of 0.05 nT/m is
   5e-11 T/m. A 3 nT noise is a variance of 9e-18 T². An absolute lower
   limit on a variance would refuse valid magnetic data.
5. Each map release has a correlated error that requires a bias state or
   an error budget in the observation variance.

Altitude has a related problem. The filter state is in a local tangent
plane. At 50 km from the plane origin, the plane `down` axis is
approximately 196 m away from the height above the ellipsoid. A
flat-plane altitude model has this error.

A camera frame or a magnetic measurement can also arrive after newer
measurements. Applying a delayed measurement at the current state time
gives a lag error. At 70 m/s, a delay of 400 ms gives a position bias of
28 m along the track.

## Decision

### Generic state dimension and propagator seam

- `KalmanState<N>` and `PreparedUpdate<M, N>` are generic over the state
  dimension `N`. The six-state layout stays the default.
- The crate-private trait `Propagator<N>` owns the dynamics of one state
  layout. `ConstantVelocityPropagator` implements `Propagator<6>`.
- `FogmBias` is the seam for bias states. It gives the discrete
  first-order Gauss–Markov step: the mean goes to `exp(−dt/τ)·x` and the
  process variance is `σ²·(1 − exp(−2·dt/τ))`. The steady-state variance
  is `σ²` for every step size. A bias-carrying propagator adds one state
  for each such bias, such as the error of one map release.

### Model admissibility

`MeasurementModel<M, N>` has the method `admissible`. The filter calls it
on the propagated state before the update and before the chi-square
gate. The default accepts every state. A map model uses it for two
checks:

1. The linearization region. The filter adds the offset of the state
   from the linearization point to `linearization_sigma_multiplier`
   horizontal 1-sigma values. If the sum is more than the valid radius,
   the filter refuses the observation with
   `RejectionReason::LinearizationRegionExceeded`.
2. The information ratio. The ratio is `η = gᵀ P_pos g / R`. Here `g` is
   the gradient in plane axes, `P_pos` is the position covariance, and
   `R` is the measurement variance. The ratio has no unit, so it does
   not depend on the magnetic scale. If `η` is below
   `min_information_ratio` or is not finite, the filter refuses the
   observation with `RejectionReason::UninformativeGradient`.

A refused observation does not change the state, the silence time, the
assessment window, or the redundancy (ADR-0004). Thus flat-cell
measurements cannot make the integrity better.

A negative or non-finite value of `linearization_sigma_multiplier` or
`min_information_ratio` makes the check fail. The checks fail closed.

### Altitude and magnetic anomaly models

- `ObservationValue::Altitude` is a height above the WGS84 ellipsoid. The
  model converts the state position to geodetic coordinates and compares
  the altitudes. The Jacobian is the local up direction, rotated into the
  plane axes. The model has no curvature error at any distance from the
  plane origin.
- `ObservationValue::MagneticAnomaly` is pre-linearized by the source
  adapter. The adapter supplies the measured anomaly, the map value and
  the map gradient at a linearization point, the valid radius, and the
  variance. All values are SI: tesla, tesla per meter, meters, and tesla².
  The gradient arrives in the NED axes at the linearization point. The
  model rotates it into the plane axes. The prediction is
  `map_t_at_lin + g · (p − p_lin)`.
- Admission requires a finite value and a strictly positive variance. It
  has no lower scale limit, so a variance of 9e-18 T² is valid.
- `scalar_gate_chi2` is the gate with 1 degree of freedom for range,
  altitude, and magnetic anomaly.

### Bounded delayed-measurement history

- The filter keeps up to `history_capacity` admitted updates. Each entry
  holds the core before the update and the observation value. The
  filter also keeps the core at initialization. `reset` clears the
  history.
- An observation at or after the state time is in order. The filter
  updates a candidate core and records the entry when it admits the
  observation.
- An observation before the state time is delayed. It must pass the same
  clock, composition, value, freshness, and per-source ordering checks.
  The filter then rewinds a scratch core to the entry before the first
  later update. It applies the delayed observation at its acquisition
  time with `admissible` and the chi-square gate.
- The filter then replays the later entries in time order. A replay does
  not run `admissible` or the gate again, so the replay is deterministic.
  A replay builds each model against the replayed core, so a replay that
  crosses a re-anchor uses the correct plane.
- The filter changes its core and its history only when every step
  passes. A refused delayed observation changes nothing.
- The filter refuses a delayed observation with
  `RejectionReason::OutsideHistory` in these conditions: the observation
  is before initialization, `history_capacity` is 0, or the history
  evicted the updates before the acquisition time.
- When the observations arrive in a different order, the result is the
  same as the in-order result.

### Innovation statistics

The filter keeps `InnovationStats` for each `MeasurementKind`: the
admitted count, the mean and the last NIS, and the number of consecutive
gate refusals. The NIS is the chi-square statistic of an admitted update.
The position fix that initializes the filter has no innovation, so it is
not in the statistics. A replay does not change the statistics.
`NavigationFilter::innovation_stats` gives the values. `reset` clears
them.

### Sensor class and crate boundary

- `SensorClass::MagneticAnomaly` is the composition class of magnetic
  anomaly observations (ADR-0003). Its bit is `1 << 9`. A magnetic
  observation and a GNSS observation have disjoint compositions, so
  they can give `Redundancy::IndependentClasses` only when the filter
  admits both.
- `navigate-magnetic` owns the sensor domain: platform compensation, the
  anomaly map, and the map interpolation. It does not depend on
  `navigate-fusion`.
- `navigate-magnetic-fusion` is the adapter. It reads the map at the
  linearization point and makes an `ObservationValue::MagneticAnomaly`.
  It declares the valid radius and the total variance, which includes
  the map error.
- `navigate-fusion` does not read a map. It knows only the linearized
  measurement.

## Consequences

- MagNav and altitude aiding use the same update path as the other
  models. The filter core does not know magnetic physics.
- Flat-cell or out-of-region magnetic data is refused and counted. It
  cannot hide GNSS silence and cannot add redundancy.
- A delayed observation inside the history has no lag bias. The cost is
  proportional to the number of entries after the acquisition time. The
  capacity bounds this cost.
- A delayed observation older than the history is refused. The adapter
  must deliver observations inside the history depth.
- Without a magnetic bias state in the six-state propagator, the adapter
  must include the map error in `variance_t2`. A propagator with bias
  states models that error with `FogmBias`.

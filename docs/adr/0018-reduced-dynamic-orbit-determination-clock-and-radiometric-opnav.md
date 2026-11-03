# ADR-0018: Reduced-dynamic orbit determination, receiver clock states, and radiometric and optical navigation

- Status: Proposed
- Date: 2026-11-03

## Context

ADR-0017 puts the inertial filter in one body-centered Cartesian frame.
The filter has these limits in space:

1. Only IMU samples and IMU increments propagate the state. A
   spacecraft in a coast often has no useful accelerometer data. A
   long coast needs a force model.
2. The gravity has the point mass, `J2`, and third bodies. Orbit
   determination in a low orbit also needs the higher harmonics, the
   solar radiation pressure, and the drag.
3. The filter has no state for the forces that the model does not hold.
   A force model without such a state is too confident.
4. A one-way range, such as a GNSS pseudorange or the signal of a lunar
   surface beacon, includes the bias and the drift of the receiver
   clock. The filter has no clock states, so ADR-0008 keeps
   `Pseudorange` as reserved vocabulary.
5. The filter has no model for the two-way range and the two-way range
   rate of a deep-space network station.
6. The filter has no model for a direction to a star or to a target,
   which a star tracker and an optical navigation camera measure.

## Decision

### Gravity and forces in `navigate-astrodynamics`

- `SphericalHarmonicField<'a>` borrows the Stokes coefficients `C_nm`
  and `S_nm` up to the degree `MAX_INLINE_HARMONIC_DEGREE`, which is 36.
  The coefficients are unnormalized or fully normalized. The field
  uses the Cartesian solid harmonics `V_nm` and `W_nm` of Montenbruck
  and Gill in the body-fixed axes. The recursion uses `x`, `y`, and `z`
  and no angle, so it has no limit at a pole.
- The field gives the acceleration and its exact gradient. Each
  derivative of a solid harmonic is a sum of solid harmonics of the next
  degree, so the gradient needs the degrees up to `N + 2`. The work
  table is on the stack and has a fixed size.
- An optional orientation rotates the frame axes into the body-fixed
  axes, so the field is also correct in an inertial frame.
- `SpaceGravityModel::SphericalHarmonics` holds a field with `'static`
  coefficients and third bodies. Thus `SpaceEnvironment` stays `Copy`
  and `'static`.
- `CannonballSrp` gives the solar radiation pressure of a sphere. The
  pressure at one astronomical unit is `4.56e-6` N/m², and it decreases
  with the square of the distance from the Sun. The central body casts a
  cylindrical shadow. In the shadow the acceleration is zero.
- `ExponentialDrag` gives the drag in an atmosphere with an exponential
  density. The atmosphere turns with a given rate relative to the frame,
  so the drag uses the velocity relative to the air. The model gives the
  Jacobians on the position and on the velocity.
- `SpaceEnvironment` has the optional fields `srp` and `drag`.
  `SpaceEnvironment::inertial` and `SpaceEnvironment::from_body` set
  them to `None`. `with_srp` and `with_drag` set them.
- `OrbitalForceModel<'a>` adds the gravity, the centrifugal and the
  Coriolis terms, the solar radiation pressure, and the drag. It gives
  the acceleration relative to the frame and its Jacobians. It has a
  fourth-order Runge–Kutta step. `SpaceEnvironment::force_model` gives
  the model of an environment.
- An accelerometer senses the solar radiation pressure and the drag. Thus
  `GravityField::gravity` of an environment does not include them.

### Twenty-two error states

The error state of `InertialNavigationFilter` has twenty-two states:

| Index | State | Unit |
|---|---|---|
| 0–2 | Position error | m |
| 3–5 | Velocity error | m/s |
| 6–8 | Attitude error in the body axes | rad |
| 9–11 | Accelerometer bias error | m/s² |
| 12–14 | Gyroscope bias error | rad/s |
| 15 | Barometric altitude bias error | m |
| 16 | Scalar magnetic map bias error | T |
| 17–19 | Empirical acceleration error of the dynamic model compensation, in the frame axes | m/s² |
| 20 | Receiver clock range bias error `c·δt` | m |
| 21 | Receiver clock range-rate drift error `c·δṫ` | m/s |

- `InertialConfig::dmc` is an optional first-order Gauss–Markov model of
  each axis of the empirical acceleration.
- `InertialConfig::clock_model` is an optional `ClockNoiseModel`. The
  white frequency noise density `S_f` drives the bias. The random-walk
  frequency noise density `S_g` drives the drift. Over `Δt` the
  transition is `[[1, Δt], [0, 1]]`, and the discrete noise is
  `[[S_f·Δt + S_g·Δt³/3, S_g·Δt²/2], [S_g·Δt²/2, S_g·Δt]]`. The Van
  Loan step of the filter gives the same values.
- When a model is `None`, its rows and columns in the dynamics, in the
  noise, and in the initial covariance are zero. Thus the variance stays
  zero, the Kalman gain of the state stays zero, and the state does not
  couple into the other states.
- A frame change turns the empirical acceleration with the axes. The
  clock does not change.
- `ERROR_STATES`, `ErrorMatrix`, `initial_error_covariance`, and
  `error_transition` keep their signatures. A covariance analysis, such
  as the bound of `navigate-magnav`, gets the new dimension.

### Propagation modes

- `PropagationMode::SensedImu` is the mode of ADR-0017. IMU samples and
  IMU increments drive the state. The accelerometers sense each force,
  so the empirical acceleration does not enter the velocity.
- `PropagationMode::ForceModel` holds a `ForceModelCoast`: the longest
  Runge–Kutta step, a constant inertial body rate, and the white noise
  densities of the forces that the model does not hold and of the body
  rate. The position and the velocity follow
  `v̇ = a(p, v) + a_dmc` with fourth-order Runge–Kutta steps. The
  empirical acceleration decays in each step. The attitude turns with
  the body rate.
- In the force-model mode, the covariance uses `∂a/∂r` and `∂a/∂v` of
  the force model, the Coriolis term, the coupling `I` of the empirical
  acceleration into the velocity, and the clock transition. The Van Loan
  step at the midpoint of each Runge–Kutta step gives `Φ` and `Q_d`.
- An observation, a publication, and `coast_to` propagate the state in
  the force-model mode. No IMU data is necessary.
- `set_propagation_mode` propagates to its time in the active mode and
  then changes the mode. `apply_impulsive_burn` propagates to its time,
  adds the velocity change, and adds its covariance to the velocity
  block. Both are history events, so a delayed observation replays
  through them. A time before the state time gives
  `AcquisitionTimeRegression`.
- The force-model mode refuses an IMU sample and an IMU increment. The
  sensed-IMU mode refuses a burn and a coast, because the accelerometers
  already sense a burn. Each refusal is
  `RejectionReason::PropagationModeMismatch`, and it has a counter.

### Measurement models

Each new model is available in each regime, because the state is
always Cartesian. Each value is in the frame of the filter.

- `Pseudorange`: the emitter position, the pseudorange, the sum of the
  Sagnac and media terms, and the variance. The model is
  `|p − p_e| + c·δt + Δρ`. The row is `uᵀ` on the position and `1` on
  the clock bias.
- `PseudorangeRate`: the emitter position and velocity, the range rate,
  and the variance. The model is `u·(v − v_e) + c·δṫ`.
- Without a clock model, the filter refuses each one-way model with
  `RejectionReason::ClockStateDisabled`, which has a counter. A one-way
  range without the clock states puts the clock error into the
  position.
- `TwoWayRange`: the transmitting and the receiving station positions,
  an optional body for the Shapiro delay, the round-trip range, the
  turnaround delay, and the variance. Each leg adds
  `2μ/c²·ln((r₁ + r₂ + r₁₂)/(r₁ + r₂ − r₁₂))`. The row includes the
  derivative of the delay. The round trip cancels the vehicle clock, so
  the model does not use the clock states. The caller gives each station
  at its own transmission or reception time, which applies the light
  time.
- `TwoWayRangeRate`: the station positions and velocities, the
  round-trip range rate, and the variance. The model is the sum of the
  uplink and the downlink range rates. It updates the position and the
  velocity.
- `LineOfSight`: a measured unit direction in the body axes, a
  `LineOfSightReference`, and the angular variance. The update uses an
  orthonormal basis `B` of the plane that is orthogonal to the predicted
  direction `û_b`, so the covariance `σ²·I₂` has full rank.
  - `InertialDirection` is a direction to a far source, such as a star.
    The row is `Bᵀ·[û_b×]` on the attitude.
  - `TargetPosition` is a target at a known position, such as Mars. The
    rows are `Bᵀ·[û_b×]` on the attitude and `−Bᵀ·Cᵀ/|d|` on the
    position, with `C` the rotation from the body axes to the frame axes
    and `|d|` the distance to the target.
  - `FusionConfig::line_of_sight_gate_chi2` is the gate with two
    degrees of freedom. Its default is 13.82, the 99.9% point. A
    measured direction 90° or more from the prediction is refused.

## Consequences

- One filter does orbit determination in a coast and inertial
  navigation in a burn, an entry, or a descent. A burn that no sensor
  measures is an impulsive burn in the force-model mode.
- The covariance matrix has 22 × 22 values, which is 3.87 KiB. The
  history keeps one covariance for each event, so the filter needs more
  memory than with seventeen states. The filter still does not allocate.
- A pure IMU mission or a two-way tracking mission does not configure
  the empirical acceleration or the clock. Those states then have no
  effect.
- `ObservationValue::Pseudorange` changes its fields to
  `emitter_position_m`, `pseudorange_m`, `sagnac_and_media_correction_m`,
  and `variance_m2`. The emitter position is in the frame of the filter.
  `NavigationFilter` still refuses it with `UnsupportedMeasurement`.
- `MeasurementKind`, `ObservationValue`, and `RejectionReason` get new
  variants. Each enum is `#[non_exhaustive]`. `RejectionCounters` gets
  two counters. `CONTRACT_VERSION` does not change, because the contract
  types do not change.
- The spherical harmonic field uses unnormalized solid harmonics. Their
  range in `f64` limits the degree. The inline limit of 36 stays far
  from that range.
- The caller supplies the time-dependent values: the third-body
  positions, the Sun position of the pressure model, the orientation of
  a harmonic field in an inertial frame, the station positions and
  velocities, and the direction to a star. The caller updates the
  environment with `update_space_environment`.

## Alternatives considered

- A lifetime parameter on `InertialNavigationFilter`, so that the filter
  can borrow coefficients. The history keeps a copy of the state before
  each event. With a borrowed field, each type of the filter gets the
  lifetime. A `'static` field in the environment and a borrowed field
  in `OrbitalForceModel<'a>` give both uses without that cost.
- A separate filter for orbit determination. Two filters need a
  transfer of the state and the covariance at each change between a
  burn and a coast. One filter with two modes keeps one history.
- A clock model with three states. Two states hold the bias and the
  drift of the clocks that the missions use. A drift rate is not
  observable in a short pass.

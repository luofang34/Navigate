# ADR-0017: One body-centered Cartesian inertial core and general frame changes

- Status: Proposed
- Date: 2026-11-03

## Context

ADR-0014 gives the inertial filter two regimes:

- In the body-surface regime, the state is in a tangent plane. The
  plane is anchored at the initialization position on a rotating body.
- In the Cartesian space regime, the state is in the frame of a
  `SpaceEnvironment`.

The two regimes cause these problems:

1. The plane axes are the local NED axes only at the anchor. Each
   geodetic model and the published solution must rotate between the
   plane axes and the local axes at the vehicle. The difference grows
   with the distance from the anchor.
2. A flight over a pole or a long flight needs a new anchor. The filter
   has no anchor change, so the flight must stop near the anchor.
3. A filter cannot go from the surface regime to the space regime. A
   launch vehicle, a lunar lander, or a vehicle on an entry trajectory
   needs one filter for the full flight.
4. `recenter_space_frame` is exact only for two frames with the same
   rotation rate. A change from the rotating frame of a body to an
   inertial frame needs the transport term `ω × r` in the velocity and
   in the covariance. The caller cannot put the covariance term into a
   velocity shift.
5. Many inertial measurement units give integrated increments with the
   coning and sculling corrections, not mean rates. The filter accepts
   only mean rates.

## Decision

### One Cartesian state

- The nominal state and the seventeen-state error covariance of
  `InertialNavigationFilter` are always in one body-centered Cartesian
  frame: the frame of a `SpaceEnvironment`. The position is in meters
  from the body center. The velocity is relative to the frame, in the
  frame axes. The attitude rotates the body axes into the frame axes.
- `SpaceEnvironment::from_body` gives the rotating body-fixed frame of
  a `PlanetaryBody`: the frame of its ellipsoid, the rotation
  `[0, 0, ω]`, and the point mass and `J2` gravity of the body. The
  environment adds the centrifugal term, so its gravity is the apparent
  gravity of the body.
- `new_surface` uses this environment and attaches the ellipsoid of
  the body. `new_space` attaches no ellipsoid.
  `attach_surface_ellipsoid` attaches an ellipsoid to the active frame
  or removes it. The ellipsoid must be on the active frame.
- The filter has no tangent plane and no anchor. The mechanization is
  the same in each frame.

### Local axes without a pole limit

- `ned_to_body_fixed_dcm` gives the rotation from the local NED axes to
  the body-fixed axes at a geodetic latitude and longitude. The north,
  east, and down vectors come directly from the sines and cosines. No
  term divides by `cos φ`, so the rotation is orthonormal at each
  latitude, the poles included.
- `initialize_surface` converts the geodetic position, the NED
  velocity, the attitude to the NED axes, and the NED covariances into
  the body-fixed frame with this rotation. The attitude error is in the
  body axes, so the filter rotates its prior into the body axes.
- `initialize_space` sets a Cartesian state in the active frame. Each
  regime accepts it.

### Geodetic models

- A geodetic model is available when the regime has an ellipsoid. Each
  model converts its value with the local axes at the nominal position
  or at the measured position:
  - A position fix converts exactly into the body-fixed frame, with
    `H = I`. Its covariance rotates from the local axes at the fix.
  - A velocity fix rotates from the local axes at the vehicle.
  - An altitude uses the outward normal of the ellipsoid as its row,
    which is the exact gradient of the ellipsoidal height.
  - A range, a terrain clearance, a magnetic anomaly, an attitude fix,
    and a visual pose use the same local axes.
- The Cartesian models are available in each regime, because the state
  is always Cartesian. The frame gate still applies.
- An undeclared frame is accepted only on the WGS84 surface of the
  Earth. Each other regime requires a declared frame.

### Publication

- `tick` publishes a `NavigationSolution` when the regime has an
  ellipsoid. The filter converts the body-fixed state at the vehicle:
  geodetic position, NED velocity, NED attitude, and NED covariances.
  The horizontal and vertical 1-sigma values come from the NED position
  covariance. `tick` replaces `tick_surface` of ADR-0014.
- `tick_space` publishes a `CartesianSolution` in each regime. Thus one
  filter gives `tick_space` in orbit and `tick` near a surface.

### General frame change

- `FrameChange` holds the new environment, an optional ellipsoid, the
  position and the velocity of the new center in the active frame, the
  rotation `C` from the active axes to the new axes, and the angular
  rate `ω_rel` of the new axes relative to the active axes.
- `FrameChange::between_environments` computes
  `ω_rel = C_new→old·ω_new − ω_old` for two frames that rotate relative
  to one inertial space.
- `change_frame` applies the exact change:
  `p' = C·(p − o)`, `v' = C·(v − ȯ − ω_rel × (p − o))`, and
  `q' = q_C ⊗ q`. The error Jacobian has `J_pp = C`, `J_vv = C`, and
  `J_vp = −C·[ω_rel×]`. The attitude and bias blocks are the identity.
  The covariance is `J·P·Jᵀ`.
- The change is a history event. A delayed observation before the
  change replays in the frame of its acquisition time.
- `recenter_space_frame` is a `FrameChange` with `ω_rel = 0` and no
  ellipsoid.
- `update_space_environment` and `attach_surface_ellipsoid` are history
  events at the state time. A reset keeps the active frame.

### IMU increments

- `ImuIncrement` holds the coning-corrected rotation increment `Δθ`
  and the sculling-corrected velocity increment `Δv` in the body axes,
  the interval, the stamp, the composition, and the declared frame.
- `ingest_imu_increment` runs the gates of `ingest_imu`. The step of a
  full interval uses the bias-corrected increments and the rotation
  correction `Δv_rot = Δṽ + ½·(Δθ̃ × Δṽ)`. The frame rotation over the
  step is exact: `q' = exp(−ω·Δt) ⊗ q ⊗ exp(Δθ̃)`. The step rotates
  `Δv_rot` into the frame axes with the frame rotation at the midpoint.
  The gravity and the Coriolis terms use midpoint RK2.
- A delayed observation inside an interval splits it. Each part uses
  the mean rates `Δv/Δt` and `Δθ/Δt`, as an `ImuSample` does.

### Covariance analysis

- `CrlbTracker` propagates its bound in the rotating body-fixed frame
  of the body. A `ReferencePoint` holds a geodetic position, a NED
  velocity, and a NED attitude. The tracker rotates the map gradient
  and reports the position bounds in the local NED axes at the most
  recent reference point.

## Consequences

- A flight over a pole needs no anchor change. The state and the
  covariance stay continuous. The published NED heading and longitude
  change by π at the pole, as the local axes do.
- One filter goes from a launch site through inertial frames to the
  surface of another body. Each change keeps the covariance exact.
- Close to a pole, a small position error turns the local axes. A NED
  velocity fix and an attitude fix then have little value. A position
  fix and the Cartesian models do not have this limit.
- The body-fixed position is about `10⁷` m on the Earth. A finite
  difference of the position must use a step that is large relative to
  the rounding of `f64` at this size.
- A caller of `tick_surface` must use `tick`. A caller of
  `ReferencePoint` must give a geodetic position and NED values. A
  surface filter accepts the Cartesian models and `initialize_space`.
  `CONTRACT_VERSION` does not change, because the contract types do not
  change.
- `SurfaceFrame` stays in `navigate-astrodynamics` as a gravity field
  of a tangent plane. The inertial filter does not use it.

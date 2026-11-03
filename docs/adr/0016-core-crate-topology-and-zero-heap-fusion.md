# ADR-0016: Core crate topology and zero-heap fusion

- Status: Proposed
- Date: 2026-11-03

## Context

ADR-0015 makes the contract, geodesy, route, guidance, and terrain
crates `#![no_std]` without heap allocation. The navigation filters
in `navigate-fusion` still need `std`. They use these heap types:

- `HashMap` for the source table and for the innovation statistics.
- `Vec` for the assessment window, the particles, and the grid map.
- `VecDeque` for the delayed-measurement histories.
- `Arc` for the space environment of the inertial filter.

Thus the filters cannot link into flight software on a bare-metal
target.

The crate names and the crate scopes also have these problems:

1. `navigate-fpl` and `navigate-egpws` use the names of Earth aircraft
   functions. The crates operate on any body.
2. `navigate-fusion` holds the planetary gravity models, which are not
   filter logic. A guidance or route crate that needs a body model must
   then depend on the filters.
3. `navigate-fusion` holds the Tolles–Lawson calibrator, the grid map,
   and the Cramér–Rao bound. These items are specific to magnetic
   anomaly navigation.
4. The coordinate time conversions of TCG, TCB, and TCL are only in
   `navigate-celestial`, which needs `std` and `hifitime`. A flight
   computer cannot convert its time scales.
5. The terrain assessment subtracts a terrain elevation from an
   ellipsoidal altitude. It does not state the vertical datum of the
   elevation. On the Earth, the geoid is up to approximately 100 m from
   the WGS84 ellipsoid, so an orthometric elevation causes an error of
   that size.
6. The workspace sets the `std` feature of `nalgebra` for all crates.
   Cargo feature unification then enables `std` in a `libm` build.

## Decision

### Crate topology

- `navigate-fpl` is `navigate-route`. `ExecutionConfig::surface_gravity_mps2`
  gives the surface gravity of the body of the route.
  `MARS_SURFACE_GRAVITY_MPS2` (`3.72076 m/s²`) and
  `TITAN_SURFACE_GRAVITY_MPS2` (`1.352 m/s²`) are named constants.
- `navigate-egpws` is `navigate-terrain-safety`. The assessment is
  `assess_clearance`. The terrain seam is `TerrainElevation`, and it
  gives ellipsoidal heights. `OrthometricTerrain` combines an
  `OrthometricElevation` source with a `navigate_geodesy::GeoidUndulation`
  model: `h = H + N`. When the model has no undulation at a position,
  the adapter gives no height, and the assessment refuses with
  `NoCoverage`.
- `navigate-astrodynamics` is a new crate. It holds these modules:
  - `body`: `PlanetaryBody` with `EARTH_WGS84`, `MOON_ME`, `MOON_PA`, and
    `MARS_IAU`, the `J2` apparent gravity, and the Somigliana normal
    gravity.
  - `gravity`: the `GravityField` trait, `PointMassBody`,
    `CentralBodyGravity`, `SurfaceFrame`, `SpaceGravityModel`, and
    `SpaceEnvironment`. `ThirdBodySet` holds up to `MAX_THIRD_BODIES`
    (8) third bodies inline. All these types are `Copy`.
  - `time_scales`: conversions of a `PhysicalInstant` between TAI, TT,
    GPST, TCG, TDB, TCB, TCL, and UT1. The secular rates `L_G`, `L_B`,
    and `ΔL_L` use exact integer arithmetic. TDB − TT uses the two
    largest Fairhead–Bretagnon terms. `EarthOrientation` gives
    UT1 − TAI and the polar motion.
  - `frame_rotation`: `FrameTransform` rotates a position and a velocity
    and adds the transport velocity `−ω × r`. `era_rad` gives the
    IAU 2000 Earth rotation angle, and `polar_motion_matrix` gives
    `W(x_p, y_p)`. `FrameTransform::axial_spin` gives the rotation of a
    body that turns at a uniform rate.
- `navigate-magnav` is a new crate. It holds `OnlineTlCalibrator`,
  `BilinearGridMap<'a>`, and `CrlbTracker`. The grid map borrows its
  values. The tracker uses the public covariance model of the inertial
  filter: `initial_error_covariance`, `error_transition`, and
  `joseph_covariance_update`.
- `navigate-fusion` re-exports `PlanetaryBody`, `SpaceEnvironment`,
  `SpaceGravityModel`, `CentralBodyGravity`, `PointMassBody`, and
  `ThirdBodySet`.

### Zero-heap fusion

- `navigate-fusion` is `#![no_std]` and does not allocate. Its modules
  are `kalman`, `kinematic`, `inertial`, `particle`, `admission`,
  `observation`, `integrity`, `provenance`, `rejection`, and `config`.
- The `ProvenanceLedger` source table holds `MAX_SOURCES` (32) sources.
  The first observation of one more source is refused with
  `RejectionReason::SourceCapacityExceeded`, because the filter cannot
  check the order of a source that it does not track.
- The assessment window holds `MAX_WINDOW_ADMISSIONS` (128) admissions.
  A full window drops its oldest admission. Thus the window can show
  less redundancy, but never more.
- The innovation statistics have 16 inline slots, one slot for each
  measurement kind.
- The kinematic history holds `MAX_KINEMATIC_HISTORY` (64) updates. The
  inertial history holds `MAX_INERTIAL_HISTORY` (32) events. A larger
  `FusionConfig::history_capacity` keeps the inline bound. The default
  inertial capacity is 32.
- A delayed update replays the history two times. The first pass checks
  that every update applies. The second pass writes the new states.
  Thus a refused delayed update does not change the history, and the
  history needs no second buffer.
- `MarginalizedParticleFilter<'a>` uses a `ParticlePool<'a>` of two
  caller-owned slices of `MpfParticle`. An update writes into the
  scratch slice, checks the weights, and then swaps the two slices in
  constant time. The pool sets the number of particles.

### Features

- The workspace dependency on `nalgebra` has no default features. Each
  `std` crate sets `features = ["std"]`.
- `navigate-astrodynamics`, `navigate-fusion`, and `navigate-magnav`
  have the features `std` (the default) and `libm`. Each feature
  forwards to `nalgebra`, `navigate-contract`, and `navigate-geodesy`.
  A build without one of the two features stops with a compile error.
- The float functions come from `navigate_geodesy::float_math`, so one
  feature list selects `std` or `libm` for every core crate.

## Consequences

- The eight core crates build for `thumbv8m.main-none-eabihf` and
  `riscv32imac-unknown-none-elf` with `--no-default-features --features
  libm`.
- A flight computer can run the kinematic, inertial, and particle
  filters with a static memory budget. The inline capacities set that
  budget at compile time.
- A delayed observation must arrive within 32 inertial events. At a
  high IMU rate, the caller must apply a delayed observation soon or
  accept an `OutsideHistory` refusal.
- A route, a guidance law, or a terrain assessment on another body
  uses the body model of that body, not the model of the Earth.
- Consumers of the renamed crates and types must change their
  dependencies and their imports. The renames are `navigate-fpl` to
  `navigate-route`, `navigate-egpws` to `navigate-terrain-safety`,
  `assess` to `assess_clearance`, and the `Egpws` prefix to the
  `Clearance` prefix. `CONTRACT_VERSION` does not change, because the
  contract types do not change.

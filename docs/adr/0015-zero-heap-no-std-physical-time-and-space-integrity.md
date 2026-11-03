# ADR-0015: Zero-heap `no_std` core, physical coordinate time, explicit body geodesy, and space integrity

- Status: Proposed
- Date: 2026-11-03

## Context

Flight computers for aircraft, reentry vehicles, satellites, and
interplanetary spacecraft run deterministic control loops without heap
allocation. A navigation library that requires `std` or `alloc` cannot
link directly into bare-metal flight software on embedded targets.

Across planetary and interplanetary regimes, four further limits occur
when terrestrial assumptions stay implicit:

1. Hardware monotonic nanoseconds (`MonotonicNanos`) measure onboard
   proper time (`SCLK`) only. Ephemerides, planetary rotation, and
   deep-space radiometric light-time equations require astronomical
   coordinate time scales (`TAI`, `TT`, `TDB`, `TCG`, `TCB`, `TCL`,
   `UT1`, `UTC`, and `GPST`) with sub-nanosecond precision. A 64-bit
   float seconds offset from `J2000.0` loses picosecond precision at a
   few decades, which corrupts X-band and Ka-band Doppler observables.
2. When functions default silently to `Ellipsoid::WGS84` or Earth
   standard gravity (`9.80665 m/s²`), a lunar or Martian caller that
   omits an ellipsoid argument computes Earth distances, Earth terrain
   clearance, or Earth turn anticipation radii without a compiler error.
3. Radar and optical sensors on irregular small bodies intersect rays
   with a 3D surface shape where ellipsoidal normal coordinates do not
   represent craters or concavities. Surface elevations also refer to an
   equipotential datum (geoid, selenoid, or areoid) that differs from
   the reference ellipsoid.
4. A space vehicle in orbit has no local horizontal or vertical plane.
   Reporting its 3D uncertainty as identical `horizontal_1sigma_m` and
   `vertical_1sigma_m` fields hides the physical ratio between radial,
   along-track, and cross-track orbit errors. In addition, an innovation
   gate that rejects a run of measurements leaves the covariance small
   even when the filter state and the sensors disagree.

## Decision

### Zero-heap `no_std` foundation

- `navigate-contract`, `navigate-geodesy`, `navigate-fpl`,
  `navigate-guidance`, and `navigate-egpws` compile with `#![no_std]`
  and zero heap allocation (`no alloc`).
- Crates that evaluate transcendental functions expose `std` (default)
  and `libm` Cargo features.
- `navigate_contract::Ident` stores a non-empty UTF-8 identifier of up
  to 32 bytes (`MAX_IDENT_BYTES`) inline and implements `Copy`.
- `Waypoint` and `PlanValidationError` use `Ident` and implement `Copy`.
- `FlightPlan<'a>` borrows a caller-owned slice `&'a [Waypoint]` and
  allocates no heap memory.
- `navigate_celestial::stars::stars_ecef_tt_into` writes star sights
  into a caller-supplied `&mut [StarSight]` buffer without allocation.
- `navigate_contract::FrameId` is the single frame identifier across
  `navigate-contract` and `navigate-celestial`.

### Dual-layer time model

- Layer 1 keeps `(ClockDomainId, MonotonicNanos)` for onboard hardware
  ordering, staleness checks, and deterministic sensor synchronization.
- Layer 2 adds `PhysicalInstant`, `SignedDuration`, `TimeScale`,
  `ClockAnchor`, and `RadiometricStamps` in `navigate-contract`.
- `SignedDuration` stores an `i64` count of seconds and a canonical
  sub-second `i64` count of attoseconds in `[0, 10^18)`. One attosecond
  is `10^-18 s`, which gives `0.3 nm` light-time resolution across all
  astronomical epochs.
- `PhysicalInstant` pairs a `TimeScale` (`Tai`, `Tt`, `Tdb`, `Tcg`,
  `Tcb`, `Tcl`, `Ut1`, `Utc`, or `Gpst`) with a `SignedDuration` offset
  from `J2000.0` in that scale. `PhysicalInstant::elapsed_since` refuses
  to subtract instants from different time scales.
- `ClockAnchor` maps hardware readings from one `ClockDomainId` to a
  `PhysicalInstant` with a fractional rate offset and an uncertainty
  bound.
- `ObservationStamp` and `SolutionStamp` carry `physical_at:
  Option<PhysicalInstant>`. `RadiometricStamps` carries transmit,
  turnaround, and receive instants for one-way, two-way, and three-way
  radiometric tracking.
- `navigate_celestial::time_bridge` converts between `PhysicalInstant`
  and `hifitime::Epoch`, including the IAU 2024 Lunar Coordinate Time
  (`Tcl`) secular drift rate relative to `Tt`.

### Explicit body geodesy, vertical datum, and surface shape

- `NavigationSolution::new`, `FlightPlan::new`, and
  `AidingValue::PositionFix` require an explicit `Ellipsoid` argument.
- `TerrainDatabase::ellipsoid` has no default implementation. Every
  terrain database implementation declares its `Ellipsoid` explicitly.
- Primary functions at the root of `navigate-geodesy`
  (`geodetic_to_body_fixed`, `body_fixed_to_geodetic`,
  `LocalTangentPlane::new`, `distance_m`, `cross_track_m`,
  `along_track_m`, and `cross_track_from_course_m`) require an explicit
  `BodyEllipsoid`. Earth WGS84 convenience functions live in
  `navigate_geodesy::wgs84`.
- `navigate_fpl::ExecutionConfig` carries `surface_gravity_mps2` so
  fly-by turn anticipation uses the gravity of the active body.
- `GuidanceConfig::expected_ellipsoid` checks that a `NavigationSolution`
  matches the active plan ellipsoid and refuses a mismatch with
  `GuidanceRefusal::EllipsoidMismatch`.
- `navigate_geodesy::GeoidUndulation` defines the height offset between
  a reference ellipsoid and an equipotential vertical datum.
- `navigate_geodesy::SurfaceShape` defines body-fixed ray intersection
  (`SurfaceHit`) for ellipsoids and irregular small-body meshes.

### Space integrity and innovation-gate storm degradation

- `CartesianSolution` carries `SpaceIntegrityAssessment` instead of the
  surface `IntegrityAssessment`.
- `SpaceUncertainty::OrbitalRtn` reports 1-sigma position uncertainties
  along the radial (`R`), along-track (`T`), and cross-track (`N`)
  orbital axes when `|r| >= 1.0 m` and `r × v` defines an orbit plane.
- `SpaceUncertainty::PrincipalAxes` reports the sorted 1-sigma principal
  axes when the vehicle hovers or moves along a radial line.
- `ProvenanceLedger` counts consecutive innovation-gate rejections since
  the last admitted aiding observation. When that count reaches
  `FusionConfig::max_consecutive_gate_rejections`, integrity derivation
  caps `SolutionQuality` at `Degraded` until an aiding observation
  passes admission.

## Consequences

- Core contract, geodesy, flight-plan, surface guidance, and terrain
  safety crates build for bare-metal `thumbv8m.main-none-eabihf` and
  `riscv32imac-unknown-none-elf` targets without a heap allocator.
- Callers cannot accidentally apply WGS84 or Earth gravity to Moon,
  Mars, or small-body trajectories through omitted parameters.
- Deep-space radiometric observables preserve exact attosecond epochs
  and explicit relativistic time scales across crate boundaries.
- Orbital consumers inspect radial, along-track, and cross-track
  uncertainties directly, and sustained innovation-gate disagreement
  surfaces immediately in `SolutionQuality`.

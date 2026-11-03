# ADR-0013: Body frames and reference ellipsoids are explicit

- Status: Proposed
- Date: 2026-11-02

## Context

A geodetic position has a meaning only on one reference ellipsoid. The
same latitude, longitude, and height name different points on the
Earth, the Moon, and Mars. They also name different points on two
realizations of one body, such as the lunar mean Earth frame and the
lunar principal axis frame.

When the ellipsoid is implicit, these errors are silent:

1. A lunar position enters a filter that converts it on WGS84. The
   metric scale is wrong by a factor of approximately 3.7.
2. A fix in the lunar principal axis frame enters a filter in the lunar
   mean Earth frame. The two frames differ by several hundred meters
   on the lunar surface.
3. A terrain database on one ellipsoid gives the elevation below a
   solution on a different ellipsoid.
4. A visual search prior on an Earth map frame receives a solution on a
   different body.

A navigation system for many bodies must refuse each of these
conditions. It must not convert the values.

The NAIF SPICE system gives an integer code to each body and to each
frame orientation. Flight dynamics tools use these codes. The codes are
a stable and exact name for a frame.

## Decision

### Frame identity

- `navigate_contract::FrameId` names one frame with two NAIF codes:
  `center_naif_id` for the body at the origin and
  `orientation_naif_id` for the orientation.
- Two frames are equal only when both codes are equal. Thus
  `FrameId::MOON_ME` (301, 31001) and `FrameId::MOON_PA` (301, 31000)
  are different frames.
- `FrameId::is_inertial` is true for the J2000 orientation (code 1).
- The presets are `SSB_J2000`, `SUN_J2000`, `EARTH_J2000`,
  `EARTH_ITRF93`, `MOON_J2000`, `MOON_PA`, `MOON_ME`, `MARS_J2000`, and
  `MARS_IAU`. The module `navigate_contract::frame::naif` holds the
  codes.

### Reference ellipsoid

- `navigate_contract::Ellipsoid` holds a body-fixed `frame`, the
  `semi_major_axis_m`, and the `flattening`. A flattening of 0 gives a
  sphere.
- The ellipsoid defines the planetographic latitude along the
  ellipsoidal normal, the east-positive longitude, and the height above
  the ellipsoid.
- The presets are `WGS84`, `GRS80`, `MOON_ME`, `MOON_PA`, and
  `MARS_IAU`.
- `Ellipsoid::is_plausible` requires a finite and positive semi-major
  axis, a flattening in `[0, 1)`, and a frame that is not inertial.

### Published solutions

- `NavigationSolution::ellipsoid` names the ellipsoid of the published
  position. `NavigationSolution::new` sets `Ellipsoid::WGS84`.
  `NavigationSolution::with_ellipsoid` sets a different ellipsoid.
- `NavigationSolution::attitude_cov` carries an attitude covariance
  when it is known.
- `CartesianSolution` is a solution in a body-centered Cartesian frame,
  for example an inertial frame during a transfer orbit. It names its
  `frame`.
- `SensorClass::SurfaceAltimeter` is the composition class of a radar
  or laser altimeter that measures the range to the terrain surface.
  Its bit is `1 << 10`.

### Geodesy on any body

- `navigate_geodesy::BodyEllipsoid` holds a plausible contract
  ellipsoid and its derived constants. `BodyEllipsoid::try_from_contract`
  refuses an implausible ellipsoid with
  `GeodesyError::ImplausibleEllipsoid`.
- `geodetic_to_body_fixed` and `body_fixed_to_geodetic` convert on any
  `BodyEllipsoid`. The inverse uses two Bowring passes. No step divides
  by the eccentricity, so a sphere gives an exact result. A position
  closer to the polar axis than `1e-12·a` takes a direct branch.
- `LocalTangentPlane::with_ellipsoid` anchors a plane on an ellipsoid.
  The plane keeps the ellipsoid, and every conversion of the plane uses
  it. `LocalTangentPlane::new` uses WGS84.
- The great-circle functions with the `_on_m` form use the mean-radius
  sphere of a `BodyEllipsoid`. The functions without that form use
  WGS84.
- `BodyEllipsoid` converts between planetocentric and planetographic
  latitude. `west_to_east_lon_rad` converts a west-positive longitude,
  such as an IAU Mars longitude, to the east-positive form.

### Frame admission in the filter

- `Observation::frame` declares the frame of the observation value.
  `Observation::new` declares no frame. `Observation::in_frame` and
  `Observation::with_frame` declare a frame.
- `NavigationFilter::new` uses `BodyEllipsoid::WGS84`.
  `NavigationFilter::new_for_ellipsoid` uses a different ellipsoid and
  sets `FusionConfig::require_explicit_frame` to `true`.
- The frame gate runs after the clock-domain gate. A declared frame
  that is not the frame of the filter ellipsoid gives
  `RejectionReason::FrameMismatch`. A missing frame gives
  `RejectionReason::MissingFrame` when the configuration requires an
  explicit frame. Each refusal increments one counter and changes
  nothing else.
- Each tangent plane of the filter uses the filter ellipsoid: the
  initial plane, the re-anchored plane, the velocity rotation planes,
  the altitude Jacobian plane, and the magnetic linearization plane.
- Each published solution carries the filter ellipsoid.

### Consumers of a solution

- `navigate-guidance` admits the solution ellipsoid with
  `BodyEllipsoid::try_from_contract`. An implausible ellipsoid gives
  `GuidanceRefusal::ImplausibleEllipsoid`. The check runs after the age
  check. The cross-track and the remaining distance use the mean-radius
  sphere of the solution ellipsoid.
- `TerrainDatabase::ellipsoid` names the ellipsoid of a terrain
  database. The default is `Ellipsoid::WGS84`. `navigate_egpws::assess`
  refuses a solution on a different ellipsoid with
  `EgpwsUnavailable::EllipsoidMismatch`. The check runs immediately
  after the database check.
- `navigate_visual::LocalFrame` is an Earth frame on WGS84.
  `navigate_visual_fusion::search_prior` refuses a solution on a
  different ellipsoid with `VisualFusionError::FrameDatumMismatch`.
  `VisualFixSource::convert` declares `FrameId::EARTH_ITRF93` on each
  observation.

## Consequences

- A mix of bodies or of realizations is refused and counted. It cannot
  move the state.
- An Earth consumer that uses the default constructors keeps WGS84 and
  accepts observations without a declared frame.
- A filter for a different body must receive a declared frame on every
  observation. A source adapter for that body must declare the frame.
- The filter compares frame identity only. It does not transform
  between frames. A transformation between `MOON_PA` and `MOON_ME`, or
  between body-fixed and inertial frames, is the responsibility of the
  source adapter.
- The great-circle guidance geometry stays spherical. On Mars the
  flattening is larger than on the Earth, so the spherical error budget
  is approximately 1.8 times larger.

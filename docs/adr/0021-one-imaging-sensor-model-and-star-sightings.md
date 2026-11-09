# ADR-0021: One imaging sensor model, and star sightings enter fusion as lines of sight

- Status: Proposed
- Date: 2026-11-09

## Context

Navigate reads images from three kinds of sensor:

- a visible camera for terrain-relative navigation (TRN),
- a long-wave infrared (LWIR) camera for TRN at night,
- a star camera for attitude and astro-inertial navigation.

The visual stack has four camera descriptions with different conventions:

- `navigate_visual::CameraModel` and `navigate_visual::CameraPose`. The pose
  rotates the eye axes into the world. The eye axes are right, up, and back.
- `navigate_visual_flight::CameraIntrinsics`, with a calibration mode.
- `navigate_match::VirtualViewCamera`. It holds a row-major rotation from the
  local east, north, up (ENU) axes into the camera axes. The camera axes are
  right, down, and forward.
- `navigate_locate::corridor::KinematicState`, with the same axes as
  `VirtualViewCamera`.

A host that renders frames, such as a simulator, must know each
convention. A star camera and an LWIR camera need the same intrinsics and
the same mount as a visible camera. A copy of each type for each sensor
adds more conventions.

Navigate has no star camera pipeline. `navigate-fusion` has
`ObservationValue::LineOfSight` and `ObservationValue::AttitudeFix`, but no
crate makes them from an image. No crate uses `navigate-celestial`.

`AttitudeFix` on a body surface rotates the body axes into the local north,
east, down (NED) axes at the vehicle. A star camera measures the attitude
relative to the stars. To make an `AttitudeFix`, an adapter must know the
vehicle position. That position comes from the filter. Then the fix and
the filter share error, and the update is not independent.

## Decision

### One camera model crate

`navigate-camera` is a new leaf crate. It is `no_std` and does not
allocate. It depends on `nalgebra`, `navigate-contract`, `navigate-geodesy`
(for the float functions), and `thiserror`. It holds
the one description of an imaging sensor:

- `Intrinsics`: a pinhole camera. Pixel centres have integer coordinates.
  Rows increase down the image. The principal point can be off centre.
- `Distortion`: no distortion, or the Brown–Conrady model with three radial
  and two tangential terms. It maps normalized image coordinates both ways.
- `OpticalPose`: the camera optical centre and the rotation from the
  world axes into the camera axes. The camera axes are `+X` right, `+Y`
  down, and `+Z` forward (the OpenCV axes). The caller declares the world
  frame.
- `CameraMount`: the lever arm in the body axes and the rotation from the
  body axes into the camera axes. The body axes are forward, right, and down
  (FRD), as the `navigate-fusion` attitude models use.
- `SpectralBand`: visible, near infrared, or long-wave infrared.
- `Shutter`: a global shutter, or a rolling shutter with a line period.
- `ImagerModel`: intrinsics, distortion, band, and shutter.
- `ImageView` and `Pixels`: borrowed row-strided pixels. The formats are
  8-bit gray, 16-bit gray with a count of significant bits, and 8-bit RGB.
  16-bit gray holds raw LWIR counts and raw star camera counts.

The other camera types stay. Each type gets lossless conversions to and
from the `navigate-camera` types. New code uses the `navigate-camera`
types in public signatures. A later change can remove an old type when it
has no user.

### LWIR frames

An LWIR frame enters as `Pixels::Gray16` with `SpectralBand::LongWaveInfrared`.
`ImageView::to_gray8_percentile` maps raw counts to 8-bit gray with a
percentile stretch. It is the automatic gain control (AGC) for matchers
that accept 8-bit gray only. The stretch is a function of the image only,
so a replay gives the same pixels.

`navigate_visual_flight::LiveFrameInput` has a `band` field. `Segformer`
learned visible colours only. Thus a frame outside the visible band does
not use the semantic sky mask. The static mask and the propeller mask
stay.

### Star sightings

`navigate-startrack` is a new crate. It makes fusion observations from a
star camera image:

1. Detection finds bright pixel groups above the local background and
   gives intensity-weighted centroids. A body rotation during the
   exposure makes each star a streak. Detection keeps a streak up to a
   maximum length. The centroid of a streak is the star direction at the
   middle of the exposure. Thus the observation stamp is the mid-exposure
   instant. Detection drops a group that touches the image edge, because
   its centroid moves toward the image centre.
2. Identification has two modes:
   - With an attitude prior, it projects the catalogue of
     `navigate-celestial` through the prior. It matches each centroid to
     the nearest predicted star inside a gate. A match must be unique in
     both directions. A check of the angles between matched stars removes
     wrong matches.
   - Without an attitude prior (lost-in-space), it uses a `PairCatalogue`:
     the catalogue star pairs closer than the diagonal field of view,
     sorted by their angle. The host makes this table once. For each frame,
     a pyramid search takes a triangle of bright centroids and finds the
     catalogue triangles with the same side angles by binary search. A
     catalogue triangle must have the same handedness, so a mirror image
     of the sky does not match. A fourth centroid must confirm one unique
     catalogue triangle. The attitude of this pyramid then acts as a prior
     for the first mode. When no triangle gives a unique confirmed pyramid,
     identification returns an error.
   - `StarTracker` chooses the mode of each frame. It uses the prior of
     the host only while it holds a lock and the prior is narrow. It uses
     lost-in-space at the start, after a configured number of frames in a
     row without a solution, and for a wide prior. A prior that drifted
     during a gap therefore cannot keep the tracker from recovering.
3. The attitude solution uses the Davenport q-method on the matched pairs.
4. Each matched star becomes one `ObservationValue::LineOfSight` with
   `LineOfSightReference::InertialDirection`. The direction is the apparent
   star direction in Earth-centred, Earth-fixed (ECEF) axes at the
   acquisition time.

`LineOfSight` needs no vehicle position, so the star camera stays an
independent source. The inertial filter gets the position information
through the coupling of the attitude error and the gravity direction
(astro-inertial navigation).

An `AttitudeFix` in local NED axes needs a position. The crate gives it
only through a function that takes the position as an argument. The
function documentation states the shared error.

The pyramid search is bounded. It uses a fixed number of the brightest
centroids, and it does not use a triangle with too many catalogue
candidates. The pair table holds at most `MAX_PAIRS` pairs.

### Resource cost

Navigate stays sans-I/O (ADR-0002). The star and camera crates do not read
a clock. The host measures time and memory around each call.

## Consequences

- A host converts its camera once into `navigate-camera` types.
- Visible, LWIR, and star cameras share one mount and one projection.
- `navigate-celestial` gets its first user in the workspace.
- A star camera gives observations without an attitude prior. The host
  keeps one `PairCatalogue` for its camera and magnitude limit.
- Lens distortion has a model. The visual matchers still need undistorted
  images. The host can use `Distortion` to remove distortion.

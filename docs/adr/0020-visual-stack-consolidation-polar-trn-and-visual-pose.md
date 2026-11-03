# ADR-0020: Visual stack consolidation, polar terrain-relative navigation, and the visual pose

- Status: Proposed
- Date: 2026-11-03

## Context

Terrain-relative navigation (TRN) finds the vehicle position from images
of the terrain below it. A lunar lander, a Mars descent stage, and a
polar aircraft use TRN near a pole.

The map tile frames of `navigate_visual::LocalFrame` are Earth frames on
WGS84. Web Mercator tiles address them, so they refuse latitudes past
85 degrees. A frame with that limit cannot hold a TRN camera at the lunar
south pole (89.9 degrees south). A lunar navigation solution cannot
center a visual search in an Earth frame.

ADR-0014 gives `ObservationValue::VisualPose` a measurement model in
`InertialNavigationFilter`. A position fix (ADR-0007) does not carry the
camera attitude of a visual estimate to the filter.

Four parts of the visual stack have one consumer each, or a strong
coupling to one other part:

- The camera session and the fusion adapter use `navigate-visual` types
  in all public signatures.
- The semantic segmentation uses the same `burn` runtimes and the same
  browser WebGPU device as the matchers of `navigate-match`.
- The place index and the search plan serve only the `navigate-locate`
  search pipeline.

A separate crate for each of these four parts adds manifests, feature
mappings, and version pins without an independent consumer.

## Decision

### Polar-safe body tangent frame

- `LocalFrame::BodyTangentEnu { ellipsoid: BodyEllipsoid, origin:
  GeodeticPosition }` is a new frame variant. Its axes are east, north,
  and up at `origin` on `ellipsoid`.
- A conversion goes through the body-fixed frame of the ellipsoid
  (`geodetic_to_body_fixed` and `body_fixed_to_geodetic`) and one
  rotation. The frame has no latitude limit. It is valid up to the
  poles on `BodyEllipsoid::WGS84`, `BodyEllipsoid::MOON_PA`,
  `BodyEllipsoid::MARS_IAU`, and every other plausible ellipsoid.
- At a pole, the origin longitude selects the east axis.
- `LocalFrame::local` refuses a conversion that does not go back to its
  input within 1 mm.
- `LocalFrame::reference_ellipsoid` gives the ellipsoid of the geodetic
  positions of a frame. The map tile frames give `BodyEllipsoid::WGS84`.
- The map tile frames (`Wgs84Enu`, `SphericalEnu`, and `AnchorMercator`)
  keep the 85-degree limit, because Web Mercator tiles address them.
- `navigate-visual-render` reads Web Mercator Earth tiles. Its renderers
  refuse a `BodyTangentEnu` frame. A host supplies a reference view in
  a `BodyTangentEnu` frame through the `ReferenceRenderer` port
  (ADR-0010).

### Visual pose observation

- `navigate_visual::fusion_bridge::visual_pose_observation` converts one
  `Estimate` into one `ObservationValue::VisualPose`. The position is the
  camera optical centre. The attitude is the body attitude in the local
  north, east, down axes at the position.
- `VisualAttitudeBudget` gives the camera mounting (`eye_to_body`) and
  the angular error terms of the mounting and of the map. Every term must
  be positive.
- The image geometry gives one joint covariance of position and
  attitude. A visual pose has one block for each. Two times each
  diagonal block bounds the joint covariance, so the builder doubles both
  blocks before it adds the budgets.
- `position_fix_observation` gives the position fix of ADR-0007 from the
  same estimate. `VisualFixSource::convert` and
  `VisualFixSource::convert_pose` apply one duplicate screen and one
  frame-order screen to both conversions. One frame gives one
  observation.
- Each observation declares the body-fixed frame of
  `LocalFrame::reference_ellipsoid`. A lunar `InertialNavigationFilter`
  admits a visual pose from a lunar frame. An Earth filter refuses it
  with `RejectionReason::FrameMismatch`.
- `search_prior` accepts a solution on the reference ellipsoid of the
  visual frame. A lunar solution centers a search in a lunar frame.

### Crate consolidation

- `navigate-visual` contains the camera session (`session`) and the
  fusion conversion (`fusion_bridge`). The default `fusion` feature
  enables `fusion_bridge` and the `navigate-fusion` dependency.
- `navigate-match` contains the semantic segmentation (`segment`). The
  `flex`, `wgpu`, `webgpu`, `metal`, `vulkan`, and `cuda` runtime
  features apply to the matchers and to the segmentation. The `optim`
  feature enables fine-tuning.
- `navigate-locate` contains the place index and the search plan
  (`place`).
- `navigate-match-kernels` stays a separate crate. The CubeCL `#[cube]`
  macro expands to `allow(clippy::all)`, and only that crate relaxes the
  lints for it.
- The model preparation tool has the package name
  `navigate-visual-model`, the same prefix as `navigate-visual-pipeline`.
  The end-to-end corridor tests are in `tests/e2e-corridor`, with the
  kebab-case form of all other workspace directories.
- The root `[workspace.dependencies]` table names every library crate of
  the workspace. Members use `workspace = true`.
- `CONTRACT_VERSION` does not change. The decision adds no contract
  type.

## Consequences

- A lunar or Martian TRN camera gives poses in an exact frame at all
  latitudes. The test `polar_visual_pose` solves a camera pose at
  89.9 degrees south on `MOON_PA` from pixel matches, converts it into a
  visual pose, and applies it to a lunar `InertialNavigationFilter`.
- The filter gets six degrees of freedom from each visual estimate. The
  attitude update needs a correct camera mounting. An incorrect
  `eye_to_body` rotation gives an incorrect body attitude. The mounting
  budget does not detect it.
- A host that renders lunar or Martian reference views supplies its own
  renderer, because `navigate-visual-render` reads Web Mercator Earth
  tiles only.
- The visual stack has six crates: `navigate-visual`,
  `navigate-visual-render`, `navigate-match`, `navigate-match-kernels`,
  `navigate-locate`, and `navigate-locate-web`. Consumers use the module
  paths `navigate_visual::session`, `navigate_visual::fusion_bridge`,
  `navigate_match::segment`, and `navigate_locate::place`.
- Scripts and workflows name the packages `navigate-match` and
  `navigate-visual-model` and the directory `tests/e2e-corridor`.

## Alternatives considered

- Extend the Web Mercator frames past 85 degrees. Web Mercator
  coordinates are not finite at a pole, and they have no meaning on a
  body that has no Web Mercator tiles.
- Keep a separate `navigate-visual-fusion` crate. Its only types wrap
  `navigate-visual` and `navigate-fusion` types, and the visual pose
  needs the camera frame conventions of `navigate-visual`.
- Send the camera attitude as a separate `AttitudeFix`. Two
  observations from one image share all their errors. One visual pose
  keeps the position and the attitude in one update of dimension six.

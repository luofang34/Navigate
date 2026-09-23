# ADR-0007: Visual positions enter fusion through one adapter with a declared error budget

- Status: Proposed
- Date: 2026-09-22

## Context

`navigate-visual` finds the pose of a camera. It matches one frame to a
reference view. A reference view is an image and a depth surface that a
renderer makes from a map release. The result is an `Estimate`.

Before this decision, no crate converted an `Estimate` into a fusion
`Observation`. Four problems stopped a correct conversion:

1. The pose had no declared frame. The tools named the frame "east,
   north, up". The renderer really uses a flat Web Mercator plane that
   has its scale at the map anchor. This plane has no Earth curvature.
   The error of the plane is approximately 196 m at 50 km from the
   anchor.
2. The frame stamp had no source, no epoch, and no clock domain. The
   filter refuses an observation without these three items.
3. The pose covariance comes from image geometry only. It does not
   include map registration error, elevation error, camera calibration
   error, or vertical datum error. If the filter uses this covariance
   directly, the filter overstates its confidence.
4. A frame can be evaluated again with a different candidate. A second
   evaluation of one frame is not a second measurement.

Two more facts apply. The filter has no attitude state. Pilotage pins
`navigate-contract` and `navigate-fusion` at a fixed revision, so a
change to these crates is a contract change (ADR-0005).

## Decision

### The reference declares its frame

`navigate-visual` defines `LocalFrame`. `ReferenceView` and `Estimate`
carry a `LocalFrame`. The only frame now is `LocalFrame::AnchorMercator`.
`LocalFrame` owns the exact conversion to and from latitude, longitude,
and altitude. It also gives `model_error_m`, which is the curvature drop
plus the scale drift at a position. The tools use `LocalFrame` and do
not keep their own copies of the conversion.

### One adapter crate converts an estimate

The new crate `navigate-visual-fusion` contains `VisualFixSource`. It
converts one `Estimate` into one `ObservationValue::PositionFix`. It
does not change `navigate-contract` or `navigate-fusion`.

- The host supplies `VisualSourceIdentity`: the source, the epoch, and
  the clock domain of the capture stream.
- The host supplies `VisualErrorBudget`. Each term must be finite and
  more than zero. Unknown error is not zero error. The adapter refuses
  a budget with a zero term.
- The adapter turns the frame axes into north, east, down. It adds the
  budget and the frame model error to the covariance.
- The adapter converts altitude to height above the WGS84 ellipsoid.
  The host declares the vertical datum of the map.
- The composition is `SensorClass::VisualLandmark`.
- The adapter refuses frame evidence that it converted before, and a
  frame that does not follow the last frame in capture time.
- Each fix names the `MapRevision` and the frame evidence digest. The
  host records these with the fix.

Attitude is not transferred. The fix is the position of the camera
optical centre. The host removes the camera lever arm, or includes it in
the calibration term.

### Visual fixes go to Navigate fusion only

A visual fix is not flight-controller aiding. The aiding schema stays
draft until the joint RFC (ADR-0005).

## Consequences

- A host can feed visual fixes to `NavigationFilter` today. A test
  shows that the filter admits a fix from the adapter.
- The budget is the host's claim. The adapter does not measure map
  error. The filter cannot detect a budget that is too small.
- Fixes from one map release share map error. The filter treats each
  fix as independent. The budget must therefore be conservative. A
  correlated measurement model is future work. It needs a new
  `ObservationValue` variant, and ADR-0003 records such a change as a
  new measurement model.
- The flat frame limits the useful range. Far from the anchor, the
  model error term makes the fix weak. A globe-mode reference with a
  true local frame is a new `LocalFrame` variant. It does not change the
  adapter interface.
- Pilotage owns camera calibration and map release identity. The host
  fills `MapRevision` from its package identity, and it fills the
  calibration term from its calibration records.

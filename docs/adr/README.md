# Architecture Decision Records

One file per decision, numbered in acceptance order. See
[ADR-0001](0001-record-architecture-decisions.md) for the format and
lifecycle rules.

## Index

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions as versioned files in this repository | Accepted |
| [0002](0002-sans-io-deterministic-workspace.md) | One Cargo workspace with sans-IO deterministic core crates | Accepted |
| [0003](0003-typed-observations-modular-filter.md) | The navigation filter fuses typed, admission-gated observations with pluggable propagation | Accepted |
| [0004](0004-integrity-honesty.md) | Every solution carries an integrity assessment that never overstates redundancy | Accepted |
| [0005](0005-contract-crate-boundary.md) | External consumers pin a pure-types contract crate; aiding stays draft until a joint RFC | Accepted |
| [0006](0006-one-leg-vocabulary-for-path-terminators-and-vertical-legs.md) | One leg vocabulary carries path terminators and vertical procedure legs | Accepted |
| [0007](0007-visual-positions-enter-fusion-through-one-adapter.md) | Visual positions enter fusion through one adapter with a declared error budget | Proposed |
| [0008](0008-navigation-means-join-through-measurement-models.md) | Navigation means join through measurement models and one propagator seam | Proposed |
| [0009](0009-the-navigation-estimate-narrows-the-visual-search.md) | The navigation estimate narrows the visual search and never admits a fix | Proposed |
| [0010](0010-reference-views-come-from-a-navigate-owned-renderer-port.md) | Reference views come from a renderer port that Navigate owns | Proposed |
| [0011](0011-matchers-and-the-locate-pipeline-have-an-asynchronous-form.md) | Matchers and the locate pipeline have an asynchronous form | Proposed |
| [0012](0012-scalar-map-models-and-bounded-state-history.md) | Scalar map models and a bounded state history extend the filter core | Proposed |
| [0013](0013-body-frames-and-ellipsoids-are-explicit.md) | Body frames and reference ellipsoids are explicit | Proposed |
| [0014](0014-strapdown-inertial-propagation-and-marginalized-particle-filter.md) | Strapdown inertial propagation and a marginalized particle filter extend fusion to attitude, biases, space frames, and multimodal maps | Proposed |
| [0015](0015-zero-heap-no-std-physical-time-and-space-integrity.md) | Zero-heap `no_std` core, physical coordinate time, explicit body geodesy, and space integrity | Proposed |
| [0016](0016-core-crate-topology-and-zero-heap-fusion.md) | Core crate topology: route, terrain safety, astrodynamics, and magnetic navigation crates, and zero-heap `no_std` fusion | Proposed |
| [0017](0017-unified-cartesian-inertial-core-and-frame-transitions.md) | One body-centered Cartesian inertial core with pole-free local axes, general frame changes, and IMU increments | Proposed |
| [0018](0018-reduced-dynamic-orbit-determination-clock-and-radiometric-opnav.md) | Reduced-dynamic orbit determination with a force-model coast, receiver clock states, and radiometric and optical navigation models in the inertial filter | Proposed |
| [0019](0019-space-trajectory-and-multiphase-guidance.md) | Space trajectory laws, space guidance setpoints, and mission-phase orchestration from orbit to a runway or to touchdown | Proposed |
| [0020](0020-visual-stack-consolidation-polar-trn-and-visual-pose.md) | Six crates hold the visual stack; an exact body tangent frame serves polar terrain-relative navigation; a camera pose enters the inertial filter as one visual pose | Proposed |
| [0021](0021-one-imaging-sensor-model-and-star-sightings.md) | One imaging sensor model, and star sightings enter fusion as lines of sight | Proposed |
| [0022](0022-navigation-data-contract-mission-packages-and-the-onboard-boundary.md) | One navigation data contract, content-addressed mission packages, and one onboard boundary | Proposed |

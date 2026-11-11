# Appendix B: Crate Topology and ADR Traceability Matrix

This appendix maps every crate in the `Navigate` Cargo workspace to its runtime constraints, responsibilities, book chapters, and authoritative Architecture Decision Records (ADRs).

## 1. Workspace crate topology

| Crate | `#![no_std]` & Zero-Heap | Primary responsibility | Book chapters |
|---|---|---|---|
| `navigate-contract` | Yes | Pure boundary types: solutions, integrity, stamps, compositions, plans, guidance setpoints, physical time | Ch 1, 2, 3, 17, 18 |
| `navigate-camera` | Yes | Unified imaging sensor model: pinhole intrinsics, Brown–Conrady distortion, FRD mount, visible/LWIR/star pixel views | Ch 2, 11, 12 |
| `navigate-geodesy` | Yes | Multi-body reference ellipsoids, Bowring geodetic $\leftrightarrow$ body-fixed conversions, local tangent planes, great-circle geometry, vertical datums | Ch 2, 15, 17 |
| `navigate-astrodynamics` | Yes | Planetary bodies, degree-36 solid spherical harmonics, 3rd-body tides, cannonball SRP, exponential drag, coordinate time scales, frame rotations | Ch 2, 5, 16 |
| `navigate-fusion` | Yes | 6-state `NavigationFilter`, 22-state `InertialNavigationFilter`, `MarginalizedParticleFilter`, 9-stage admission funnel, OOSM history ring, integrity engine | Ch 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 16 |
| `navigate-magnav` | Yes | 18-term Tolles–Lawson online calibrator, borrowed bilinear anomaly grid maps, posterior Cramér–Rao bound tracker (`CrlbTracker`) | Ch 6, 14 |
| `navigate-route` | Yes | Multi-body flight-plan execution, RNAV leg sequencing, gravity-scaled fly-by turn anticipation, procedure selection | Ch 17 |
| `navigate-trajectory` | Yes | Space trajectory laws: velocity-to-go burns, Clohessy–Wiltshire LVLH rendezvous, drag-energy atmospheric entry, polynomial E-guidance powered descent | Ch 18 |
| `navigate-guidance` | Yes | Integrity-gated surface guidance (`L1`, velocity, glideslope), space guidance admission, and multi-phase `step_mission` orchestration | Ch 17, 18 |
| `navigate-terrain-safety` | Yes | Multi-body terrain clearance assessment, geoid/selenoid/areoid datum adapters, and fail-closed alerting vocabulary | Ch 2, 15 |
| `flamsteed` | Yes | Embedded Yale Bright Star Catalogue (5th rev.), 3D linear space motion, and 133 ORB6 visual binary star orbits (`f64` / `f32`) | Ch 10 |
| `eclipse-canon` | Yes | Embedded NASA Five Millennium Canon of 11,898 solar and 12,064 lunar eclipses ($-1999$ to $+3000$) | Ch 10 |
| `navigate-celestial` | `wasm32` + host | DE431-fitted analytic ephemerides, ANISE adapter, light-time & relativistic aberration, $\Delta T$, Earth rotation, lunar time `TCL` | Ch 2, 10, 11 |
| `navigate-startrack` | Host | Star centroiding with streak covariance, Pyramid lost-in-space & prior-aided identification, Davenport q-method, `LineOfSight` fusion adapter | Ch 8, 11 |
| `navigate-visual` | Host | Camera pose estimation, `L1-GNC` PnP solver, spatial holdout, local scene bundle refinement, odometry session, `BodyTangentEnu`, `fusion_bridge` | Ch 8, 13 |
| `navigate-visual-render` | Host (`wgpu` / CPU) | Synthetic terrain DEM and orthoimagery reference view and depth renderer | Ch 12 |
| `navigate-match` | `wasm32` + host | Pure-Rust `burn` (and optional `ort`) inference for `LoFTR`, `XFeat`, `LighterGlue`, `DINOv2`, `SegFormer-B0` 4-class segmentation, and `Kashin` banks | Ch 12, 20 |
| `navigate-match-kernels` | Host / GPU | CubeCL custom GPU compute kernels for learned matchers | Ch 12 |
| `navigate-locate` | `wasm32` + host | Single-photo localization, Hilbert-sharded `DINOv2` place search, and flight corridor tracking | Ch 13, 20 |
| `navigate-visual-flight` | Host | Real-time 30 Hz zero-disk-IO flight visual engine, multi-hypothesis spatial basin checks, amortized masking, delayed covariance propagation | Ch 7, 12, 13 |
| `navigate-locate-web` | `wasm32` | Browser Web Worker visual locator with WebNN, WebGPU, and CPU fallback | Ch 20 |
| `navigate-data` | `no_std` + `alloc` | Navigation data contract: canonical integer-only package manifests, tile index blobs, content-addressed blobs, verification, ranged read and fetch ports, correlation keys by source and cell, vertical datum | Ch 21 |
| `navigate-onboard` | Host | Onboard boundary: `Navigator` and `FlightDirector` traits, declared sensor suite, acceptance records, reference inertial navigator with GNSS, barometric, and star-camera aiding, route director | Ch 17, 21 |
| `navigate-recording` | Host | Visual recording wire contracts, session processor, and revisit loop closures | Ch 20 |
| `tools/navrun` | Host CLI | Deterministic end-to-end closed-loop scenario runner | Ch 19 |
| `tools/visual-model` | Host CLI | Deterministic checkpoint converter for `LoFTR`, `XFeat`, and `LighterGlue` | Ch 20 |
| `tools/visual-pipeline` | Host CLI | Dataset preparation, `SegFormer-B0` training, reference bank building, corridor tracking, and `sokoly-debrief` export | Ch 20 |

## 2. Architecture Decision Record (ADR) traceability matrix

| ADR | Title | Primary chapters |
|---|---|---|
| [ADR-0001](../adr/0001-record-architecture-decisions.md) | Record architecture decisions as versioned files in this repository | Introduction |
| [ADR-0002](../adr/0002-sans-io-deterministic-workspace.md) | One Cargo workspace with sans-IO deterministic core crates | Ch 1, 3, 19 |
| [ADR-0003](../adr/0003-typed-observations-modular-filter.md) | The navigation filter fuses typed, admission-gated observations with pluggable propagation | Ch 3, 4 |
| [ADR-0004](../adr/0004-integrity-honesty.md) | Every solution carries an integrity assessment that never overstates redundancy | Ch 1, 3, 4, 17 |
| [ADR-0005](../adr/0005-contract-crate-boundary.md) | External consumers pin a pure-types contract crate; aiding stays draft until a joint RFC | Ch 1, 3 |
| [ADR-0006](../adr/0006-one-leg-vocabulary-for-path-terminators-and-vertical-legs.md) | One leg vocabulary carries path terminators and vertical procedure legs | Ch 17 |
| [ADR-0007](../adr/0007-visual-positions-enter-fusion-through-one-adapter.md) | Visual positions enter fusion through one adapter with a declared error budget | Ch 8, 13 |
| [ADR-0008](../adr/0008-navigation-means-join-through-measurement-models.md) | Navigation means join through measurement models and one propagator seam | Ch 4, 8 |
| [ADR-0009](../adr/0009-the-navigation-estimate-narrows-the-visual-search.md) | The navigation estimate narrows the visual search and never admits a fix | Ch 3, 8, 13 |
| [ADR-0010](../adr/0010-reference-views-come-from-a-navigate-owned-renderer-port.md) | Reference views come from a renderer port that Navigate owns | Ch 12 |
| [ADR-0011](../adr/0011-matchers-and-the-locate-pipeline-have-an-asynchronous-form.md) | Matchers and the locate pipeline have an asynchronous form | Ch 12, 13, 20 |
| [ADR-0012](../adr/0012-scalar-map-models-and-bounded-state-history.md) | Scalar map models and a bounded state history extend the filter core | Ch 4, 7, 14, 15 |
| [ADR-0013](../adr/0013-body-frames-and-ellipsoids-are-explicit.md) | Body frames and reference ellipsoids are explicit | Ch 2, 13, 15 |
| [ADR-0014](../adr/0014-strapdown-inertial-propagation-and-marginalized-particle-filter.md) | Strapdown inertial propagation and a marginalized particle filter extend fusion to attitude, biases, space frames, and multimodal maps | Ch 5, 6, 7, 14, 15 |
| [ADR-0015](../adr/0015-zero-heap-no-std-physical-time-and-space-integrity.md) | Zero-heap `no_std` core, physical coordinate time, explicit body geodesy, and space integrity | Ch 2, 3, 17, 18 |
| [ADR-0016](../adr/0016-core-crate-topology-and-zero-heap-fusion.md) | Core crate topology: route, terrain safety, astrodynamics, and magnetic navigation crates, and zero-heap `no_std` fusion | Ch 3, 14, 15, 16, 17 |
| [ADR-0017](../adr/0017-unified-cartesian-inertial-core-and-frame-transitions.md) | One body-centered Cartesian inertial core with pole-free local axes, general frame changes, and IMU increments | Ch 2, 5, 7, 15 |
| [ADR-0018](../adr/0018-reduced-dynamic-orbit-determination-clock-and-radiometric-opnav.md) | Reduced-dynamic orbit determination with a force-model coast, receiver clock states, and radiometric and optical navigation models in the inertial filter | Ch 5, 8, 9, 16 |
| [ADR-0019](../adr/0019-space-trajectory-and-multiphase-guidance.md) | Space trajectory laws, space guidance setpoints, and mission-phase orchestration from orbit to a runway or to touchdown | Ch 18 |
| [ADR-0020](../adr/0020-visual-stack-consolidation-polar-trn-and-visual-pose.md) | Visual stack consolidation, polar terrain-relative navigation, and 6-DOF visual pose fusion | Ch 2, 8, 12, 13 |
| [ADR-0021](../adr/0021-one-imaging-sensor-model-and-star-sightings.md) | One imaging sensor model, and star sightings enter fusion as lines of sight | Ch 8, 11, 12 |
| [ADR-0022](../adr/0022-navigation-data-contract-mission-packages-and-the-onboard-boundary.md) | One navigation data contract, content-addressed mission packages, and one onboard boundary | Ch 21 |

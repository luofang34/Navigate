# Image-only local reconstruction

Build `navigate-visual` with the `reconstruction` feature.
Supply undistorted image coordinates and one common `CameraModel`.
Use digests from `Frame::evidence_sha256` for the source observations.
The host must retain the source stream, calibration, and matcher identities.

An image matcher can supply `PixelMatch` values to `DenseTrackBuilder`.
A point tracker can supply `ImageTracks` directly.
The dense adapter discards abandoned tracks with only two observations.
It retains active two-image links and completed tracks with three or more observations.
Feature IDs remain stable when other tracks are removed.

Call `propose_seeds` for a selected camera pair, or supply a `ReconstructionSeed`.
Call `reconstruct` separately for each seed.
The solver keeps all camera rotation axes free.
It reports cameras without enough support in `unresolved_camera_indices`.
A group accepts 2 to 129 cameras and up to 65,536 feature tracks.

The browser worker exposes the same library through `SceneTracks`.
The main upload worker stores image groups in OPFS with a checksum.
It uses bounded groups with eight shared boundary observations.
It reuses adjacent-image matches for terrain-based tracking.
A failed map match does not remove image observations from these groups.
The upload worker initializes each group from its image observations.
Its initial camera pairs cover separate parts of the group.
The upload path checks each initial camera pair in its search budget.
Complete camera coverage does not establish that a scene estimate is correct.
A later initial pair can produce a different complete estimate.
The result records camera pairs that the search budget did not examine.
It then checks shared camera poses to align compatible scene estimates.
Parent poses supply a fallback when image initialization has no result.
The worker continues at most three candidates into the next group.
It retains different local solutions and parent coordinate frames in this schedule.
It also uses camera coverage and image fit to order work.
This schedule does not accept or reject a geographic alternative.
All other candidates remain stored with an explicit deferred status.
A coordinate-frame ID identifies shared axes, not a unique camera path.
Use the scene record and its exact parent record to follow one alternative.
Do not combine cameras from different alternatives to report path coverage.
The worker stores each local point cloud once.
Each alignment record identifies that cloud, its parent scene, and its transform.
It retains incompatible overlaps as separate coordinate frames.
The browser can propose a map registration for a connected scene.
Registration fits scale, rotation, and translation to rendered map surface points.
The browser keeps scene paths separate from map pose acceptances.

The upload path can sample a disconnected image interval again.
It adds decoded frames around a break between supported scene parts.
One pass uses at most two groups and 24 extra frames.
A group cannot exceed 129 observations.
The worker checks the pixels and identities of reused observations.
Two seek requests for one decoded frame do not add two observations.
The host saves new pixels before it publishes their scene estimates.
Reprocessing a completed refinement does not add evidence.
Unprocessed intervals remain explicit when a budget is insufficient.

A complete group can remain separate when its overlap alignment fails.
The browser retries that boundary with a wider source-image window.
The window contains at most 129 existing observations.
It preserves their image identities and the original scene alternatives.
A failed retry remains unresolved. It does not join the coordinate frames.

The dense matcher adapter caches exact adjacent-image pairs.
Its keys use source digests that include pixels and calibration.
The cache has byte and entry limits.
The adapter reports cache hits separately from inference work.
The geometric solver and acceptance policy do not use cache hits as support.

Scene coordinates have arbitrary scale.
The result contains estimated geometry and estimated camera poses.
It does not provide a geographic fix, a map update, or an accuracy bound.
Shared observations and unknown correlations must not be treated as independent evidence.
Keep geographic alternatives separate.

Use `align_scenes` to convert one estimated group to another.
The input groups must share observation identities and enough camera motion.
The fit residual describes consistency between estimates, not absolute accuracy.

Use `align_scenes_with_points` when shared cameras have too little movement.
Supply proposed `ScenePointAssociation` values from the host adapter.
The optional `associate_scene_points` adapter uses mutual nearest image pixels.
It requires the same observation identities in both groups.
It does not assume that point IDs match between groups.
The geometric fit checks shared image support and camera rotations.
Estimated points constrain scale relative to shared camera origins.
The fit also checks point extent and camera position consistency.
It rejects repeated point associations and repeated image evidence.
Distinct supported scales remain separate candidates.
The result reports when a search or output budget excludes alternatives.
Unknown errors and shared image evidence remain explicit in the browser result.
Reprocessing a saved group does not add evidence or confidence.

The browser restores parent points from the exact saved local solution.
It checks scene identities, parent lineage, and transformed camera values.
Missing or changed saved data produces an error.
Point-supported alignment runs in the same worker as reconstruction.
It does not require a specific image matcher or execution backend.

## Conditional map registration

Call `propose_scene_registration` with a local scene and a `ReferenceView`.
Supply the query processing digest, the common camera model, and `PixelMatch` values.
Any matcher adapter can supply the correspondences.
The solver links query pixels to reconstructed points.
It uses only finite positive reference depth.
Repeated point IDs and repeated source observations are invalid.
Repeated matches cannot increase support.
The result retains distinct geometric proposals and their inlier associations.
The result reports when its candidate budget removes other supported proposals.
A bounded search cannot establish a unique location.

The map surface, reconstructed scene, and calibration are estimates.
Their errors and correlations remain unknown.
The point fit residual is not an absolute geographic accuracy measurement.
Apply navigation priors and geographic acceptance in a separate policy.
Keep the observation, reference render, and map release identities.

The browser schedules up to three source observations for each coordinate frame.
Each source can come from any connected group in the selected paths.
A later group uses its stored cloud and exact saved alignment.
Its registration applies only to paths that contain that group.
It cannot supply a registration for a sibling path.
The schedule reports paths without a registration source.
The schedule prefers map hypotheses that passed single-image geometry.
It then spreads the source observations across the input sequence.
Nearby high-inlier frames do not consume the full registration budget.
This spacing controls work. It does not establish independent evidence.
A traced reference proposal can also initialize the separate scene solve.
A rejected single-image pose does not become a map acceptance.
The worker renders that hypothesis and runs the selected image matcher again.
The Rust solver then proposes scene registration from the current rendered depth.
OPFS stores the registration, source scene identity, and reference hashes.
Paths with equal camera coverage keep the solver work order in the preview.
This display order is not a confidence score.
The preview transforms each exact parent chain separately.
It does not join alternative chains that share a coordinate frame.
At most 32 registered paths expand into preview poses for one upload.
Each registration receives a share of the remaining preview budget.
Unused slots remain available to later registrations.
Other registration results and scene branches remain stored.
The result reports deferred preview paths.
The original map hypotheses also remain available.

A scene without a supported registration remains in local coordinates.
A map registration does not validate every reconstructed camera or surface.
The map display still shows imagery and terrain.
It does not yet draw reconstructed facade geometry.

## Conditional camera fitting

Call `refit_scene_camera` to fit an observation to estimated scene points.
Supply one `CameraModel`, the scene digest, and the observation digest.
Supply an initial `LocalScenePose` and `ScenePointMatch` values.
Each match contains a scene feature ID, a 3D point, and an image pixel.
A learned matcher or a classical tracker can supply these links.
Model loading and device selection stay in that adapter.
The geometry function does not require a specific execution backend.

The scene stays fixed during this calculation.
This does not make its geometry certain.
The solver can change all camera rotation and translation axes.
It rejects repeated feature IDs and repeated image pixels.
It returns `None` if this fitter cannot find enough support.
A supported result retains the scene and observation identities.
It reports the inlier feature IDs and an image residual.
These values do not establish geographic acceptance or accuracy.
Keep other scene alternatives and apply acceptance policy separately.

The WASM adapter exposes `resect_scene_camera`.
Run this synchronous function in a browser worker.
The native and WASM paths use the same Rust geometric implementation.
A host must verify the source scene before it supplies its points.
Reprocessing the same points does not create independent evidence.
A changed camera path requires a separate check of its map alignment.

## Conditional point triangulation

Call `triangulate_scene_tracks` with an `ImageTracks` graph and camera estimates.
The camera order and observation digests must match the graph.
Each call handles one scene alternative.
The function keeps camera positions, rotations, and fixed flags unchanged.
It returns supported points and the feature IDs that remain unresolved.
Points need positive depth, sufficient parallax, and consistent image support.
The input graph retains each original image link.

The WASM adapter exposes `triangulate_scene_points`.
Run this synchronous function in a worker.
The adapter keeps the supplied camera records and coordinate gauge.
The solver does not load models or select devices.
A matcher adapter can supply the graph through Navigate-owned types.
Call the separate point refinement step with fixed camera estimates if needed.
Fixed estimates do not become ground truth.
A denser point set does not establish independent evidence or geographic accuracy.

Keep map anchor observations when selecting a bounded camera set.
A uniform time sample can remove an observation with useful map support.
Triangulation cannot recover its map links if that camera is absent.
Use a new map registration for changed scene geometry.
Keep rejected registrations and alternative transforms separate.

## Bounded whole-path refinement

Use `ImageTrackMerger` to join verified image groups for one connected candidate.
Supply one camera model and append source groups in path order.
The merger requires unique pixel agreement in at least two shared observations.
All other shared pixels must agree within one pixel.
Ambiguous associations remain separate.
The merger retains each source group and feature ID.
A repeated group ID is invalid.

Use `select` to obtain a bounded `ImageTracks` graph and its source associations.
Keep map anchor observations in the camera selection.
Use `initialize_scene_tracks` when traced point estimates are available.
Supply `ScenePointSeed` values in the same coordinates as the camera estimates.
Strict triangulation has priority.
A remaining point estimate needs two positive-depth projections within 12 pixels.
This threshold only controls initialization. It does not accept geometry.
Run joint refinement and separate camera fitting checks on the result.

The browser adapter exposes `SceneTrackGraph` and `initialize_scene_points`.
Its graph remains in the worker during repeated camera fits.
`set_scene` returns the digest of the exact supplied scene JSON.
`fit_camera` retains that digest and the source observation identity.
A successful group append invalidates the selected point set.
Unknown point IDs and repeated point evidence produce an error.
Image links with identical query pixels remain ambiguous and are omitted.

The upload flow refines at most three saved path alternatives.
It starts with at most 112 source cameras, including map anchors.
It adds real source frames at disconnected support intervals.
It can also add cameras whose point fitting failed.
Each path uses at most eight passes and 129 cameras.
A failed or bounded solve retains its unresolved frames and original alternatives.
No frame pose is interpolated to fill a reconstruction gap.
Playback interpolation remains a separate display operation.

The worker refines a denser point set with the camera estimates held fixed.
It stores the joint scene, dense scene, source associations, and camera fit evidence.
The map-registration step checks this stored evidence before preview expansion.
It transforms long paths in batches of at most 129 cameras.
The reference model, rendered depth validity checks, and final acceptance policy remain separate.
The map display still needs reconstructed surfaces to show a close facade.

These library inputs do not name a model or device.
A native adapter can use the same merger, initializer, and camera fitter.
A browser adapter runs those Rust functions through WASM in a worker.
Image matching can use its own GPU or device runtime.
The joint geometric solve does not claim GPU or ANE execution.
Changing an execution backend does not change evidence identities or geographic acceptance.

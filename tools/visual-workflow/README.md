# Visual workflow

This Rust program makes reference feature packets and estimates camera poses.
It can extract reference features at startup or load feature packets.
One query extraction serves all supplied reference candidates.
XFeat reuses matches within a query when reference feature contents are identical.
Each candidate still runs its own geometry checks. The result field
`matching_reused_from` identifies the source candidate when reuse occurs.
Reused matches do not constitute independent observations.
LoFTR uses an image-pair path because its computation depends on both images.
All paths use the same Navigate pose verifier.

## Build

From the Navigate root:

```sh
cargo build --locked --release -p navigate-workflow
```

To build the ORT program with the pinned Nix dependencies:

```sh
nix build .#workflow
result/bin/navigate-workflow locate config.json job.json result.json
```

The Nix package contains the program. Runtime libraries, models, and reference
packets are separate inputs. On macOS, build the rustnn Core ML variant with
`nix build .#rustnn`; it installs the same `navigate-workflow` command and accepts
`engine: "rustnn"` configurations. The flake supplies the required Rust compiler
and protocol-buffer compiler. `nix flake check` builds and tests the programs
available on the current system.

For the rustnn Core ML adapter, use macOS and Rust 1.97 or later:

```sh
cargo build --locked --release -p navigate-workflow --features rustnn-coreml
```

Supply an ONNX Runtime dynamic library with API 22 or later. The pinned runtime
can be obtained with `nix build .#runtime --no-link --print-out-paths`. Set the
configuration's `runtime` path to that output's `lib/libonnxruntime.dylib` on macOS
or `lib/libonnxruntime.so` on Linux. It does not require a Python environment.
Supply model files separately. The program does not download weights. Keep their
licenses with them.

## Execution settings

Paths in this file are relative to the file. Use an absolute `profile_prefix`
when you need an ORT execution profile.

```json
{
  "runtime": "runtime/libonnxruntime.dylib",
  "engine": "ort",
  "detector": "models/xfeat.onnx",
  "graph": "sparse",
  "matcher": "models/lighterglue.onnx",
  "extraction": {"provider": "cpu", "threads": 4, "allow_cpu_fallback": true},
  "matching": {"provider": "cpu", "threads": 4, "allow_cpu_fallback": true},
  "keypoints": 512,
  "limits": {
    "image_pixels": 4000000,
    "features_per_image": 512,
    "candidates": 8,
    "reference_bytes": 134217728,
    "query_feature_bytes": 4194304
  }
}
```

Providers are `cpu`, `coreml_gpu`, and `coreml_ane`. Selection is explicit.
`allow_cpu_fallback: false` rejects ORT graphs that need the ORT CPU provider.
Core ML can still select CPU kernels inside its graph.

The graph contracts are:

| Graph | Input | Output |
| --- | --- | --- |
| `sparse` | XFeat `images`, RGB, 800 × 600 | Sparse XFeat features |
| `dense` | XFeat `image`, normalized grayscale, 800 × 576 | Dense XFeat maps |
| `dense_image` | XFeat `images`, grayscale or RGB, 800 × 600 | Dense maps with preprocessing in the graph |
| `loftr` | LoFTR `image0` and `image1`, grayscale, 640 × 480 | Matched coordinates |

The matcher accepts the original `log_assignment` output, an `assignment`
probability matrix, or compact `row_indices`, `column_indices`, and `row_scores`
outputs. Use [model preparation](../visual-model/README.md) to build a supported
fixed-size matcher with compact outputs. Its feature count must match both inputs;
changing the feature budget requires the corresponding prepared model.

Use `matcher: null` for LoFTR. Its optional `refinement_patches: true` setting
runs the full view and four crops. Budget all five inference calls.

For rustnn, set `engine` to `rustnn`, `graph` to `dense_image`, and `detector`
to the `.webnn` graph produced by the flake's `converter` package; see
[model preparation](../visual-model/README.md). Keep its external weights beside it. The graph must return
`dense_descriptors`, `keypoint_logits`, and `reliability`, with shapes
`[1,64,72,100]`, `[1,65,72,100]`, and `[1,1,72,100]`. Matching still uses ORT.
The extraction provider selects the Core ML compute-unit preference. ORT thread,
profile, and fallback settings apply only to ORT stages.

## Inputs and commands

Make a reference packet:

```sh
target/release/navigate-workflow pack config.json reference.png reference.navxft
```

The packet binds the decoded source pixels, model contract, and feature limit.
The loader checks its checksum, lengths, values, and descriptor norms.
Do not reuse it with a different encoder contract. Color input uses the `image`
crate's grayscale conversion before model preprocessing.

A localization job supplies camera calibration and immutable reference evidence:

```json
{
  "query": {"path": "query.png", "sha256": "SHA256_OF_IMAGE_FILE"},
  "camera": {"size": [640, 480], "focal": [600, 600], "principal": [319.5, 239.5]},
  "prior": {"position": [0, 0, 100], "rotation": [0, 0, 0, 1]},
  "position_radius_m": 50,
  "attitude_radius_rad": 0.2,
  "anchor_lat_lon": [35.8, -114.4],
  "map_release": "map-release-id",
  "map_sha256": "SHA256_OF_MAP_MANIFEST",
  "references": [{
    "id": "view-01",
    "image": {"path": "reference.png", "sha256": "SHA256_OF_IMAGE_FILE"},
    "depth": "reference.depth",
    "depth_sha256": "SHA256_OF_DEPTH_FILE",
    "pose": {"position": [0, 0, 100], "rotation": [0, 0, 0, 1]},
    "features": "reference.navxft"
  }]
}
```

Replace all digest placeholders with lowercase SHA-256 digests. Job paths are
relative to the job file. Depth is row-major, little-endian `f32`, in metres
along the camera optical axis. Zero means no surface. Its size must match the
image. Poses use `[x,y,z,w]` quaternions. `coordinate_frame` defaults to
`anchor_mercator`. Use `spherical_enu` for Cartesian east, north, up axes at
the anchor on the mean-radius sphere. Its Cartesian z is not geographic altitude
away from the anchor. Keep rendering coordinates and camera poses in one frame.
A file reference can include its own `camera` with the same fields as the job
camera. Use this for a different image size or focal length. If omitted, the
reference uses the job camera. Generated references use the job camera and
reject a reference camera override. Reference depth uses the reference camera;
pose fitting and query errors use the query camera. A photo anchor supplies
conditional map geometry. Keep its source history outside the feature packet;
do not use the query or its derived poses as an independent anchor.

Set `features` to `null` to extract each reference at startup. LoFTR requires
this setting and cannot consume XFeat packets.

```sh
target/release/navigate-workflow locate config.json job.json result.json --iterations 10
```

For an orthographic map, use a separate surface job:

```sh
target/release/navigate-workflow prepare-surface arrays surface
target/release/navigate-workflow locate-surface config.json surface/job.json surface/result.json --iterations 3
```

`prepare-surface` reads four row-major numeric NPY files from `arrays`:

- `image_query.npy` and `image_dop.npy`: RGB bytes with shape `[height,width,3]`.
- `intrinsics.npy`: a float32, unskewed pinhole matrix with shape `[3,3]`.
- `dsm.npy`: float32 map coordinates with shape `[height,width,3]`.

The digital surface model (`dsm`) must match the reference image dimensions.
Its first two coordinates must form one affine horizontal grid, in metres.
Its third coordinate is height in the same frame. Non-finite heights have no
surface evidence. Negative heights are valid. Camera calibration applies to
the query image. Preparation reads no query pose or query point map.
Keep independent evaluation data outside these inputs.

The output directory must not exist. It receives images, a height file, and a
job with file digests. The job uses a map-centre seed 100 metres above the
highest finite surface. Its broad pose bounds are experiment settings.
Set `prior`, `position_radius_m`, and `attitude_radius_rad` for the deployment.
The surface result is a local metric pose. It has no geographic registration.

`reference_quarter_turns` lists unique clockwise quarter turns from 0 through 3.
Preparation selects all four. An omitted field selects only 0. Each orientation
counts against the candidate and reference memory limits. Sparse matching keeps
reference features in memory and shares one query extraction across orientations.
LoFTR runs each image pair. Results retain all candidates without choosing a
navigation fix. Without a window, reference pixels use the original map grid.
Set `record_matches` to true to save correspondences for separate evaluation.
Reported iteration times exclude file loading and model/reference setup.
Reference byte counts exclude model memory and temporary inference allocations.

The optional `reference.window` field is `[x, y, width, height]` in source pixels.
The window must fit the source and be at least 32 by 32 pixels. It keeps the
source pixel spacing and heights. Result correspondences use window coordinates;
add `x` and `y` to obtain source coordinates. Metric poses keep the source frame.
The loader reads the full source before it selects the window. The memory limit
must contain both the source raster and the window during this operation.

Use `acquire-surface config.json job.json result.json` to test bounded map search.
Supply the full map and quarter turns `[0, 1, 2, 3]`. The map must be at least
2048 by 2048 pixels. Set `limits.candidates` to at least 76. This is the maximum
search size, including four rotations per window. Use a sparse or dense-image
feature backend. This command
loads the model and query once. It checks a coarse map, then at most nine
adjacent 2048-pixel windows. If all candidates fail, it checks nine 1024-pixel
windows inside the parent with most raw matches. It retains every decision.
The first pass uses search order, not a known query position. A geometry pass
does not establish a correct or unique position. The result includes loading,
encoding, matching, geometry, and time to the first pass. Memory counts exclude
model and runtime allocations. The full source map stays in memory.

Use `prepare-acquisition config.json surface.json output-directory` to prepare
map feature packets without query images or poses. Set `matcher` to `null`
in the preparation config to load only the encoder. This does not change the
feature contract. The output directory must not exist. The input uses the
`reference` fields from a surface job, with no `window`. The command verifies and reads the map image. Height
bytes are not read; their digest and the map geometry bind the packet namespace.
Preparation covers the full image and all local windows allowed by the search
rules, including clamped boundary windows. Add `--include-fine` to include all
fine child windows. Each window has four rotations and its own encoder context.
The command writes `catalog.json`, checksummed `.navfeat` packets, and preparation
measurements. Each packet must preserve its feature values through a decode
round trip. The catalog records coverage and the encoder contract. This command
prepares files. Catalog loading is not part of `evaluate-acquisition`.

Use `evaluate-acquisition config.json sequence.json result.json` to combine full-map
search and tracking. Use the same sequence fields as `evaluate-surface`, but
supply a full-map job with the acquisition budget. Omit `keyframe_interval`.
The selected search estimate and its map pixels initialize tracking directly.
A failed track, scheduled refresh, or gap over 200 ms starts full-map search
from the supplied prior. Each new map estimate starts a separate motion segment.
The process retains the model and source map. Use `--reference-cache-mib 32`
to retain exact crop features in a least-recently-used cache. The default is zero.
The maximum is 64 MiB. The cache also obeys `limits.reference_bytes`, including
source pixels, crop pixels, rotated pixels, and feature handles. It reserves
4 MiB for each new feature handle and rejects larger handles. The budget can
evict entries before the cache reaches its own limit. A hit checks the input
pixel digest and reuses the validated host feature handle. Each immutable crop
shares one pixel digest across its four rotation keys. A hit skips rotation
and encoding. Matching still uses the backend tensor interface. Reports include
cache events, retained bytes, peak accounted reference bytes, and adapter
tensor counters before and after each acquisition. These counters exclude
model loading and driver-internal work. They reset with the adapter lifecycle.
Use `--reload-model-on-recovery` only for a model residency cost comparison.
This control reloads the model sessions before each acquisition after the first
one. It keeps one process-wide runtime with the same native library identity.
It retains the same source map, search order, geometry checks, and tracker.
Results are conditional pose proposals. They do not establish unique fixes or
real-time navigation. The default `--arrival-policy offline` processes all declared frames.
Use `--arrival-policy fifo --queue-capacity 64` to test a bounded first-in,
first-out queue. The capacity is 1 to 128 records. Overflow removes the oldest
pending record. Use `--arrival-policy latest` to retain one pending record.
Use `--arrival-policy catch-up --queue-capacity 128` to retain bounded history.
This policy selects the latest retained frame within 200 ms of the current
accepted tracking frame. It discards earlier pending frames. If this history
is missing, or no accepted frame exists, it selects the latest arrival for
recovery. The tracker still applies the same time and geometry checks.
These modes use measured wall time and the declared capture timestamps.
They do not use stored inference times. The default `--capture-start ready`
starts capture when the models and map are ready. Use `--capture-start setup`
to start capture when the workflow starts, before it loads the job and models.
The first declared capture timestamp is the clock origin. Config-file loading
occurs before this clock. Image files remain on disk; the selected image is
verified and decoded within processing time. These queues retain file records,
not decoded image buffers. The queue drains after the final capture timestamp.
The report records each drop, queue depth, input age, result age, and drain time.
A pose retains its original capture timestamp. The test does not extrapolate
poses or measure live camera costs. Final output serialization is excluded.

Use `evaluate-surface config.json sequence.json result.json` for one fixed map
region and an ordered image sequence. Its manifest has `job`, `refresh_interval`,
and `frames`. Each frame has `image` (path and file digest), `frame_id`, and
`capture_time_ns`. Frame IDs and capture times must increase. Keep evaluation
truth outside the manifest. The job supplies one camera calibration for all frames.

This command requires the `gpu` build feature and uses wgpu optical flow.
An interval of 1 matches the map on every frame. Larger intervals track fixed
map points between refreshes. A failed track or a gap above 200 ms requests map
matching. A scheduled refresh skips tracking; a failed refresh clears the track.
The result records all map candidates, the selected source, and the original
map observation for each track. These tracks are conditional map evidence.
`motion_continuity` contains session odometry. Reacquisition without a verified
motion step starts a new segment. Do not compute velocity across segment changes.
The local map pose stays separate; this command submits no geographic map anchors.
Startup time includes loading and reference preparation. Frame times include
image decoding. The command does not include coarse region search or map download.

Use `replay` to process observations in time order. It keeps the models in memory.
A reference cache keeps prepared references for reuse. The cache removes the
least recently used references when it reaches a limit. Use an XFeat feature-stage
configuration. Create a JSONL manifest with one record per observation:

```jsonl
{"job":"frames/first.json","capture_time_ns":1000000000}
{"job":"frames/second.json","capture_time_ns":1100000000}
```

```sh
target/release/navigate-workflow replay config.json observations.jsonl results.jsonl
```

Job paths are relative to the manifest; timestamps must strictly increase. Each
record is limited to 16 KiB. The result file is written and flushed incrementally,
with `start`, `frame`, and `end` records. A malformed record or failed frame is
reported and processing continues. A failed frame has no pose. Failure to read
the manifest or write results stops the run. Geometry rejections remain valid
processed results, separate from frame failures.

The reference cache must stay within `candidates` and `reference_bytes`.
It keeps validated reference data. The cache key includes the reference data,
pose, map, calibration, and rendering settings. Change the declared digests when
you replace input data. The program does not read cached files again for each
frame. It keeps only one terrain renderer. It creates the renderer when a
reference needs rendering.

Per-frame `elapsed_ms` includes job loading, image loading, reference preparation,
inference, and geometry. The `start` record measures backend initialization.
These times exclude result serialization and writing. Replay does not calculate
a trajectory. It does not use its own estimates as new independent priors.

To render references from an installed `photo-reference-catalog`, omit `image`,
`depth`, `depth_sha256`, and `features` from each generated reference. Keep its
`id` and `pose`. Add this job field:

```json
"render_scene": {
  "catalog": "map/catalog.json",
  "catalog_sha256": "SHA256_OF_CATALOG_FILE",
  "imagery_tile_pixels": 128,
  "scene_bytes": 268435456,
  "far_m": 40000,
  "steps": 100,
  "refinements": 12
}
```

Set `map_sha256` to the catalog digest and the anchor to the midpoint of its
bounding box. The loader verifies each tile's size and digest. Imagery and
Terrarium elevation tiles must be 256 × 256. `imagery_tile_pixels` is a power
of two from 1 through 256; a lower value reduces imagery detail and memory.
Each assembled grid must fit 16384 pixels per side, the hardware adapter's texture
limit, and the scene byte limit. The renderer requests the dimensions it needs.
GPU imagery uses two bytes per pixel; terrain uses four, including missing coverage.
Missing coverage produces zero depth. Rays must observe an above-ground sample
before crossing terrain; coarse depth steps can miss narrow terrain features.

`locate` uploads the scene once and generates references in memory. To save
references for a separate deployment or feature-packet preparation stage:

```sh
target/release/navigate-workflow render job.json prepared --iterations 10
target/release/navigate-workflow locate config.json prepared/job.json result.json
```

The output directory must not exist. It contains a reusable job, reference PNGs,
little-endian depth files, and render measurements. `render` needs a hardware
GPU but no inference runtime or model. Both paths use wgpu with explicit scene
and view limits. Camera positions must stay within 100 km of the local origin.
The renderer reads image and depth back to host memory for the inference adapter.

Output files must not exist. A geometric rejection is a valid result. Input,
backend, and resource failures return an error. Results keep every candidate's
acceptance decision; the caller must resolve competing accepted hypotheses.

Each iteration measures resident query extraction, matching, and pose verification.
Model loading and reference preparation have separate measurements. Image decoding
is outside iteration timing. These times exclude reference rendering, retrieval,
and camera acquisition. The `render` result records rendering and readback costs
separately. Scene texture limits exclude host loading buffers and driver overhead;
upload staging can require additional memory of the same order as the scene.
Host reference budgets exclude model weights, backend
workspaces, and allocator overhead. They are not process memory limits.

`adapter_tensor_io` is null when counters are unavailable. rustnn counters include
reference extraction and all iterations. Zero dispatch-copy counters do not mean
zero host transfers or zero driver-internal copies. Use the host read/write counts
and process memory measurements with the timing results.

## Pipeline evaluation

Use `evaluate` to compare local map matching with temporal point tracking:

```sh
navigate-workflow evaluate config.json experiment.json results.json
```

Both `evaluate` and `evaluate-surface` accept optional `keyframe_interval`
(2 through 30 accepted frames; omitted disables correction). Adjacent-frame
predictions initialize full-resolution alignment to an appearance keyframe.
Corrections over two pixels or failed patch checks retain the original prediction;
corrected geometry must pass the unchanged gate or the prediction fit is retained.
Map refresh replaces the keyframe, and loss clears it. This retains one extra
grayscale image and reports patch/geometry failures and selection separately.
`prediction_fit_evaluated` distinguishes skipped prediction fits from successful
ones. Correction selection refers to the tracking proposal; a scheduled map
refresh can subsequently replace it. This option does not establish accuracy.

An experiment uses one fixed reference from a localization job. It accepts up to
1000 frames. Frame times must increase. Paths are relative to the experiment.

```json
{
  "job": "job.json",
  "frames": [
    {
      "image": {"path": "frame.png", "sha256": "SHA256_OF_FRAME_FILE"},
      "capture_time_ns": 0
    }
  ],
  "fold": 0,
  "refresh_interval": 5,
  "query_focal_scale": 1.0,
  "retain_valid_track": true,
  "tracking_backend": "wgpu",
  "max_map_track_disagreement_m": null
}
```

The Nix workflow packages include GPU tracking. For Cargo builds, use
`--features gpu` to select `tracking_backend: "wgpu"`. The GPU device and
tracking kernel stay loaded across frames. A missing GPU returns an error. Use
`"cpu"` for the CPU comparison; an omitted field selects CPU. This choice does
not change the neural inference backend. Results record the active tracker.

For a tracker consistency diagnostic, set `tracking_probe_frames` to at most
16 increasing, unique frame indices. Each selected frame runs CPU and wgpu on
the same preceding pixels and ordered point set, recording every output slot
and both geometry decisions. Neither arm updates the trajectory. Probe time is
validation time, excluded from `pipeline_ms`; fresh diagnostic devices make
these timings unsuitable for backend performance comparisons. This requires
the `gpu` build feature.

Run folds 0 through 4 separately. Each fold excludes fixed 64-pixel map cells
from fitting and tracking. A four-pixel cell border is excluded. The score uses
all test matches with valid depth, without removing pose outliers. Optional frame
`controls` contain `[reference_x, reference_y, query_x, query_y]` arrays. Copy the
output controls into another experiment to use the same test pairs. Controls
never enter fitting. They still share map and matcher errors; they are not truth.

Set `fold` to null for a coverage and timing run with all matches. This mode has
no test score. `query_focal_scale` changes query rays only; map geometry stays
fixed. Compare several scales to measure sensitivity to calibration.

For a fixed-match comparison, set `focal_probe_scales` to a list of 1 through 21
values in `[0.8, 1.2]`. Each value scales the effective query focal lengths.
On each map-matching frame, the probe reuses the same training matches, reference
geometry, prior, and test controls. It runs no extra neural inference.
Results include the match digests, poses, test errors, and geometry times.
Probe time is part of `validation_ms`, outside `pipeline_ms`.
No probe result changes tracking, acceptance, or session state. A focal sweep
measures conditional sensitivity; it does not establish calibrated intrinsics.

If query images were cropped and resized, set `query_transform` to
`{"crop":[left,top,width,height],"size":[output_width,output_height]}`.
The job camera must describe the uncropped source image. The program transforms
its focal lengths and principal point into the output pixel grid, then applies
`query_focal_scale`. This field changes calibration only; supply images with the
crop and resize already applied. It does not change reference calibration.
Set each reference's `camera` explicitly when its image grid differs from the
source query grid. Omit `query_transform` when the job camera already describes
the supplied query images. Results include the effective `query_camera`.
Keep source calibration tied to the actual capture mode.
Keep fitted candidate intrinsics separate from capture calibration. Validate
them independently before reuse across frames. A shared image size or
camera model does not establish equal photo and video fields of view.

Each frame can include `position_check` with `source`, `latitude_longitude`,
`reported_horizontal_error_m`, and `associated_capture_time_ns`.
Use `null` when the source does not report horizontal error.
The result records horizontal separation and the capture time difference.
This check does not change fitting, candidate selection, or acceptance.
The program does not verify source accuracy or independence from map preparation
and priors. It does not compare altitude. Keep this check separate from truth labels.

Set `navigation` to an object with `map_horizontal_sigma_m`,
`map_vertical_sigma_m`, `max_object_height_m`, `capture_time_error_bound_ns`,
and `max_located_bound_m` to evaluate the session state layer. All fields are
required. Map sigma values must be positive. Object height can be zero.
The session uses the standard drift and anchor policies. The lens, mount,
vertical datum, and map age are unknown. No age growth is modelled.
These declarations are assumptions; session bounds are not measured accuracy.

The session receives each geometric map fit before the map-track distance gate.
It receives relative motion from propagated map points, before a new map fit
replaces them. This motion shares map error. A lost track starts a new segment.
Session output does not change frontend matching, tracking, or acceptance.
Each frame records the causal session pose, continuous odometry, anchor decision,
trajectory revisions, and permitted uses. A revision is not vehicle motion.
`stage_ms.navigation_ms` is included in `pipeline_ms`. External position checks
also score the causal session pose without entering session decisions.
Omit `navigation` to disable this evaluation path.

A refresh interval of 1 requests map matching for every frame. Larger intervals
permit tracking between map requests. Tracking stops across a gap above 200 ms.
With `retain_valid_track`, a rejected map result does not remove a valid track.
A finite `max_map_track_disagreement_m` also prevents a conflicting map result
from replacing the track. This bound is an experiment input, not a measured
uncertainty. Results record the conflict. Propagated map points do not provide
new independent geographic evidence. `accepted` means geometric acceptance only.

`setup_ms` includes reference preparation and model loading. `pipeline_ms` includes
image decoding, inference or tracking, geometry, and track-state updates. It
excludes result writing and test scoring. `stage_ms` separates tracking, geometry,
and map matching; `frame_loading_ms` records image loading and validation. The
first inference can include backend startup work. Acquisition, global retrieval, and map download are outside this
local experiment. Report rejected frames and startup costs with timing results.

For split LoFTR execution, set `graph` to `loftr_split`, `detector` to the coarse
model, and `matcher` to the fixed-batch fine model. `extraction` selects the coarse
provider and `matching` selects the fine provider. Prepare both models with the
[model tool](../visual-model/README.md#split-loftr-execution). The runtime pads
partial batches and retains only valid matches. Keep the sessions resident.

## Offline scene refinement

Use `navigate-workflow refine-scene scene.json refined.json` to refine initialized
poses and points. This command does not load a neural model or accept a navigation fix.
The input has `cameras`, `points`, `iterations` (1 through 100), and `coordinate_gauge`.
Each camera has `observation_sha256`, `camera`, `pose`, and `fixed`. Camera and pose
fields use the localization job formats. Each point has a unique integer `feature_id`,
a three-coordinate `position`, and `observations` with `camera_index` and `pixel`.
All positions must use the same coordinate frame. Each camera keeps its own pixel
size and intrinsics. Intrinsics stay fixed during refinement.
A point can have a `position_prior` with a three-coordinate `position` and positive
`scale` in scene units. This adds a squared distance penalty. Keep the prior center
from its source when you refine the scene again. The scale is a tuning parameter,
not a calibrated uncertainty. Position priors require a null `coordinate_gauge`.

Set `coordinate_gauge` to null when each connected component has an observed baseline
between fixed cameras. Otherwise, use `[origin_camera, scale_camera]` to define an
arbitrary coordinate origin and scale. Only the origin camera is fixed in this mode.
The output retains the input observations and includes the source file digest, fitted
scene, objective costs, and solve time. Fixed poses and source points remain estimates.
Keep test observations outside the input graph. A lower fitting cost does not establish
geographic accuracy or independent confidence.

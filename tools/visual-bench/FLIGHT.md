# Original video replay

The `flight` command reads an original video with FFmpeg. It uses the video
presentation timestamps. Models and the renderer stay in memory. The output
contains separate candidate paths. A rejected frame makes a gap in the path.

Build the CLI:

```sh
cargo build --release --manifest-path tools/visual-bench/Cargo.toml
```

Run a local experiment:

```sh
visual-bench --backend gpu flight package.json flight.mp4 \
  --prior source-camera-and-prior.json \
  --candidates estimated-candidates.json \
  --model loftr.onnx --runtime libonnxruntime.dylib \
  --device coreml-ane --width 640 --fps 5 \
  --timeline-cache flight-timeline.json \
  --output flight.jsonl --track flight.geojson
```

The source camera file uses the `image` command format. Its camera dimensions
must match the original video. The CLI scales the intrinsics with the image.
The optional candidate file is an array of objects with `position_enu_m` and
`eye_to_enu_xyzw`. These are estimated search seeds. They do not replace the
navigation prior. The CLI can refine these candidates. It does not run a new
wide-area search if all candidates fail.

The timeline cache contains a digest of the complete video. A different video
cannot reuse that cache. The output records the video, model, runtime, map,
reference image, reference depth, and observation identities. The renderer
revision names the minimum supported fork revision. It is not the identity of
uncommitted renderer changes. Record the source revisions and dirty diff with
an experiment report.

## Motion and map checks

The default relative solver estimates all six pose components. `--fixed-tilt`
holds the reference camera tilt constant. It estimates translation and heading.
Use this option only when the scene and camera justify that assumption. The
option does not measure attitude or set attitude error to zero. It retains the
depth, image support, and navigation-prior checks. Other views can use the full
pose solver.

Map checks use the full pose solver. `--map-interval 5` makes a check every five
video seconds. These checks are separate results. They do not move a supported
relative path. `--reanchor` permits an explicit restart at a supported map pose.
The output retains the relative alternative and marks a break in continuity.
Set `--map-interval 0` to measure tracking without periodic map checks.
An image-matcher failure is a separate error result. A failed diagnostic map
check does not discard a supported relative pose. If all matchers fail, the
observation has no supported pose. The next observation can still run.

A rendered terrain surface is not verified scene geometry. Buildings and trees
can cause systematic error. A supported relative pose is conditional on the
initial map hypothesis, calibration, and rendered depth. It is not an independent
geographic fix. Repeated processing does not add independent evidence.

## Available processing time

`--adaptive --utilization 0.7 --realtime` uses measured processing cost to select
frames. The fraction describes the host time available to this processing lane.
It is not a measurement of GPU or NPU utilization. A host can change the fraction
with `--budget-file budget.json`. The file contains a JSON number from zero to
one. Replace it atomically. Zero pauses admission. The host can derive the budget
from navigation deadlines, thermal limits, or other work.

The controller admits one observation at a time. It does not create a processing
queue. Paced replay drops obsolete frames. An increase in cost reduces the rate
immediately. A decrease in cost increases the rate gradually. Rejected work also
counts as cost. The output reports a timing shortfall when the requested interval
exceeds one second. A smaller budget can cause loss of image overlap and tracking.
The controller cannot prevent that loss or interrupt an active inference call.

The measured cost includes the selected frame decode, rendering, matching, and
geometry. The summary also records all decode time. Decode work for discarded
frames and result writes is outside the controller estimate. Use the complete
wall time when you assess a host budget.

Cold acquisition and steady tracking have different costs. The default paced
run includes acquisition stalls. `--warm-start --realtime` starts paced playback
only after the first frame has an accepted map anchor. Its warmup record has no
paced completion age. This option measures an established track. It does not
prove that live cold acquisition can meet the same deadline.

`navigate_visual::AdaptiveSampler` has no device, model, or clock dependency.
Native and WASM hosts can call `begin`, process the newest admitted frame, and
call `complete` with measured cost. They can change the budget with
`set_utilization`. The image matcher and renderer remain replaceable adapters.
The CLI uses a classical CPU or wgpu matcher first and a dense ORT matcher on
failure. A provider request such as `coreml-ane` does not prove ANE execution.
Use the provider compute plan to verify operator placement.

FFmpeg is a CLI decoder dependency. Python is not part of this execution path.
The browser uses its own decoder and model adapter. The native CLI is excluded
from the browser bundle.

# Visual workflow

This Rust program makes reference feature packets and estimates camera poses.
It can extract reference features at startup or load feature packets.
One query extraction serves all supplied reference candidates.
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
packets are separate inputs. The rustnn feature uses the Cargo command below.

For the rustnn Core ML adapter, use macOS and Rust 1.97 or later:

```sh
cargo build --locked --release -p navigate-workflow --features rustnn-coreml
```

Supply an ONNX Runtime dynamic library with API 22 or later. Supply model files
separately. The program does not download weights. Keep their licenses with them.

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
| `dense_image` | XFeat `images`, RGB, 800 × 600 | Dense maps with preprocessing in the graph |
| `loftr` | LoFTR `image0` and `image1`, grayscale, 640 × 480 | Matched coordinates |

Use `matcher: null` for LoFTR. Its optional `refinement_patches: true` setting
runs the full view and four crops. Budget all five inference calls.

For rustnn, set `engine` to `rustnn`, `graph` to `dense_image`, and `detector`
to the `.webnn` graph. Keep its external weights beside it. The graph must return
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
image. Poses use Navigate's anchor Mercator frame and `[x,y,z,w]` quaternions.
Keep rendering coordinates and camera calibration in that same frame.

Set `features` to `null` to extract each reference at startup. LoFTR requires
this setting and cannot consume XFeat packets.

```sh
target/release/navigate-workflow locate config.json job.json result.json --iterations 10
```

Output files must not exist. A geometric rejection is a valid result. Input,
backend, and resource failures return an error. Results keep every candidate's
acceptance decision; the caller must resolve competing accepted hypotheses.

Each iteration measures resident query extraction, matching, and pose verification.
Model loading and reference preparation have separate measurements. Image decoding
is outside iteration timing. These times exclude reference rendering, retrieval,
and camera acquisition. Host reference budgets exclude model weights, backend
workspaces, and allocator overhead. They are not process memory limits.

`adapter_tensor_io` is null when counters are unavailable. rustnn counters include
reference extraction and all iterations. Zero dispatch-copy counters do not mean
zero host transfers or zero driver-internal copies. Use the host read/write counts
and process memory measurements with the timing results.

# Native LoFTR adapter

`LoFtrMatcher` implements `navigate_visual::ImageMatcher`. The adapter loads the
fixed 640 by 480 LoFTR outdoor dual-softmax export. It owns image resizing,
letterboxing, tensor layouts, model scores, and runtime selection. Its output is
`Vec<PixelMatch>`. The host passes these matches to `PoseVerifier`.

Initialize the ONNX runtime once with `initialize_blocking`. Load one matcher
with `LoFtrMatcher::load_blocking(model, execution)`. Keep it loaded across frames.
Use `match_images_blocking` for image pairs. Use `PoseVerifier::evaluate` for map
candidates and `PoseVerifier::track` for conditional frame tracking. The geometry
library does not depend on ONNX Runtime or this adapter.

`ExecutionConfig` controls device selection. The model digest and requested
provider are part of the adapter identity. A requested provider does not prove
that all operators run on that device. Keep a CPU path. Validate each model and
precision variant on its target hardware. RK3588, Jetson, and Hailo board results
are not available.

## Build an asset

The build tool needs Torch 2.8.0, Kornia 0.8.1, ONNX 1.18.0, Pillow, NumPy, and
ONNX Runtime. These are asset-build dependencies. The deployed Rust application
and browser do not need a Python runtime.

```sh
python tools/visual-inference/export_loftr.py \
  /path/to/loftr_outdoor.ckpt /path/to/new-loftr.onnx \
  /path/to/reference.png /path/to/query.png
```

The tool checks output parity with the original model on the image pair and
blank inputs. The public asset identity is in
`tools/visual-bench/model-assets/loftr.json`. LoFTR and Kornia use Apache-2.0.
The export uses dual softmax. It does not include optimal transport code from
SuperGlue. Image-pair parity is not a camera-accuracy measurement.

## Check local image pairs

The example reads files named `NAME-reference.png` and `NAME-query.png`.
It writes matches and timing data. It does not accept a geographic pose.

```sh
cargo run --release --manifest-path tools/visual-inference/Cargo.toml \
  --example dense_pairs -- \
  --library /path/to/libonnxruntime.dylib --model /path/to/loftr.onnx \
  --pairs /path/to/pairs --output /path/to/new-results.json --device coreml-ane
```

With float32 weights, the measured Core ML compute plan used CPU operations.
A separate mixed-precision experiment put backbone operations on ANE. It changed
some correspondences. Full float16 conversion changed more matches and had a
runtime shutdown failure. Do not substitute a precision variant based on timing
alone. The public browser asset remains float32.

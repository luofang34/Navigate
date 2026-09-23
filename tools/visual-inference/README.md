# Native visual matchers and inference probe

The Rust library target is `navigate_visual_onnx`. It supplies concrete
implementations of `navigate_visual::ImageMatcher`: XFeat with mutual descriptor
matching, XFeat with LighterGlue, and SuperPoint with SuperGlue. The host supplies
ONNX files. The library does not download or embed model weights.

Initialize the runtime once in the host. Keep an adapter resident for repeated
calls. Model selection and execution-provider selection are separate. The adapter
owns image resizing, pixel transforms, detector decoding, feature limits, model
loading, and runtime setup. It returns `PixelMatch` values in input-image pixels.
It does not return a geographic confidence score.

Add the package as a path dependency. Its package name is
`visual-inference-probe`; its library name is `navigate_visual_onnx`.

```rust,no_run
use navigate_visual::{Localizer, LocalizerConfig};
use navigate_visual_onnx::{ExecutionConfig, MatcherFiles, OnnxMatcher, initialize_blocking};
use std::path::Path;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
initialize_blocking(Path::new("/path/to/libonnxruntime.dylib"))?;
let matcher = OnnxMatcher::load_blocking(
    MatcherFiles::XFeat { model: "/path/to/xfeat.onnx".into() },
    ExecutionConfig::default(),
    2048,
)?;
let localizer = Localizer::new(matcher, LocalizerConfig::default())?;
# let _ = localizer;
# Ok(())
# }
```

A different model or hardware implementation can implement `ImageMatcher`
directly. It need not use ONNX. The shared `PoseVerifier` retains the geometry
policy. Search candidates and map rendering remain outside the matcher. The
adapter cannot change prior bounds or convert repeated fits into new evidence.
The native library is not linked into the web bundle.

## Local model comparison

```sh
cargo run --release --manifest-path tools/visual-inference/Cargo.toml \
  --bin matcher_compare -- /path/to/suite.json /path/to/new-report.json
```

The suite defines `library`, `models`, and `cases`. Paths are relative to the
suite file. A model has `kind: "xfeat"` and `path`, or `kind: "superglue"`
with `detector`, `matcher`, and `image_size`. Use `xfeat_lighterglue` with
`detector` and `matcher` for sparse XFeat plus LighterGlue. Use `xfeat_dense`
with a dense `detector` and an optional LighterGlue `matcher`.
The SuperGlue size must match the image size embedded in its coordinate
normalization. The
adapter handles the detector's separate fixed or dynamic input size.

Each case supplies a query image, reference image, optical depth file, camera
calibration, reference pose, navigation prior, map identity, and content digests.
See `src/comparison/input.rs` for the exact input fields. Depth is little-endian
float32. Digests bind decoded grayscale pixels and raw depth bytes. Unknown
fields are errors. The tool rejects mismatched evidence before inference.

The tool first checks identical images, a known translation, blank input, and
incompatible dimensions. All models then use the same cases and the same Rust
pose verifier. Execution uses four CPU threads. The default `keypoints` limit
is 2048. The optional suite `provider` is `cpu`, `coreml_ane`, `coreml_gpu`,
`cuda`, or `tensor_rt`. NVIDIA suite runs use device zero.
Core ML takes only static-shape graphs. Dynamic matching stays on the CPU.
The report separates model loading, matching, and geometry time. It records model
identities, reference identities, rejections, and runtime failures. A runtime
failure makes the command fail. A geometric rejection is a measured outcome.

This comparison uses supplied reference candidates. It does not measure search
recall. Candidates from one matcher can bias the comparison. Test wide-area
retrieval separately, with the same prior and candidate budget for each matcher.
Rendered depth is not independently verified geometry. An accepted fit is not
an independent geographic accuracy measurement. The three map screenshots in
the supplied folder need an image-alignment evaluation, not an airborne-pose
accuracy claim.

## Model execution probe

This separate tool measures an ONNX model through Rust `ort`. It supports CPU,
Core ML GPU, Core ML Neural Engine, CUDA, and TensorRT selection. It does not link into the
navigation library or the browser bundle.

Supply a compatible ONNX Runtime shared library with the required execution
provider. The probe loads this library at run time. Use an ONNX Runtime
distribution for your target. The probe records model load time, warm inference times,
outputs, and provider placement. It creates a new output directory.

```sh
cargo build --release --manifest-path tools/visual-inference/Cargo.toml
tools/visual-inference/target/release/visual-inference-probe \
  --library /absolute/path/to/libonnxruntime.dylib \
  --model /absolute/path/to/model.onnx \
  --inputs /absolute/path/to/inputs.json \
  --output target/new-probe-run \
  --provider coreml-ane
```

The input manifest is a JSON array. Each item has `name`, `shape`, and `path`.
Paths are relative to the manifest. Data files contain little-endian float32
values. Use `cpu`, `coreml-gpu`, `coreml-ane`, `cuda`, or `tensor-rt` for
`--provider`. NVIDIA runs accept `--device-id`. TensorRT runs accept
`--workspace-mib`, with a default of 256 MiB. This limits builder workspace,
not total GPU memory. The probe keeps engine caches in its new output directory.

`export_models.py` exports the supplied external SuperPoint and SuperGlue models.
Run it with `--help` for its inputs. SuperPoint timings cover the dense backbone
and descriptor normalization. They exclude keypoint selection. SuperGlue timings
cover the matcher with nonempty input keypoints. The exported keypoint axes are
dynamic. Empty keypoint inputs require a separate host branch.

`mixed_precision.py` keeps descriptor normalization in FP32. Full FP16 conversion
can overflow the normalization and produce zero descriptors. Run
`validate_precision.py` before accepting accelerator timings. It compares the
features with FP32 and measures the same workload with PyTorch MPS. It requires
the external model source and an Apple GPU. The upstream
[SuperGlue license](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE)
limits these models to noncommercial research.

Core ML selection permits CPU operations. Read its compute-plan log before
claiming GPU or Neural Engine execution. Provider registration alone is not proof
of hardware placement. Load time and warm inference time measure different work.
Neither measures image decoding, data retrieval, or camera fitting.

For target runtime choices and measured device coverage, see
[accelerator evaluation](ACCELERATORS.md). `wgpu` targets GPUs. It does not select
ANE, RKNN, or Hailo devices. The navigation library's wgpu patch matcher is
separate from learned model inference.

For visual localization, first reduce candidate count with the prior and cached
map descriptors. Keep models and the map resident. Use the previous accepted
pose as a proposal, with independent acceptance bounds. Then measure keypoint
count, precision, and matcher choice. A faster backbone alone does not remove
the regional search cost. Test LightGlue as a separate matcher candidate; do not
assume its results or speed match this probe.

## Dense XFeat and LighterGlue assets

See [accelerator evaluation](ACCELERATORS.md) for measured results and target
runtime choices. Model choice and hardware execution remain separate.

The dense adapter uses an 800 by 576 convolutional graph. Rust applies image
normalization, peak selection, reliability interpolation, and bicubic descriptor
sampling. It keeps these operations in float32. The convolutional graph can use
float16. The adapter maps all matches back into the input image coordinates.
The optional LighterGlue model returns log assignment scores. The adapter checks
finite outputs, mutual assignment, and the model score threshold. The geometric
verifier does not receive these scores.

The export tools require PyTorch, Kornia, ONNX, ONNX Runtime, NumPy, Pillow, and
onnxconverter-common. They are asset-build tools. Native inference uses Rust and
the ONNX Runtime library. Browser inference does not call Python.

```sh
python export_xfeat_dense.py /path/to/modules/model.py /path/to/xfeat.pt \
  /path/to/reference.png /path/to/new-export-directory
python export_lighterglue.py /path/to/xfeat-lighterglue.pt \
  /path/to/new-lighterglue.onnx --validation-inputs /path/to/inputs.json
```

The dense tool checks its patch-convolution rewrite against upstream XFeat. It
writes float32 and float16 ONNX files, input tensors, and a hash report. The
float16 artifact still needs device and matching tests. The LighterGlue tool
checks the export against the original layer operations. It checks finite
outputs for extreme logits. An optional real-input manifest adds a regression
case. The tool requires the same selected matches and bounded score drift.
The manifest uses the inference probe's input format.

Use `--save-outputs` with `visual-inference-probe` to save float32 output tensors.
The report names each tensor file and gives its shape. Writes occur outside the
timed model execution. Compare these values with a CPU reference. A provider
name or a low inference time does not prove correct output.

## Browser model probe

The probe is outside the production web bundle. It runs inference in a browser
worker. It compares real feature tensors with native CPU assignment results.
It requires finite scores, identical selected matches, and GPU compute dispatches.
The input manifest must contain at least 1024 features for each image.

```sh
python prepare_browser_probe.py /path/to/lighterglue.onnx \
  /path/to/inputs.json /path/to/new-browser-inputs
node browser/serve.mjs /path/to/new-browser-inputs \
  ../visual-bench/webapp/runtime /path/to/lighterglue.onnx /path/to/new-report.json
```

Open the printed loopback URL in a WebGPU browser. The page writes its result to
the given local report path. It does not send tensors to an external service.
This probe checks the matcher model. Test the full search and pose pipeline
separately before changing the production browser adapter.

## NVIDIA runtime selection

The native adapter supports `Provider::Cuda` and `Provider::TensorRt`.
TensorRT has priority over CUDA. ONNX Runtime can send remaining operators to
CPU. An explicit CUDA or TensorRT request must register its provider. A missing
provider is an error. It does not silently become a CPU benchmark.
The application host owns these settings. They are not navigation inputs.

```rust,no_run
use navigate_visual_onnx::{ExecutionConfig, Provider};

let execution = ExecutionConfig {
    provider: Provider::TensorRt,
    nvidia_device_id: 0,
    tensor_rt_workspace_bytes: 256 * 1024 * 1024,
    ..ExecutionConfig::default()
};
```

The optional `engine_cache_directory` enables persistent TensorRT engines.
The adapter uses model content for the cache prefix. The host must supply a new
cache directory when the GPU, ONNX Runtime, or TensorRT version changes.
Engine caches are not portable model assets. Keep sessions resident during use.
No automatic float16 or integer conversion is enabled by this adapter.

Use a GPU-enabled ONNX Runtime build that matches the Jetson JetPack, CUDA,
TensorRT, and cuDNN versions. The Rust ARM64 Linux compile check passed.
A local runtime without NVIDIA providers passed a negative execution test:
CPU computation worked, and both explicit NVIDIA requests failed.
This test does not prove execution or speed on Jetson.
Physical-board tests must check operator placement, numeric outputs, selected
matches, pose decisions, memory, and latency.

Run the provider-failure test with a runtime that has no NVIDIA providers:

```sh
NAVIGATE_TEST_ORT=/absolute/path/to/libonnxruntime.so \
  cargo test --manifest-path tools/visual-inference/Cargo.toml --lib \
  unavailable_accelerators_cannot_become_cpu_success -- --ignored
```

The test fixture is a generated ONNX `Add(input, input)` graph. It has no learned
weights or user imagery. Ordinary unit tests do not require a runtime library.

[ONNX Runtime TensorRT requirements](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
and [CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
define the native runtime dependencies.

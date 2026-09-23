# Native visual matchers and inference probe


The Rust library target is `navigate_visual_onnx`. It supplies two concrete
implementations of `navigate_visual::ImageMatcher`: XFeat with mutual descriptor
matching, and SuperPoint with SuperGlue. The host supplies ONNX files. The library
does not download or embed model weights.

Initialize the runtime once in the host. Keep an adapter resident for repeated
calls. Model selection and execution-provider selection are separate. The adapter
owns image resizing, pixel transforms, detector decoding, feature limits, model
loading, and runtime setup. It returns `PixelMatch` values in input-image pixels.
It does not return a geographic confidence score.

Add the package as a path dependency. Its package name is
`visual-inference-probe`; its library name is `navigate_visual_onnx`.

```rust,no_run
use navigate_visual::{ImageMatcher, Localizer, LocalizerConfig};
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
with `detector`, `matcher`, and `image_size`. The latter size must be the
image size embedded in the SuperGlue export's coordinate normalization. The
adapter handles the detector's separate fixed or dynamic input size.

Each case supplies a query image, reference image, optical depth file, camera
calibration, reference pose, navigation prior, map identity, and content digests.
See `src/comparison/input.rs` for the exact input fields. Depth is little-endian
float32. Digests bind decoded grayscale pixels and raw depth bytes. Unknown
fields are errors. The tool rejects mismatched evidence before inference.

The tool first checks identical images, a known translation, blank input, and
incompatible dimensions. Both models then use the same cases and the same Rust
pose verifier. CPU execution uses four threads and a 2048-keypoint limit.
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
Core ML GPU, and Core ML Neural Engine selection. It does not link into the
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
values. Use `cpu`, `coreml-gpu`, or `coreml-ane` for `--provider`.

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

Backend routes for deployment:

| Target | Rust route | Required validation |
| --- | --- | --- |
| Apple GPU / Neural Engine | `ort` with Core ML | Operator placement and mixed-precision feature quality |
| NVIDIA GPU | `ort` with CUDA / TensorRT | Export support, static shape profiles and engine cache |
| Cross-platform GPU / browser | Burn with wgpu | Model import, operator support, transfer cost and shader performance |
| Qualcomm NPU | ONNX Runtime QNN | Target SDK, quantization and supported operator coverage |
| Intel NPU | ONNX Runtime OpenVINO | Device support, shapes and placement |
| Coral Edge TPU | Compiled TensorFlow Lite model and a runtime adapter | Integer quantization, static dimensions and compiler coverage |

The wgpu route targets GPUs. It does not select an NPU or an Edge TPU. The
navigation library's existing wgpu patch matcher is not SuperGlue inference.
Burn model import and learned matching on wgpu are not implemented by this probe.
CUDA, TensorRT, QNN, OpenVINO and Edge TPU need tests on their target hardware.

Primary implementation references:

- [Rust ort](https://github.com/pykeio/ort)
- [Core ML provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [TensorRT provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [Burn wgpu and CubeCL](https://burn.dev/blog/release-0.20.0/)
- [QNN provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [OpenVINO provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)
- [Edge TPU model requirements](https://coral.ai/docs/edgetpu/models-intro/)
- [LightGlue ONNX](https://github.com/fabio-sim/LightGlue-ONNX)

For visual localization, first reduce candidate count with the prior and cached
map descriptors. Keep models and the map resident. Use the previous accepted
pose as a proposal, with independent acceptance bounds. Then measure keypoint
count, precision, and matcher choice. A faster backbone alone does not remove
the regional search cost. Test LightGlue as a separate matcher candidate; do not
assume its results or speed match this probe.

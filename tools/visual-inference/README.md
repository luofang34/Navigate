# Rust inference probe

This separate tool measures an ONNX model through Rust `ort`. It supports CPU,
Core ML GPU, and Core ML Neural Engine selection. It does not link into the
navigation library or the browser bundle.

Supply a compatible ONNX Runtime shared library with the required execution
provider. The probe loads this library at run time. The Python `onnxruntime`
package contains a suitable library on the tested Apple host. Use the file in
its `capi` directory. The probe records model load time, warm inference times,
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

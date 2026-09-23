# Visual matcher accelerator evaluation

XFeat with LighterGlue is the current candidate for a common matcher. Both model
projects use Apache-2.0. The native adapter and a browser WebGPU model test pass
on the available Apple M3 Max. The production browser search still uses XFeat
with mutual descriptor matching. This evaluation does not change that path.

## Measured quality

The local suite contains 11 airborne query images. Every matcher uses the same
reference pixels, rendered depth, camera calibration, prior, and pose verifier.
The minimum remains 20 inliers. No acceptance threshold was reduced.

| Adapter | Keypoint limit | Geometric acceptances |
| --- | ---: | ---: |
| Sparse XFeat with mutual descriptors | 2048 | 9/11 |
| SuperPoint with SuperGlue | 2048 | 10/11 |
| Sparse XFeat with LighterGlue | 512 | 10/11 |
| Sparse XFeat with LighterGlue | 1024 | 10/11 |
| Dense XFeat with LighterGlue, CPU float32 | 1024 | 10/11 |
| Dense XFeat with LighterGlue, ANE float16 detector | 1024 | 10/11 |
| Dense XFeat with LighterGlue, CPU float32 | 2048 | 10/11 |

`DJI_0029_frame_5` remains rejected. LighterGlue with 2048 keypoints gives 19
inliers for that case. At 512 keypoints, two other cases are close to the minimum.
Use 1024 as the next integration candidate. A higher feature limit can be tested
for difficult candidates. A repeated fit is not new independent evidence.

These are supplied-candidate tests. SuperGlue produced the candidate poses, which
can bias this comparison. The results do not measure wide-area retrieval recall.
They do not establish independent geographic accuracy. Rendered depth is not
independently verified scene geometry. The three map screenshots in the source
folder are not airborne pose cases. Night scenes have not been tested.

## Measured execution

The host is an Apple M3 Max with 48 GiB of memory. Native CPU sessions use four
threads. Sessions stay loaded. The values below are medians across the 11 cases,
with model loading and geometric verification outside the matching time.

| Dense XFeat plus LighterGlue | Keypoints | Matching time |
| --- | ---: | ---: |
| CPU, ONNX Runtime 1.22 | 1024 | 90.4 ms |
| ANE detector plus CPU matcher, ONNX Runtime 1.30 | 1024 | 75.1 ms |
| CPU, ONNX Runtime 1.22 | 2048 | 189.0 ms |

These are different runtime builds. The timing difference is not an isolated
measurement of the accelerator alone. These runs are not board benchmarks.

The native inference probe measured the dense float16 detector at 5.56 ms per
image over 30 warm calls. Its Core ML compute plan assigned 59 operations to ANE
and 6 to CPU. All output tensors were finite. The maximum absolute difference
from the float32 CPU output was 0.0222 for descriptors, 0.1359 for logits, and
0.00455 for reliability. The complete matcher retained 10/11 acceptances.
A compute plan describes placement. It is not hardware utilization telemetry.

Core ML must take only static shapes in this adapter. Without that restriction,
the dynamic matcher caused unsupported-partition attempts and took more time.
ONNX Runtime 1.22 did not place this float16 detector's convolutions on ANE.
The tested dense path with 1.30 exited cleanly. A separate SuperGlue comparison
had an intermittent 1.30 shutdown failure. That failure is not resolved here.

Chrome WebGPU produced the same selected LighterGlue matches as native CPU for
real translated-image features at 512 and 1024 keypoints. The tests recorded
691 GPU compute dispatches per inference. All scores were finite. Maximum model
score differences were below 0.000151. This checks model execution and assignment.
It does not check browser image search, reference rendering, or camera movement.

## Deployment boundary

Keep Navigate's application, data management, image transforms, geometric
verification, and navigation estimator in Rust. Keep device runtimes inside
adapters. `ImageMatcher` still returns Navigate pixel correspondences. It does
not expose ONNX tensors, model scores, or device handles to geometry.

`ort` is a Rust interface to the C/C++ ONNX Runtime. It is not a pure Rust
inference engine. Vendor runtimes are also native libraries. A Rust application
can use these without a Python process during inference.

| Target | Runtime path to validate | Artifact | Status |
| --- | --- | --- | --- |
| Web | ONNX Runtime Web, WebGPU with WASM fallback | ONNX | LighterGlue model parity and GPU dispatch checked |
| Apple native | Rust `ort`, Core ML and CPU | Dense float16 ONNX plus float32 matcher | Native image suite checked |
| Jetson | Rust `ort`, TensorRT, then CUDA, then CPU | ONNX and device-specific engine cache | No board measurement |
| RK3588 | RKNN Runtime through a Rust adapter | RKNN compiled for the board | No compiler or board measurement |
| Raspberry Pi with Hailo | HailoRT through a Rust adapter | HEF compiled for the installed Hailo device | No compiler or board measurement |

The documented ONNX Runtime RKNPU provider targets RK1808. It is not proof of
RK3588 support. Use RKNN Toolkit2 and RKNN Runtime for RK3588. HailoRT supplies
native C/C++ interfaces. The Hailo device type and runtime version must match the
compiled artifact. No RKNN or Hailo placeholder adapter is added here.

The next device work should compile the fixed-shape convolutional detector first.
Keep keypoint selection, interpolation, normalization, and sparse matching on the
host until a measured device path is available. Jetson can also accelerate the
matcher. Test quantization with representative query and reference images. Check
selected matches and pose acceptance as well as tensor error and latency.

Hide backend selection from the normal user workflow. A packaged application
should select a tested artifact and runtime from detected device capabilities.
It should use CPU when the accelerator is absent or incompatible. Diagnostics
must state the actual fallback. The current evaluation tool uses explicit
provider selection so a benchmark cannot silently change devices.

WebGPU and `wgpu` target graphics/compute GPUs. They do not replace ANE, RKNN,
or Hailo runtime interfaces. A pure Rust GPU implementation can be another
adapter if measurements justify it. Training a new model is not required by
these results.

## Export checks and source references

The dense export folds patch extraction and the first keypoint layer into an
8 by 8 stride-8 convolution. The export checks this against upstream XFeat.
Rust preserves the upstream bicubic sampling and float32 normalization.

The LighterGlue export uses a stable log-sigmoid expression. A direct
`log(sigmoid(x))` export produced infinities and NaNs for a real translation case.
The export tool now checks extreme ONNX logits and real input tensors. It bounds
score drift and requires identical selected correspondences.

- [XFeat source and licence](https://github.com/verlab/accelerated_features)
- [LightGlue source and licence](https://github.com/cvg/LightGlue)
- [ONNX Runtime Core ML options](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [ONNX Runtime TensorRT provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [ONNX Runtime RKNPU supported platform](https://onnxruntime.ai/docs/execution-providers/community-maintained/RKNPU-ExecutionProvider.html)
- [RKNN Toolkit2 and Runtime](https://github.com/airockchip/rknn-toolkit2)
- [HailoRT](https://github.com/hailo-ai/hailort)

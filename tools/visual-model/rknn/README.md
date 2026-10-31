# RK3588 `XFeat` compiler checks

`compile_rknn.py` compiles a dense `float32` `XFeat` ONNX graph into an RK3588 `.rknn` artifact.
The compiler uses `float16` operations and does not quantize weights to integers.
Supply input tensors that already use the normalization from `navigate_match::XFeat::prepare_dense_input`.
Do not apply image normalization a second time in RKNN Runtime.

## Technical names

- An **RKNN artifact** (`xfeat-rk3588-f16.rknn`) is a compiled Rockchip NPU graph produced by RKNN Toolkit2.
- A **simulator check** is a host numerical comparison between the RKNN Toolkit2 simulator and the ONNX Runtime CPU provider.

## Build the compiler container

Run Docker from the repository root:

```sh
docker build -f tools/visual-model/rknn/rknn.Dockerfile \
  -t navigate-rknn-compiler tools/visual-model/rknn
```

## Compile and check one model

Mount the model directory, input directory, and output parent directory with absolute host paths.
Supply a little-endian `float32` input file with shape `[1, 1, 576, 800]`.
Ensure that the output directory does not exist before you run the container:

```sh
docker run --rm --network none --cpus 4 --memory 4g \
  -v "$PWD/tools/visual-model/rknn:/tool:ro" \
  -v /absolute/models:/models:ro \
  -v /absolute/inputs:/inputs:ro \
  -v /absolute/results:/results \
  navigate-rknn-compiler python /tool/compile_rknn.py \
  /models/xfeat-dense-f32.onnx /results/new-rk3588-check \
  --validate-input /inputs/input.f32
```

Run the simulator comparison unit tests in the same container image:

```sh
docker run --rm --network none \
  -v "$PWD/tools/visual-model/rknn:/tool:ro" \
  -w /tool navigate-rknn-compiler python -m unittest test_rknn_compile
```

The simulator compares `descriptors`, `keypoint_logits`, and `reliability` against ONNX Runtime CPU execution.
It rejects missing outputs, wrong tensor shapes, non-finite values, and absolute differences above `0.2`.
Run selected-match and camera-pose checks on physical RK3588 hardware before deployment through `navigate_match::VendorNpuSession`.

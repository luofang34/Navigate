# Model preparation (`navigate-model`)

`navigate-model` is a Rust command-line tool that prepares `XFeat`, `LighterGlue`, and `LoFTR` model files.
It converts verified source files into `burn` `.safetensors` weights or fixed-shape ONNX graphs.
It checks the input SHA-256 digest before it runs a graph recipe.
Keep the upstream model license with each prepared output file.

## Technical names

- A **source export** is a pinned upstream `.onnx` or `.ckpt` file identified by its SHA-256 digest.
- A **recipe** is a named deterministic weight conversion or ONNX graph rewrite.
- A **feature capacity** is the fixed maximum keypoint count per image in a prepared `LighterGlue` ONNX graph.

## Build

Build the binary from the repository root:

```sh
cargo build --locked --release -p navigate-model
```

The build uses pure-Rust protobuf definitions and does not require `protoc`, a GPU, or an inference runtime.

## Burn `.safetensors` weights

Convert verified exports into `.safetensors` files for `navigate-match`:

```sh
target/release/navigate-model xfeat.onnx xfeat-f16.safetensors --recipe xfeat-burn-f16
target/release/navigate-model xfeat.onnx xfeat-f32.safetensors --recipe xfeat-burn-f32
target/release/navigate-model lighterglue.onnx lighterglue-f16.safetensors --recipe lighterglue-burn-f16
target/release/navigate-model lighterglue.onnx lighterglue-f32.safetensors --recipe lighterglue-burn-f32
target/release/navigate-model loftr_outdoor.ckpt loftr-f16.safetensors --recipe loftr-burn-f16
target/release/navigate-model loftr_outdoor.ckpt loftr-f32.safetensors --recipe loftr-burn-f32
```

## Fixed-shape `XFeat` ONNX graphs

Prepare a dense `XFeat` ONNX graph for `navigate-match` (`--features ort`) or RK3588 compilation (`tools/visual-model/rknn/`):

```sh
target/release/navigate-model xfeat.onnx xfeat-dense.onnx --recipe xfeat-dense
target/release/navigate-model xfeat.onnx xfeat-gray.onnx --recipe xfeat-gray
target/release/navigate-model xfeat.onnx xfeat-gray-2x.onnx --recipe xfeat-gray-2x
```

The source export must have SHA-256 `41336466bbbb09b701b815e33932c7da69cff6ab75a068230418468430482912`.
`xfeat-dense` accepts `float32` NCHW `images` (`[1, 3, 600, 800]`) and emits `dense_descriptors` (`[1, 64, 72, 100]`), `keypoint_logits` (`[1, 65, 72, 100]`), and `reliability` (`[1, 1, 72, 100]`).
`xfeat-gray` accepts `[1, 1, 600, 800]` grayscale input and repeats the channel inside the graph.
`xfeat-gray-2x` accepts `[1, 1, 1200, 1600]` grayscale input and emits a `148 x 200` coarse grid.

## Fixed-shape `LighterGlue` and `LoFTR` ONNX graphs

Prepare fixed-capacity `LighterGlue` ONNX graphs (`32..=4096` features, source SHA-256 `170b10e74763fbfca483245cfa279972fc6747444065db2538500e22fc7ce9ca`):

```sh
target/release/navigate-model lighterglue.onnx matcher-2048.onnx --features 2048
target/release/navigate-model lighterglue.onnx matcher-masked.onnx --recipe lighterglue-masked --features 2048
```

Prepare split `LoFTR` ONNX stage graphs (`1..=1024` fine batch, source SHA-256 `49901002e682c650dcc1cd596d4effa980f12ffcc383746b77651ac89ed88559`):

```sh
target/release/navigate-model loftr.onnx coarse.onnx --recipe loftr-coarse
target/release/navigate-model loftr.onnx fine.onnx --recipe loftr-fine --batch 256
```

## Numerical parity check

Compare the original and prepared `XFeat` graphs with ONNX Runtime:

```sh
cargo test --release -p navigate-model dense_recipe_matches_original_model_intermediates -- --ignored
```

Set `NAVIGATE_TEST_ORT` to the ONNX Runtime shared library, `NAVIGATE_XFEAT_MODEL` to `xfeat.onnx`, and `NAVIGATE_XFEAT_INPUT` to little-endian `float32` NCHW input bytes before you run the test.

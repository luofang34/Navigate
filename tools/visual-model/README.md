# Model preparation

This Rust program prepares supported XFeat and LighterGlue ONNX exports.
It retains the weights and source metadata. It records the source digest and recipe in the output model. Keep the upstream model license with the
output. The input digest is checked before a named graph rewrite can run.

Build with `cargo build --locked --release -p navigate-model` or `nix build .#model`.
The Cargo build needs `protoc`; the Nix package supplies it. No inference runtime
or GPU is required for preparation.

## Matching

```sh
target/release/navigate-model lighterglue.onnx matcher-2048.onnx --features 2048
```

The output path must not exist. The supported input has SHA-256
`170b10e74763fbfca483245cfa279972fc6747444065db2538500e22fc7ce9ca`.
Feature counts must be 32 through 4096. The `lighterglue` recipe requires that
exact count from each image. To accept fewer features with a fixed input shape:

```sh
target/release/navigate-model lighterglue.onnx matcher-masked.onnx \
  --recipe lighterglue-masked --features 2048
```

The masked recipe adds `key_bias0` and `key_bias1` float32 inputs with shape
`[1, 1, capacity]`. `navigate-workflow` supplies zero for valid features and
`-10000` for padding. It masks keys in each attention layer and both assignment
normalizations. It removes padded indices from the output. Empty feature sets
return no matches without inference. Set the workflow feature limit at or below
the model capacity. Reference packets retain only valid features. Full-capacity
feature arrays are borrowed; shorter arrays require temporary padded buffers.

The recipe expresses assignment probabilities as the product of two softmax
terms and two sigmoid terms. It removes logarithms from assignment and expresses
negation with multiplication. This preserves the mathematical mutual-match rule
and the strict probability threshold of 0.1. Floating-point output must still be
checked on the target backend and the intended images.

The default output has `row_indices`, `column_indices`, and `row_scores`. It uses
ArgMax with the first-index tie rule and returns 20 bytes per feature instead of
the full quadratic matrix. `--output-kind probability` returns the full float32
`assignment` matrix for backends without the required reductions or for numerical
comparison. Both contracts are accepted by `navigate-workflow`.

Set `matcher` in the workflow configuration to the prepared model path. Set
`matching.allow_cpu_fallback` to `false` when the chosen ORT provider must own
the full graph. Use an execution profile to check graph placement; a Core ML
partition does not identify each Core ML kernel's physical compute unit.

## Feature extraction

```sh
target/release/navigate-model xfeat.onnx xfeat-dense.onnx --recipe xfeat-dense
```

The supported input has SHA-256
`41336466bbbb09b701b815e33932c7da69cff6ab75a068230418468430482912`.
The graph accepts float32 NCHW `images` with shape 1 × 3 × 600 × 800 and emits
`dense_descriptors`, `keypoint_logits`, and `reliability`, each at 72 × 100 pixels
with 64, 65, and 1 channels respectively. It retains image normalization and
resizing, removes sparse point selection, and folds the 8 × 8 patch expansion
into a strided convolution without changing learned weight values.

Use `--recipe xfeat-gray` for grayscale input with shape 1 × 1 × 600 × 800.
This recipe repeats the channel inside the graph. It retains the channel mean
operation and its floating-point order. The host transfers one channel.
The model digest changes, so build new feature packets for this graph.

For ORT, set the workflow's `detector` to this file and `graph` to `dense_image`.
Feature selection remains in the workflow, so this recipe does not accept a
`--features` setting. For rustnn, build and validate the WebNN representation with
the pinned Rust converter:

```sh
O2W_CACHE_DIR="$PWD/prepared" nix run .#converter -- convert \
  --input "$PWD/xfeat-dense.onnx" --optimize --output --validate
```

The converter package supplies its ONNX Runtime library. It compares the exported
WebNN graph with ONNX execution on fixed test inputs. The flake and the converter
lockfile fix the sources, dependencies, and compatibility patches. The converter
is a separate offline tool. The flight program does not depend on it.

Use the emitted `.webnn` file under `prepared/webnn` as the workflow's `detector`.
Keep its adjacent `.safetensors` weights when copying it to another machine.
The emitted filenames depend on the source path. Conversion validation checks
CPU numerical equivalence; validate actual image features and geometry on the
chosen deployment backend as well.

The numerical regression test loads the original model and compares its dense
intermediates with the prepared model through ORT. Set `NAVIGATE_TEST_ORT` to
the runtime library, `NAVIGATE_XFEAT_MODEL` to the source model, and
`NAVIGATE_XFEAT_INPUT` to little-endian float32 NCHW input bytes, then run:

```sh
cargo test --release -p navigate-model dense_recipe_matches_original_model_intermediates -- --ignored
```

# Model preparation

This Rust program prepares the supported LighterGlue ONNX export for a fixed
feature count. It retains the weights and source metadata. It records the source
digest and recipe in the output model. Keep the upstream model license with the
output. The input digest is checked before a named graph rewrite can run.

Build with `cargo build --locked --release -p navigate-model` or `nix build .#model`.
The Cargo build needs `protoc`; the Nix package supplies it. No inference runtime
or GPU is required for preparation.

```sh
target/release/navigate-model lighterglue.onnx matcher-2048.onnx --features 2048
```

The output path must not exist. The supported input has SHA-256
`170b10e74763fbfca483245cfa279972fc6747444065db2538500e22fc7ce9ca`.
Feature counts must be 32 through 4096. Both input images must supply that exact
count; this tool does not pad features or change the feature selection policy.

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

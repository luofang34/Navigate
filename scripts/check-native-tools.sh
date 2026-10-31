#!/usr/bin/env bash
# Gates for native tools outside the workspace. Accelerator runtime tests
# require the opt-in environment documented by the inference crate.
set -euo pipefail
cd "$(dirname "$0")/.."

manifest=tools/visual-inference/Cargo.toml
echo "== $manifest =="
cargo fmt --manifest-path "$manifest" --check
cargo clippy --manifest-path "$manifest" --all-targets -- -D warnings
cargo test --manifest-path "$manifest" --all-targets
RUSTDOCFLAGS="-D missing_docs -D rustdoc::broken_intra_doc_links" \
    cargo doc --manifest-path "$manifest" --no-deps
cargo build --manifest-path "$manifest" --release

#!/usr/bin/env bash
# Full local quality gate. CI runs the same steps; run this before pushing.
set -euo pipefail
cd "$(dirname "$0")"

echo "== structure checks =="
./scripts/check-structure.sh

echo "== cargo fmt --check =="
cargo fmt --all --check

echo "== cargo clippy =="
cargo clippy --all-targets -- -D warnings

echo "== wasm32 (navigate-celestial) =="
cargo clippy -p navigate-celestial --target wasm32-unknown-unknown -- -D warnings
cargo clippy -p navigate-celestial --features long-range --target wasm32-unknown-unknown -- -D warnings

echo "== navigate-celestial long-range feature =="
cargo clippy -p navigate-celestial --features long-range --all-targets -- -D warnings
cargo test -p navigate-celestial --features long-range

echo "== cargo test =="
cargo test --all-targets

echo "== cargo doc =="
RUSTDOCFLAGS="-D missing_docs -D rustdoc::broken_intra_doc_links" \
    cargo doc --no-deps --workspace

echo "== visual tools =="
./scripts/check-visual-tools.sh

echo "== native tools =="
./scripts/check-native-tools.sh

echo "== cargo build --release =="
cargo build --release --workspace

echo "ALL GATES GREEN"

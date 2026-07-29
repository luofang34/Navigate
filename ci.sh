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

echo "== cargo test =="
cargo test --all-targets

echo "== cargo doc =="
RUSTDOCFLAGS="-D missing_docs -D rustdoc::broken_intra_doc_links" \
    cargo doc --no-deps --workspace

echo "== cargo build --release =="
cargo build --release --workspace

echo "ALL GATES GREEN"

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

echo "== flamsteed (wasm32, no_std targets, crates.io package) =="
cargo clippy -p flamsteed --target wasm32-unknown-unknown -- -D warnings
cargo clippy -p flamsteed --no-default-features --features libm --all-targets -- -D warnings
for features in libm libm,magnitude-5 libm,all-stars std std,magnitude-5; do
    cargo test -p flamsteed --no-default-features --features "$features"
done
for target in thumbv8m.main-none-eabihf riscv32imac-unknown-none-elf thumbv6m-none-eabi; do
    cargo clippy -p flamsteed --no-default-features --features libm --target "$target" -- -D warnings
done
for harness in qemu-riscv qemu-arm; do
    cargo fmt --manifest-path "crates/flamsteed/$harness/Cargo.toml" --check
    (cd "crates/flamsteed/$harness" && cargo clippy --release --locked -- -D warnings)
done
# The QEMU runs and the nightly feature are in the weekly hardware workflow
# (.github/workflows/flamsteed-hardware.yml); run the script directly when
# flamsteed changes.
cargo package -p flamsteed --allow-dirty

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

echo "== visual pipeline (Python) =="
if command -v uv >/dev/null; then
    uv run --locked --project tools/visual-pipeline python -m unittest discover -s tools/visual-pipeline -p 'test_*.py'
else
    echo "SKIPPED: visual pipeline tests — uv is not installed"
fi

echo "== native tools =="
./scripts/check-native-tools.sh

echo "== cargo build --release =="
cargo build --release --workspace

echo "ALL GATES GREEN"

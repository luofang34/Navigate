#!/usr/bin/env bash
# Gates for the visual tools outside the workspace. The tools build against the
# sibling MapLibre fork. tools/visual-bench/MAPLIBRE_REVISION names the oldest
# fork commit that the tools support. The fork checkout must contain it.
set -euo pipefail
cd "$(dirname "$0")/.."

fork="../maplibre-rs-experimental"
pin=$(tr -d '[:space:]' < tools/visual-bench/MAPLIBRE_REVISION)

if [ ! -d "$fork/.git" ] && [ ! -f "$fork/.git" ]; then
    echo "SKIPPED: visual tools — no MapLibre fork at $fork (needs $pin)"
    exit 0
fi
if ! git -C "$fork" cat-file -e "$pin^{commit}" 2>/dev/null; then
    echo "FAIL: MapLibre fork at $fork does not contain pinned commit $pin" >&2
    exit 1
fi
if ! git -C "$fork" merge-base --is-ancestor "$pin" HEAD; then
    echo "FAIL: MapLibre fork HEAD does not descend from pinned commit $pin" >&2
    exit 1
fi
if [ -n "$(git -C "$fork" status --porcelain -- maplibre)" ]; then
    echo "WARNING: MapLibre fork has uncommitted changes under maplibre/; results include them"
fi

for manifest in tools/visual-bench/Cargo.toml tools/visual-bench/wasm-preview/Cargo.toml; do
    echo "== $manifest =="
    cargo fmt --manifest-path "$manifest" --check
    cargo clippy --manifest-path "$manifest" --all-targets -- -D warnings
    cargo test --manifest-path "$manifest" --all-targets
done

echo "== wasm-preview (wasm32) =="
cargo clippy --manifest-path tools/visual-bench/wasm-preview/Cargo.toml \
    --target wasm32-unknown-unknown -- -D warnings

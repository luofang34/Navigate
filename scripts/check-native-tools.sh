#!/usr/bin/env bash
# Gates for the tools that link the system GDAL library: the imagery provider
# and the visual service. They sit outside the workspace.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v gdal-config >/dev/null 2>&1; then
    echo "SKIPPED: native tools — GDAL is not installed (gdal-config not found)"
    exit 0
fi
echo "GDAL $(gdal-config --version)"

for manifest in tools/imagery-provider/Cargo.toml tools/visual-service/Cargo.toml; do
    echo "== $manifest =="
    cargo fmt --manifest-path "$manifest" --check
    cargo clippy --manifest-path "$manifest" --all-targets -- -D warnings
    cargo test --manifest-path "$manifest" --all-targets
done

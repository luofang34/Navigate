#!/usr/bin/env bash
# Gates for the workspace crates that .github/workflows/release.yml can
# publish: every crate that does not set `publish = false`. crates.io keeps a
# published version for ever, so a crate must carry its metadata and package
# its README before a release tag can name it.
set -euo pipefail
cd "$(dirname "$0")/.."

fields=(description license rust_version readme repository)
crates=$(cargo metadata --no-deps --format-version 1 --locked \
    | jq -c '.packages[] | select(.publish == null)')
if [ -z "$crates" ]; then
    echo "no publishable workspace crate"
    exit 0
fi

status=0
while IFS= read -r manifest; do
    name=$(jq -r .name <<<"$manifest")
    errors=$status
    status=0
    for field in "${fields[@]}"; do
        if [ "$(jq -r --arg f "$field" '.[$f] // ""' <<<"$manifest")" = "" ]; then
            echo "$name: Cargo.toml has no ${field//_/-}"
            status=1
        fi
    done
    readme=$(jq -r '.readme // ""' <<<"$manifest")
    if [ -n "$readme" ] && ! cargo package -p "$name" --list --locked --allow-dirty \
        | grep -qx "$(basename "$readme")"; then
        echo "$name: the package does not include $readme"
        status=1
    fi
    if [ "$status" -eq 0 ]; then
        echo "$name: publishable"
    fi
    status=$((status | errors))
done <<<"$crates"
exit "$status"

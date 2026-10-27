#!/usr/bin/env bash
# Structural conventions: no mod.rs, bounded file sizes, lib.rs stays a
# re-export surface, every lib.rs opens with a crate-level doc comment,
# each lockfile resolves one wgpu major version, and the root LICENSE is the
# GNU AGPL version 3.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0

while IFS= read -r f; do
    echo "FAIL: $f — mod.rs is banned; use foo.rs + foo/ instead" >&2
    fail=1
done < <(find crates tools -path '*/target' -prune -o -name mod.rs -type f -print 2>/dev/null)

while IFS= read -r f; do
    lines=$(wc -l < "$f")
    if [ "$lines" -gt 500 ]; then
        echo "FAIL: $f — $lines lines exceeds the 500-line limit" >&2
        fail=1
    fi
done < <(find crates tools -path '*/target' -prune -o -name '*.rs' -type f -print 2>/dev/null)

while IFS= read -r f; do
    lines=$(wc -l < "$f")
    if [ "$lines" -gt 100 ]; then
        echo "FAIL: $f — lib.rs is $lines lines; keep it under 100 (re-exports and module declarations only)" >&2
        fail=1
    fi
    if ! head -1 "$f" | grep -q '^//!'; then
        echo "FAIL: $f — lib.rs must open with a crate-level //! doc comment" >&2
        fail=1
    fi
done < <(find crates tools -path '*/target' -prune -o -name lib.rs -type f -print 2>/dev/null)

# The reference renderer and the learned matcher share one GPU device only
# when every crate resolves the same wgpu major version.
while IFS= read -r f; do
    majors=$(awk '
        /^name = / { wgpu = ($0 == "name = \"wgpu\"") }
        wgpu && /^version = / { split($3, v, "."); gsub(/"/, "", v[1]); print v[1]; wgpu = 0 }
    ' "$f" | sort -u | tr '\n' ' ')
    if [ "$(wc -w <<< "$majors")" -gt 1 ]; then
        echo "FAIL: $f — wgpu resolves to more than one major version: $majors" >&2
        fail=1
    fi
done < <(git ls-files '*Cargo.lock')

# The repository is free software under the GNU AGPL, version 3 or later.
if ! head -2 LICENSE 2>/dev/null | grep -q 'GNU AFFERO GENERAL PUBLIC LICENSE' \
    || ! head -2 LICENSE | grep -q 'Version 3, 19 November 2007'; then
    echo "FAIL: LICENSE — the repository root must hold the GNU AGPL version 3 text" >&2
    fail=1
fi

exit "$fail"

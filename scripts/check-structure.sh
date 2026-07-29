#!/usr/bin/env bash
# Structural conventions: no mod.rs, bounded file sizes, lib.rs stays a
# re-export surface, every lib.rs opens with a crate-level doc comment.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0

while IFS= read -r f; do
    echo "FAIL: $f — mod.rs is banned; use foo.rs + foo/ instead" >&2
    fail=1
done < <(find crates tools -name mod.rs -type f 2>/dev/null)

while IFS= read -r f; do
    lines=$(wc -l < "$f")
    if [ "$lines" -gt 500 ]; then
        echo "FAIL: $f — $lines lines exceeds the 500-line limit" >&2
        fail=1
    fi
done < <(find crates tools -name '*.rs' -type f 2>/dev/null)

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
done < <(find crates tools -name lib.rs -type f 2>/dev/null)

exit "$fail"

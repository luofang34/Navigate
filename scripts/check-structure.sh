#!/usr/bin/env bash
# Structural conventions: no mod.rs, bounded file sizes, lib.rs stays a
# re-export surface, every lib.rs opens with a crate-level doc comment,
# each lockfile resolves one wgpu major version, the root LICENSE is the
# GNU AGPL version 3, and git holds no build output and no file over 5 MiB.
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

# Build output never enters the repository, and no file inflates the clone.
while IFS= read -r path; do
    echo "FAIL: $path — build output in a target/ directory must not be in git" >&2
    fail=1
done < <(git ls-files | grep -E '(^|/)target/' || true)

# JavaScript, TypeScript, HTML, CSS, and npm manifests must not enter git.
while IFS= read -r path; do
    case "$path" in
        *.js|*.mjs|*.cjs|*.ts|*.tsx|*.jsx|*.html|*.css|package.json|*/package.json)
            echo "FAIL: $path — web script, markup, stylesheet, and package.json files are banned" >&2
            fail=1
            ;;
    esac
done < <(git ls-files)

# Every GitHub Actions uses: reference must pin a 40-character lowercase hex commit SHA.
for wf in .github/workflows/*.yml; do
    [ -f "$wf" ] || continue
    while IFS= read -r bad_line; do
        [ -n "$bad_line" ] || continue
        echo "FAIL: $bad_line" >&2
        fail=1
    done < <(awk -v file="$wf" '
        /^[[:space:]]*(-[[:space:]]+)?uses:[[:space:]]*/ {
            line = $0
            sub(/^[[:space:]]*(-[[:space:]]+)?uses:[[:space:]]*/, "", line)
            sub(/[[:space:]]+#.*/, "", line)
            sub(/^#.*/, "", line)
            gsub(/^[[:space:]"\x27]+|[[:space:]"\x27]+$/, "", line)
            if (line !~ /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.\/-]+@[0-9a-f]{40}$/) {
                printf "%s:%d — uses: %s must pin a 40-hex commit SHA\n", file, NR, line
            }
        }
    ' "$wf")
done
max_bytes=$((5 * 1024 * 1024))
while IFS= read -r path; do
    [ -f "$path" ] || continue
    size=$(wc -c < "$path")
    if [ "$size" -gt "$max_bytes" ]; then
        echo "FAIL: $path — $size bytes exceeds the 5 MiB limit for a tracked file" >&2
        fail=1
    fi
done < <(git ls-files)

# A crate for crates.io states its licence and ships its licence text and
# README, so that the package stands alone.
while IFS= read -r manifest; do
    dir=$(dirname "$manifest")
    grep -q '^publish = true' "$manifest" || continue
    if ! grep -q '^license = ' "$manifest"; then
        echo "FAIL: $manifest — a published crate must declare its license" >&2
        fail=1
    fi
    for file in LICENSE README.md; do
        if [ ! -f "$dir/$file" ]; then
            echo "FAIL: $dir — a published crate must ship $file" >&2
            fail=1
        fi
    done
done < <(find crates tools -path '*/target' -prune -o -name Cargo.toml -type f -print 2>/dev/null)

# A generated Rust source names its generator, relative to its crate. The
# generator must exist and be Rust, so that the tables can be rebuilt with
# the crate's own toolchain.
while IFS= read -r f; do
    generator=$(head -1 "$f" | sed -nE 's/^\/\/! .*[Gg]enerated by `([^`]+)`.*/\1/p')
    [ -n "$generator" ] || continue
    crate=$(dirname "$f")
    while [ ! -f "$crate/Cargo.toml" ] && [ "$crate" != . ]; do
        crate=$(dirname "$crate")
    done
    case "$generator" in
        *.rs) ;;
        *)
            echo "FAIL: $f — generated by $generator; generators must be Rust" >&2
            fail=1
            continue
            ;;
    esac
    if [ ! -f "$crate/$generator" ]; then
        echo "FAIL: $f — its generator $crate/$generator does not exist" >&2
        fail=1
    fi
done < <(find crates tools -path '*/target' -prune -o -name '*.rs' -type f -print 2>/dev/null)

exit "$fail"

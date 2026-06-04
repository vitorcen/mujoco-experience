#!/bin/bash
# Apply local modifications to third-party submodules as reviewable git patches.
#
# Convention (mirrors ../isaaclab-experience/patches):
#   patches/<submodule_name>/NNNN-description.patch
#   -> applied with `git -C dependencies/<submodule_name> apply`
#
# Each patch is a plain `git diff` taken at the submodule root, so paths inside
# the patch are relative to the submodule (a/foo b/foo). This keeps the diff
# visible/reviewable and lets `git apply` detect conflicts when the upstream
# pinned commit moves — unlike copying whole vendored files.
#
# Idempotent: a patch that is already applied (reverse-check passes) is skipped.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_ROOT="$REPO_ROOT/patches"
DEPS="$REPO_ROOT/dependencies"

applied=0; skipped=0; failed=0
for sm_dir in "$PATCH_ROOT"/*/; do
    sm="$(basename "$sm_dir")"
    target="$DEPS/$sm"
    [ -e "$target/.git" ] || { echo "⚠️  skip $sm: $target not a submodule checkout"; continue; }

    for patch in "$sm_dir"*.patch; do
        [ -e "$patch" ] || continue
        name="$sm/$(basename "$patch")"
        if git -C "$target" apply --reverse --check "$patch" >/dev/null 2>&1; then
            echo "= already applied: $name"; skipped=$((skipped+1)); continue
        fi
        if git -C "$target" apply --check "$patch" >/dev/null 2>&1; then
            git -C "$target" apply "$patch"
            echo "✅ applied: $name"; applied=$((applied+1))
        else
            echo "❌ FAILED (conflict — upstream drifted?): $name"; failed=$((failed+1))
        fi
    done
done

echo "-----------------------------------"
echo "applied=$applied skipped=$skipped failed=$failed"
[ "$failed" -eq 0 ]

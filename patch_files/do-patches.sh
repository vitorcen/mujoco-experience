#!/bin/bash

# Patch script for mujoco_menagerie submodule
# Applies local modifications to submodule files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Applying patches to mujoco_menagerie ==="

# Patch trs_so_arm100 model
echo "Patching trs_so_arm100..."
cp -rf "$SCRIPT_DIR/mujoco_menagerie/trs_so_arm100/"* \
       "$PROJECT_ROOT/dependencies/mujoco_menagerie/trs_so_arm100/"

echo "✓ Patches applied successfully"

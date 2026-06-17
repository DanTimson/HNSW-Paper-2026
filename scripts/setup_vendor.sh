#!/usr/bin/env bash
# Clone the reference HNSW-merge implementation into vendor_repo/.
set -euo pipefail
REPO="https://github.com/aponom84/merging-navigable-graphs.git"
DEST="$(dirname "$0")/../vendor_repo"
if [ -d "$DEST/.git" ] || [ -f "$DEST/hnsw.py" ]; then
  echo "vendor_repo already present at $DEST"
else
  git clone --depth 1 "$REPO" "$DEST"
  echo "cloned reference impl into $DEST"
fi

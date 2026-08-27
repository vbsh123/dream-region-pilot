#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAWN_DIR="$REPO_ROOT/external/DAWN"
DAWN_REVISION="19c32c28b5bf0475ccdfad853c74fc885f6410cd"

mkdir -p "$REPO_ROOT/external"
if [[ ! -d "$DAWN_DIR/.git" ]]; then
  git clone https://github.com/lizhuo-luo/DAWN.git "$DAWN_DIR"
fi
git -C "$DAWN_DIR" fetch origin "$DAWN_REVISION"
git -C "$DAWN_DIR" checkout --detach "$DAWN_REVISION"

echo "Pinned official DAWN source at $DAWN_REVISION"

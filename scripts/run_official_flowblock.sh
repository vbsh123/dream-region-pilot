#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOWBLOCK_DIR="$REPO_ROOT/external/FlowBlock"
MODEL_PATH="${1:?usage: run_official_flowblock.sh LLADA2.1_MODEL_PATH [FlowBlock args]}"
shift

if [[ ! -f "$FLOWBLOCK_DIR/run_eval.sh" ]]; then
  echo "Official checkout missing; run scripts/setup_vast.sh first." >&2
  exit 1
fi

echo "Official FlowBlock documents ~80 GB VRAM; a 24 GB RTX 4090 is not supported."
cd "$FLOWBLOCK_DIR"
exec bash run_eval.sh \
  --method flowblock,llada2.1 \
  --task math \
  --model-path "$MODEL_PATH" \
  "$@"

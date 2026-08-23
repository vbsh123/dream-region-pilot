#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

OUTPUT_DIR="${1:-outputs/gsm8k_graph_probe_2}"
shift || true

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m dream_region_pilot.run_gsm8k \
  --config configs/gsm8k_50.yaml \
  --output-dir "$OUTPUT_DIR" \
  --limit 2 \
  --strategies async_lag0 async_lag1 async_lag2 async_lag4 \
  --dependency-recompute-interval 1 \
  --diagnostic-examples 2 \
  --diagnostic-snapshot-interval 1 \
  "$@"

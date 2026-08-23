#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

OUTPUT_DIR="${1:-outputs/gsm8k_dynamic_graph_probe_2}"
shift || true

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m dream_region_pilot.run_gsm8k \
  --config configs/gsm8k_50.yaml \
  --output-dir "$OUTPUT_DIR" \
  --limit 2 \
  --strategies wavefront_probe \
  --probe-window 8 \
  --spawn-readiness 0.15 \
  --readiness-confidence-threshold 0.5 \
  --dependency-recompute-interval 4 \
  --diagnostic-examples 2 \
  --diagnostic-snapshot-interval 4 \
  "$@"

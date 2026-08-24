#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

CONFIG="${1:-configs/gsm8k_cot_official_50.yaml}"
OUTPUT_DIR="${2:-outputs/gsm8k_expanded_probe_2}"
if (( $# >= 2 )); then
  shift 2
else
  set --
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m dream_region_pilot.run_gsm8k \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --limit 2 \
  --strategies \
    vanilla \
    flowblock_proxy \
    loose_wavefront \
    mean_field_repro \
    controlled_position \
    controlled_dapd \
    controlled_jsd \
    controlled_combo \
  --probe-window 8 \
  --spawn-readiness 0.15 \
  --readiness-confidence-threshold 0.5 \
  --max-progress-gap 8 \
  --edge-persistence 2 \
  --dependency-recompute-interval 4 \
  --diagnostic-examples 2 \
  --diagnostic-snapshot-interval 4 \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

CONFIG="${1:?usage: run_balanced_attribution_50.sh CONFIG OUTPUT_DIR [extra args]}"
OUTPUT_DIR="${2:?usage: run_balanced_attribution_50.sh CONFIG OUTPUT_DIR [extra args]}"
shift 2

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m dream_region_pilot.run_gsm8k \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --limit 50 \
  --strategies \
    vanilla \
    vanilla_steps32 \
    vanilla_steps64 \
    vanilla_steps72 \
    always_on \
    always_on_tail_guard \
    always_on_coupled_defer_tail_guard \
    loose_wavefront \
    controlled_position \
    controlled_position_tail_guard \
  --probe-window 8 \
  --spawn-readiness 0.15 \
  --readiness-confidence-threshold 0.5 \
  --max-progress-gap 4 \
  --deferral-confidence-threshold 0.4 \
  --deferral-until-revealed-tokens 2 \
  --max-global-deferral-iterations 4 \
  --diagnostic-examples 0 \
  "$@"

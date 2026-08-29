#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

OUTPUT_DIR="${1:-outputs/gsm8k_expanded_50_r32}"
shift || true

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m dream_region_pilot.run_gsm8k \
  --config configs/gsm8k_cot_official_50.yaml \
  --output-dir "$OUTPUT_DIR" \
  --limit 50 \
  --strategies \
    vanilla \
    flowblock_proxy \
    loose_wavefront \
    mean_field_repro \
    controlled_position \
    controlled_position_tail_guard \
    always_on_tail_guard \
    always_on_coupled_defer_tail_guard \
  --probe-window 8 \
  --spawn-readiness 0.15 \
  --readiness-confidence-threshold 0.5 \
  --max-progress-gap 4 \
  --deferral-confidence-threshold 0.4 \
  --deferral-until-revealed-tokens 2 \
  --max-global-deferral-iterations 4 \
  --diagnostic-examples 3 \
  "$@"

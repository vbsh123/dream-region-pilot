#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

OUTPUT_DIR="${1:-outputs/gsm8k_vanilla_step_sweep_2}"
LIMIT="${2:-2}"
if (( $# >= 2 )); then
  shift 2
else
  set --
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m dream_region_pilot.run_gsm8k \
  --config configs/gsm8k_cot_official_50.yaml \
  --output-dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --strategies \
    vanilla \
    vanilla_steps128 \
    vanilla_steps96 \
    vanilla_steps72 \
    vanilla_steps64 \
    vanilla_steps32 \
  --diagnostic-examples 0 \
  "$@"

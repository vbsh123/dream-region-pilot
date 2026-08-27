#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-.venv}"
DAPD_DIR="external/DAPD"
DAPD_REVISION="05727b08da4cb4008a275123d7d9885dd5714f7c"
FLOWBLOCK_DIR="external/FlowBlock"
FLOWBLOCK_REVISION="8f730a2173140792a4324736efdcba27a2bdee75"
DAWN_DIR="external/DAWN"
DAWN_REVISION="19c32c28b5bf0475ccdfad853c74fc885f6410cd"
HUMANEVAL_DIR="external/HumanEval"
HUMANEVAL_REVISION="6d43fb980f9fee3c892a914eda09951f772ad10d"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "$TORCH_INDEX_URL" "torch==2.5.1"
python -m pip install -e ".[evaluation]"

mkdir -p external
if [[ ! -d "$DAPD_DIR/.git" ]]; then
  git clone https://github.com/quasar529/DAPD.git "$DAPD_DIR"
fi
git -C "$DAPD_DIR" fetch origin "$DAPD_REVISION"
git -C "$DAPD_DIR" checkout --detach "$DAPD_REVISION"

# Source-only checkout avoids OpenAI HumanEval's legacy setup.py, which imports
# pkg_resources and fails in modern isolated pip build environments. The pilot
# imports only its standard-library execution module.
if [[ ! -d "$HUMANEVAL_DIR/.git" ]]; then
  git clone https://github.com/openai/human-eval.git "$HUMANEVAL_DIR"
fi
git -C "$HUMANEVAL_DIR" fetch origin "$HUMANEVAL_REVISION"
git -C "$HUMANEVAL_DIR" checkout --detach "$HUMANEVAL_REVISION"

# Source-only checkout for reproducibility. Do not install FlowBlock into this
# environment: its LLaDA/SGLang stack has different pins and needs ~80 GB VRAM.
if [[ ! -d "$FLOWBLOCK_DIR/.git" ]]; then
  git clone https://github.com/Red-EAD/FlowBlock.git "$FLOWBLOCK_DIR"
fi
git -C "$FLOWBLOCK_DIR" fetch origin "$FLOWBLOCK_REVISION"
git -C "$FLOWBLOCK_DIR" checkout --detach "$FLOWBLOCK_REVISION"

# The official DAWN Dream fork exposes averaged late-layer attention in the
# same forward that produces logits. Regional DAWN strategies import this
# source checkout but continue loading the pinned public Dream weights.
if [[ ! -d "$DAWN_DIR/.git" ]]; then
  git clone https://github.com/lizhuo-luo/DAWN.git "$DAWN_DIR"
fi
git -C "$DAWN_DIR" fetch origin "$DAWN_REVISION"
git -C "$DAWN_DIR" checkout --detach "$DAWN_REVISION"

echo "Vast environment prepared. Activate with: source $ENV_DIR/bin/activate"

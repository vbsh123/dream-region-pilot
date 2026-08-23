#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-.venv}"
DAPD_DIR="external/DAPD"
DAPD_REVISION="05727b08da4cb4008a275123d7d9885dd5714f7c"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "$TORCH_INDEX_URL" "torch==2.5.1"
python -m pip install -e .

mkdir -p external
if [[ ! -d "$DAPD_DIR/.git" ]]; then
  git clone https://github.com/quasar529/DAPD.git "$DAPD_DIR"
fi
git -C "$DAPD_DIR" fetch origin "$DAPD_REVISION"
git -C "$DAPD_DIR" checkout --detach "$DAPD_REVISION"

echo "Vast environment prepared. Activate with: source $ENV_DIR/bin/activate"

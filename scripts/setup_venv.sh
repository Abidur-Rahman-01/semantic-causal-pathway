#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3.11}"
  "$PYTHON_BIN" -m venv .venv
fi

export HF_HOME="$ROOT/.venv/hf_home"
export PIP_CACHE_DIR="$ROOT/.venv/pip_cache"
mkdir -p "$HF_HOME" "$ROOT/.venv/data"
mkdir -p "$PIP_CACHE_DIR"
./.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
./.venv/bin/python -m ipykernel install --sys-prefix --name semantic-circuits --display-name "Python (Semantic Circuits)"
printf 'Environment ready: %s/.venv\nModel cache: %s\n' "$ROOT" "$HF_HOME"

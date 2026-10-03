#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo "Project environment is missing. Follow the README setup steps first." >&2
  exit 1
fi
exec ./.venv/bin/python scripts/run_pilot.py "$@"

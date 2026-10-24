#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHON=".venv/bin/python"

# Refactored reproduction: robustness evaluation now lives in experiments/.
# Optional first argument selects a fresh run directory; second may be --reuse.
RUN_DIR="${1:-experiments/robustness}"
if [ "${2:-}" = "--reuse" ]; then
    "$PYTHON" experiments/robustness/run.py --output-dir "$RUN_DIR" --reuse
else
    "$PYTHON" experiments/robustness/run.py --output-dir "$RUN_DIR"
fi
"$PYTHON" experiments/robustness/finalize.py --output-dir "$RUN_DIR"

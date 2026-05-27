#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Compute Korean and Arabic MGSM sanity metrics from raw runs.
# Inputs:  data/raw_runs/extra_language_checks/
# Outputs: artifacts/extra_language/

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
elif command -v py >/dev/null 2>&1 && py -3.12 -c "import sys" >/dev/null 2>&1; then
  PYTHON_CMD=(py -3.12)
else
  PYTHON_CMD=(python)
fi

"${PYTHON_CMD[@]}" -m src.evaluation.compute_main_metrics --task extra_language --no-canonical-prefix

echo "Wrote:"
echo "  ${REPO_ROOT}/artifacts/extra_language/metrics.csv"
echo "  ${REPO_ROOT}/artifacts/extra_language/bootstrap_ci.csv"
echo "  ${REPO_ROOT}/artifacts/extra_language/manifest.json"

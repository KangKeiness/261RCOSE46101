#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Compute main Chinese MGSM metrics from raw runs.
# Inputs:  data/raw_runs/main_chinese_mgsm/
#          data/raw_runs/composition_path_control/
#          data/raw_runs/trajectory_rationale_followup/
#          data/raw_runs/source_specific_patch_control/
# Outputs: artifacts/main/

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
elif command -v py >/dev/null 2>&1 && py -3.12 -c "import sys" >/dev/null 2>&1; then
  PYTHON_CMD=(py -3.12)
else
  PYTHON_CMD=(python)
fi

"${PYTHON_CMD[@]}" -m src.evaluation.compute_main_metrics --task main --no-canonical-prefix

echo "Wrote:"
echo "  ${REPO_ROOT}/artifacts/main/main_metrics.csv"
echo "  ${REPO_ROOT}/artifacts/main/bootstrap_ci.csv"
echo "  ${REPO_ROOT}/artifacts/main/diagnostics.csv"
echo "  ${REPO_ROOT}/artifacts/main/repetition_robustness.csv"
echo "  ${REPO_ROOT}/artifacts/main/manifest.json"

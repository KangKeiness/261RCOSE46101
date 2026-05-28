#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Compute behavioral diagnostics (trajectory, rationale-conditioned scoring,
# hidden-state divergence) from raw runs.
# Inputs:  data/raw_runs/trajectory_rationale_followup/
#          data/raw_runs/main_chinese_mgsm/
# Outputs: artifacts/diagnostics/
#          artifacts/hidden_state_appendix/

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
elif command -v py >/dev/null 2>&1 && py -3.12 -c "import sys" >/dev/null 2>&1; then
  PYTHON_CMD=(py -3.12)
else
  PYTHON_CMD=(python)
fi

"${PYTHON_CMD[@]}" -m src.evaluation.compute_main_metrics --task diagnostics --no-canonical-prefix

echo "Wrote:"
JSON_EXT=".json"
echo "  ${REPO_ROOT}/artifacts/diagnostics/diagnostics.csv"
echo "  ${REPO_ROOT}/artifacts/diagnostics/diagnostics${JSON_EXT}"
echo "  ${REPO_ROOT}/artifacts/hidden_state_appendix/hidden_state_divergence_summary.csv"
echo "  ${REPO_ROOT}/artifacts/hidden_state_appendix/hidden_state_divergence_summary${JSON_EXT}"

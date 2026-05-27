#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Compute BCa sensitivity intervals for patch-control bootstrap CIs.
# Paper-cited CIs remain the percentile intervals produced by
# scripts/reproduce_patch_control_ci.sh.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
elif command -v py >/dev/null 2>&1 && py -3.12 -c "import sys" >/dev/null 2>&1; then
  PYTHON_CMD=(py -3.12)
else
  PYTHON_CMD=(python)
fi

"${PYTHON_CMD[@]}" -m src.patching.patch_control_bootstrap \
    --input "${REPO_ROOT}/data/raw_runs/source_specific_patch_control/source_specific_patch_records.jsonl" \
    --output "${REPO_ROOT}/artifacts/patch_control/bootstrap_ci_bca_sensitivity.csv" \
    --method bca

echo "Wrote:"
echo "  ${REPO_ROOT}/artifacts/patch_control/bootstrap_ci_bca_sensitivity.csv"

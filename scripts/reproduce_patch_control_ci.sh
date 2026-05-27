#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Compute patch-control bootstrap confidence intervals.
# Inputs:  data/raw_runs/source_specific_patch_control/source_specific_patch_records.jsonl
#          artifacts/patch_control/s_broken_ids.txt
# Outputs: artifacts/patch_control/bootstrap_ci.csv

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
    --output "${REPO_ROOT}/artifacts/patch_control/bootstrap_ci.csv"

echo "Wrote:"
echo "  ${REPO_ROOT}/artifacts/patch_control/bootstrap_ci.csv"

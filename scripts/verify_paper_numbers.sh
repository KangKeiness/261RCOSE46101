#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Verify that all artifacts/ CSVs reproduce every paper-cited number.
# Exit 0 on full pass (prints PAPER NUMBERS VERIFIED), exit 1 on any mismatch.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
elif command -v py >/dev/null 2>&1 && py -3.12 -c "import sys" >/dev/null 2>&1; then
  PYTHON_CMD=(py -3.12)
else
  PYTHON_CMD=(python)
fi

"${PYTHON_CMD[@]}" -m src.evaluation.verify_paper_numbers

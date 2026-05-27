#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Generate paper figures from artifacts/ CSVs.
# Inputs:  artifacts/main/main_metrics.csv
#          artifacts/diagnostics/diagnostics.csv
# Outputs: artifacts/figures/*.pdf
#          artifacts/figures/*.png

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
elif command -v py >/dev/null 2>&1 && py -3.12 -c "import sys" >/dev/null 2>&1; then
  PYTHON_CMD=(py -3.12)
else
  PYTHON_CMD=(python)
fi

"${PYTHON_CMD[@]}" -m src.figures.generate_paper_figures

echo "Wrote:"
echo "  ${REPO_ROOT}/artifacts/figures/rationale_conditioned_heatmap.pdf"
echo "  ${REPO_ROOT}/artifacts/figures/rationale_conditioned_heatmap.png"
echo "  ${REPO_ROOT}/artifacts/figures/transition_flow_simple.pdf"
echo "  ${REPO_ROOT}/artifacts/figures/transition_flow_simple.png"
echo "  ${REPO_ROOT}/artifacts/figures/transition_sankey.pdf"
echo "  ${REPO_ROOT}/artifacts/figures/transition_sankey.png"
echo "  ${REPO_ROOT}/artifacts/figures/figure_latex_snippets.tex"

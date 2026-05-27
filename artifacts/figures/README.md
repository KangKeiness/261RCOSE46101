# Generated Paper Figures

Standalone candidate figures for the COSE461 final project paper on answer redistribution under direct middle-layer replacement.

## Files

- `rationale_conditioned_heatmap.pdf/png`: 2x2 rationale-conditioned scoring heatmap. Safest replacement for Table 3.
- `transition_flow_simple.pdf/png`: static transition decomposition flow. Alternative replacement candidate for Table 1.
- `transition_sankey.pdf/png`: experimental Sankey-style transition flow. The Sankey-style flow was generated as `transition_sankey.pdf/png`.
- `figure_latex_snippets.tex`: LaTeX snippets and replacement notes.
- `generate_paper_figures.py`: regeneration script.

## Input values used

### Rationale-conditioned scoring

Rows are model condition; columns are rationale source. Values are mean margin for the baseline answer.
All values loaded from `artifacts/diagnostics/diagnostics.csv` and verified against paper_constants.py.

Supplemental CIs recorded for paper text or appendix:

| Model condition | Rationale source | CI |
| --- | --- | --- |
| Baseline model | Baseline rationale | [2.894, 3.487] |
| Baseline model | Direct-swap rationale | [-2.438, -1.774] |
| Direct-swap model | Baseline rationale | [2.446, 2.935] |
| Direct-swap model | Direct-swap rationale | [-2.187, -1.616] |

### Transition decomposition

Chinese MGSM. All values loaded from `artifacts/main/main_metrics.csv` and verified against paper_constants.py.

## How to regenerate

```bash
bash scripts/reproduce_figures.sh
```

The script requires `matplotlib`, `numpy`, and Pillow.

## Recommended figure choice

- Safest replacement: heatmap for Table 3.
- Alternative replacement: simple flow for Table 1.
- Experimental alternative: Sankey-style flow.

## Warnings

- Do not include both an old table and its replacement figure in the main paper unless explicitly needed.
- Replacing Table 3 loses explicit CI display unless CIs are mentioned in text or appendix.
- Replacing Table 1 with a flow diagram should preserve all transition counts in labels.
- Do not change the paper numbers or introduce new analysis when swapping figures into the paper.

# Reproduction Guide

This guide explains how to reproduce every paper-cited number from the raw
runs included in this bundle. No model downloads are required for steps 2–8;
all inputs are pre-computed JSONL/CSV files under `data/raw_runs/`.

---

## Prerequisites

- Python 3.10 or later
- A Unix-like shell (bash) or Git Bash on Windows

---

## Steps

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Compute main Chinese MGSM metrics

```bash
bash scripts/reproduce_main_metrics.sh
```

Reads `data/raw_runs/main_chinese_mgsm/` and writes:
- `artifacts/main/main_metrics.csv`
- `artifacts/main/bootstrap_ci.csv`
- `artifacts/main/main_metrics.json`

Expected to complete in minutes on CPU; exact timing depends on bootstrap
parameters (default: 10000 resamples, seed 20260517).

### 3. Compute Korean and Arabic sanity metrics

```bash
bash scripts/reproduce_extra_language.sh
```

Reads `data/raw_runs/extra_language_checks/` and writes:
- `artifacts/extra_language/metrics.csv`
- `artifacts/extra_language/bootstrap_ci.csv`

Expected to complete in minutes on CPU.

### 4. Compute patch-control bootstrap confidence intervals

```bash
bash scripts/reproduce_patch_control_ci.sh
```

Reads `data/raw_runs/source_specific_patch_control/source_specific_patch_records.jsonl`
and the S\_broken sample list `artifacts/patch_control/s_broken_ids.txt` and
writes:
- `artifacts/patch_control/bootstrap_ci.csv`

Expected to complete in minutes on CPU.

For a BCa sensitivity interval over the same n=39 paired sample units:

```bash
bash scripts/reproduce_patch_control_ci_bca.sh
```

This writes `artifacts/patch_control/bootstrap_ci_bca_sensitivity.csv`.
The paper-cited patch-control CIs remain the percentile bootstrap intervals
from `scripts/reproduce_patch_control_ci.sh`.

### 5. Compute behavioral diagnostics

```bash
bash scripts/reproduce_diagnostics.sh
```

Reads `data/raw_runs/trajectory_rationale_followup/` and writes:
- `artifacts/diagnostics/diagnostics.csv`
- `artifacts/hidden_state_appendix/hidden_state_divergence_summary.csv`

Expected to complete in minutes on CPU.

### 6. Verify paper numbers

```bash
bash scripts/verify_paper_numbers.sh
```

Loads every paper-cited CSV from `artifacts/` and compares each value to the
constants in `src/evaluation/paper_constants.py`. Must print:

```
PAPER NUMBERS VERIFIED
```

If any value mismatches, the script prints the CSV path, column, expected
value, and actual value, then exits with code 1.

### 7. Reproduce figures

```bash
bash scripts/reproduce_figures.sh
```

Reads `artifacts/main/main_metrics.csv` and `artifacts/diagnostics/diagnostics.csv`
and writes PDF and PNG figures to `artifacts/figures/`. The script also runs
the same paper-number assertions as step 6 before drawing; if values differ
from `paper_constants.py`, it exits 1 before rendering any figure.

### 8. Run the test suite

```bash
pytest tests/
```

Runs parametrized tests against all eight paper-cited numerical clusters.
Must pass with no failures.

---

## Notes on the S\_broken subset (patch control)

The patch-control bootstrap (step 4) subsets to the n=39 canonical S\_broken
samples in `artifacts/patch_control/s_broken_ids.txt` before resampling. See
`docs/artifact_lineage.md` for the explanation of why the raw run shows n=40
while the paper reports on n=39.

## Notes on trajectory diagnostics partition

The `data/raw_runs/trajectory_rationale_followup/numeric_trace_summary.csv`
reflects a partition of n=155. The paper reports on the final-evaluator
reconciled n=151 partition after signed-answer parsing. Step 5 produces the
reconciled values. See
`docs/artifact_lineage.md` for details.

## Patch-Control Bootstrap Sensitivity

The paper reports percentile bootstrap CIs over the n=39 canonical S\_broken
subset. For sensitivity, BCa intervals are also computed and shipped at
`artifacts/patch_control/bootstrap_ci_bca_sensitivity.csv`. Point estimates
are identical by construction; interval endpoints differ mildly at n=39 and
should be read as a small-subgroup robustness check rather than a replacement
for the paper-cited percentile intervals.

---

## What this bundle reproduces vs what it does not

### What works end-to-end from this bundle

The following entrypoints are fully self-contained and can be re-executed from
the bundle without any additional files:

```bash
python -m src.evaluation.compute_main_metrics --task main
python -m src.evaluation.compute_main_metrics --task diagnostics
python -m src.evaluation.verify_paper_numbers
python -m src.patching.patch_control_bootstrap
python -m src.figures.generate_paper_figures
pytest tests/
```

These scripts derive every paper-cited metric from the pre-computed JSONL/CSV
files under `data/raw_runs/` and `artifacts/`. No model loading or dataset
download is required.

### What this bundle does NOT reproduce

The `src/runs/` scripts (`run_main_chinese_mgsm.py`, `run_extra_languages.py`)
are included as **provenance reference only**. They document the original
inference pipeline that produced `data/raw_runs/`, but they cannot be
re-executed from this bundle because several of their dependencies were not
included in the release:

- `src/evaluation/generation.py`
- `src/evaluation/activation_cache.py`
- `src/evaluation/mechanism_common.py`
- `src/data/data_loader.py`
- `src/patching/patching_utils.py`
- Several utility modules under `src/utils/`

These modules are part of the full SOURCE repository and require the complete
development environment to run. Re-running the original inference pipeline is
not necessary to verify paper results; all paper-cited numbers can be derived
from the pre-computed artifacts included in this bundle.

The Korean and Arabic extra-language artifacts (`artifacts/extra_language/`)
are included pre-computed in this bundle. The behavioral diagnostics entrypoint
recomputes `artifacts/diagnostics/diagnostics.csv` from shipped trajectory and
rationale CSV records.

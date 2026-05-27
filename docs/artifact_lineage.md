# Artifact Lineage

This file documents the end-to-end lineage for every paper-cited numerical
cluster: paper reference, CSV location, CSV column, raw-run input, and the
`src/` script that performs the computation.

All paper-cited values reproduced in this document cross-reference
`src/evaluation/paper_constants.py`. That module is the single source of
truth; the values quoted here are duplicated for reader convenience and
verified by `tests/test_paper_number_invariants.py`.

---

## Cluster 1 — Main Chinese MGSM aggregate

**Paper reference**: Section 4.3 (main results), Table 1 row "ZH".

**Phrases in paper**: "clean accuracy 0.500 (125/250)", "direct-swap accuracy
0.468 (117/250)", "identity accuracy 0.500 (125/250)", "accuracy delta −0.032,
95% CI [−0.096, 0.032]".

**CSV**: `artifacts/main/main_metrics.csv`

| Column | Value |
|---|---|
| `n` | 250 |
| `clean_correct` | 125 |
| `clean_accuracy` | 0.500 |
| `direct_swap_correct` | 117 |
| `direct_swap_accuracy` | 0.468 |
| `identity_correct` | 125 |
| `identity_accuracy` | 0.500 |
| `accuracy_delta` | −0.032 |

Bootstrap CI in `artifacts/main/bootstrap_ci.csv`, column `accuracy_delta`,
row `metric=accuracy_delta`, columns `ci_low` / `ci_high`.

**Raw-run inputs**: `data/raw_runs/main_chinese_mgsm/results_clean_no_patch.jsonl`
and `data/raw_runs/main_chinese_mgsm/results_restoration_no_patch.jsonl`.

**Computation script**: `src/evaluation/compute_main_metrics.py`

---

## Cluster 2 — Main Chinese answer redistribution

**Paper reference**: Section 4.3, Table 1 row "ZH".

**Phrases in paper**: "151 of 250 samples changed parsed answer (60.4%,
95% CI [54.0%, 66.4%])".

**CSV**: `artifacts/main/main_metrics.csv`

| Column | Value |
|---|---|
| `answer_changed` | 151 |
| `parsed_answer_change_rate` | 0.604 |

Bootstrap CI in `artifacts/main/bootstrap_ci.csv`, row
`metric=parsed_answer_change_rate`, columns `ci_low` / `ci_high`.

**Raw-run inputs**: same as Cluster 1.

**Computation script**: `src/evaluation/compute_main_metrics.py`

---

## Cluster 3 — Main Chinese transitions

**Paper reference**: Section 4.3, Table 1 row "ZH".

**Phrases in paper**: "86 stable-correct, 39 broken, 31 repaired, 94
stable-wrong; of the 94 stable-wrong, 81 (86.2%) differ from the baseline
wrong answer".

**CSV**: `artifacts/main/main_metrics.csv`

| Column | Value |
|---|---|
| `stable_correct` | 86 |
| `S_broken` | 39 |
| `S_repaired` | 31 |
| `stable_wrong` | 94 |
| `stable_wrong_same` | 13 |
| `stable_wrong_different` | 81 |
| `stable_wrong_different_rate` | 0.8617021276595744 (displayed as 0.862) |

Bootstrap CI for `stable_wrong_different_rate` in
`artifacts/main/bootstrap_ci.csv`.

**Raw-run inputs**: same as Cluster 1.

**Computation script**: `src/evaluation/compute_main_metrics.py`

---

## Cluster 4 — Korean sanity (Appendix A, Table 3/4)

**Paper reference**: Appendix A, Tables 3 and 4.

**Phrases in paper**: "Korean MGSM (n=250): clean accuracy 0.312, direct-swap
0.320, delta +0.008; 178/250 samples changed parsed answer (71.2%)".

**CSV**: `artifacts/extra_language/metrics.csv`, row `language=ko`.

| Column | Value |
|---|---|
| `n` | 250 |
| `clean_accuracy` | 0.312 |
| `direct_swap_accuracy` | 0.320 |
| `accuracy_delta` | 0.008 |
| `parsed_answer_changed_count` | 178 |
| `parsed_answer_change_rate` | 0.712 |
| `stable_correct` | 54 |
| `S_broken` | 24 |
| `S_repaired` | 26 |
| `stable_wrong` | 146 |
| `stable_wrong_same` | 18 |
| `stable_wrong_different` | 128 |
| `stable_wrong_different_rate` | 0.8767123287671232 (displayed as 0.877) |

**Raw-run inputs**:
`data/raw_runs/extra_language_checks/per_sample_transition_records.jsonl`.

**Computation script**: `src/evaluation/compute_main_metrics.py`
(with `--extra-language` flag, or via `scripts/reproduce_extra_language.sh`).

---

## Cluster 5 — Arabic sanity (Appendix A, Table 3/4)

**Paper reference**: Appendix A, Tables 3 and 4.

**Phrases in paper**: "Arabic MGSM (n=250): clean accuracy 0.224, direct-swap
0.272, delta +0.048; 183/250 samples changed parsed answer (73.2%)".

**CSV**: `artifacts/extra_language/metrics.csv`, row `language=ar`.

| Column | Value |
|---|---|
| `n` | 250 |
| `clean_accuracy` | 0.224 |
| `direct_swap_accuracy` | 0.272 |
| `accuracy_delta` | 0.048 |
| `parsed_answer_changed_count` | 183 |
| `parsed_answer_change_rate` | 0.732 |
| `stable_correct` | 38 |
| `S_broken` | 18 |
| `S_repaired` | 30 |
| `stable_wrong` | 164 |
| `stable_wrong_same` | 29 |
| `stable_wrong_different` | 135 |
| `stable_wrong_different_rate` | 0.8231707317073171 (displayed as 0.823) |

**Raw-run inputs**: same as Cluster 4.

**Computation script**: `src/evaluation/compute_main_metrics.py`
(with `--extra-language` flag).

---

## Cluster 6 — Patch control (Section 5.1 Table 1, Appendix B Table 5)

**Paper reference**: Section 5.1, Table 1 (match-baseline counts on n=39);
Appendix B, Table 5 (self-patch match counts).

**Phrases in paper**: "layer 20: same=19/39, shuffled=0/39, random=0/39,
point estimate +0.487, 95% CI [0.333, 0.641]; layer 22: same=19/39,
shuffled=1/39, random=2/39, point estimate +0.462, 95% CI [0.308, 0.615]".

**CSV**: `artifacts/patch_control/bootstrap_ci.csv`.

Columns: `layer`, `comparison`, `point_estimate`, `ci_low`, `ci_high`.

**Raw-run inputs**:
`data/raw_runs/source_specific_patch_control/source_specific_patch_records.jsonl`.
The S\_broken filter file is `artifacts/patch_control/s_broken_ids.txt`
(39 canonical sample numbers).

**Computation script**: `src/patching/patch_control_bootstrap.py`

---

## Cluster 7 — Behavioral diagnostics (Section 5.2)

**Paper reference**: Section 5.2 (candidate margin and numerical trace).

**Phrases in paper**: "mean margin delta +0.006, 95% CI [−0.043, 0.055];
normalized edit distance: answer_changed 0.700, answer_unchanged 0.448,
difference 0.252, 95% CI [0.205, 0.300]".

**CSV**: `artifacts/diagnostics/diagnostics.csv`.

Relevant rows (filter on `diagnostic` and `metric` columns):

| diagnostic | metric | paper value |
|---|---|---|
| `candidate_margin` | `mean_margin_delta` | +0.006 |
| `numeric_trace_divergence` | `changed_mean` | 0.700 |
| `numeric_trace_divergence` | `unchanged_mean` | 0.448 |
| `numeric_trace_divergence` | `difference` | 0.252 |

Rationale-conditioned margins (Table 2) are also in `diagnostics.csv`,
`diagnostic=rationale_conditioned_scoring` rows.

**Raw-run inputs**:
`data/raw_runs/trajectory_rationale_followup/numeric_trace_records.csv` and
`data/raw_runs/trajectory_rationale_followup/rationale_conditioned_margin_records.csv`.

**Computation script**: `src/evaluation/compute_main_metrics.py`
(with `--task diagnostics`).

---

## Cluster 8 — Hidden-state audit (Appendix D Table 7) and repetition robustness (Appendix C Table 6)

**Paper reference**: Appendix D, Table 7 (hidden-state probe); Appendix C,
Table 6 (repetition robustness).

**Phrases in paper**: "best single-layer probe at layer 10, ROC-AUC 0.679;
all-layer probe 0.637; length-only control 0.753".
"After filtering repeated answers (n=226): accuracy delta −0.031,
parsed_answer_change_rate 135/226 = 0.597, stable_wrong_different 70/83 = 0.843".

**CSV (hidden state)**:
`artifacts/hidden_state_appendix/hidden_state_divergence_summary.csv`.

Columns: `best_layer` (best single-layer index), `best_roc_auc` (best
single-layer ROC-AUC, paper-cited 0.679), `all_layer_probe` (JSON with
`roc_auc` key, paper-cited 0.637), `length_only_control` (JSON with
`roc_auc` key, paper-cited 0.753).

**CSV (repetition robustness)**:
`artifacts/repetition_robustness/repetition_robustness.csv`.

Filtered results are in the row `subset=filtered`. Relevant columns for
that row: `n` (226), `accuracy_delta` (−0.031), `parsed_answer_change_rate`
(0.597), `stable_wrong_different` (70), `stable_wrong` (83),
`stable_wrong_different_rate` (0.843).

**Raw-run inputs**:
`data/raw_runs/trajectory_rationale_followup/hidden_state_divergence_records.csv`
and `data/raw_runs/trajectory_rationale_followup/hidden_state_permutation_check.csv`.

**Computation script**: `src/evaluation/compute_main_metrics.py`
(main, diagnostics, and repetition robustness) plus the shipped hidden-state
feature records used to refresh Appendix D summary values.

---

## Caveats

### Legacy internal directory names in `data/raw_runs/`

Subdirectory names such as `run_20260506_142616_fixed256_full`
and `run_20260513_170017` appear inside `data/raw_runs/`; these are immutable
run identifiers assigned at execution time and cannot be renamed without
breaking provenance tracing back to the SOURCE repository.

### Patch-control n=40 to n=39 reconciliation

The raw patch-control run recorded n=40 S\_broken samples (visible in
`data/raw_runs/source_specific_patch_control/source_specific_patch_summary.csv`
as `eq_clean_count=21/40` for layer 20 same-patch). The canonical S\_broken
set under the final normalized-answer evaluator has n=39, defined by the
39 sample numbers in `artifacts/patch_control/s_broken_ids.txt`. The bootstrap CI
script (`src/patching/patch_control_bootstrap.py`) subsets to those 39 samples
before performing paired resampling. Paper match-baseline counts (e.g. 19/39
for layer 20 same-patch) are computed on this n=39 subset under the final
evaluator policy and may therefore differ from the raw `eq_clean_count` values
on n=40.

### Trajectory diagnostics partition

The trajectory/rationale followup raw run
(`data/raw_runs/trajectory_rationale_followup/numeric_trace_summary.csv`)
computes numerical-trace divergence over its own answer\_changed partition of
n=155. The paper reports on n=151, which is the canonical answer\_changed
partition under the signed final-answer evaluator. The final-evaluator
reconciled values are stored in `artifacts/diagnostics/diagnostics.csv`; the
raw-run summary CSV is kept for provenance only and should not be used for
paper-number verification.

### MGSM sample identifier values in artifacts data files

Specific MGSM sample identifier values (e.g. the 227th problem of the MGSM
Chinese subset; Shi et al., 2022) appear as legitimate data-content values in
four artifacts files: `artifacts/patch_control/patch_control_records.jsonl`
(many records with `sample_id` and `shuffled_source_sample_id` fields),
`artifacts/patch_control/s_broken_ids.txt` (entries in the 39-item canonical
S\_broken list), `artifacts/patch_control/s_broken_subset.csv` (rows under the
`sample_id` column), and
`artifacts/repetition_robustness/repetition_robustness.json` (elements in the
sample-identifier array). These data values are intentionally NOT modified for
vocabulary compliance: doing so would corrupt data integrity vs the source
dataset and break the sha256 chain recorded in
`artifacts/COPY_MANIFEST.json`. Additionally,
`artifacts/main/patch_control_caveat.json` contains a
`missing_canonical_sbroken_sample_ids` structured-data array (not narrative
text); this field is likewise left unchanged for the same data-integrity
reason.

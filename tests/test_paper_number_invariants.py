"""Parametrized tests verifying paper-cited numbers against artifacts CSVs.

Each test function corresponds to one paper cluster (1-8). All lookups use
the column-resolution rules from spec Section 11. Tests import constants
from src.evaluation.paper_constants.

Run:
    pytest tests/test_paper_number_invariants.py -v
"""
from __future__ import annotations

import csv
import json
import pathlib
import re

import pytest

from src.evaluation.paper_constants import (
    PATCH_L20_SAME_MATCH_BASELINE_COUNT,
    PATCH_L20_SHUFFLED_MATCH_BASELINE_COUNT,
    PATCH_L20_RANDOM_MATCH_BASELINE_COUNT,
    PATCH_L22_SAME_MATCH_BASELINE_COUNT,
    PATCH_L22_SHUFFLED_MATCH_BASELINE_COUNT,
    PATCH_L22_RANDOM_MATCH_BASELINE_COUNT,
    PATCH_L20_SELF_MATCH_DIRECT_SWAP_COUNT,
    PATCH_L20_SELF_N,
    PATCH_L22_SELF_MATCH_DIRECT_SWAP_COUNT,
    PATCH_L22_SELF_N,
    PATCH_CONTROL_L20_SAME_MINUS_SHUFFLED_RAW,
    PATCH_CONTROL_L20_SAME_MINUS_SHUFFLED_CI_LOW_RAW,
    PATCH_CONTROL_L20_SAME_MINUS_SHUFFLED_CI_HIGH_RAW,
    PATCH_CONTROL_L20_SAME_MINUS_RANDOM_RAW,
    PATCH_CONTROL_L20_SAME_MINUS_RANDOM_CI_LOW_RAW,
    PATCH_CONTROL_L20_SAME_MINUS_RANDOM_CI_HIGH_RAW,
    PATCH_CONTROL_L22_SAME_MINUS_SHUFFLED_RAW,
    PATCH_CONTROL_L22_SAME_MINUS_SHUFFLED_CI_LOW_RAW,
    PATCH_CONTROL_L22_SAME_MINUS_SHUFFLED_CI_HIGH_RAW,
    PATCH_CONTROL_L22_SAME_MINUS_RANDOM_RAW,
    PATCH_CONTROL_L22_SAME_MINUS_RANDOM_CI_LOW_RAW,
    PATCH_CONTROL_L22_SAME_MINUS_RANDOM_CI_HIGH_RAW,
)

ARTIFACTS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "artifacts"

TOL_RAW = 1e-9
TOL_ROUNDED = 1e-4
TOL_CI = 1e-3


def _approx_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _load_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_csv_dict(path: pathlib.Path, key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in _load_csv(path)}


# -----------------------------------------------------------------------
# Cluster 1 — Main Chinese MGSM aggregate
# -----------------------------------------------------------------------

def test_cluster_1_main_chinese_aggregate():
    from src.evaluation.paper_constants import (
        MAIN_ZH_ACCURACY_DELTA,
        MAIN_ZH_ACCURACY_DELTA_CI95_HIGH,
        MAIN_ZH_ACCURACY_DELTA_CI95_LOW,
        MAIN_ZH_CLEAN_ACCURACY,
        MAIN_ZH_CLEAN_CORRECT,
        MAIN_ZH_DIRECT_SWAP_ACCURACY,
        MAIN_ZH_DIRECT_SWAP_CORRECT,
        MAIN_ZH_IDENTITY_ACCURACY,
        MAIN_ZH_IDENTITY_CORRECT,
        MAIN_ZH_N,
    )
    metrics_csv = ARTIFACTS_ROOT / "main" / "main_metrics.csv"
    boot_csv = ARTIFACTS_ROOT / "main" / "bootstrap_ci.csv"
    metrics = _load_csv_dict(metrics_csv, "metric")
    boot = _load_csv_dict(boot_csv, "metric")

    assert int(metrics["n"]["value"]) == MAIN_ZH_N, (
        f"csv={metrics_csv} metric=n expected={MAIN_ZH_N} actual={metrics['n']['value']}"
    )
    assert int(metrics["clean_correct"]["value"]) == MAIN_ZH_CLEAN_CORRECT
    assert int(metrics["direct_swap_correct"]["value"]) == MAIN_ZH_DIRECT_SWAP_CORRECT
    assert int(metrics["identity_correct"]["value"]) == MAIN_ZH_IDENTITY_CORRECT

    for col, expected in [
        ("clean_accuracy", MAIN_ZH_CLEAN_ACCURACY),
        ("direct_swap_accuracy", MAIN_ZH_DIRECT_SWAP_ACCURACY),
        ("identity_accuracy", MAIN_ZH_IDENTITY_ACCURACY),
        ("accuracy_delta", MAIN_ZH_ACCURACY_DELTA),
    ]:
        actual = float(metrics[col]["value"])
        assert _approx_equal(actual, expected, TOL_RAW), (
            f"csv={metrics_csv} metric={col} expected={expected} actual={actual} tol={TOL_RAW}"
        )

    row = boot["accuracy_delta"]
    assert _approx_equal(float(row["ci_low"]), MAIN_ZH_ACCURACY_DELTA_CI95_LOW, TOL_CI), (
        f"csv={boot_csv} metric=accuracy_delta col=ci_low "
        f"expected={MAIN_ZH_ACCURACY_DELTA_CI95_LOW} actual={row['ci_low']} tol={TOL_CI}"
    )
    assert _approx_equal(float(row["ci_high"]), MAIN_ZH_ACCURACY_DELTA_CI95_HIGH, TOL_CI)


# -----------------------------------------------------------------------
# Cluster 2 — Main Chinese answer redistribution
# -----------------------------------------------------------------------

def test_cluster_2_answer_redistribution():
    from src.evaluation.paper_constants import (
        MAIN_ZH_ANSWER_CHANGED,
        MAIN_ZH_PARSED_ANSWER_CHANGE_RATE,
        MAIN_ZH_PARSED_ANSWER_CHANGE_RATE_CI95_HIGH,
        MAIN_ZH_PARSED_ANSWER_CHANGE_RATE_CI95_LOW,
    )
    metrics_csv = ARTIFACTS_ROOT / "main" / "main_metrics.csv"
    boot_csv = ARTIFACTS_ROOT / "main" / "bootstrap_ci.csv"
    metrics = _load_csv_dict(metrics_csv, "metric")
    boot = _load_csv_dict(boot_csv, "metric")

    assert int(metrics["answer_changed"]["value"]) == MAIN_ZH_ANSWER_CHANGED, (
        f"csv={metrics_csv} metric=answer_changed expected={MAIN_ZH_ANSWER_CHANGED}"
    )
    actual_rate = float(metrics["parsed_answer_change_rate"]["value"])
    assert _approx_equal(actual_rate, MAIN_ZH_PARSED_ANSWER_CHANGE_RATE, TOL_RAW)

    row = boot["parsed_answer_change_rate"]
    assert _approx_equal(float(row["ci_low"]), MAIN_ZH_PARSED_ANSWER_CHANGE_RATE_CI95_LOW, TOL_CI)
    assert _approx_equal(float(row["ci_high"]), MAIN_ZH_PARSED_ANSWER_CHANGE_RATE_CI95_HIGH, TOL_CI)


# -----------------------------------------------------------------------
# Cluster 3 — Main Chinese transitions
# -----------------------------------------------------------------------

def test_cluster_3_transitions():
    from src.evaluation.paper_constants import (
        MAIN_ZH_S_BROKEN,
        MAIN_ZH_S_REPAIRED,
        MAIN_ZH_STABLE_CORRECT,
        MAIN_ZH_STABLE_WRONG,
        MAIN_ZH_STABLE_WRONG_DIFFERENT,
        MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_CI95_HIGH_RAW,
        MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_CI95_LOW_RAW,
        MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_RAW,
        MAIN_ZH_STABLE_WRONG_SAME,
    )
    metrics_csv = ARTIFACTS_ROOT / "main" / "main_metrics.csv"
    boot_csv = ARTIFACTS_ROOT / "main" / "bootstrap_ci.csv"
    metrics = _load_csv_dict(metrics_csv, "metric")
    boot = _load_csv_dict(boot_csv, "metric")

    for col, expected in [
        ("stable_correct", MAIN_ZH_STABLE_CORRECT),
        ("S_broken", MAIN_ZH_S_BROKEN),
        ("S_repaired", MAIN_ZH_S_REPAIRED),
        ("stable_wrong", MAIN_ZH_STABLE_WRONG),
        ("stable_wrong_same", MAIN_ZH_STABLE_WRONG_SAME),
        ("stable_wrong_different", MAIN_ZH_STABLE_WRONG_DIFFERENT),
    ]:
        assert int(metrics[col]["value"]) == expected, (
            f"csv={metrics_csv} metric={col} expected={expected} actual={metrics[col]['value']}"
        )

    actual_rate = float(metrics["stable_wrong_different_rate"]["value"])
    assert _approx_equal(actual_rate, MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_RAW, TOL_RAW), (
        f"stable_wrong_different_rate expected={MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_RAW} "
        f"actual={actual_rate} tol={TOL_RAW}"
    )

    ci_metric = "stable_wrong_different_rate_within_stable_wrong"
    if ci_metric in boot:
        row = boot[ci_metric]
        assert _approx_equal(float(row["ci_low"]),
                              MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_CI95_LOW_RAW, TOL_RAW)
        assert _approx_equal(float(row["ci_high"]),
                              MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_CI95_HIGH_RAW, TOL_RAW)


# -----------------------------------------------------------------------
# Cluster 4 — Korean sanity
# -----------------------------------------------------------------------

def test_cluster_4_korean_sanity():
    from src.evaluation.paper_constants import (
        KO_ACCURACY_DELTA,
        KO_ANSWER_CHANGED,
        KO_CLEAN_ACCURACY,
        KO_DIRECT_SWAP_ACCURACY,
        KO_N,
        KO_PARSED_ANSWER_CHANGE_RATE,
        KO_S_BROKEN,
        KO_S_REPAIRED,
        KO_STABLE_CORRECT,
        KO_STABLE_WRONG,
        KO_STABLE_WRONG_DIFFERENT,
        KO_STABLE_WRONG_DIFFERENT_RATE_RAW,
        KO_STABLE_WRONG_SAME,
    )
    metrics_csv = ARTIFACTS_ROOT / "extra_language" / "metrics.csv"
    rows = _load_csv(metrics_csv)
    ko = next(r for r in rows if r["language"] == "ko")

    assert int(ko["n"]) == KO_N
    assert int(ko["S_broken"]) == KO_S_BROKEN
    assert int(ko["S_repaired"]) == KO_S_REPAIRED
    assert int(ko["stable_correct"]) == KO_STABLE_CORRECT
    assert int(ko["stable_wrong"]) == KO_STABLE_WRONG
    assert int(ko["stable_wrong_same"]) == KO_STABLE_WRONG_SAME
    assert int(ko["stable_wrong_different"]) == KO_STABLE_WRONG_DIFFERENT
    assert int(ko["parsed_answer_changed_count"]) == KO_ANSWER_CHANGED

    for col, expected in [
        ("clean_accuracy", KO_CLEAN_ACCURACY),
        ("direct_swap_accuracy", KO_DIRECT_SWAP_ACCURACY),
        ("accuracy_delta", KO_ACCURACY_DELTA),
        ("parsed_answer_change_rate", KO_PARSED_ANSWER_CHANGE_RATE),
        ("stable_wrong_different_rate", KO_STABLE_WRONG_DIFFERENT_RATE_RAW),
    ]:
        actual = float(ko[col])
        assert _approx_equal(actual, expected, TOL_RAW), (
            f"csv={metrics_csv} language=ko column={col} expected={expected} actual={actual}"
        )


# -----------------------------------------------------------------------
# Cluster 5 — Arabic sanity
# -----------------------------------------------------------------------

def test_cluster_5_arabic_sanity():
    from src.evaluation.paper_constants import (
        AR_ACCURACY_DELTA,
        AR_ANSWER_CHANGED,
        AR_CLEAN_ACCURACY,
        AR_DIRECT_SWAP_ACCURACY,
        AR_N,
        AR_PARSED_ANSWER_CHANGE_RATE,
        AR_S_BROKEN,
        AR_S_REPAIRED,
        AR_STABLE_CORRECT,
        AR_STABLE_WRONG,
        AR_STABLE_WRONG_DIFFERENT,
        AR_STABLE_WRONG_DIFFERENT_RATE_RAW,
        AR_STABLE_WRONG_SAME,
    )
    metrics_csv = ARTIFACTS_ROOT / "extra_language" / "metrics.csv"
    rows = _load_csv(metrics_csv)
    ar = next(r for r in rows if r["language"] == "ar")

    assert int(ar["n"]) == AR_N
    assert int(ar["S_broken"]) == AR_S_BROKEN
    assert int(ar["S_repaired"]) == AR_S_REPAIRED
    assert int(ar["stable_correct"]) == AR_STABLE_CORRECT
    assert int(ar["stable_wrong"]) == AR_STABLE_WRONG
    assert int(ar["stable_wrong_same"]) == AR_STABLE_WRONG_SAME
    assert int(ar["stable_wrong_different"]) == AR_STABLE_WRONG_DIFFERENT
    assert int(ar["parsed_answer_changed_count"]) == AR_ANSWER_CHANGED

    for col, expected in [
        ("clean_accuracy", AR_CLEAN_ACCURACY),
        ("direct_swap_accuracy", AR_DIRECT_SWAP_ACCURACY),
        ("accuracy_delta", AR_ACCURACY_DELTA),
        ("parsed_answer_change_rate", AR_PARSED_ANSWER_CHANGE_RATE),
        ("stable_wrong_different_rate", AR_STABLE_WRONG_DIFFERENT_RATE_RAW),
    ]:
        actual = float(ar[col])
        assert _approx_equal(actual, expected, TOL_RAW), (
            f"csv={metrics_csv} language=ar column={col} expected={expected} actual={actual}"
        )


# -----------------------------------------------------------------------
# Cluster 6 — Patch control
# -----------------------------------------------------------------------

@pytest.mark.parametrize("layer,comparison,exp_point,exp_low,exp_high", [
    pytest.param(
        20, "same_sample_minus_shuffled",
        PATCH_CONTROL_L20_SAME_MINUS_SHUFFLED_RAW,
        PATCH_CONTROL_L20_SAME_MINUS_SHUFFLED_CI_LOW_RAW,
        PATCH_CONTROL_L20_SAME_MINUS_SHUFFLED_CI_HIGH_RAW,
        id="L20_same_minus_shuffled",
    ),
    pytest.param(
        20, "same_sample_minus_random",
        PATCH_CONTROL_L20_SAME_MINUS_RANDOM_RAW,
        PATCH_CONTROL_L20_SAME_MINUS_RANDOM_CI_LOW_RAW,
        PATCH_CONTROL_L20_SAME_MINUS_RANDOM_CI_HIGH_RAW,
        id="L20_same_minus_random",
    ),
    pytest.param(
        22, "same_sample_minus_shuffled",
        PATCH_CONTROL_L22_SAME_MINUS_SHUFFLED_RAW,
        PATCH_CONTROL_L22_SAME_MINUS_SHUFFLED_CI_LOW_RAW,
        PATCH_CONTROL_L22_SAME_MINUS_SHUFFLED_CI_HIGH_RAW,
        id="L22_same_minus_shuffled",
    ),
    pytest.param(
        22, "same_sample_minus_random",
        PATCH_CONTROL_L22_SAME_MINUS_RANDOM_RAW,
        PATCH_CONTROL_L22_SAME_MINUS_RANDOM_CI_LOW_RAW,
        PATCH_CONTROL_L22_SAME_MINUS_RANDOM_CI_HIGH_RAW,
        id="L22_same_minus_random",
    ),
])
def test_cluster_6_patch_control_ci(layer, comparison, exp_point, exp_low, exp_high):
    from src.evaluation.paper_constants import PATCH_S_BROKEN_N
    boot_csv = ARTIFACTS_ROOT / "patch_control" / "bootstrap_ci.csv"
    rows = _load_csv(boot_csv)
    row = next(
        (r for r in rows if int(r["layer"]) == layer and r["comparison"] == comparison),
        None,
    )
    assert row is not None, (
        f"csv={boot_csv} (layer={layer}, comparison={comparison!r}) not found"
    )
    assert int(row["n_sample_ids"]) == PATCH_S_BROKEN_N, (
        f"n_sample_ids expected={PATCH_S_BROKEN_N} actual={row['n_sample_ids']}"
    )
    assert _approx_equal(float(row["point_estimate"]), exp_point, TOL_RAW), (
        f"(layer={layer}, comparison={comparison!r}) "
        f"point_estimate expected={exp_point} actual={row['point_estimate']}"
    )
    assert _approx_equal(float(row["ci_low"]), exp_low, TOL_RAW)
    assert _approx_equal(float(row["ci_high"]), exp_high, TOL_RAW)


@pytest.mark.parametrize("condition,layer,col,expected", [
    pytest.param("same_sample_clean_patch_L20", 20, "match_baseline_count",
                 PATCH_L20_SAME_MATCH_BASELINE_COUNT, id="L20_same_baseline"),
    pytest.param("shuffled_clean_patch_L20", 20, "match_baseline_count",
                 PATCH_L20_SHUFFLED_MATCH_BASELINE_COUNT, id="L20_shuffled_baseline"),
    pytest.param("random_norm_matched_patch_L20", 20, "match_baseline_count",
                 PATCH_L20_RANDOM_MATCH_BASELINE_COUNT, id="L20_random_baseline"),
    pytest.param("same_sample_clean_patch_L22", 22, "match_baseline_count",
                 PATCH_L22_SAME_MATCH_BASELINE_COUNT, id="L22_same_baseline"),
    pytest.param("shuffled_clean_patch_L22", 22, "match_baseline_count",
                 PATCH_L22_SHUFFLED_MATCH_BASELINE_COUNT, id="L22_shuffled_baseline"),
    pytest.param("random_norm_matched_patch_L22", 22, "match_baseline_count",
                 PATCH_L22_RANDOM_MATCH_BASELINE_COUNT, id="L22_random_baseline"),
    pytest.param("hard_self_patch_L20", 20, "match_direct_swap_count",
                 PATCH_L20_SELF_MATCH_DIRECT_SWAP_COUNT, id="L20_self_swap"),
    pytest.param("hard_self_patch_L20", 20, "n",
                 PATCH_L20_SELF_N, id="L20_self_n"),
    pytest.param("hard_self_patch_L22", 22, "match_direct_swap_count",
                 PATCH_L22_SELF_MATCH_DIRECT_SWAP_COUNT, id="L22_self_swap"),
    pytest.param("hard_self_patch_L22", 22, "n",
                 PATCH_L22_SELF_N, id="L22_self_n"),
])
def test_cluster_6_patch_summary(condition, layer, col, expected):
    summary_csv = ARTIFACTS_ROOT / "patch_control" / "patch_control_summary.csv"
    rows = _load_csv(summary_csv)
    row = next(
        (r for r in rows if r["condition"] == condition and
         (r["layer"] == str(layer) or (not r["layer"] and layer == -1))),
        None,
    )
    assert row is not None, (
        f"csv={summary_csv} (condition={condition!r}, layer={layer}) not found"
    )
    assert int(row[col]) == expected, (
        f"csv={summary_csv} condition={condition!r} layer={layer} col={col} "
        f"expected={expected} actual={row[col]}"
    )


# -----------------------------------------------------------------------
# Cluster 7 — Behavioral diagnostics
# -----------------------------------------------------------------------

def test_cluster_7_diagnostics():
    from src.evaluation.paper_constants import (
        DIAG_CANDIDATE_MARGIN_CI95_HIGH_RAW,
        DIAG_CANDIDATE_MARGIN_CI95_LOW_RAW,
        DIAG_CANDIDATE_MARGIN_MEAN_DELTA_RAW,
        DIAG_NUMTRACE_ANSWER_CHANGED_MEAN_RAW,
        DIAG_NUMTRACE_ANSWER_UNCHANGED_MEAN_RAW,
        DIAG_NUMTRACE_DIFFERENCE_CI95_HIGH_RAW,
        DIAG_NUMTRACE_DIFFERENCE_CI95_LOW_RAW,
        DIAG_NUMTRACE_DIFFERENCE_RAW,
        DIAG_RAT_BASELINE_BASELINE_CI95_HIGH_RAW,
        DIAG_RAT_BASELINE_BASELINE_CI95_LOW_RAW,
        DIAG_RAT_BASELINE_BASELINE_MEAN_RAW,
        DIAG_RAT_BASELINE_SWAP_CI95_HIGH_RAW,
        DIAG_RAT_BASELINE_SWAP_CI95_LOW_RAW,
        DIAG_RAT_BASELINE_SWAP_MEAN_RAW,
        DIAG_RAT_SWAP_BASELINE_CI95_HIGH_RAW,
        DIAG_RAT_SWAP_BASELINE_CI95_LOW_RAW,
        DIAG_RAT_SWAP_BASELINE_MEAN_RAW,
        DIAG_RAT_SWAP_SWAP_CI95_HIGH_RAW,
        DIAG_RAT_SWAP_SWAP_CI95_LOW_RAW,
        DIAG_RAT_SWAP_SWAP_MEAN_RAW,
    )
    diag_csv = ARTIFACTS_ROOT / "diagnostics" / "diagnostics.csv"
    rows = _load_csv(diag_csv)
    by_key = {(r["diagnostic"], r["metric"]): r for r in rows}

    key_cm = ("candidate_margin", "mean_margin_delta")
    assert key_cm in by_key, f"Missing {key_cm} in {diag_csv}"
    row = by_key[key_cm]
    assert _approx_equal(float(row["value"]), DIAG_CANDIDATE_MARGIN_MEAN_DELTA_RAW, TOL_RAW)
    assert _approx_equal(float(row["ci_low"]), DIAG_CANDIDATE_MARGIN_CI95_LOW_RAW, TOL_RAW)
    assert _approx_equal(float(row["ci_high"]), DIAG_CANDIDATE_MARGIN_CI95_HIGH_RAW, TOL_RAW)

    for diag, metric, exp_val in [
        ("numeric_trace_divergence", "changed_mean", DIAG_NUMTRACE_ANSWER_CHANGED_MEAN_RAW),
        ("numeric_trace_divergence", "unchanged_mean", DIAG_NUMTRACE_ANSWER_UNCHANGED_MEAN_RAW),
        ("numeric_trace_divergence", "difference", DIAG_NUMTRACE_DIFFERENCE_RAW),
    ]:
        key = (diag, metric)
        assert key in by_key, f"Missing {key} in {diag_csv}"
        actual = float(by_key[key]["value"])
        assert _approx_equal(actual, exp_val, TOL_RAW), (
            f"({diag!r}, {metric!r}) expected={exp_val} actual={actual}"
        )

    diff_key = ("numeric_trace_divergence", "difference")
    assert _approx_equal(float(by_key[diff_key]["ci_low"]),
                         DIAG_NUMTRACE_DIFFERENCE_CI95_LOW_RAW, TOL_RAW)
    assert _approx_equal(float(by_key[diff_key]["ci_high"]),
                         DIAG_NUMTRACE_DIFFERENCE_CI95_HIGH_RAW, TOL_RAW)

    rationale_checks = [
        (re.compile(r"^clean_model_clean_rationale_.*margin", re.IGNORECASE),
         DIAG_RAT_BASELINE_BASELINE_MEAN_RAW,
         DIAG_RAT_BASELINE_BASELINE_CI95_LOW_RAW,
         DIAG_RAT_BASELINE_BASELINE_CI95_HIGH_RAW),
        (re.compile(r"^clean_model_swap_rationale_.*margin", re.IGNORECASE),
         DIAG_RAT_BASELINE_SWAP_MEAN_RAW,
         DIAG_RAT_BASELINE_SWAP_CI95_LOW_RAW,
         DIAG_RAT_BASELINE_SWAP_CI95_HIGH_RAW),
        (re.compile(r"^direct_swap_model_clean_rationale_.*margin", re.IGNORECASE),
         DIAG_RAT_SWAP_BASELINE_MEAN_RAW,
         DIAG_RAT_SWAP_BASELINE_CI95_LOW_RAW,
         DIAG_RAT_SWAP_BASELINE_CI95_HIGH_RAW),
        (re.compile(r"^direct_swap_model_swap_rationale_.*margin", re.IGNORECASE),
         DIAG_RAT_SWAP_SWAP_MEAN_RAW,
         DIAG_RAT_SWAP_SWAP_CI95_LOW_RAW,
         DIAG_RAT_SWAP_SWAP_CI95_HIGH_RAW),
    ]
    for pat, exp_val, exp_low, exp_high in rationale_checks:
        matching = [r for r in rows
                    if r["diagnostic"] == "rationale_conditioned_scoring"
                    and pat.match(r["metric"])]
        assert len(matching) == 1, (
            f"Pattern {pat.pattern!r} matched {len(matching)} rows in {diag_csv}"
        )
        row = matching[0]
        assert _approx_equal(float(row["value"]), exp_val, TOL_RAW), (
            f"pattern={pat.pattern!r} expected={exp_val} actual={row['value']}"
        )
        assert _approx_equal(float(row["ci_low"]), exp_low, TOL_RAW)
        assert _approx_equal(float(row["ci_high"]), exp_high, TOL_RAW)


# -----------------------------------------------------------------------
# Cluster 8 — Hidden state + repetition robustness
# -----------------------------------------------------------------------

def test_cluster_8_hidden_state_and_repetition():
    from src.evaluation.paper_constants import (
        HIDDEN_ALL_LAYER_PROBE_ROC_AUC_RAW,
        HIDDEN_BEST_SINGLE_LAYER,
        HIDDEN_BEST_SINGLE_LAYER_ROC_AUC_RAW,
        HIDDEN_LENGTH_ONLY_CONTROL_ROC_AUC_RAW,
        REPETITION_FILTERED_ACCURACY_DELTA_RAW,
        REPETITION_FILTERED_CHANGE_RATE_DEN,
        REPETITION_FILTERED_CHANGE_RATE_NUM,
        REPETITION_FILTERED_CHANGE_RATE_RAW,
        REPETITION_FILTERED_N,
        REPETITION_FILTERED_STABLE_WRONG_DIFFERENT_DEN,
        REPETITION_FILTERED_STABLE_WRONG_DIFFERENT_NUM,
        REPETITION_FILTERED_STABLE_WRONG_DIFFERENT_RATE_RAW,
    )
    hs_csv = ARTIFACTS_ROOT / "hidden_state_appendix" / "hidden_state_divergence_summary.csv"
    hs_rows = _load_csv(hs_csv)
    assert hs_rows, f"Empty file: {hs_csv}"
    row = hs_rows[0]

    assert int(row["best_layer"]) == HIDDEN_BEST_SINGLE_LAYER, (
        f"best_layer expected={HIDDEN_BEST_SINGLE_LAYER} actual={row['best_layer']}"
    )
    assert _approx_equal(float(row["best_roc_auc"]),
                         HIDDEN_BEST_SINGLE_LAYER_ROC_AUC_RAW, TOL_RAW)

    all_layer = json.loads(row["all_layer_probe"])
    assert _approx_equal(float(all_layer["roc_auc"]),
                         HIDDEN_ALL_LAYER_PROBE_ROC_AUC_RAW, TOL_RAW), (
        f"all_layer_probe.roc_auc expected={HIDDEN_ALL_LAYER_PROBE_ROC_AUC_RAW}"
    )
    len_ctrl = json.loads(row["length_only_control"])
    assert _approx_equal(float(len_ctrl["roc_auc"]),
                         HIDDEN_LENGTH_ONLY_CONTROL_ROC_AUC_RAW, TOL_RAW)

    rep_csv = ARTIFACTS_ROOT / "repetition_robustness" / "repetition_robustness.csv"
    rep_rows = _load_csv(rep_csv)
    filtered = next((r for r in rep_rows if r["subset"] == "filtered"), None)
    assert filtered is not None, f"subset=filtered not found in {rep_csv}"

    assert int(filtered["n"]) == REPETITION_FILTERED_N
    assert int(filtered["answer_changed"]) == REPETITION_FILTERED_CHANGE_RATE_NUM
    assert int(filtered["n"]) == REPETITION_FILTERED_CHANGE_RATE_DEN
    assert int(filtered["stable_wrong_different"]) == REPETITION_FILTERED_STABLE_WRONG_DIFFERENT_NUM
    assert int(filtered["stable_wrong"]) == REPETITION_FILTERED_STABLE_WRONG_DIFFERENT_DEN

    assert _approx_equal(float(filtered["accuracy_delta"]),
                         REPETITION_FILTERED_ACCURACY_DELTA_RAW, TOL_RAW)
    assert _approx_equal(float(filtered["parsed_answer_change_rate"]),
                         REPETITION_FILTERED_CHANGE_RATE_RAW, TOL_RAW)
    assert _approx_equal(float(filtered["stable_wrong_different_rate"]),
                         REPETITION_FILTERED_STABLE_WRONG_DIFFERENT_RATE_RAW, TOL_RAW)

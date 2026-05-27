"""Verify that artifacts/  CSVs reproduce every paper-cited number.

Usage:
    python -m src.evaluation.verify_paper_numbers
    python -m src.evaluation.verify_paper_numbers --artifacts-root /path/to/artifacts

Exit 0 on full pass, 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from typing import List, Tuple

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
    HIDDEN_ALL_LAYER_PROBE_ROC_AUC_RAW,
    HIDDEN_BEST_SINGLE_LAYER,
    HIDDEN_BEST_SINGLE_LAYER_ROC_AUC_RAW,
    HIDDEN_LENGTH_ONLY_CONTROL_ROC_AUC_RAW,
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
    MAIN_ZH_ACCURACY_DELTA,
    MAIN_ZH_ACCURACY_DELTA_CI95_HIGH,
    MAIN_ZH_ACCURACY_DELTA_CI95_LOW,
    MAIN_ZH_ANSWER_CHANGED,
    MAIN_ZH_CLEAN_ACCURACY,
    MAIN_ZH_CLEAN_CORRECT,
    MAIN_ZH_DIRECT_SWAP_ACCURACY,
    MAIN_ZH_DIRECT_SWAP_CORRECT,
    MAIN_ZH_IDENTITY_ACCURACY,
    MAIN_ZH_IDENTITY_CORRECT,
    MAIN_ZH_N,
    MAIN_ZH_PARSED_ANSWER_CHANGE_RATE,
    MAIN_ZH_PARSED_ANSWER_CHANGE_RATE_CI95_HIGH,
    MAIN_ZH_PARSED_ANSWER_CHANGE_RATE_CI95_LOW,
    MAIN_ZH_S_BROKEN,
    MAIN_ZH_S_REPAIRED,
    MAIN_ZH_STABLE_CORRECT,
    MAIN_ZH_STABLE_WRONG,
    MAIN_ZH_STABLE_WRONG_DIFFERENT,
    MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_RAW,
    MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_CI95_HIGH_RAW,
    MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_CI95_LOW_RAW,
    MAIN_ZH_STABLE_WRONG_SAME,
    PATCH_L20_RANDOM_MATCH_BASELINE_COUNT,
    PATCH_L20_SAME_MATCH_BASELINE_COUNT,
    PATCH_L20_SAME_MINUS_RANDOM_CI95_HIGH_RAW,
    PATCH_L20_SAME_MINUS_RANDOM_CI95_LOW_RAW,
    PATCH_L20_SAME_MINUS_RANDOM_POINT_RAW,
    PATCH_L20_SAME_MINUS_SHUFFLED_CI95_HIGH_RAW,
    PATCH_L20_SAME_MINUS_SHUFFLED_CI95_LOW_RAW,
    PATCH_L20_SAME_MINUS_SHUFFLED_POINT_RAW,
    PATCH_L20_SELF_MATCH_DIRECT_SWAP_COUNT,
    PATCH_L20_SELF_N,
    PATCH_L20_SHUFFLED_MATCH_BASELINE_COUNT,
    PATCH_L22_RANDOM_MATCH_BASELINE_COUNT,
    PATCH_L22_SAME_MATCH_BASELINE_COUNT,
    PATCH_L22_SAME_MINUS_RANDOM_CI95_HIGH_RAW,
    PATCH_L22_SAME_MINUS_RANDOM_CI95_LOW_RAW,
    PATCH_L22_SAME_MINUS_RANDOM_POINT_RAW,
    PATCH_L22_SAME_MINUS_SHUFFLED_CI95_HIGH_RAW,
    PATCH_L22_SAME_MINUS_SHUFFLED_CI95_LOW_RAW,
    PATCH_L22_SAME_MINUS_SHUFFLED_POINT_RAW,
    PATCH_L22_SELF_MATCH_DIRECT_SWAP_COUNT,
    PATCH_L22_SELF_N,
    PATCH_L22_SHUFFLED_MATCH_BASELINE_COUNT,
    PATCH_S_BROKEN_N,
    REPETITION_FILTERED_ACCURACY_DELTA_RAW,
    REPETITION_FILTERED_CHANGE_RATE_DEN,
    REPETITION_FILTERED_CHANGE_RATE_NUM,
    REPETITION_FILTERED_CHANGE_RATE_RAW,
    REPETITION_FILTERED_N,
    REPETITION_FILTERED_STABLE_WRONG_DIFFERENT_DEN,
    REPETITION_FILTERED_STABLE_WRONG_DIFFERENT_NUM,
    REPETITION_FILTERED_STABLE_WRONG_DIFFERENT_RATE_RAW,
)

TOL_RAW = 1e-9
TOL_ROUNDED = 1e-4
TOL_CI = 1e-3


def _approx_equal(a: float, b: float, tol: float) -> bool:
    """Return True if |a - b| <= tol."""
    return abs(a - b) <= tol


def _load_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_csv_as_dict(path: pathlib.Path, key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in _load_csv(path)}


def _f(val: str) -> float:
    return float(val)


def _i(val: str) -> int:
    return int(val)


def _fmt_mismatch(
    cluster: int,
    csv_path: pathlib.Path,
    row_key: str,
    column: str,
    expected: float,
    actual: float,
    tol: float,
) -> str:
    return (
        f"MISMATCH cluster={cluster} csv={csv_path} row_key={row_key!r} "
        f"column={column!r} expected={expected!r} actual={actual!r} tol={tol}"
    )


def _check_cluster_1(artifacts_root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    metrics_csv = artifacts_root / "main" / "main_metrics.csv"
    boot_csv = artifacts_root / "main" / "bootstrap_ci.csv"
    metrics = _load_csv_as_dict(metrics_csv, "metric")
    boot = _load_csv_as_dict(boot_csv, "metric")

    checks_int = [
        ("n", MAIN_ZH_N),
        ("clean_correct", MAIN_ZH_CLEAN_CORRECT),
        ("direct_swap_correct", MAIN_ZH_DIRECT_SWAP_CORRECT),
        ("identity_correct", MAIN_ZH_IDENTITY_CORRECT),
    ]
    for col, expected in checks_int:
        if col not in metrics:
            errors.append(f"MISSING cluster=1 csv={metrics_csv} metric={col!r}")
            continue
        actual = _i(metrics[col]["value"])
        if actual != expected:
            errors.append(_fmt_mismatch(1, metrics_csv, col, "value", expected, actual, 0))

    checks_float = [
        ("clean_accuracy", MAIN_ZH_CLEAN_ACCURACY),
        ("direct_swap_accuracy", MAIN_ZH_DIRECT_SWAP_ACCURACY),
        ("identity_accuracy", MAIN_ZH_IDENTITY_ACCURACY),
        ("accuracy_delta", MAIN_ZH_ACCURACY_DELTA),
    ]
    for col, expected in checks_float:
        if col not in metrics:
            errors.append(f"MISSING cluster=1 csv={metrics_csv} metric={col!r}")
            continue
        actual = _f(metrics[col]["value"])
        if not _approx_equal(actual, expected, TOL_RAW):
            errors.append(_fmt_mismatch(1, metrics_csv, col, "value", expected, actual, TOL_RAW))

    # CI for accuracy_delta
    if "accuracy_delta" not in boot:
        errors.append(f"MISSING cluster=1 csv={boot_csv} metric='accuracy_delta'")
    else:
        row = boot["accuracy_delta"]
        for col, expected in [
            ("ci_low", MAIN_ZH_ACCURACY_DELTA_CI95_LOW),
            ("ci_high", MAIN_ZH_ACCURACY_DELTA_CI95_HIGH),
        ]:
            actual = _f(row[col])
            if not _approx_equal(actual, expected, TOL_CI):
                errors.append(_fmt_mismatch(1, boot_csv, "accuracy_delta", col, expected, actual, TOL_CI))

    return errors


def _check_cluster_2(artifacts_root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    metrics_csv = artifacts_root / "main" / "main_metrics.csv"
    boot_csv = artifacts_root / "main" / "bootstrap_ci.csv"
    metrics = _load_csv_as_dict(metrics_csv, "metric")
    boot = _load_csv_as_dict(boot_csv, "metric")

    if "answer_changed" not in metrics:
        errors.append(f"MISSING cluster=2 csv={metrics_csv} metric='answer_changed'")
    else:
        actual = _i(metrics["answer_changed"]["value"])
        if actual != MAIN_ZH_ANSWER_CHANGED:
            errors.append(_fmt_mismatch(2, metrics_csv, "answer_changed", "value", MAIN_ZH_ANSWER_CHANGED, actual, 0))

    if "parsed_answer_change_rate" not in metrics:
        errors.append(f"MISSING cluster=2 csv={metrics_csv} metric='parsed_answer_change_rate'")
    else:
        actual = _f(metrics["parsed_answer_change_rate"]["value"])
        if not _approx_equal(actual, MAIN_ZH_PARSED_ANSWER_CHANGE_RATE, TOL_RAW):
            errors.append(_fmt_mismatch(2, metrics_csv, "parsed_answer_change_rate", "value",
                                        MAIN_ZH_PARSED_ANSWER_CHANGE_RATE, actual, TOL_RAW))

    if "parsed_answer_change_rate" in boot:
        row = boot["parsed_answer_change_rate"]
        for col, expected in [
            ("ci_low", MAIN_ZH_PARSED_ANSWER_CHANGE_RATE_CI95_LOW),
            ("ci_high", MAIN_ZH_PARSED_ANSWER_CHANGE_RATE_CI95_HIGH),
        ]:
            actual = _f(row[col])
            if not _approx_equal(actual, expected, TOL_CI):
                errors.append(_fmt_mismatch(2, boot_csv, "parsed_answer_change_rate", col, expected, actual, TOL_CI))

    return errors


def _check_cluster_3(artifacts_root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    metrics_csv = artifacts_root / "main" / "main_metrics.csv"
    boot_csv = artifacts_root / "main" / "bootstrap_ci.csv"
    metrics = _load_csv_as_dict(metrics_csv, "metric")
    boot = _load_csv_as_dict(boot_csv, "metric")

    checks_int = [
        ("stable_correct", MAIN_ZH_STABLE_CORRECT),
        ("S_broken", MAIN_ZH_S_BROKEN),
        ("S_repaired", MAIN_ZH_S_REPAIRED),
        ("stable_wrong", MAIN_ZH_STABLE_WRONG),
        ("stable_wrong_same", MAIN_ZH_STABLE_WRONG_SAME),
        ("stable_wrong_different", MAIN_ZH_STABLE_WRONG_DIFFERENT),
    ]
    for col, expected in checks_int:
        if col not in metrics:
            errors.append(f"MISSING cluster=3 csv={metrics_csv} metric={col!r}")
            continue
        actual = _i(metrics[col]["value"])
        if actual != expected:
            errors.append(_fmt_mismatch(3, metrics_csv, col, "value", expected, actual, 0))

    if "stable_wrong_different_rate" in metrics:
        actual = _f(metrics["stable_wrong_different_rate"]["value"])
        if not _approx_equal(actual, MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_RAW, TOL_RAW):
            errors.append(_fmt_mismatch(3, metrics_csv, "stable_wrong_different_rate", "value",
                                        MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_RAW, actual, TOL_RAW))

    ci_metric = "stable_wrong_different_rate_within_stable_wrong"
    if ci_metric in boot:
        row = boot[ci_metric]
        for col, expected in [
            ("ci_low", MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_CI95_LOW_RAW),
            ("ci_high", MAIN_ZH_STABLE_WRONG_DIFFERENT_RATE_CI95_HIGH_RAW),
        ]:
            actual = _f(row[col])
            if not _approx_equal(actual, expected, TOL_RAW):
                errors.append(_fmt_mismatch(3, boot_csv, ci_metric, col, expected, actual, TOL_RAW))

    return errors


def _check_extra_language_lang(
    artifacts_root: pathlib.Path,
    lang: str,
    cluster_id: int,
    checks: dict,
) -> list[str]:
    """Shared helper for cluster 4 (KO) and cluster 5 (AR) point-estimate validation.

    Master prompt §2 Clusters 4-5 cite point estimates only; CIs not asserted.
    """
    errors: list[str] = []
    metrics_csv = artifacts_root / "extra_language" / "metrics.csv"
    metrics_rows = _load_csv(metrics_csv)
    lang_metrics = {r["language"]: r for r in metrics_rows}

    int_cols = {"n", "clean_correct", "direct_swap_correct", "identity_correct",
                "parsed_answer_changed_count", "stable_correct", "S_broken", "S_repaired",
                "stable_wrong", "stable_wrong_same", "stable_wrong_different"}

    if lang not in lang_metrics:
        errors.append(f"MISSING cluster={cluster_id} csv={metrics_csv} language={lang!r}")
        return errors

    row = lang_metrics[lang]
    for col, expected in checks.items():
        if col not in row:
            errors.append(
                f"MISSING cluster={cluster_id} csv={metrics_csv} language={lang!r} column={col!r}"
            )
            continue
        if col in int_cols:
            actual = _i(row[col])
            if actual != int(expected):
                errors.append(
                    _fmt_mismatch(cluster_id, metrics_csv, f"language={lang}", col, expected, actual, 0)
                )
        else:
            actual = _f(row[col])
            if not _approx_equal(actual, expected, TOL_RAW):
                errors.append(
                    _fmt_mismatch(cluster_id, metrics_csv, f"language={lang}", col, expected, actual, TOL_RAW)
                )
    return errors


def _check_cluster_4(artifacts_root: pathlib.Path) -> list[str]:
    """Korean sanity checks (Cluster 4).

    Master prompt §2 Clusters 4-5 cite point estimates only; CIs not asserted.
    """
    return _check_extra_language_lang(
        artifacts_root,
        lang="ko",
        cluster_id=4,
        checks={
            "n": KO_N,
            "clean_accuracy": KO_CLEAN_ACCURACY,
            "direct_swap_accuracy": KO_DIRECT_SWAP_ACCURACY,
            "accuracy_delta": KO_ACCURACY_DELTA,
            "parsed_answer_changed_count": KO_ANSWER_CHANGED,
            "parsed_answer_change_rate": KO_PARSED_ANSWER_CHANGE_RATE,
            "stable_correct": KO_STABLE_CORRECT,
            "S_broken": KO_S_BROKEN,
            "S_repaired": KO_S_REPAIRED,
            "stable_wrong": KO_STABLE_WRONG,
            "stable_wrong_same": KO_STABLE_WRONG_SAME,
            "stable_wrong_different": KO_STABLE_WRONG_DIFFERENT,
            "stable_wrong_different_rate": KO_STABLE_WRONG_DIFFERENT_RATE_RAW,
        },
    )


def _check_cluster_5(artifacts_root: pathlib.Path) -> list[str]:
    """Arabic sanity checks (Cluster 5).

    Master prompt §2 Clusters 4-5 cite point estimates only; CIs not asserted.
    """
    return _check_extra_language_lang(
        artifacts_root,
        lang="ar",
        cluster_id=5,
        checks={
            "n": AR_N,
            "clean_accuracy": AR_CLEAN_ACCURACY,
            "direct_swap_accuracy": AR_DIRECT_SWAP_ACCURACY,
            "accuracy_delta": AR_ACCURACY_DELTA,
            "parsed_answer_changed_count": AR_ANSWER_CHANGED,
            "parsed_answer_change_rate": AR_PARSED_ANSWER_CHANGE_RATE,
            "stable_correct": AR_STABLE_CORRECT,
            "S_broken": AR_S_BROKEN,
            "S_repaired": AR_S_REPAIRED,
            "stable_wrong": AR_STABLE_WRONG,
            "stable_wrong_same": AR_STABLE_WRONG_SAME,
            "stable_wrong_different": AR_STABLE_WRONG_DIFFERENT,
            "stable_wrong_different_rate": AR_STABLE_WRONG_DIFFERENT_RATE_RAW,
        },
    )


def _check_cluster_6(artifacts_root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    boot_csv = artifacts_root / "patch_control" / "bootstrap_ci.csv"
    summary_csv = artifacts_root / "patch_control" / "patch_control_summary.csv"
    boot_rows = _load_csv(boot_csv)
    summary_rows = _load_csv(summary_csv)

    boot_by_key = {(int(r["layer"]), r["comparison"]): r for r in boot_rows}
    summary_by_key = {(r["condition"], int(r["layer"]) if r["layer"] else -1): r for r in summary_rows}

    boot_checks = [
        (20, "same_sample_minus_shuffled",
         PATCH_L20_SAME_MINUS_SHUFFLED_POINT_RAW,
         PATCH_L20_SAME_MINUS_SHUFFLED_CI95_LOW_RAW,
         PATCH_L20_SAME_MINUS_SHUFFLED_CI95_HIGH_RAW),
        (20, "same_sample_minus_random",
         PATCH_L20_SAME_MINUS_RANDOM_POINT_RAW,
         PATCH_L20_SAME_MINUS_RANDOM_CI95_LOW_RAW,
         PATCH_L20_SAME_MINUS_RANDOM_CI95_HIGH_RAW),
        (22, "same_sample_minus_shuffled",
         PATCH_L22_SAME_MINUS_SHUFFLED_POINT_RAW,
         PATCH_L22_SAME_MINUS_SHUFFLED_CI95_LOW_RAW,
         PATCH_L22_SAME_MINUS_SHUFFLED_CI95_HIGH_RAW),
        (22, "same_sample_minus_random",
         PATCH_L22_SAME_MINUS_RANDOM_POINT_RAW,
         PATCH_L22_SAME_MINUS_RANDOM_CI95_LOW_RAW,
         PATCH_L22_SAME_MINUS_RANDOM_CI95_HIGH_RAW),
    ]
    for layer, comparison, exp_point, exp_low, exp_high in boot_checks:
        key = (layer, comparison)
        if key not in boot_by_key:
            errors.append(f"MISSING cluster=6 csv={boot_csv} (layer={layer}, comparison={comparison!r})")
            continue
        row = boot_by_key[key]
        if _i(row["n_sample_ids"]) != PATCH_S_BROKEN_N:
            errors.append(_fmt_mismatch(6, boot_csv, f"(layer={layer}, comparison={comparison!r})",
                                        "n_sample_ids", PATCH_S_BROKEN_N, _i(row["n_sample_ids"]), 0))
        for col, expected in [
            ("point_estimate", exp_point),
            ("ci_low", exp_low),
            ("ci_high", exp_high),
        ]:
            actual = _f(row[col])
            if not _approx_equal(actual, expected, TOL_RAW):
                errors.append(_fmt_mismatch(6, boot_csv, f"(layer={layer}, comparison={comparison!r})",
                                            col, expected, actual, TOL_RAW))

    summary_checks = [
        ("same_sample_clean_patch_L20", 20, "match_baseline_count", PATCH_L20_SAME_MATCH_BASELINE_COUNT),
        ("shuffled_clean_patch_L20", 20, "match_baseline_count", PATCH_L20_SHUFFLED_MATCH_BASELINE_COUNT),
        ("random_norm_matched_patch_L20", 20, "match_baseline_count", PATCH_L20_RANDOM_MATCH_BASELINE_COUNT),
        ("same_sample_clean_patch_L22", 22, "match_baseline_count", PATCH_L22_SAME_MATCH_BASELINE_COUNT),
        ("shuffled_clean_patch_L22", 22, "match_baseline_count", PATCH_L22_SHUFFLED_MATCH_BASELINE_COUNT),
        ("random_norm_matched_patch_L22", 22, "match_baseline_count", PATCH_L22_RANDOM_MATCH_BASELINE_COUNT),
        ("hard_self_patch_L20", 20, "match_direct_swap_count", PATCH_L20_SELF_MATCH_DIRECT_SWAP_COUNT),
        ("hard_self_patch_L20", 20, "n", PATCH_L20_SELF_N),
        ("hard_self_patch_L22", 22, "match_direct_swap_count", PATCH_L22_SELF_MATCH_DIRECT_SWAP_COUNT),
        ("hard_self_patch_L22", 22, "n", PATCH_L22_SELF_N),
    ]
    for condition, layer, col, expected in summary_checks:
        key = (condition, layer)
        if key not in summary_by_key:
            errors.append(f"MISSING cluster=6 csv={summary_csv} (condition={condition!r}, layer={layer})")
            continue
        actual = _i(summary_by_key[key][col])
        if actual != expected:
            errors.append(_fmt_mismatch(6, summary_csv, f"(condition={condition!r}, layer={layer})",
                                        col, expected, actual, 0))

    return errors


def _check_cluster_7(artifacts_root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    diag_csv = artifacts_root / "diagnostics" / "diagnostics.csv"
    rows = _load_csv(diag_csv)

    diag_by_key = {(r["diagnostic"], r["metric"]): r for r in rows}

    simple_checks = [
        ("candidate_margin", "mean_margin_delta",
         DIAG_CANDIDATE_MARGIN_MEAN_DELTA_RAW,
         DIAG_CANDIDATE_MARGIN_CI95_LOW_RAW,
         DIAG_CANDIDATE_MARGIN_CI95_HIGH_RAW),
        ("numeric_trace_divergence", "changed_mean",
         DIAG_NUMTRACE_ANSWER_CHANGED_MEAN_RAW, None, None),
        ("numeric_trace_divergence", "unchanged_mean",
         DIAG_NUMTRACE_ANSWER_UNCHANGED_MEAN_RAW, None, None),
        ("numeric_trace_divergence", "difference",
         DIAG_NUMTRACE_DIFFERENCE_RAW,
         DIAG_NUMTRACE_DIFFERENCE_CI95_LOW_RAW,
         DIAG_NUMTRACE_DIFFERENCE_CI95_HIGH_RAW),
    ]
    for diag, metric, exp_val, exp_low, exp_high in simple_checks:
        key = (diag, metric)
        if key not in diag_by_key:
            errors.append(f"MISSING cluster=7 csv={diag_csv} (diagnostic={diag!r}, metric={metric!r})")
            continue
        row = diag_by_key[key]
        actual_val = _f(row["value"])
        if not _approx_equal(actual_val, exp_val, TOL_RAW):
            errors.append(_fmt_mismatch(7, diag_csv, f"({diag!r}, {metric!r})",
                                        "value", exp_val, actual_val, TOL_RAW))
        if exp_low is not None and row.get("ci_low"):
            actual_low = _f(row["ci_low"])
            if not _approx_equal(actual_low, exp_low, TOL_RAW):
                errors.append(_fmt_mismatch(7, diag_csv, f"({diag!r}, {metric!r})",
                                            "ci_low", exp_low, actual_low, TOL_RAW))
        if exp_high is not None and row.get("ci_high"):
            actual_high = _f(row["ci_high"])
            if not _approx_equal(actual_high, exp_high, TOL_RAW):
                errors.append(_fmt_mismatch(7, diag_csv, f"({diag!r}, {metric!r})",
                                            "ci_high", exp_high, actual_high, TOL_RAW))

    rationale_patterns = [
        (re.compile(r"^clean_model_clean_rationale_.*margin", re.IGNORECASE),
         DIAG_RAT_BASELINE_BASELINE_MEAN_RAW,
         DIAG_RAT_BASELINE_BASELINE_CI95_LOW_RAW,
         DIAG_RAT_BASELINE_BASELINE_CI95_HIGH_RAW,
         "clean_model_clean_rationale"),
        (re.compile(r"^clean_model_swap_rationale_.*margin", re.IGNORECASE),
         DIAG_RAT_BASELINE_SWAP_MEAN_RAW,
         DIAG_RAT_BASELINE_SWAP_CI95_LOW_RAW,
         DIAG_RAT_BASELINE_SWAP_CI95_HIGH_RAW,
         "clean_model_swap_rationale"),
        (re.compile(r"^direct_swap_model_clean_rationale_.*margin", re.IGNORECASE),
         DIAG_RAT_SWAP_BASELINE_MEAN_RAW,
         DIAG_RAT_SWAP_BASELINE_CI95_LOW_RAW,
         DIAG_RAT_SWAP_BASELINE_CI95_HIGH_RAW,
         "direct_swap_model_clean_rationale"),
        (re.compile(r"^direct_swap_model_swap_rationale_.*margin", re.IGNORECASE),
         DIAG_RAT_SWAP_SWAP_MEAN_RAW,
         DIAG_RAT_SWAP_SWAP_CI95_LOW_RAW,
         DIAG_RAT_SWAP_SWAP_CI95_HIGH_RAW,
         "direct_swap_model_swap_rationale"),
    ]
    for pat, exp_val, exp_low, exp_high, label in rationale_patterns:
        matching = [(r["diagnostic"], r["metric"]) for r in rows
                    if r["diagnostic"] == "rationale_conditioned_scoring" and pat.match(r["metric"])]
        if len(matching) != 1:
            errors.append(
                f"PATTERN_ERROR cluster=7 csv={diag_csv} pattern={pat.pattern!r} "
                f"found {len(matching)} matches (expected exactly 1)"
            )
            continue
        key = matching[0]
        row = diag_by_key[key]
        actual_val = _f(row["value"])
        if not _approx_equal(actual_val, exp_val, TOL_RAW):
            errors.append(_fmt_mismatch(7, diag_csv, str(key), "value", exp_val, actual_val, TOL_RAW))
        for col, exp_ci in [("ci_low", exp_low), ("ci_high", exp_high)]:
            if row.get(col):
                actual_ci = _f(row[col])
                if not _approx_equal(actual_ci, exp_ci, TOL_RAW):
                    errors.append(_fmt_mismatch(7, diag_csv, str(key), col, exp_ci, actual_ci, TOL_RAW))

    return errors


def _check_cluster_8(artifacts_root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    hs_csv = artifacts_root / "hidden_state_appendix" / "hidden_state_divergence_summary.csv"
    rep_csv = artifacts_root / "repetition_robustness" / "repetition_robustness.csv"

    hs_rows = _load_csv(hs_csv)
    if not hs_rows:
        errors.append(f"MISSING cluster=8 csv={hs_csv} (empty)")
    else:
        row = hs_rows[0]
        actual_layer = _i(row["best_layer"])
        if actual_layer != HIDDEN_BEST_SINGLE_LAYER:
            errors.append(_fmt_mismatch(8, hs_csv, "row0", "best_layer",
                                        HIDDEN_BEST_SINGLE_LAYER, actual_layer, 0))
        actual_roc = _f(row["best_roc_auc"])
        if not _approx_equal(actual_roc, HIDDEN_BEST_SINGLE_LAYER_ROC_AUC_RAW, TOL_RAW):
            errors.append(_fmt_mismatch(8, hs_csv, "row0", "best_roc_auc",
                                        HIDDEN_BEST_SINGLE_LAYER_ROC_AUC_RAW, actual_roc, TOL_RAW))
        try:
            all_layer = json.loads(row["all_layer_probe"])
            actual_all = _f(all_layer["roc_auc"])
            if not _approx_equal(actual_all, HIDDEN_ALL_LAYER_PROBE_ROC_AUC_RAW, TOL_RAW):
                errors.append(_fmt_mismatch(8, hs_csv, "row0", "all_layer_probe.roc_auc",
                                            HIDDEN_ALL_LAYER_PROBE_ROC_AUC_RAW, actual_all, TOL_RAW))
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            errors.append(f"PARSE_ERROR cluster=8 csv={hs_csv} column=all_layer_probe: {e}")
        try:
            len_ctrl = json.loads(row["length_only_control"])
            actual_len = _f(len_ctrl["roc_auc"])
            if not _approx_equal(actual_len, HIDDEN_LENGTH_ONLY_CONTROL_ROC_AUC_RAW, TOL_RAW):
                errors.append(_fmt_mismatch(8, hs_csv, "row0", "length_only_control.roc_auc",
                                            HIDDEN_LENGTH_ONLY_CONTROL_ROC_AUC_RAW, actual_len, TOL_RAW))
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            errors.append(f"PARSE_ERROR cluster=8 csv={hs_csv} column=length_only_control: {e}")

    rep_rows = _load_csv(rep_csv)
    filtered = next((r for r in rep_rows if r.get("subset") == "filtered"), None)
    if filtered is None:
        errors.append(f"MISSING cluster=8 csv={rep_csv} subset='filtered'")
    else:
        checks = [
            ("n", REPETITION_FILTERED_N, True),
            ("accuracy_delta", REPETITION_FILTERED_ACCURACY_DELTA_RAW, False),
            ("parsed_answer_change_rate", REPETITION_FILTERED_CHANGE_RATE_RAW, False),
            ("stable_wrong_different_rate", REPETITION_FILTERED_STABLE_WRONG_DIFFERENT_RATE_RAW, False),
            ("answer_changed", REPETITION_FILTERED_CHANGE_RATE_NUM, True),
            ("stable_wrong_different", REPETITION_FILTERED_STABLE_WRONG_DIFFERENT_NUM, True),
            ("stable_wrong", REPETITION_FILTERED_STABLE_WRONG_DIFFERENT_DEN, True),
        ]
        for col, expected, is_int in checks:
            if col not in filtered:
                errors.append(f"MISSING cluster=8 csv={rep_csv} subset=filtered column={col!r}")
                continue
            if is_int:
                actual = _i(filtered[col])
                if actual != int(expected):
                    errors.append(_fmt_mismatch(8, rep_csv, "subset=filtered", col, expected, actual, 0))
            else:
                actual = _f(filtered[col])
                if not _approx_equal(actual, expected, TOL_RAW):
                    errors.append(_fmt_mismatch(8, rep_csv, "subset=filtered", col, expected, actual, TOL_RAW))
        # Also verify DEN
        if "n" in filtered and _i(filtered["n"]) != REPETITION_FILTERED_CHANGE_RATE_DEN:
            errors.append(_fmt_mismatch(8, rep_csv, "subset=filtered", "n",
                                        REPETITION_FILTERED_CHANGE_RATE_DEN, _i(filtered["n"]), 0))

    return errors


def main(artifacts_root: pathlib.Path | None = None) -> int:
    if artifacts_root is None:
        artifacts_root = pathlib.Path(__file__).resolve().parents[2] / "artifacts"
    all_errors: list[str] = []
    for fn in [
        _check_cluster_1,
        _check_cluster_2,
        _check_cluster_3,
        _check_cluster_4,
        _check_cluster_5,
        _check_cluster_6,
        _check_cluster_7,
        _check_cluster_8,
    ]:
        all_errors.extend(fn(artifacts_root))

    if all_errors:
        for e in all_errors:
            print(e, file=sys.stderr)
        return 1
    print("PAPER NUMBERS VERIFIED")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify paper numbers against artifacts CSVs.")
    parser.add_argument(
        "--artifacts-root",
        type=pathlib.Path,
        default=None,
        help="Path to artifacts/ directory. Defaults to <repo_root>/artifacts.",
    )
    args = parser.parse_args()
    sys.exit(main(args.artifacts_root))

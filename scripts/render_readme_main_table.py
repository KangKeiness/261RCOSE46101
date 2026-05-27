#!/usr/bin/env python3
"""Render the README main result table from artifacts/main/ CSVs.

Loads artifacts/main/main_metrics.csv and artifacts/main/bootstrap_ci.csv
(paths anchored to bundle root), then prints a markdown table to stdout.

Usage:
    python scripts/render_readme_main_table.py

Exit 0 on success.
"""
from __future__ import annotations

import csv
import pathlib
import sys

# Configure stdout to UTF-8 so em-dash (U+2014) and right-arrow (U+2192)
# in rendered table strings do not crash on Windows cp949 consoles.
sys.stdout.reconfigure(encoding="utf-8")


BUNDLE_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_METRICS_CSV = BUNDLE_ROOT / "artifacts" / "main" / "main_metrics.csv"
BOOTSTRAP_CI_CSV = BUNDLE_ROOT / "artifacts" / "main" / "bootstrap_ci.csv"


def _load_metrics(path: pathlib.Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            metrics[row["metric"]] = row["value"]
    return metrics


def _load_ci(path: pathlib.Path) -> dict[str, dict[str, str]]:
    ci: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ci[row["metric"]] = {
                "point": row["point"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
            }
    return ci


def render_table(metrics: dict[str, str], ci: dict[str, dict[str, str]]) -> str:
    n = int(metrics["n"])
    clean_correct = int(metrics["clean_correct"])
    clean_acc = float(metrics["clean_accuracy"])
    swap_correct = int(metrics["direct_swap_correct"])
    swap_acc = float(metrics["direct_swap_accuracy"])
    delta = float(metrics["accuracy_delta"])
    changed = int(metrics["answer_changed"])
    change_rate = float(metrics["parsed_answer_change_rate"])
    stable_correct = int(metrics["stable_correct"])
    s_broken = int(metrics["S_broken"])
    s_repaired = int(metrics["S_repaired"])
    stable_wrong = int(metrics["stable_wrong"])
    stable_wrong_same = int(metrics["stable_wrong_same"])
    stable_wrong_diff = int(metrics["stable_wrong_different"])
    swd_rate = float(metrics["stable_wrong_different_rate"])

    delta_ci = ci.get("accuracy_delta", {})
    delta_lo = float(delta_ci.get("ci_low", 0))
    delta_hi = float(delta_ci.get("ci_high", 0))

    change_ci = ci.get("parsed_answer_change_rate", {})
    change_lo = float(change_ci.get("ci_low", 0))
    change_hi = float(change_ci.get("ci_high", 0))

    swd_ci = ci.get("stable_wrong_different_rate_within_stable_wrong", {})
    swd_lo = float(swd_ci.get("ci_low", 0))
    swd_hi = float(swd_ci.get("ci_high", 0))

    lines: list[str] = []
    lines.append(f"**Chinese MGSM (n={n}) — main result**")
    lines.append("")
    lines.append("| Metric | Value | 95% CI |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Clean accuracy | {clean_acc:.3f} ({clean_correct}/{n})"
        f" | [{float(ci.get('baseline_accuracy', {}).get('ci_low', 0)):.3f},"
        f" {float(ci.get('baseline_accuracy', {}).get('ci_high', 0)):.3f}] |"
    )
    lines.append(
        f"| Direct-swap accuracy | {swap_acc:.3f} ({swap_correct}/{n})"
        f" | [{float(ci.get('direct_swap_accuracy', {}).get('ci_low', 0)):.3f},"
        f" {float(ci.get('direct_swap_accuracy', {}).get('ci_high', 0)):.3f}] |"
    )
    lines.append(
        f"| Accuracy delta | {delta:+.3f}"
        f" | [{delta_lo:.3f}, {delta_hi:.3f}] |"
    )
    lines.append(
        f"| Parsed-answer change rate | {change_rate:.3f} ({changed}/{n})"
        f" | [{change_lo:.3f}, {change_hi:.3f}] |"
    )
    lines.append("")
    lines.append("**Transition breakdown**")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---|")
    lines.append(f"| Stable-correct | {stable_correct} |")
    lines.append(f"| Broken (correct → wrong) | {s_broken} |")
    lines.append(f"| Repaired (wrong → correct) | {s_repaired} |")
    lines.append(f"| Stable-wrong, same answer | {stable_wrong_same} |")
    lines.append(f"| Stable-wrong, different answer | {stable_wrong_diff} |")
    lines.append(
        f"| Stable-wrong-different rate | {swd_rate:.3f} ({stable_wrong_diff}/{stable_wrong})"
        f" | [{swd_lo:.3f}, {swd_hi:.3f}] |"
    )
    lines.append("")
    lines.append(
        "Bootstrap CIs: paired nonparametric percentile, 10000 resamples, seed 20260517."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        metrics = _load_metrics(MAIN_METRICS_CSV)
        ci = _load_ci(BOOTSTRAP_CI_CSV)
        print(render_table(metrics, ci))
        sys.exit(0)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

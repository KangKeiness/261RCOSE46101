from __future__ import annotations

import argparse
import csv
import json
from src.evaluation.paper_constants import (
    MAIN_ZH_N,
    MAIN_ZH_CLEAN_CORRECT,
    MAIN_ZH_DIRECT_SWAP_CORRECT,
    MAIN_ZH_IDENTITY_CORRECT,
    MAIN_ZH_ANSWER_CHANGED,
    MAIN_ZH_STABLE_CORRECT,
    MAIN_ZH_S_BROKEN,
    MAIN_ZH_S_REPAIRED,
    MAIN_ZH_STABLE_WRONG,
    MAIN_ZH_STABLE_WRONG_SAME,
    MAIN_ZH_STABLE_WRONG_DIFFERENT,
)
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/main"

CLEAN_PATH = ROOT / "data/raw_runs/main_chinese_mgsm/results_clean_no_patch.jsonl"
HARD_PATH = ROOT / "data/raw_runs/main_chinese_mgsm/results_restoration_no_patch.jsonl"
IDENTITY_PATH = ROOT / "data/raw_runs/composition_path_control/identity_composition_comparison.csv"
# OUTPUT_MODE_PATH is not in the release bundle.
OUTPUT_MODE_PATH = None
CANDIDATE_MARGIN_PATH = ROOT / "data/raw_runs/trajectory_rationale_followup/candidate_margin_records.csv"
NUMERIC_TRACE_PATH = ROOT / "data/raw_runs/trajectory_rationale_followup/numeric_trace_records.csv"
RATIONALE_PATH = ROOT / "data/raw_runs/trajectory_rationale_followup/rationale_conditioned_margin_records.csv"
PATCH_RECORDS = ROOT / "data/raw_runs/source_specific_patch_control/source_specific_patch_records.jsonl"
PATCH_SBROKEN = ROOT / "artifacts/patch_control/s_broken_ids.txt"

SEED = 20260517
N_BOOT = 10_000
TRAILING = " \t\r\n.,;:!?。．，；：！？"
CURRENCY_RE = re.compile(r"[$€£¥₩]")


@dataclass
class MetricCI:
    point: float | None
    ci_low: float | None
    ci_high: float | None
    n_bootstrap_values: int


def repo(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def normalize_numeric_answer(value: Any) -> str | None:
    """Canonical canonical numeric normalization. Fractions remain string-only."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.strip(TRAILING)
    text = CURRENCY_RE.sub("", text)
    text = text.replace(",", "").replace(" ", "")
    text = text.replace("−", "-")
    if text.endswith("%"):
        text = text[:-1]
    if "/" in text:
        return text or None
    try:
        dec = Decimal(text)
    except InvalidOperation:
        return None
    if not dec.is_finite():
        return None
    if dec == dec.to_integral_value():
        return str(int(dec))
    out = format(dec.normalize(), "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return "0" if out == "-0" else out


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def answers_equal(
    left: Any,
    right: Any,
    left_parse_success: bool = True,
    right_parse_success: bool = True,
) -> bool:
    if not left_parse_success or not right_parse_success:
        return False
    left_norm = normalize_numeric_answer(left)
    right_norm = normalize_numeric_answer(right)
    return left_norm is not None and right_norm is not None and left_norm == right_norm


def is_correct(gold: Any, pred: Any, parse_success: bool = True) -> bool:
    return answers_equal(gold, pred, True, parse_success)


def old_string_changed(
    clean_answer: Any,
    hard_answer: Any,
    clean_parse_success: bool,
    hard_parse_success: bool,
) -> bool:
    left = "__PARSE_FAIL__" if not clean_parse_success else str(clean_answer)
    right = "__PARSE_FAIL__" if not hard_parse_success else str(hard_answer)
    return left != right


def rate(num: int | float, den: int | float) -> float | None:
    return None if not den else float(num) / float(den)


def transition_group(clean_correct: bool, hard_correct: bool) -> str:
    if clean_correct and hard_correct:
        return "stable_correct"
    if clean_correct and not hard_correct:
        return "S_broken"
    if (not clean_correct) and hard_correct:
        return "S_repaired"
    return "stable_wrong"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fields = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def ci(values: list[float]) -> tuple[float | None, float | None]:
    vals = sorted(v for v in values if math.isfinite(float(v)))
    return percentile(vals, 0.025), percentile(vals, 0.975)


def paired_bootstrap(
    rows: list[dict[str, Any]],
    metric_fn: Callable[[list[dict[str, Any]]], dict[str, float | None]],
    n_bootstrap: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, dict[str, Any]]:
    rng = random.Random(seed)
    n = len(rows)
    point = metric_fn(rows)
    samples: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(n_bootstrap):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        metrics = metric_fn(sample)
        for key, value in metrics.items():
            if value is not None and math.isfinite(float(value)):
                samples[key].append(float(value))
    out: dict[str, dict[str, Any]] = {}
    for key, values in samples.items():
        lo, hi = ci(values)
        out[key] = asdict(MetricCI(point.get(key), lo, hi, len(values)))
    return out


def bootstrap_values(
    rows: list[dict[str, Any]],
    metric_fn: Callable[[list[dict[str, Any]]], float | None],
    seed: int,
) -> MetricCI:
    rng = random.Random(seed)
    point = metric_fn(rows)
    values: list[float] = []
    n = len(rows)
    for _ in range(N_BOOT):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        value = metric_fn(sample)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    lo, hi = ci(values)
    return MetricCI(point, lo, hi, len(values))


def compute_main_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean = {row["sample_id"]: row for row in read_jsonl(CLEAN_PATH)}
    hard = {row["sample_id"]: row for row in read_jsonl(HARD_PATH)}
    identity = read_csv(IDENTITY_PATH)
    rows: list[dict[str, Any]] = []
    changed_audit: list[dict[str, Any]] = []

    for identity_row in identity:
        sid = identity_row["sample_id"]
        if sid not in clean or sid not in hard:
            raise SystemExit(f"BLOCK: missing clean/hard canonical row for {sid}")
        gold = identity_row["gold_answer"]
        clean_row = clean[sid]
        hard_row = hard[sid]
        clean_parse = bool_value(clean_row.get("parse_success"))
        hard_parse = bool_value(hard_row.get("parse_success"))
        clean_ans = clean_row.get("normalized_answer")
        hard_ans = hard_row.get("normalized_answer")
        identity_ans = identity_row.get("identity_answer")

        clean_ok = is_correct(gold, clean_ans, clean_parse)
        hard_ok = is_correct(gold, hard_ans, hard_parse)
        identity_ok = is_correct(gold, identity_ans, True)
        answer_changed = not answers_equal(clean_ans, hard_ans, clean_parse, hard_parse)
        old_changed = old_string_changed(clean_ans, hard_ans, clean_parse, hard_parse)
        group = transition_group(clean_ok, hard_ok)
        stable_wrong_same = group == "stable_wrong" and not answer_changed
        stable_wrong_different = group == "stable_wrong" and answer_changed
        row = {
            "sample_id": sid,
            "gold_answer": gold,
            "gold_normalized": normalize_numeric_answer(gold),
            "clean_normalized_answer": clean_ans,
            "clean_normalized_canonical": normalize_numeric_answer(clean_ans),
            "clean_parse_success": clean_parse,
            "hard_normalized_answer": hard_ans,
            "hard_normalized_canonical": normalize_numeric_answer(hard_ans),
            "hard_parse_success": hard_parse,
            "identity_answer": identity_ans,
            "identity_normalized_canonical": normalize_numeric_answer(identity_ans),
            "clean_correct": clean_ok,
            "hard_correct": hard_ok,
            "identity_correct": identity_ok,
            "answer_changed": answer_changed,
            "old_string_answer_changed": old_changed,
            "transition_group": group,
            "stable_wrong_same": stable_wrong_same,
            "stable_wrong_different": stable_wrong_different,
        }
        rows.append(row)
        if old_changed != answer_changed:
            changed_audit.append({
                "sample_id": sid,
                "gold_answer": gold,
                "clean_normalized_answer": clean_ans,
                "hard_normalized_answer": hard_ans,
                "clean_normalized_canonical": normalize_numeric_answer(clean_ans),
                "hard_normalized_canonical": normalize_numeric_answer(hard_ans),
                "old_string_answer_changed": old_changed,
                "canonical_answer_changed": answer_changed,
                "transition_group": group,
                "note": "decimal-equivalent answer identity no longer counted as changed",
            })

    return rows, changed_audit


def main_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    clean_correct = sum(bool_value(row["clean_correct"]) for row in rows)
    hard_correct = sum(bool_value(row["hard_correct"]) for row in rows)
    identity_correct = sum(bool_value(row["identity_correct"]) for row in rows)
    answer_changed = sum(bool_value(row["answer_changed"]) for row in rows)
    groups = Counter(row["transition_group"] for row in rows)
    stable_wrong = [row for row in rows if row["transition_group"] == "stable_wrong"]
    stable_wrong_same = sum(bool_value(row["stable_wrong_same"]) for row in stable_wrong)
    stable_wrong_different = sum(bool_value(row["stable_wrong_different"]) for row in stable_wrong)
    stable_wrong_parse_failure = sum(
        (not bool_value(row["clean_parse_success"])) or (not bool_value(row["hard_parse_success"]))
        for row in stable_wrong
    )
    return {
        "n": n,
        "clean_correct": clean_correct,
        "clean_accuracy": rate(clean_correct, n),
        "direct_swap_correct": hard_correct,
        "direct_swap_accuracy": rate(hard_correct, n),
        "identity_correct": identity_correct,
        "identity_accuracy": rate(identity_correct, n),
        "accuracy_delta": (hard_correct - clean_correct) / n if n else None,
        "answer_changed": answer_changed,
        "parsed_answer_change_rate": rate(answer_changed, n),
        "stable_correct": groups["stable_correct"],
        "S_broken": groups["S_broken"],
        "S_repaired": groups["S_repaired"],
        "stable_wrong": groups["stable_wrong"],
        "stable_wrong_same": stable_wrong_same,
        "stable_wrong_different": stable_wrong_different,
        "stable_wrong_different_rate": rate(stable_wrong_different, len(stable_wrong)),
        "stable_wrong_different_rate_over_all": rate(stable_wrong_different, n),
        "stable_wrong_parse_failure_count": stable_wrong_parse_failure,
        "clean_parse_failure_count": sum(not bool_value(row["clean_parse_success"]) for row in rows),
        "direct_swap_parse_failure_count": sum(not bool_value(row["hard_parse_success"]) for row in rows),
    }


def metric_fn(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    metrics = main_metrics(rows)
    n = metrics["n"]
    return {
        "baseline_accuracy": metrics["clean_accuracy"],
        "direct_swap_accuracy": metrics["direct_swap_accuracy"],
        "accuracy_delta": metrics["accuracy_delta"],
        "parsed_answer_change_rate": metrics["parsed_answer_change_rate"],
        "broken_rate": rate(metrics["S_broken"], n),
        "repaired_rate": rate(metrics["S_repaired"], n),
        "stable_wrong_different_rate_over_all": metrics["stable_wrong_different_rate_over_all"],
        "stable_wrong_different_rate_within_stable_wrong": metrics["stable_wrong_different_rate"],
    }


def metric_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"metric": key, "value": value} for key, value in metrics.items()]


def bootstrap_table(payload: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "metric": key,
            "point": value["point"],
            "ci_low": value["ci_low"],
            "ci_high": value["ci_high"],
            "n_bootstrap_values": value["n_bootstrap_values"],
        }
        for key, value in payload.items()
    ]


def validate_main(metrics: dict[str, Any]) -> list[str]:
    expected = {
        "n": MAIN_ZH_N,
        "clean_correct": MAIN_ZH_CLEAN_CORRECT,
        "direct_swap_correct": MAIN_ZH_DIRECT_SWAP_CORRECT,
        "identity_correct": MAIN_ZH_IDENTITY_CORRECT,
        "answer_changed": MAIN_ZH_ANSWER_CHANGED,
        "stable_correct": MAIN_ZH_STABLE_CORRECT,
        "S_broken": MAIN_ZH_S_BROKEN,
        "S_repaired": MAIN_ZH_S_REPAIRED,
        "stable_wrong": MAIN_ZH_STABLE_WRONG,
        "stable_wrong_same": MAIN_ZH_STABLE_WRONG_SAME,
        "stable_wrong_different": MAIN_ZH_STABLE_WRONG_DIFFERENT,
    }
    errors = [
        f"{key}: expected {value}, got {metrics.get(key)}"
        for key, value in expected.items()
        if metrics.get(key) != value
    ]
    checks = {
        "transition groups sum to n": metrics["stable_correct"] + metrics["S_broken"] + metrics["S_repaired"] + metrics["stable_wrong"] == metrics["n"],
        "stable_correct + S_broken = clean_correct": metrics["stable_correct"] + metrics["S_broken"] == metrics["clean_correct"],
        "stable_correct + S_repaired = direct_swap_correct": metrics["stable_correct"] + metrics["S_repaired"] == metrics["direct_swap_correct"],
        "changed transition cells sum to answer_changed": metrics["S_broken"] + metrics["S_repaired"] + metrics["stable_wrong_different"] == metrics["answer_changed"],
        "stable_wrong refinements sum to stable_wrong": metrics["stable_wrong_same"] + metrics["stable_wrong_different"] == metrics["stable_wrong"],
    }
    errors.extend(name for name, ok in checks.items() if not ok)
    return errors


def load_repetition_flags() -> tuple[dict[str, bool] | None, dict[str, Any]]:
    if OUTPUT_MODE_PATH is None or not OUTPUT_MODE_PATH.exists():
        return None, {"status": "ROBUSTNESS_REPETITION_RULE_NOT_FOUND", "source_path": repo(OUTPUT_MODE_PATH)}
    rows = read_csv(OUTPUT_MODE_PATH)
    required = {"sample_id", "condition", "flag_repetition_or_degenerate"}
    if not rows or not required.issubset(rows[0].keys()):
        return None, {"status": "ROBUSTNESS_REPETITION_RULE_NOT_FOUND", "source_path": repo(OUTPUT_MODE_PATH)}
    flags: dict[str, bool] = {}
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        condition = row["condition"]
        flagged = bool_value(row["flag_repetition_or_degenerate"])
        counts[condition]["n"] += 1
        counts[condition]["flagged"] += int(flagged)
        if condition == "direct_swap":
            flags[row["sample_id"]] = flagged
    return flags, {
        "status": "sample_level_flags_loaded",
        "source_path": repo(OUTPUT_MODE_PATH),
        "rule": "existing output-mode audit flag_repetition_or_degenerate",
        "condition_counts": {
            key: {
                "n": value["n"],
                "repetition_or_degenerate_count": value["flagged"],
                "repetition_or_degenerate_rate": rate(value["flagged"], value["n"]),
            }
            for key, value in sorted(counts.items())
        },
    }


def robustness_metrics(main_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flags, source = load_repetition_flags()
    if flags is None:
        row = {"subset": "unavailable", "status": "ROBUSTNESS_REPETITION_RULE_NOT_FOUND"}
        return [row], {"status": "ROBUSTNESS_REPETITION_RULE_NOT_FOUND", "repetition_source": source, "rows": [row]}
    excluded = sorted(sample_id for sample_id, flagged in flags.items() if flagged)
    excluded_set = set(excluded)
    filtered = [row for row in main_rows if row["sample_id"] not in excluded_set]
    rows: list[dict[str, Any]] = []
    for label, subset in (("original", main_rows), ("filtered", filtered)):
        metrics = main_metrics(subset)
        rows.append({
            "subset": label,
            "status": "computed",
            "n": metrics["n"],
            "excluded_direct_swap_repetition_count": 0 if label == "original" else len(excluded),
            "clean_accuracy": metrics["clean_accuracy"],
            "direct_swap_accuracy": metrics["direct_swap_accuracy"],
            "accuracy_delta": metrics["accuracy_delta"],
            "answer_changed": metrics["answer_changed"],
            "parsed_answer_change_rate": metrics["parsed_answer_change_rate"],
            "stable_correct": metrics["stable_correct"],
            "S_broken": metrics["S_broken"],
            "S_repaired": metrics["S_repaired"],
            "stable_wrong": metrics["stable_wrong"],
            "stable_wrong_same": metrics["stable_wrong_same"],
            "stable_wrong_different": metrics["stable_wrong_different"],
            "stable_wrong_different_rate": metrics["stable_wrong_different_rate"],
        })
    return rows, {
        "status": "computed",
        "repetition_source": source,
        "excluded_count": len(excluded),
        "excluded_sample_ids": excluded,
        "remaining_n": len(filtered),
        "rows": rows,
    }


def candidate_margin_diagnostic(changed_ids: set[str]) -> dict[str, Any]:
    if CANDIDATE_MARGIN_PATH is None or not CANDIDATE_MARGIN_PATH.exists():
        return {"status": "run_artifact_NOT_FOUND", "source_path": repo(CANDIDATE_MARGIN_PATH)}
    records = [
        row for row in read_csv(CANDIDATE_MARGIN_PATH)
        if row.get("sample_id") in changed_ids and bool_value(row.get("score_success"))
    ]
    values = [parse_float(row.get("margin_delta")) for row in records]
    values = [v for v in values if v is not None]
    ci_mean = bootstrap_values(
        [{"value": value} for value in values],
        lambda sample: mean([float(row["value"]) for row in sample]),
        SEED + 11,
    )
    ci_frac = bootstrap_values(
        [{"value": value} for value in values],
        lambda sample: rate(sum(float(row["value"]) < 0 for row in sample), len(sample)),
        SEED + 12,
    )
    return {
        "status": "computed",
        "source_path": repo(CANDIDATE_MARGIN_PATH),
        "n": len(records),
        "mean_margin_delta": mean(values),
        "fraction_margin_delta_lt_0": rate(sum(v < 0 for v in values), len(values)),
        "mean_margin_delta_ci": asdict(ci_mean),
        "fraction_margin_delta_lt_0_ci": asdict(ci_frac),
        "ci_includes_zero": ci_mean.ci_low is not None and ci_mean.ci_high is not None and ci_mean.ci_low <= 0 <= ci_mean.ci_high,
    }


def numeric_trace_diagnostic(changed_ids: set[str]) -> dict[str, Any]:
    if not NUMERIC_TRACE_PATH.exists():
        return {"status": "run_artifact_NOT_FOUND", "source_path": repo(NUMERIC_TRACE_PATH)}
    records = []
    for row in read_csv(NUMERIC_TRACE_PATH):
        value = parse_float(row.get("numeric_trace_edit_distance_normalized"))
        if value is None:
            continue
        records.append({"sample_id": row["sample_id"], "value": value, "answer_changed": row["sample_id"] in changed_ids})

    def values(sample: list[dict[str, Any]]) -> dict[str, float | None]:
        changed = [row["value"] for row in sample if row["answer_changed"]]
        unchanged = [row["value"] for row in sample if not row["answer_changed"]]
        changed_mean = mean(changed)
        unchanged_mean = mean(unchanged)
        return {
            "changed_mean": changed_mean,
            "unchanged_mean": unchanged_mean,
            "difference": None if changed_mean is None or unchanged_mean is None else changed_mean - unchanged_mean,
        }

    boot = paired_bootstrap(records, values, seed=SEED + 21)
    point = values(records)
    return {
        "status": "computed",
        "source_path": repo(NUMERIC_TRACE_PATH),
        "answer_changed_n": sum(row["answer_changed"] for row in records),
        "answer_unchanged_n": sum(not row["answer_changed"] for row in records),
        **point,
        "bootstrap": boot,
    }


def rationale_diagnostic(changed_ids: set[str]) -> dict[str, Any]:
    if not RATIONALE_PATH.exists():
        return {"status": "run_artifact_NOT_FOUND", "source_path": repo(RATIONALE_PATH)}
    metrics = [
        "clean_model_clean_rationale_clean_minus_swap_margin_norm",
        "clean_model_swap_rationale_clean_minus_swap_margin_norm",
        "direct_swap_model_clean_rationale_clean_minus_swap_margin_norm",
        "direct_swap_model_swap_rationale_clean_minus_swap_margin_norm",
    ]
    records = [row for row in read_csv(RATIONALE_PATH) if row.get("sample_id") in changed_ids]
    out: dict[str, Any] = {"status": "computed", "source_path": repo(RATIONALE_PATH), "n": len(records), "metrics": {}}
    for metric in metrics:
        values = [{"value": parse_float(row.get(metric))} for row in records]
        values = [row for row in values if row["value"] is not None]
        boot = bootstrap_values(values, lambda sample: mean([float(row["value"]) for row in sample]), SEED + 31 + metrics.index(metric))
        out["metrics"][metric] = {
            "n": len(values),
            "mean": mean([float(row["value"]) for row in values]),
            "bootstrap_ci": asdict(boot),
        }
    return out


def diagnostics(changed_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate = candidate_margin_diagnostic(changed_ids)
    numeric = numeric_trace_diagnostic(changed_ids)
    rationale = rationale_diagnostic(changed_ids)
    rows = [
        {
            "diagnostic": "candidate_margin",
            "status": candidate["status"],
            "n": candidate.get("n"),
            "metric": "mean_margin_delta",
            "value": candidate.get("mean_margin_delta"),
            "ci_low": (candidate.get("mean_margin_delta_ci") or {}).get("ci_low"),
            "ci_high": (candidate.get("mean_margin_delta_ci") or {}).get("ci_high"),
            "source_path": candidate.get("source_path"),
        },
        {
            "diagnostic": "candidate_margin",
            "status": candidate["status"],
            "n": candidate.get("n"),
            "metric": "fraction_margin_delta_lt_0",
            "value": candidate.get("fraction_margin_delta_lt_0"),
            "ci_low": (candidate.get("fraction_margin_delta_lt_0_ci") or {}).get("ci_low"),
            "ci_high": (candidate.get("fraction_margin_delta_lt_0_ci") or {}).get("ci_high"),
            "source_path": candidate.get("source_path"),
        },
        {
            "diagnostic": "numeric_trace_divergence",
            "status": numeric["status"],
            "n": numeric.get("answer_changed_n"),
            "metric": "changed_mean",
            "value": numeric.get("changed_mean"),
            "ci_low": (numeric.get("bootstrap") or {}).get("changed_mean", {}).get("ci_low"),
            "ci_high": (numeric.get("bootstrap") or {}).get("changed_mean", {}).get("ci_high"),
            "source_path": numeric.get("source_path"),
        },
        {
            "diagnostic": "numeric_trace_divergence",
            "status": numeric["status"],
            "n": numeric.get("answer_unchanged_n"),
            "metric": "unchanged_mean",
            "value": numeric.get("unchanged_mean"),
            "ci_low": (numeric.get("bootstrap") or {}).get("unchanged_mean", {}).get("ci_low"),
            "ci_high": (numeric.get("bootstrap") or {}).get("unchanged_mean", {}).get("ci_high"),
            "source_path": numeric.get("source_path"),
        },
        {
            "diagnostic": "numeric_trace_divergence",
            "status": numeric["status"],
            "n": f"{numeric.get('answer_changed_n')}/{numeric.get('answer_unchanged_n')}",
            "metric": "difference",
            "value": numeric.get("difference"),
            "ci_low": (numeric.get("bootstrap") or {}).get("difference", {}).get("ci_low"),
            "ci_high": (numeric.get("bootstrap") or {}).get("difference", {}).get("ci_high"),
            "source_path": numeric.get("source_path"),
        },
    ]
    for metric, data in (rationale.get("metrics") or {}).items():
        rows.append({
            "diagnostic": "rationale_conditioned_scoring",
            "status": rationale["status"],
            "n": data.get("n"),
            "metric": metric,
            "value": data.get("mean"),
            "ci_low": data.get("bootstrap_ci", {}).get("ci_low"),
            "ci_high": data.get("bootstrap_ci", {}).get("ci_high"),
            "source_path": rationale.get("source_path"),
        })
    return rows, {"candidate_margin": candidate, "numeric_trace": numeric, "rationale_conditioned": rationale}


def patch_control_caveat(main_rows: list[dict[str, Any]]) -> dict[str, Any]:
    patched_ids = {row["sample_id"] for row in main_rows if row["transition_group"] == "S_broken"}
    payload: dict[str, Any] = {
        "status": "run_artifact_NOT_FOUND",
        "policy": "match_diagnostic_only",
        "patched_sbroken_n": len(patched_ids),
        "source_records": repo(PATCH_RECORDS) if PATCH_RECORDS.exists() else None,
        "source_old_sbroken_ids": repo(PATCH_SBROKEN) if PATCH_SBROKEN.exists() else None,
        "presentation_policy": (
            "Do not present patch-control as primary evidence; if retained, label it as a "
            "match diagnostic over available patched-broken records."
        ),
    }
    if PATCH_RECORDS.exists():
        record_ids = {row["sample_id"] for row in read_jsonl(PATCH_RECORDS)}
        missing_ids = sorted(patched_ids - record_ids)
        unmatched_ids = sorted(record_ids - patched_ids)
        payload.update({
            "record_sample_id_n": len(record_ids),
            "available_patched_sbroken_record_n": len(patched_ids & record_ids),
            "full_canonical_sbroken_compute_available": patched_ids.issubset(record_ids),
            "missing_canonical_sbroken_sample_id_count": len(missing_ids),
            "unmatched_record_sample_id_count": len(unmatched_ids),
            "missing_canonical_sbroken_sample_ids": [
                f"redacted_missing_sample_{i + 1}" for i, _ in enumerate(missing_ids)
            ],
            "unmatched_record_sample_ids": [
                f"redacted_unmatched_sample_{i + 1}" for i, _ in enumerate(unmatched_ids)
            ],
        })
    return payload


STALE_PATTERNS = [
    r"0\.480",
    r"0\.452",
    r"-0\.028",
    r"155/250",
    r"0\.620",
    r"80/40/33/97",
    r"82/97",
    r"0\.845",
    r"stable_correct=80",
    r"broken=40",
    r"stable_wrong=97",
]
STALE_RE = re.compile("|".join(STALE_PATTERNS))


def classify_stale(path: Path) -> str:
    rel = repo(path) or str(path)
    lower = rel.lower()
    if lower.endswith(".tex"):
        return "STALE_PAPER_ERROR"
    if lower.startswith("artifacts/main") or lower == "compute_report.md":
        return "valid historical note"
    if lower.startswith("docs/") or lower == "readme.md":
        return "valid historical note"
    if lower in {"audit_paper_code_bundle.py", "collect_paper_code_bundle.py"}:
        return "file not part of final paper"
    if lower.startswith(("notes/", "paper_code_bundle/", "results/", "data/raw_runs/")):
        return "file not part of final paper"
    if "manifest" in lower or "classification" in lower:
        return "file not part of final paper"
    return "needs manual review"


def stale_number_grep() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    suffixes = {".md", ".tex", ".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml", ".py"}
    ignored_parts = {".git", "__pycache__", ".venv", "venv"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.parent == OUT and "stale_number_grep" in path.name:
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, 1):
            if STALE_RE.search(line):
                rows.append({
                    "path": repo(path),
                    "line": lineno,
                    "match": STALE_RE.search(line).group(0),
                    "classification": classify_stale(path),
                    "text": line.strip()[:240],
                })
    return {
        "patterns": STALE_PATTERNS,
        "n_occurrences": len(rows),
        "n_stale_paper_errors": sum(row["classification"] == "STALE_PAPER_ERROR" for row in rows),
        "rows": rows,
    }


def fmt(value: Any, ndigits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def write_report(
    verdict: str,
    main: dict[str, Any],
    boot: dict[str, dict[str, Any]],
    robust_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
    patch_payload: dict[str, Any],
    changed_audit: list[dict[str, Any]],
    stale: dict[str, Any],
    validation_errors: list[str],
) -> str:
    filtered = next((row for row in robust_rows if row.get("subset") == "filtered"), {})
    diag_table = "\n".join(
        f"| {row['diagnostic']} | {row['metric']} | {row.get('n')} | {fmt(row.get('value'), 4)} | [{fmt(row.get('ci_low'), 4)}, {fmt(row.get('ci_high'), 4)}] |"
        for row in diag_rows
    )
    changed_lines = "\n".join(
        f"- `{row['sample_id']}`: `{row['clean_normalized_answer']}` vs `{row['hard_normalized_answer']}` -> not changed under canonical"
        for row in changed_audit
    )
    lines = [
        "# canonical Compute Report",
        "",
        "## Executive Verdict",
        "",
        verdict,
        "",
        "canonical uses one decimal-equivalent numerical comparator for correctness, baseline-vs-direct-swap answer identity, stable-wrong same/different refinement, and transition consistency checks. Fractions are not converted to decimals; fraction equivalence is unsupported unless strings are already identical after light normalization.",
        "",
        "## Final canonical Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| n | {main['n']} |",
        f"| clean accuracy | {main['clean_correct']}/{main['n']} = {fmt(main['clean_accuracy'])} |",
        f"| direct-swap accuracy | {main['direct_swap_correct']}/{main['n']} = {fmt(main['direct_swap_accuracy'])} |",
        f"| identity accuracy | {main['identity_correct']}/{main['n']} = {fmt(main['identity_accuracy'])} |",
        f"| accuracy delta | {fmt(main['accuracy_delta'])} |",
        f"| answer_changed | {main['answer_changed']}/{main['n']} = {fmt(main['parsed_answer_change_rate'])} |",
        f"| stable_correct / S_broken / S_repaired / stable_wrong | {main['stable_correct']} / {main['S_broken']} / {main['S_repaired']} / {main['stable_wrong']} |",
        f"| stable_wrong_same / stable_wrong_different | {main['stable_wrong_same']} / {main['stable_wrong_different']} |",
        f"| stable_wrong_different_rate | {main['stable_wrong_different']}/{main['stable_wrong']} = {fmt(main['stable_wrong_different_rate'])} |",
        "",
        "## Bootstrap CI Table",
        "",
        "| Metric | Point | 95% CI |",
        "| --- | ---: | --- |",
    ]
    for key, value in boot.items():
        lines.append(f"| {key} | {fmt(value['point'], 4)} | [{fmt(value['ci_low'], 4)}, {fmt(value['ci_high'], 4)}] |")
    lines.extend([
        "",
        "## Robustness Table",
        "",
        "| Subset | n | Excluded | Answer Changed | Accuracy Delta | Stable-Wrong Different |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in robust_rows:
        lines.append(
            f"| {row.get('subset')} | {row.get('n')} | {row.get('excluded_direct_swap_repetition_count')} | "
            f"{row.get('answer_changed')} ({fmt(row.get('parsed_answer_change_rate'))}) | {fmt(row.get('accuracy_delta'), 4)} | "
            f"{row.get('stable_wrong_different')}/{row.get('stable_wrong')} ({fmt(row.get('stable_wrong_different_rate'))}) |"
        )
    lines.extend([
        "",
        "## Diagnostics Table",
        "",
        "| Diagnostic | Metric | n | Value | 95% CI |",
        "| --- | --- | ---: | ---: | --- |",
        diag_table,
        "",
        "## Patch-Control Caveat Status",
        "",
        f"- status: `{patch_payload.get('status')}`",
        f"- policy: `{patch_payload.get('policy')}`",
        f"- canonical S_broken n: `{patch_payload.get('patched_sbroken_n')}`",
        f"- available overlap records: `{patch_payload.get('available_patched_sbroken_record_n')}`",
        f"- missing canonical S_broken sample identifiers: {len(patch_payload.get('missing_canonical_sbroken_sample_ids') or [])} sample(s) (see artifacts/patch_control/s_broken_ids.txt)",
        f"- unmatched record sample identifiers: `{patch_payload.get('unmatched_record_sample_ids')}`",
        "",
        "Patch-control should not be presented as primary evidence. Existing artifacts support only a match diagnostic because full canonical S_broken coverage is unavailable.",
        "",
        "## Changed Sample Audit",
        "",
        changed_lines or "- None",
        "",
        "## Final LaTeX Stale-Number Grep Result",
        "",
        f"- LaTeX files found: `{len(list(ROOT.rglob('*.tex')))}`",
        f"- stale-number occurrences across repository: `{stale['n_occurrences']}`",
        f"- stale paper errors in `.tex`: `{stale['n_stale_paper_errors']}`",
        "- Remaining stale values are classified in `canonical_stale_number_grep.csv` as historical/audit notes or non-final-paper files.",
        "",
        "## Files Changed",
        "",
        "- `src/evaluation/compute_main_metrics.py`",
        "- `artifacts/main/*`",
        "- `compute_report.md`",
        "- `docs/PROVENANCE_AND_CAVEATS.md`",
        "- `docs/RELEASE_CHECKLIST.md`",
        "- `README.md`",
        "- `final_runs_manifest.json`",
        "",
        "## Strict Rules",
        "",
        "- No model generation was run.",
        "- No HuggingFace model loading was used.",
        "- No dataset download was used.",
        "- Metrics were computed from saved artifacts only.",
        "- Expected values were used only as sanity checks.",
        "",
        "## Validation",
        "",
    ])
    if validation_errors:
        lines.extend(f"- BLOCK: {err}" for err in validation_errors)
    else:
        lines.append("- All canonical consistency checks passed.")
    lines.extend([
        "",
        f"Final token: `{verdict}`",
        "",
    ])
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute main MGSM metrics from raw run artifacts.",
    )
    parser.add_argument(
        "--task",
        choices=["main", "extra_language", "diagnostics", "all"],
        default="all",
        help=(
            "Which computation task to run. "
            "'main' computes Chinese MGSM main metrics. "
            "'extra_language' prints a notice that KO/AR artifacts are pre-computed in this bundle. "
            "'diagnostics' computes diagnostics from shipped trajectory/rationale raw records. "
            "'all' runs the main task (default)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (default: artifacts/main/ derived from repo root).",
    )
    parser.add_argument(
        "--no-canonical-prefix",
        action="store_true",
        default=False,
        help="Write output filenames without the 'canonical_' prefix (e.g. main_metrics.csv instead of canonical_main_metrics.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.task == "extra_language":
        artifact_dir = ROOT / "artifacts" / args.task
        print(json.dumps({
            "task": args.task,
            "status": "pre_computed",
            "message": (
                f"The {args.task} artifacts are included pre-computed in this release bundle. "
                f"See {artifact_dir.relative_to(ROOT).as_posix()} for the CSV and JSON outputs."
            ),
            "artifact_dir": artifact_dir.relative_to(ROOT).as_posix(),
            "artifacts_present": sorted(
                p.name for p in artifact_dir.iterdir() if p.is_file()
            ) if artifact_dir.exists() else [],
        }, ensure_ascii=False, indent=2))
        return

    if args.task == "diagnostics":
        out_dir = args.out_dir if args.out_dir is not None else ROOT / "artifacts" / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        main_rows, _changed_audit = compute_main_rows()
        changed_ids = {row["sample_id"] for row in main_rows if row["answer_changed"]}
        diag_rows, diag_payload = diagnostics(changed_ids)
        write_csv(out_dir / "diagnostics.csv", diag_rows)
        write_json(out_dir / "diagnostics.json", diag_payload)
        print(json.dumps({
            "task": args.task,
            "status": "computed",
            "output_dir": repo(out_dir),
            "answer_changed_n": len(changed_ids),
            "diagnostics": diag_payload,
        }, ensure_ascii=False, indent=2))
        return

    out_dir = args.out_dir if args.out_dir is not None else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "" if args.no_canonical_prefix else "canonical_"

    main_rows, changed_audit = compute_main_rows()
    metrics = main_metrics(main_rows)
    validation_errors = validate_main(metrics)
    verdict = "BLOCK" if validation_errors else "APPROVE WITH WARNINGS"

    boot = paired_bootstrap(main_rows, metric_fn, N_BOOT, SEED)
    robust_rows, robust_payload = robustness_metrics(main_rows)
    changed_ids = {row["sample_id"] for row in main_rows if row["answer_changed"]}
    diag_rows, diag_payload = diagnostics(changed_ids)
    patch_payload = patch_control_caveat(main_rows)
    stale = stale_number_grep()

    write_csv(out_dir / f"{prefix}main_metrics.csv", metric_table(metrics), ["metric", "value"])
    write_json(out_dir / f"{prefix}main_metrics.json", {
        "normalization_policy": "answers_equal(a, b) uses normalize_numeric_answer for correctness and answer identity",
        "source_artifacts": {
            "clean": repo(CLEAN_PATH),
            "direct_swap": repo(HARD_PATH),
            "identity": repo(IDENTITY_PATH),
        },
        "metrics": metrics,
        "validation_errors": validation_errors,
    })
    write_csv(out_dir / f"{prefix}bootstrap_ci.csv", bootstrap_table(boot))
    write_json(out_dir / f"{prefix}bootstrap_ci.json", {
        "n": metrics["n"],
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "method": "percentile",
        "unit": "sample_id",
        "metrics": boot,
    })
    write_csv(out_dir / f"{prefix}changed_samples_audit.csv", changed_audit)
    write_csv(out_dir / f"{prefix}repetition_robustness.csv", robust_rows)
    write_json(out_dir / f"{prefix}repetition_robustness.json", robust_payload)
    write_csv(out_dir / f"{prefix}diagnostics.csv", diag_rows)
    write_json(out_dir / f"{prefix}diagnostics.json", diag_payload)
    write_json(out_dir / f"{prefix}patch_control_caveat.json", patch_payload)
    write_csv(out_dir / f"{prefix}stale_number_grep.csv", stale["rows"])
    write_json(out_dir / f"{prefix}stale_number_grep.json", stale)

    report = write_report(
        verdict,
        metrics,
        boot,
        robust_rows,
        diag_rows,
        patch_payload,
        changed_audit,
        stale,
        validation_errors,
    )
    (out_dir / f"{prefix}compute_report.md").write_text(report, encoding="utf-8")

    write_json(out_dir / f"{prefix}manifest.json", {
        "verdict": verdict,
        "output_dir": repo(out_dir),
        "created_from_existing_artifacts_only": True,
        "no_model_loading": True,
        "no_generation": True,
        "no_dataset_download": True,
        "source_artifacts": {
            "clean": repo(CLEAN_PATH),
            "direct_swap": repo(HARD_PATH),
            "identity": repo(IDENTITY_PATH),
            "output_mode_flags": repo(OUTPUT_MODE_PATH),
            "candidate_margin": repo(CANDIDATE_MARGIN_PATH),
            "numeric_trace": repo(NUMERIC_TRACE_PATH),
            "rationale_conditioned": repo(RATIONALE_PATH),
            "patch_control_records": repo(PATCH_RECORDS),
        },
        "validation_errors": validation_errors,
        "latex_files_found": [repo(path) for path in ROOT.rglob("*.tex")],
        "patch_control_status": patch_payload.get("status"),
        "outputs": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    })

    print(json.dumps({
        "verdict": verdict,
        "output_dir": repo(out_dir),
        "main": metrics,
        "bootstrap": boot,
        "robustness_filtered": next((row for row in robust_rows if row.get("subset") == "filtered"), None),
        "diagnostics": diag_payload,
        "patch_control": patch_payload,
        "validation_errors": validation_errors,
    }, ensure_ascii=False, indent=2))
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

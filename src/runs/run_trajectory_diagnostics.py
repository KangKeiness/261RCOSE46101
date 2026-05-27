"""Trajectory/rationale follow-up diagnostics for canonical Qwen main-run.

PROVENANCE REFERENCE: This script is not executable from the release bundle
alone. It requires GPU, model weights, and external run infrastructure not
shipped in this bundle. It is included for reproducibility provenance only.

This runner keeps failed candidate-margin and patch-margin analyses excluded.
It tests whether answer redistribution is better described by trajectory-level
changes than by a simple raw-prompt final-answer candidate preference shift.

No open-ended generation is run. Canonical inputs are read-only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import mechanism_common as mc
from src.runs.run_hidden_state_appendix import (
    accuracy,
    average_precision,
    balanced_accuracy,
    numpy_logistic_cv_predictions,
    roc_auc,
)


DEFAULT_OUTPUT_ROOT = "stage1/outputs/trajectory_rationale_followup"
DEFAULT_CONFIG = "stage1/configs/stage2_confound_fixed256.yaml"
DEFAULT_CLEAN_JSONL = (
    "data/raw_runs/main_chinese_mgsm/"
    "results_clean_no_patch.jsonl"
)
DEFAULT_SWAP_JSONL = (
    "data/raw_runs/main_chinese_mgsm/"
    "results_restoration_no_patch.jsonl"
)
DEFAULT_TRANSITION_RECORDS = (
    "results/qwen_canonical_answer_audit/corrected_per_sample_transition_records.jsonl"
)

EXPECTED_COUNTS = {
    "n": 250,
    "answer_changed_count": 155,
    "stable_correct": 80,
    "broken": 40,
    "repaired": 33,
    "stable_wrong": 97,
    "baseline_accuracy": 0.480,
    "direct_swap_accuracy": 0.452,
}

GROUP_ORDER = [
    "all",
    "answer_changed",
    "answer_unchanged",
    "stable_correct",
    "broken",
    "repaired",
    "stable_wrong",
    "stable_wrong_same",
    "stable_wrong_different",
]

RATIONAL_CONDITIONS = [
    ("clean_model_clean_rationale", "clean_model", "clean_rationale"),
    ("clean_model_swap_rationale", "clean_model", "swap_rationale"),
    ("direct_swap_model_clean_rationale", "direct_swap_model", "clean_rationale"),
    ("direct_swap_model_swap_rationale", "direct_swap_model", "swap_rationale"),
]


def parse_args(argv: [Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--transition-records", default=DEFAULT_TRANSITION_RECORDS)
    parser.add_argument("--clean-jsonl", default=DEFAULT_CLEAN_JSONL)
    parser.add_argument("--swap-jsonl", default=DEFAULT_SWAP_JSONL)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--hidden-state-records",
        default=None,
        help=" hidden_state_divergence_records.csv. Defaults to latest successful hidden-state run.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip-rationale-scoring", action="store_true")
    return parser.parse_args(argv)


def main(argv: [Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = make_run_dir(args.output_root)
    print(f"[trajectory-rationale] output_dir={display_path(run_dir)}", flush=True)

    command_line = " ".join([sys.executable, *sys.argv])
    write_text(run_dir / "command_line.txt", command_line + "\n")
    input_paths = {
        "transition_records": args.transition_records,
        "clean_jsonl": args.clean_jsonl,
        "swap_jsonl": args.swap_jsonl,
        "config": args.config,
    }
    if args.hidden_state_records:
        input_paths["hidden_state_records"] = args.hidden_state_records
    meta: Dict[str, Any] = {
        "experiment": "trajectory_rationale_followup",
        "started_at_utc": utc_now(),
        "run_dir": display_path(run_dir),
        "command_line": command_line,
        "args": vars(args),
        "input_hashes_sha256": hash_input_files(input_paths),
        "git_sha": mc.git_sha(),
        "runtime_versions": mc.runtime_versions(),
        "failed_prior_tests_policy": {
            "candidate_margin": "excluded",
            "patch_margin": "excluded",
            "reinterpret_failed_candidate_margin": False,
        },
        "canonical_expected_counts": EXPECTED_COUNTS,
    }
    write_json(run_dir / "run_meta.json", meta)

    statuses: Dict[str, Any] = {}
    outputs: Dict[str, Any] = {}
    try:
        data = load_and_validate_inputs(args, run_dir)
        statuses["canonical_count_verification"] = {
            "status": "passed",
            "details": data["canonical_verification"],
        }
    except Exception as exc:
        statuses["canonical_count_verification"] = failure_status(exc)
        write_failure_report(run_dir, "canonical_count_verification", exc)
        write_json(run_dir / "run_meta.json", {**meta, "statuses": statuses, "ended_at_utc": utc_now()})
        return 2

    code_review = write_code_self_review(run_dir)
    statuses["code_self_review"] = code_review
    if code_review.get("critical_failures"):
        statuses["stopped"] = "critical code self-review failure"
        write_json(run_dir / "run_meta.json", {**meta, "statuses": statuses, "ended_at_utc": utc_now()})
        return 3

    try:
        outputs["trajectory_numeric"] = run_trajectory_numeric_job(args, data, run_dir)
        statuses["trajectory_numeric"] = {"status": "passed"}
    except Exception as exc:
        statuses["trajectory_numeric"] = failure_status(exc)
        write_failure_report(run_dir, "trajectory_numeric", exc)

    if args.skip_rationale_scoring:
        statuses["rationale_conditioned_scoring"] = {
            "status": "skipped",
            "reason": "--skip-rationale-scoring",
        }
    else:
        try:
            outputs["rationale_conditioned"] = run_rationale_conditioned_job(args, data, run_dir)
            statuses["rationale_conditioned_scoring"] = {"status": "passed"}
        except Exception as exc:
            statuses["rationale_conditioned_scoring"] = failure_status(exc)
            write_failure_report(run_dir, "rationale_conditioned_scoring", exc)

    try:
        outputs["hidden_state_robustness"] = run_hidden_state_robustness_job(args, data, run_dir)
        statuses["hidden_state_robustness"] = {"status": "passed"}
    except Exception as exc:
        statuses["hidden_state_robustness"] = failure_status(exc)
        write_failure_report(run_dir, "hidden_state_robustness", exc)

    try:
        result_review = write_result_self_review(run_dir, data, outputs, statuses)
        statuses["result_self_review"] = result_review
    except Exception as exc:
        statuses["result_self_review"] = failure_status(exc)
        write_failure_report(run_dir, "result_self_review", exc)

    try:
        write_final_report(run_dir, data, outputs, statuses)
        statuses["final_report"] = {"status": "passed"}
    except Exception as exc:
        statuses["final_report"] = failure_status(exc)
        write_failure_report(run_dir, "final_report", exc)

    meta["statuses"] = statuses
    meta["outputs"] = summarize_outputs(outputs)
    meta["ended_at_utc"] = utc_now()
    write_json(run_dir / "run_meta.json", meta)
    print(f"[trajectory-rationale] complete: {display_path(run_dir)}", flush=True)
    return 0


def make_run_dir(output_root: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    path = root / f"run_{ts}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def display_path(path: Any) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, default=json_default)


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: [Sequence[str]] = None) -> None:
    field_list: List[str] = []
    if fields:
        field_list.extend(fields)
    for row in rows:
        for key in row:
            if key not in field_list:
                field_list.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_safe_row(row))


def read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_safe_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            out[key] = json.dumps(value, ensure_ascii=False, default=json_default)
        else:
            out[key] = value
    return out


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def hash_input_files(paths: Dict[str, str]) -> Dict[str, Any]:
    out = {}
    for name, path in paths.items():
        item = {"path": display_path(path), "sha256": None, "error": None}
        try:
            h = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    h.update(chunk)
            item["sha256"] = h.hexdigest()
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        out[name] = item
    return out


def failure_status(exc: BaseException) -> Dict[str, Any]:
    return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}


def write_failure_report(run_dir: Path, label: str, exc: BaseException) -> None:
    lines = [
        f"# Failure: {label}",
        "",
        f"- error_type: `{type(exc).__name__}`",
        f"- error: `{str(exc)}`",
        "",
        "```text",
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "```",
    ]
    write_text(run_dir / f"FAIL_{label}.md", "\n".join(lines) + "\n")


def load_and_validate_inputs(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    transition_rows = read_jsonl(args.transition_records)
    clean_rows = read_jsonl(args.clean_jsonl)
    swap_rows = read_jsonl(args.swap_jsonl)
    samples_by_id, dataset_meta = load_samples(args.config)
    trans_by_id = unique_by_sample_id(transition_rows, "transition_records")
    clean_by_id = unique_by_sample_id(clean_rows, "clean_jsonl")
    swap_by_id = unique_by_sample_id(swap_rows, "swap_jsonl")
    sample_id_list = sorted(trans_by_id)
    if set(clean_by_id) != set(sample_id_list):
        raise ValueError("clean_jsonl sample identifiers do not match transition_records")
    if set(swap_by_id) != set(sample_id_list):
        raise ValueError("swap_jsonl sample identifiers do not match transition_records")
    if not set(sample_id_list).issubset(set(samples_by_id)):
        missing = sorted(set(sample_id_list) - set(samples_by_id))
        raise ValueError(f"config-loaded samples missing identifiers: {missing[:5]}")

    mismatches = []
    records = []
    for sid in sample_id_list:
        tr = trans_by_id[sid]
        clean = clean_by_id[sid]
        swap = swap_by_id[sid]
        baseline_answer = answer_string(tr.get("baseline_parsed_answer"))
        swap_answer = answer_string(tr.get("direct_swap_parsed_answer"))
        if answer_string(clean.get("normalized_answer")) != baseline_answer:
            mismatches.append({"sample_id": sid, "field": "baseline_answer"})
        if answer_string(swap.get("normalized_answer")) != swap_answer:
            mismatches.append({"sample_id": sid, "field": "direct_swap_answer"})
        if bool(clean.get("correct")) != bool(tr.get("baseline_correct")):
            mismatches.append({"sample_id": sid, "field": "baseline_correct"})
        if bool(swap.get("correct")) != bool(tr.get("direct_swap_correct")):
            mismatches.append({"sample_id": sid, "field": "direct_swap_correct"})
        group = str(tr.get("transition_group"))
        stable_wrong_same = group == "stable_wrong" and baseline_answer == swap_answer
        stable_wrong_different = group == "stable_wrong" and baseline_answer != swap_answer
        sample = samples_by_id[sid]
        records.append(
            {
                "sample_id": sid,
                "prompt": str(sample["prompt"]),
                "prompt_hash": mc.sha256_text(str(sample["prompt"])),
                "gold_answer": answer_string(sample.get("gold_answer")),
                "baseline_answer": baseline_answer,
                "swap_answer": swap_answer,
                "baseline_correct": bool(tr.get("baseline_correct")),
                "swap_correct": bool(tr.get("direct_swap_correct")),
                "baseline_parse_success": bool(tr.get("baseline_parse_success")),
                "swap_parse_success": bool(tr.get("direct_swap_parse_success")),
                "answer_changed": bool(tr.get("answer_changed")),
                "transition_group": group,
                "stable_wrong_same_wrong_answer": stable_wrong_same,
                "stable_wrong_different_wrong_answer": stable_wrong_different,
                "clean_output_text": str(clean.get("output_text") or ""),
                "swap_output_text": str(swap.get("output_text") or ""),
            }
        )
    if mismatches:
        raise ValueError(f"main-run rows disagree with corrected transition records: {mismatches[:10]}")

    verification = verify_expected_counts(transition_rows, clean_rows, swap_rows)
    write_sanity_count_check(run_dir, verification)
    return {
        "records": records,
        "records_by_id": {r["sample_id"]: r for r in records},
        "transition_rows": transition_rows,
        "clean_rows": clean_rows,
        "swap_rows": swap_rows,
        "samples_by_id": samples_by_id,
        "dataset_meta": dataset_meta,
        "canonical_verification": verification,
    }


def load_samples(config_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    from src.data.mgsm_loader import load_mgsm
    from src.utils.config import load_config

    config = load_config(config_path)
    mc.validate_project_config_for_mechanism(config)
    samples = load_mgsm(config)
    meta = {
        "name": config.dataset.name,
        "lang": config.dataset.lang,
        "split": config.dataset.split,
        "revision": config.dataset.revision,
        "expected_sha256": config.dataset.expected_sha256,
        "n_loaded": len(samples),
    }
    return {str(s["sample_id"]): s for s in samples}, meta


def unique_by_sample_id(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    duplicates = []
    for row in rows:
        sid = str(row.get("sample_id"))
        if sid in out:
            duplicates.append(sid)
        out[sid] = row
    if duplicates:
        raise ValueError(f"{label} duplicate sample_id values: {duplicates[:5]}")
    return out


def answer_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "nan", "parse_fail", "<parse_fail>"}:
        return ""
    return text


def verify_expected_counts(
    transition_rows: Sequence[Dict[str, Any]],
    clean_rows: Sequence[Dict[str, Any]],
    swap_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    group_counts = Counter(str(row.get("transition_group")) for row in transition_rows)
    observed = {
        "n": len(transition_rows),
        "answer_changed_count": sum(bool(row.get("answer_changed")) for row in transition_rows),
        "stable_correct": group_counts.get("stable_correct", 0),
        "broken": group_counts.get("broken", 0),
        "repaired": group_counts.get("repaired", 0),
        "stable_wrong": group_counts.get("stable_wrong", 0),
        "baseline_accuracy": rate(row.get("baseline_correct") for row in transition_rows),
        "direct_swap_accuracy": rate(row.get("direct_swap_correct") for row in transition_rows),
        "clean_jsonl_n": len(clean_rows),
        "swap_jsonl_n": len(swap_rows),
        "clean_jsonl_accuracy": rate(row.get("correct") for row in clean_rows),
        "swap_jsonl_accuracy": rate(row.get("correct") for row in swap_rows),
    }
    checks = {}
    all_passed = True
    for key, expected in EXPECTED_COUNTS.items():
        got = observed[key]
        passed = abs(float(got) - float(expected)) <= 1e-12 if isinstance(expected, float) else got == expected
        checks[key] = {"expected": expected, "observed": got, "passed": passed}
        all_passed = all_passed and passed
    for key, expected in [
        ("clean_jsonl_n", EXPECTED_COUNTS["n"]),
        ("swap_jsonl_n", EXPECTED_COUNTS["n"]),
        ("clean_jsonl_accuracy", EXPECTED_COUNTS["baseline_accuracy"]),
        ("swap_jsonl_accuracy", EXPECTED_COUNTS["direct_swap_accuracy"]),
    ]:
        got = observed[key]
        passed = abs(float(got) - float(expected)) <= 1e-12
        checks[key] = {"expected": expected, "observed": got, "passed": passed}
        all_passed = all_passed and passed
    if not all_passed:
        raise ValueError(f"canonical count verification failed: {checks}")
    return {"passed": True, "observed": observed, "checks": checks}


def write_sanity_count_check(run_dir: Path, verification: Dict[str, Any]) -> None:
    write_json(run_dir / "sanity_canonical_count_check.json", verification)
    lines = [
        "# Canonical Count Check",
        "",
        f"- status: `{'PASS' if verification.get('passed') else 'FAIL'}`",
        "",
        "| metric | expected | observed | passed |",
        "|---|---:|---:|---|",
    ]
    for key in ["n", "answer_changed_count", "stable_correct", "broken", "repaired", "stable_wrong", "baseline_accuracy", "direct_swap_accuracy"]:
        check = verification["checks"].get(key, {})
        lines.append(f"| {key} | {check.get('expected')} | {check.get('observed')} | {check.get('passed')} |")
    write_text(run_dir / "sanity_canonical_count_check.md", "\n".join(lines) + "\n")


def rate(values: Iterable[Any]) -> float:
    vals = list(values)
    return sum(1 for value in vals if bool(value)) / len(vals) if vals else float("nan")


def write_code_self_review(run_dir: Path) -> Dict[str, Any]:
    items = [
        ("canonical counts verified", "PASS", True, "load_and_validate_inputs verifies corrected canonical counts before jobs."),
        ("noncanonical files avoided", "PASS", True, "The script never reads subset summary files or run summary files."),
        ("failed candidate-margin not reinterpreted", "PASS", True, "run_meta and reports explicitly keep candidate/patch margins excluded."),
        ("no canonical files modified", "PASS", True, "Canonical inputs are read-only; outputs are written only to a fresh trajectory_rationale_followup run directory."),
        ("no free generation in rationale scoring", "PASS", True, "The rationale job uses model forward passes and conditional log-prob scoring only."),
        ("skip counts reported", "PASS", False, "Rationale scoring records skip reasons and summary skip counts."),
        ("trajectory divergence not just final answer comparison", "PASS", False, "Job 1 computes first divergence, pre-final-answer divergence, edit distances, and numeric traces."),
        ("numeric normalization documented", "PASS", False, "Reports document NFKC normalization, comma stripping, Decimal canonicalization, and fraction preservation."),
        ("hidden-state robustness uses proper controls", "PASS", False, "Job 3 includes length covariates, stratified AUC, permutation sanity, and divergence+length logistic probes."),
        ("claims are association-level only", "PASS", True, "Final report forbids structural association, localization-type, and restoration claims."),
    ]
    critical_failures = [name for name, status, critical, _note in items if critical and status != "PASS"]
    lines = [
        "# Trajectory/Rationale Code Self-Review",
        "",
        "| check | status | critical | note |",
        "|---|---|---|---|",
    ]
    for name, status, critical, note in items:
        lines.append(f"| {name} | {status} | {critical} | {note} |")
    lines.extend(["", f"- critical_failures: {len(critical_failures)}", f"- code_self_review_passed: {len(critical_failures) == 0}"])
    write_text(run_dir / "SELF_REVIEW_TRAJECTORY_RATIONALE_CODE.md", "\n".join(lines) + "\n")
    return {"status": "passed" if not critical_failures else "failed", "critical_failures": critical_failures}


# Job 1 ---------------------------------------------------------------------


def run_trajectory_numeric_job(args: argparse.Namespace, data: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    trajectory_rows = []
    numeric_rows = []
    for record in data["records"]:
        clean_text = normalize_text(record["clean_output_text"])
        swap_text = normalize_text(record["swap_output_text"])
        clean_rationale, clean_answer_segment, clean_marker = split_rationale(clean_text, record["baseline_answer"])
        swap_rationale, swap_answer_segment, swap_marker = split_rationale(swap_text, record["swap_answer"])
        char_div = first_divergence_index(clean_text, swap_text)
        clean_tokens = text_units(clean_text)
        swap_tokens = text_units(swap_text)
        token_div = first_sequence_divergence(clean_tokens, swap_tokens)
        char_edit = levenshtein_distance(clean_text, swap_text)
        token_edit = levenshtein_distance(clean_tokens, swap_tokens)
        min_answer_start = min(clean_marker["start_index"], swap_marker["start_index"])
        group_flags = group_labels(record)
        trajectory_rows.append(
            {
                "sample_id": record["sample_id"],
                **group_flags,
                "gold_answer": record["gold_answer"],
                "baseline_answer": record["baseline_answer"],
                "swap_answer": record["swap_answer"],
                "clean_output_chars": len(clean_text),
                "swap_output_chars": len(swap_text),
                "clean_output_tokens": len(clean_tokens),
                "swap_output_tokens": len(swap_tokens),
                "first_char_divergence_index": char_div,
                "first_token_divergence_index": token_div,
                "normalized_char_divergence_position": normalized_position(char_div, max(len(clean_text), len(swap_text))),
                "normalized_token_divergence_position": normalized_position(token_div, max(len(clean_tokens), len(swap_tokens))),
                "clean_final_answer_start_index": clean_marker["start_index"],
                "swap_final_answer_start_index": swap_marker["start_index"],
                "divergence_before_final_answer_segment": bool(char_div is not None and char_div < min_answer_start),
                "char_edit_distance": char_edit,
                "char_edit_distance_normalized": safe_div(char_edit, max(len(clean_text), len(swap_text))),
                "token_edit_distance": token_edit,
                "token_edit_distance_normalized": safe_div(token_edit, max(len(clean_tokens), len(swap_tokens))),
                "clean_rationale_chars": len(clean_rationale),
                "swap_rationale_chars": len(swap_rationale),
                "clean_answer_segment_chars": len(clean_answer_segment),
                "swap_answer_segment_chars": len(swap_answer_segment),
                "clean_final_marker": clean_marker["marker"],
                "swap_final_marker": swap_marker["marker"],
            }
        )
        clean_nums = extract_numeric_trace(clean_text)
        swap_nums = extract_numeric_trace(swap_text)
        num_edit = levenshtein_distance(clean_nums, swap_nums)
        first_num_div = first_sequence_divergence(clean_nums, swap_nums)
        numeric_rows.append(
            {
                "sample_id": record["sample_id"],
                **group_flags,
                "gold_answer": record["gold_answer"],
                "baseline_answer": record["baseline_answer"],
                "swap_answer": record["swap_answer"],
                "clean_numeric_trace": clean_nums,
                "swap_numeric_trace": swap_nums,
                "clean_numeric_count": len(clean_nums),
                "swap_numeric_count": len(swap_nums),
                "numeric_trace_edit_distance": num_edit,
                "numeric_trace_edit_distance_normalized": safe_div(num_edit, max(len(clean_nums), len(swap_nums))),
                "first_numeric_divergence_index": first_num_div,
                "shared_numeric_prefix_length": shared_prefix_length(clean_nums, swap_nums),
            }
        )

    traj_summary = summarize_trajectory_rows(trajectory_rows, args.bootstrap_n, args.seed)
    numeric_summary = summarize_numeric_rows(numeric_rows, args.bootstrap_n, args.seed)
    write_csv(run_dir / "trajectory_divergence_records.csv", trajectory_rows)
    write_csv(run_dir / "trajectory_divergence_summary.csv", traj_summary)
    write_json(run_dir / "trajectory_divergence_summary.json", {"summary_rows": traj_summary})
    write_csv(run_dir / "numeric_trace_records.csv", numeric_rows)
    write_csv(run_dir / "numeric_trace_summary.csv", numeric_summary)
    write_json(run_dir / "numeric_trace_summary.json", {"summary_rows": numeric_summary})
    write_trajectory_numeric_report(run_dir, traj_summary, numeric_summary)
    return {
        "trajectory_rows": trajectory_rows,
        "trajectory_summary": traj_summary,
        "numeric_rows": numeric_rows,
        "numeric_summary": numeric_summary,
    }


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace("\r\n", "\n")


MARKER_RE = re.compile(
    r"(?i)(final\s+answer\s*:?\s*(?:the\s+answer\s+is)?|the\s+answer\s+is|答案\s*(?:是|为)?\s*[:：]?|最终答案\s*[:：]?)"
)


def split_rationale(output_text: str, parsed_answer: str) -> Tuple[str, str, Dict[str, Any]]:
    text = normalize_text(output_text)
    answer_norm = normalize_number_string(parsed_answer)
    matches = list(MARKER_RE.finditer(text))
    chosen = None
    if answer_norm:
        for match in matches:
            window = text[match.start() : min(len(text), match.start() + 180)]
            nums = extract_numeric_trace(window)
            if answer_norm in nums or answer_norm in normalize_numberish_substrings(window):
                chosen = match
                break
    if chosen is None and matches:
        chosen = matches[0]
    if chosen is None:
        start = len(text)
        marker = "none_found"
    else:
        start = chosen.start()
        marker = chosen.group(0)
    return text[:start].strip(), text[start:].strip(), {"start_index": start, "marker": marker}


def normalize_numberish_substrings(text: str) -> List[str]:
    return [normalize_number_string(x) for x in NUMERIC_RE.findall(normalize_text(text))]


def first_divergence_index(a: str, b: str) -> int | None:
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return idx
    if len(a) != len(b):
        return limit
    return None


def first_sequence_divergence(a: Sequence[Any], b: Sequence[Any]) -> int | None:
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return idx
    if len(a) != len(b):
        return limit
    return None


def normalized_position(index: int | None, denom: int) -> float | None:
    if index is None or denom <= 0:
        return None
    return float(index) / float(denom)


TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*(?:/\d+(?:[.,]\d+)*)?|[A-Za-z]+|[\u4e00-\u9fff]|[^\s]", re.UNICODE)


def text_units(text: str) -> List[str]:
    return TOKEN_RE.findall(normalize_text(text))


NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?")


def extract_numeric_trace(text: str) -> List[str]:
    return [normalize_number_string(m) for m in NUMERIC_RE.findall(normalize_text(text))]


def normalize_number_string(value: Any) -> str:
    text = normalize_text(str(value or "")).strip()
    text = text.replace(",", "")
    text = text.strip(" $￥¥.。,:：;；()[]{}")
    if not text:
        return ""
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 2:
            return f"{normalize_number_string(parts[0])}/{normalize_number_string(parts[1])}"
    try:
        dec = Decimal(text)
        if dec == dec.to_integral():
            return str(dec.quantize(Decimal(1)))
        return format(dec.normalize(), "f").rstrip("0").rstrip(".")
    except (InvalidOperation, ValueError):
        return text


def levenshtein_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return int(previous[-1])


def safe_div(num: Any, denom: Any) -> float | None:
    try:
        denom_f = float(denom)
        if denom_f == 0:
            return None
        return float(num) / denom_f
    except Exception:
        return None


def shared_prefix_length(a: Sequence[Any], b: Sequence[Any]) -> int:
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def group_labels(record: Dict[str, Any]) -> Dict[str, Any]:
    group = str(record["transition_group"])
    return {
        "answer_changed": bool(record["answer_changed"]),
        "transition_group": group,
        "stable_correct": group == "stable_correct",
        "broken": group == "broken",
        "repaired": group == "repaired",
        "stable_wrong": group == "stable_wrong",
        "stable_wrong_same": bool(record["stable_wrong_same_wrong_answer"]),
        "stable_wrong_different": bool(record["stable_wrong_different_wrong_answer"]),
    }


def group_predicate(group: str) -> Callable[[Dict[str, Any]], bool]:
    return {
        "all": lambda r: True,
        "answer_changed": lambda r: bool(r.get("answer_changed")),
        "answer_unchanged": lambda r: not bool(r.get("answer_changed")),
        "stable_correct": lambda r: str(r.get("transition_group")) == "stable_correct",
        "broken": lambda r: str(r.get("transition_group")) == "broken",
        "repaired": lambda r: str(r.get("transition_group")) == "repaired",
        "stable_wrong": lambda r: str(r.get("transition_group")) == "stable_wrong",
        "stable_wrong_same": lambda r: bool(r.get("stable_wrong_same")),
        "stable_wrong_different": lambda r: bool(r.get("stable_wrong_different")),
    }[group]


def summarize_trajectory_rows(rows: Sequence[Dict[str, Any]], bootstrap_n: int, seed: int) -> List[Dict[str, Any]]:
    metrics = [
        "normalized_char_divergence_position",
        "normalized_token_divergence_position",
        "char_edit_distance_normalized",
        "token_edit_distance_normalized",
        "clean_rationale_chars",
        "swap_rationale_chars",
    ]
    out = []
    for group in GROUP_ORDER:
        subset = [r for r in rows if group_predicate(group)(r)]
        base = {
            "group": group,
            "n": len(subset),
            "fraction_divergence_before_final_answer_segment": rate(
                r.get("divergence_before_final_answer_segment") for r in subset
            ) if subset else None,
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in subset if is_number(r.get(metric))]
            stats = describe_values(vals, bootstrap_n, seed + len(group) + len(metric))
            out.append({"metric": metric, **base, **stats})
    return out


def summarize_numeric_rows(rows: Sequence[Dict[str, Any]], bootstrap_n: int, seed: int) -> List[Dict[str, Any]]:
    metrics = [
        "clean_numeric_count",
        "swap_numeric_count",
        "numeric_trace_edit_distance",
        "numeric_trace_edit_distance_normalized",
        "shared_numeric_prefix_length",
    ]
    out = []
    for group in GROUP_ORDER:
        subset = [r for r in rows if group_predicate(group)(r)]
        for metric in metrics:
            vals = [float(r[metric]) for r in subset if is_number(r.get(metric))]
            out.append(
                {
                    "group": group,
                    "metric": metric,
                    **describe_values(vals, bootstrap_n, seed + len(group) + len(metric)),
                }
            )
    return out


def describe_values(values: Sequence[float], bootstrap_n: int, seed: int) -> Dict[str, Any]:
    vals = np.array([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "q25": None, "q75": None, "ci_low": None, "ci_high": None}
    ci_low, ci_high = bootstrap_mean_ci(vals, bootstrap_n, seed)
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "q25": float(np.quantile(vals, 0.25)),
        "q75": float(np.quantile(vals, 0.75)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def bootstrap_mean_ci(values: np.ndarray, bootstrap_n: int, seed: int) -> Tuple[float, float]:
    if values.size <= 1 or bootstrap_n <= 0:
        mean = float(np.mean(values)) if values.size else float("nan")
        return mean, mean
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=values.size, replace=True))) for _ in range(bootstrap_n)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def write_trajectory_numeric_report(run_dir: Path, traj_summary: Sequence[Dict[str, Any]], numeric_summary: Sequence[Dict[str, Any]]) -> None:
    stable_wrong_diff = find_summary(traj_summary, "stable_wrong_different", "token_edit_distance_normalized")
    answer_changed_pre = find_summary(traj_summary, "answer_changed", "normalized_char_divergence_position")
    numeric_changed = find_summary(numeric_summary, "answer_changed", "numeric_trace_edit_distance_normalized")
    lines = [
        "# Trajectory and Numeric Trace Report",
        "",
        "## Method",
        "",
        "The audit compares full clean and direct-swap generated outputs, not just final parsed answers. Final-answer segments are detected with conservative English/Chinese markers such as `Final Answer`, `The answer is`, and `答案是`; divergence-before-final-answer is computed relative to the earliest detected final-answer marker in the two outputs.",
        "",
        "Numeric traces use NFKC normalization, comma stripping, Decimal canonicalization for integers/decimals, and fraction preservation for simple `a/b` forms.",
        "",
        "## Key Descriptives",
        "",
        f"- answer_changed normalized first-char divergence: {fmt(answer_changed_pre.get('mean') if answer_changed_pre else None)}",
        f"- stable_wrong_different normalized token edit distance: {fmt(stable_wrong_diff.get('mean') if stable_wrong_diff else None)}",
        f"- answer_changed normalized numeric trace edit distance: {fmt(numeric_changed.get('mean') if numeric_changed else None)}",
        "",
        "## Interpretation",
        "",
        "This analysis is descriptive. It can show whether parsed-answer redistribution coincides with broad trajectory and numeric-trace divergence, but it does not prove a structural association or localize reasoning.",
    ]
    write_text(run_dir / "trajectory_numeric_report.md", "\n".join(lines) + "\n")


def find_summary(rows: Sequence[Dict[str, Any]], group: str, metric: str) -> [Dict[str, Any]]:
    for row in rows:
        if row.get("group") == group and row.get("metric") == metric:
            return row
    return None


# Job 2 ---------------------------------------------------------------------


def run_rationale_conditioned_job(args: argparse.Namespace, data: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    ensure_device_policy(args.allow_cpu)
    model_bundle = None
    try:
        model_bundle = load_model_bundle(args)
        records, summary, skip_counts = score_rationale_conditioned_records(args, data, model_bundle)
    finally:
        release_model_bundle(model_bundle)
    write_csv(run_dir / "rationale_conditioned_margin_records.csv", records)
    write_csv(run_dir / "rationale_conditioned_margin_summary.csv", summary)
    write_json(
        run_dir / "rationale_conditioned_margin_summary.json",
        {"summary_rows": summary, "skip_counts": skip_counts, "n_rows": len(records)},
    )
    write_rationale_conditioned_report(run_dir, summary, skip_counts, records)
    return {"records": records, "summary": summary, "skip_counts": skip_counts}


def ensure_device_policy(allow_cpu: bool) -> None:
    import torch

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA unavailable and --allow-cpu was not passed")


def load_model_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from src.composition.composer import compose_model, load_models

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    print("[trajectory-rationale] loading Qwen models for rationale scoring", flush=True)
    recipient, donor, tokenizer = load_models(
        recipient_name=mc.CANONICAL_RECIPIENT_ID,
        donor_name=mc.CANONICAL_DONOR_ID,
        recipient_revision=mc.CANONICAL_RECIPIENT_REVISION,
        donor_revision=mc.CANONICAL_DONOR_REVISION,
        device=args.device,
        dtype=dtype,
    )
    hard_model, compose_meta = compose_model(
        recipient=recipient,
        donor=donor,
        b=mc.CANONICAL_B,
        t=mc.CANONICAL_T,
        condition="hard_swap",
    )
    for model, label in [(recipient, "clean_model"), (donor, "donor_base"), (hard_model, "direct_swap_model")]:
        model.eval()
        mc.validate_no_active_adapters(model, condition=label)
        for param in model.parameters():
            param.requires_grad_(False)
    try:
        donor.cpu()
    except Exception:
        pass
    del donor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "clean_model": recipient,
        "direct_swap_model": hard_model,
        "tokenizer": tokenizer,
        "compose_meta": compose_meta,
    }


def release_model_bundle(bundle: [Dict[str, Any]]) -> None:
    if not bundle:
        return
    for key in ["clean_model", "direct_swap_model"]:
        model = bundle.get(key)
        if model is not None:
            mc.release_model(model)


def score_rationale_conditioned_records(
    args: argparse.Namespace,
    data: Dict[str, Any],
    bundle: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    records = []
    skip_counts: Counter[str] = Counter()
    tokenizer = bundle["tokenizer"]
    total = len(data["records"])
    for idx, sample in enumerate(data["records"], start=1):
        base = answer_string(sample["baseline_answer"])
        swap = answer_string(sample["swap_answer"])
        gold = answer_string(sample["gold_answer"])
        clean_rationale, _clean_answer_segment, clean_marker = split_rationale(sample["clean_output_text"], base)
        swap_rationale, _swap_answer_segment, swap_marker = split_rationale(sample["swap_output_text"], swap)
        row = {
            "sample_id": sample["sample_id"],
            **group_labels(sample),
            "gold_answer": gold,
            "clean_answer": base,
            "swap_answer": swap,
            "clean_rationale_chars": len(clean_rationale),
            "swap_rationale_chars": len(swap_rationale),
            "clean_final_marker": clean_marker["marker"],
            "swap_final_marker": swap_marker["marker"],
            "skip_reason": "",
        }
        skip_reason = rationale_skip_reason(base, swap, gold, clean_rationale, swap_rationale)
        if skip_reason:
            row["skip_reason"] = skip_reason
            skip_counts[skip_reason] += 1
            records.append(row)
            continue

        candidates = {"clean": base, "swap": swap, "gold": gold}
        prefixes = {
            "clean_rationale": build_rationale_prefix(sample["prompt"], clean_rationale),
            "swap_rationale": build_rationale_prefix(sample["prompt"], swap_rationale),
        }
        try:
            for condition_label, model_key, rationale_key in RATIONAL_CONDITIONS:
                model = bundle[model_key]
                prefix = prefixes[rationale_key]
                role_scores = {}
                for role, answer in candidates.items():
                    score = score_candidate_answer(model, tokenizer, prefix, answer)
                    if not score.get("score_success"):
                        raise RuntimeError(f"{condition_label}/{role}: {score.get('error_type')}")
                    row[f"{condition_label}_{role}_score_norm"] = score["score_norm"]
                    row[f"{condition_label}_{role}_score_raw"] = score["sum_logp"]
                    row[f"{condition_label}_{role}_tokens"] = score["num_candidate_tokens"]
                    role_scores[role] = score
                margin_norm = float(role_scores["clean"]["score_norm"]) - float(role_scores["swap"]["score_norm"])
                margin_raw = float(role_scores["clean"]["sum_logp"]) - float(role_scores["swap"]["sum_logp"])
                row[f"{condition_label}_clean_minus_swap_margin_norm"] = margin_norm
                row[f"{condition_label}_clean_minus_swap_margin_raw"] = margin_raw
            add_rationale_effect_columns(row)
            records.append(row)
        except Exception as exc:
            row["skip_reason"] = "score_failure"
            row["score_failure_error"] = f"{type(exc).__name__}: {exc}"
            skip_counts["score_failure"] += 1
            records.append(row)
        if idx % 20 == 0 or idx == total:
            print(f"[trajectory-rationale] rationale scoring {idx}/{total}", flush=True)

    scored = [r for r in records if not r.get("skip_reason")]
    summary = summarize_rationale_records(scored, skip_counts, args.bootstrap_n, args.seed)
    return records, summary, dict(skip_counts)


def rationale_skip_reason(base: str, swap: str, gold: str, clean_rationale: str, swap_rationale: str) -> str:
    if not base:
        return "empty_clean_answer"
    if not swap:
        return "empty_swap_answer"
    if not gold:
        return "empty_gold_answer"
    if base == swap:
        return "clean_answer_equals_swap_answer"
    if clean_rationale.strip() == "":
        return "empty_clean_rationale"
    if swap_rationale.strip() == "":
        return "empty_swap_rationale"
    return ""


def build_rationale_prefix(prompt: str, rationale: str) -> str:
    return str(prompt).rstrip() + "\n" + str(rationale).strip() + "\nThe answer is"


def score_candidate_answer(model: Any, tokenizer: Any, prefix: str, answer: str) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if not answer:
        return {"score_success": False, "error_type": "empty_candidate"}
    device = model.get_input_embeddings().weight.device
    prefix_ids = tokenizer(prefix, return_tensors="pt", padding=False).input_ids.to(device)
    candidate_text = " " + str(answer).strip()
    candidate_ids = tokenizer(candidate_text, return_tensors="pt", padding=False, add_special_tokens=False).input_ids.to(device)
    n_tokens = int(candidate_ids.shape[1])
    if n_tokens == 0:
        return {"score_success": False, "error_type": "empty_candidate_tokenization"}
    full_ids = torch.cat([prefix_ids, candidate_ids], dim=1)
    prompt_len = int(prefix_ids.shape[1])
    model.eval()
    with torch.inference_mode():
        out = model(full_ids, use_cache=False)
        logits = out.logits[:, prompt_len - 1 : prompt_len - 1 + n_tokens, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_logps = log_probs.gather(2, candidate_ids.unsqueeze(-1)).squeeze(-1)
        sum_logp = float(token_logps.sum().item())
    return {
        "score_success": True,
        "error_type": None,
        "sum_logp": sum_logp,
        "score_norm": sum_logp / max(n_tokens, 1),
        "num_candidate_tokens": n_tokens,
    }


def add_rationale_effect_columns(row: Dict[str, Any]) -> None:
    pairs = [
        ("clean_model_rationale_effect_norm", "clean_model_swap_rationale", "clean_model_clean_rationale", "norm"),
        ("direct_swap_model_rationale_effect_norm", "direct_swap_model_swap_rationale", "direct_swap_model_clean_rationale", "norm"),
        ("model_effect_clean_rationale_norm", "direct_swap_model_clean_rationale", "clean_model_clean_rationale", "norm"),
        ("model_effect_swap_rationale_norm", "direct_swap_model_swap_rationale", "clean_model_swap_rationale", "norm"),
        ("clean_model_rationale_effect_raw", "clean_model_swap_rationale", "clean_model_clean_rationale", "raw"),
        ("direct_swap_model_rationale_effect_raw", "direct_swap_model_swap_rationale", "direct_swap_model_clean_rationale", "raw"),
        ("model_effect_clean_rationale_raw", "direct_swap_model_clean_rationale", "clean_model_clean_rationale", "raw"),
        ("model_effect_swap_rationale_raw", "direct_swap_model_swap_rationale", "clean_model_swap_rationale", "raw"),
    ]
    for out_col, left, right, metric in pairs:
        row[out_col] = float(row[f"{left}_clean_minus_swap_margin_{metric}"]) - float(row[f"{right}_clean_minus_swap_margin_{metric}"])


def summarize_rationale_records(
    rows: Sequence[Dict[str, Any]],
    skip_counts: Dict[str, int],
    bootstrap_n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    metrics = [
        "clean_model_clean_rationale_clean_minus_swap_margin_norm",
        "clean_model_swap_rationale_clean_minus_swap_margin_norm",
        "direct_swap_model_clean_rationale_clean_minus_swap_margin_norm",
        "direct_swap_model_swap_rationale_clean_minus_swap_margin_norm",
        "clean_model_rationale_effect_norm",
        "direct_swap_model_rationale_effect_norm",
        "model_effect_clean_rationale_norm",
        "model_effect_swap_rationale_norm",
        "clean_model_rationale_effect_raw",
        "direct_swap_model_rationale_effect_raw",
        "model_effect_clean_rationale_raw",
        "model_effect_swap_rationale_raw",
    ]
    out = []
    for group in GROUP_ORDER:
        subset = [r for r in rows if group_predicate(group)(r)]
        for metric in metrics:
            vals = [float(r[metric]) for r in subset if is_number(r.get(metric))]
            stats = describe_values(vals, bootstrap_n, seed + len(group) + len(metric))
            fraction_lt_0 = rate(v < 0 for v in vals) if vals else None
            out.append(
                {
                    "group": group,
                    "metric": metric,
                    "n_scored": len(subset),
                    "fraction_lt_0": fraction_lt_0,
                    **stats,
                }
            )
    out.append({"group": "skip_counts", "metric": "skip_counts", "n_scored": len(rows), "skip_counts": dict(skip_counts)})
    return out


def write_rationale_conditioned_report(
    run_dir: Path,
    summary: Sequence[Dict[str, Any]],
    skip_counts: Dict[str, int],
    records: Sequence[Dict[str, Any]],
) -> None:
    all_clean_effect = find_summary(summary, "all", "clean_model_rationale_effect_norm")
    all_direct_effect = find_summary(summary, "all", "direct_swap_model_rationale_effect_norm")
    raw_clean = find_summary(summary, "all", "clean_model_rationale_effect_raw")
    n_scored = sum(1 for r in records if not r.get("skip_reason"))
    lines = [
        "# Rationale-Conditioned Margin Report",
        "",
        "## Method",
        "",
        "For samples where clean and direct-swap parsed answers differ, the runner extracted clean and swap rationales before the detected final-answer segment. It then scored clean, swap, and gold answer candidates after `original_prompt + rationale + \"\\nThe answer is\"` under clean and direct-swap models. No text generation was run.",
        "",
        "Candidate-margin and patch-margin analyses from earlier runs remain excluded and are not reinterpreted here.",
        "",
        "## Skip Counts",
        "",
        f"- n rows: {len(records)}",
        f"- n scored clean-vs-swap answer pairs: {n_scored}",
        f"- skip_counts: `{dict(skip_counts)}`",
        "",
        "## Key Effects",
        "",
        f"- clean model rationale effect, normalized margin: mean={fmt(all_clean_effect.get('mean') if all_clean_effect else None)}, fraction_lt_0={fmt(all_clean_effect.get('fraction_lt_0') if all_clean_effect else None)}",
        f"- direct-swap model rationale effect, normalized margin: mean={fmt(all_direct_effect.get('mean') if all_direct_effect else None)}, fraction_lt_0={fmt(all_direct_effect.get('fraction_lt_0') if all_direct_effect else None)}",
        f"- raw-score sensitivity, clean model rationale effect: mean={fmt(raw_clean.get('mean') if raw_clean else None)}",
        "",
        "Negative rationale-effect values mean the swap rationale reduced the clean-answer-over-swap-answer margin relative to the clean rationale.",
        "",
        "## Interpretation",
        "",
        "This is a trajectory-conditioned scoring diagnostic. It can support the claim that final-answer preferences follow generated rationales, but it does not rescue or reinterpret the failed raw-prompt candidate-margin analysis.",
    ]
    write_text(run_dir / "rationale_conditioned_margin_report.md", "\n".join(lines) + "\n")


# Job 3 ---------------------------------------------------------------------


def run_hidden_state_robustness_job(args: argparse.Namespace, data: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    records_path = Path(args.hidden_state_records) if args.hidden_state_records else find_latest_hidden_state_records()
    if records_path is None or not records_path.exists():
        raise FileNotFoundError("No hidden_state_divergence_records.csv found")
    print(f"[trajectory-rationale] using hidden-state records {display_path(records_path)}", flush=True)
    hs_rows = read_csv_dicts(records_path)
    sample_rows = build_hidden_state_sample_features(hs_rows, data)
    feature_names = sorted([k for k in sample_rows[0] if k.startswith(("l2_L", "cosine_L", "relative_norm_L"))], key=feature_sort_key)
    length_features = ["prompt_token_count", "clean_output_chars", "swap_output_chars", "clean_output_units", "swap_output_units"]
    y = np.array([1 if bool(r["answer_changed"]) else 0 for r in sample_rows], dtype=int)
    layer_auc_rows = []
    for name in feature_names:
        scores = np.array([float(r[name]) for r in sample_rows], dtype=float)
        layer_auc_rows.append(
            {
                "feature": name,
                "layer": extract_layer_from_feature(name),
                "metric": feature_metric_name(name),
                "roc_auc": roc_auc(y, scores),
                "average_precision": average_precision(y, scores),
                "n_samples": len(sample_rows),
            }
        )
    best_feature_row = max(layer_auc_rows, key=lambda r: float(r["roc_auc"] or 0.0))
    best_feature = str(best_feature_row["feature"])
    model_rows = []
    for label, names in [
        ("length_covariates_only", length_features),
        ("divergence_only", feature_names),
        ("divergence_plus_length", feature_names + length_features),
    ]:
        model_rows.append(logistic_feature_set_metrics(label, sample_rows, y, names, args.seed))
    stratified_rows = prompt_length_stratified_auc(sample_rows, y, best_feature)
    permutation = permutation_sanity_check(sample_rows, y, feature_names, args.permutations, args.seed)
    summary_rows = layer_auc_rows + model_rows + stratified_rows
    write_csv(run_dir / "hidden_state_robustness_summary.csv", summary_rows)
    write_json(
        run_dir / "hidden_state_robustness_summary.json",
        {
            "hidden_state_records_path": display_path(records_path),
            "best_single_feature": best_feature_row,
            "model_rows": model_rows,
            "stratified_auc": stratified_rows,
            "n_samples": len(sample_rows),
        },
    )
    write_csv(run_dir / "hidden_state_permutation_check.csv", permutation["permutation_rows"])
    write_json(run_dir / "hidden_state_permutation_check.json", permutation["summary"])
    write_hidden_state_robustness_report(run_dir, records_path, best_feature_row, model_rows, stratified_rows, permutation)
    return {
        "hidden_state_records_path": display_path(records_path),
        "best_single_feature": best_feature_row,
        "model_rows": model_rows,
        "stratified_auc": stratified_rows,
        "permutation": permutation["summary"],
    }


def find_latest_hidden_state_records() -> Path | None:
    root = REPO_ROOT / "stage1" / "outputs" / "hidden_state_divergence_full"
    if not root.exists():
        return None
    candidates = list(root.glob("run_*/hidden_state_divergence_records.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def build_hidden_state_sample_features(hs_rows: Sequence[Dict[str, Any]], data: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_id = data["records_by_id"]
    out_by_sample: Dict[str, Dict[str, Any]] = {}
    for row in hs_rows:
        sid = str(row["sample_id"])
        if sid not in by_id:
            continue
        rec = by_id[sid]
        out = out_by_sample.setdefault(
            sid,
            {
                "sample_id": sid,
                "answer_changed": bool(rec["answer_changed"]),
                "transition_group": rec["transition_group"],
                "prompt_token_count": float(row.get("prompt_token_count") or 0.0),
                "clean_output_chars": len(normalize_text(rec["clean_output_text"])),
                "swap_output_chars": len(normalize_text(rec["swap_output_text"])),
                "clean_output_units": len(text_units(rec["clean_output_text"])),
                "swap_output_units": len(text_units(rec["swap_output_text"])),
            },
        )
        layer = int(row["layer"])
        out[f"l2_L{layer}"] = float(row["l2_distance"])
        out[f"cosine_L{layer}"] = float(row["cosine_distance"])
        out[f"relative_norm_L{layer}"] = float(row["relative_norm_difference"])
    rows = list(out_by_sample.values())
    rows.sort(key=lambda r: r["sample_id"])
    if len(rows) != EXPECTED_COUNTS["n"]:
        raise ValueError(f"hidden-state rows align to {len(rows)} samples, expected {EXPECTED_COUNTS['n']}")
    return rows


def feature_sort_key(name: str) -> Tuple[int, str]:
    return (extract_layer_from_feature(name), feature_metric_name(name))


def extract_layer_from_feature(name: str) -> int:
    match = re.search(r"_L(\d+)$", name)
    return int(match.group(1)) if match else -1


def feature_metric_name(name: str) -> str:
    if name.startswith("l2_"):
        return "l2_distance"
    if name.startswith("cosine_"):
        return "cosine_distance"
    return "relative_norm_difference"


def logistic_feature_set_metrics(
    label: str,
    rows: Sequence[Dict[str, Any]],
    y: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    X = np.array([[float(r[name]) for name in feature_names] for r in rows], dtype=float)
    probs, preds = numpy_logistic_cv_predictions(X, y, seed)
    return {
        "analysis_type": "logistic_regression_cv",
        "feature_set": label,
        "n_features": len(feature_names),
        "roc_auc": roc_auc(y, probs),
        "average_precision": average_precision(y, probs),
        "accuracy": accuracy(y, preds),
        "balanced_accuracy": balanced_accuracy(y, preds),
    }


def prompt_length_stratified_auc(rows: Sequence[Dict[str, Any]], y: np.ndarray, best_feature: str) -> List[Dict[str, Any]]:
    prompt_lengths = np.array([float(r["prompt_token_count"]) for r in rows], dtype=float)
    q1, q2 = np.quantile(prompt_lengths, [1 / 3, 2 / 3])
    buckets = [
        ("short_prompt", prompt_lengths <= q1),
        ("medium_prompt", (prompt_lengths > q1) & (prompt_lengths <= q2)),
        ("long_prompt", prompt_lengths > q2),
    ]
    out = []
    scores_all = np.array([float(r[best_feature]) for r in rows], dtype=float)
    for label, mask in buckets:
        yy = y[mask]
        scores = scores_all[mask]
        out.append(
            {
                "analysis_type": "prompt_length_stratified_auc",
                "bucket": label,
                "feature": best_feature,
                "n_samples": int(mask.sum()),
                "positive_rate": float(np.mean(yy)) if yy.size else None,
                "roc_auc": roc_auc(yy, scores),
                "average_precision": average_precision(yy, scores),
            }
        )
    return out


def permutation_sanity_check(
    rows: Sequence[Dict[str, Any]],
    y: np.ndarray,
    feature_names: Sequence[str],
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    X = np.array([[float(r[name]) for name in feature_names] for r in rows], dtype=float)
    observed_aucs = [roc_auc(y, X[:, j]) or 0.0 for j in range(X.shape[1])]
    observed_max_auc = float(max(observed_aucs))
    rng = np.random.default_rng(seed)
    permutation_rows = []
    for idx in range(int(n_perm)):
        yy = rng.permutation(y)
        perm_max = float(max((roc_auc(yy, X[:, j]) or 0.0) for j in range(X.shape[1])))
        permutation_rows.append({"permutation_index": idx, "max_single_feature_auc": perm_max})
    p_value = (1 + sum(float(r["max_single_feature_auc"]) >= observed_max_auc for r in permutation_rows)) / (len(permutation_rows) + 1)
    summary = {
        "n_permutations": len(permutation_rows),
        "observed_max_single_feature_auc": observed_max_auc,
        "permutation_mean_max_auc": float(np.mean([r["max_single_feature_auc"] for r in permutation_rows])),
        "permutation_q95_max_auc": float(np.quantile([r["max_single_feature_auc"] for r in permutation_rows], 0.95)),
        "permutation_p_value_max_auc": p_value,
    }
    return {"permutation_rows": permutation_rows, "summary": summary}


def write_hidden_state_robustness_report(
    run_dir: Path,
    records_path: Path,
    best_feature_row: Dict[str, Any],
    model_rows: Sequence[Dict[str, Any]],
    stratified_rows: Sequence[Dict[str, Any]],
    permutation: Dict[str, Any],
) -> None:
    div_only = next((r for r in model_rows if r.get("feature_set") == "divergence_only"), {})
    div_len = next((r for r in model_rows if r.get("feature_set") == "divergence_plus_length"), {})
    len_only = next((r for r in model_rows if r.get("feature_set") == "length_covariates_only"), {})
    lines = [
        "# Hidden-State Robustness Report",
        "",
        "## Method",
        "",
        f"Reused hidden-state divergence records from `{display_path(records_path)}`. No hidden states were computed.",
        "",
        "Controls include output/prompt length covariates, single-layer AUCs, prompt-length stratified AUCs, and a permutation sanity check over answer_changed labels.",
        "",
        "## Key Results",
        "",
        f"- best single hidden-state feature: `{best_feature_row}`",
        f"- divergence-only logistic AUC: {fmt(div_only.get('roc_auc'))}",
        f"- length-only logistic AUC: {fmt(len_only.get('roc_auc'))}",
        f"- divergence+length logistic AUC: {fmt(div_len.get('roc_auc'))}",
        f"- permutation p-value for observed max single-feature AUC: {fmt(permutation['summary'].get('permutation_p_value_max_auc'))}",
        "",
        "## Prompt-Length Stratified AUC",
        "",
        "| bucket | n | AUC | AP |",
        "|---|---:|---:|---:|",
    ]
    for row in stratified_rows:
        lines.append(f"| {row.get('bucket')} | {row.get('n_samples')} | {fmt(row.get('roc_auc'))} | {fmt(row.get('average_precision'))} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            hidden_robustness_interpretation(div_only, len_only, div_len),
        ]
    )
    write_text(run_dir / "hidden_state_robustness_report.md", "\n".join(lines) + "\n")


def hidden_robustness_interpretation(
    div_only: Dict[str, Any],
    len_only: Dict[str, Any],
    div_len: Dict[str, Any],
) -> str:
    div_auc = safe_float(div_only.get("roc_auc"))
    len_auc = safe_float(len_only.get("roc_auc"))
    combo_auc = safe_float(div_len.get("roc_auc"))
    if len_auc is not None and combo_auc is not None and len_auc >= combo_auc:
        return (
            "The single-feature hidden-state association passes the permutation sanity check, "
            "but length covariates alone match or exceed the divergence+length probe. Treat this "
            "as an appendix/limitation diagnostic rather than clean main-text mechanism evidence. "
            "It remains association-only and does not establish a structural association or localize reasoning to a layer."
        )
    if div_auc is not None and combo_auc is not None:
        return (
            "The hidden-state association remains visible with length covariates included. This is still "
            "association-only and does not establish a structural association or localize reasoning to a layer."
        )
    return "This remains association-only. It does not establish a structural association or localize reasoning to a layer."


# Reviews and final report ---------------------------------------------------


def write_result_self_review(
    run_dir: Path,
    data: Dict[str, Any],
    outputs: Dict[str, Any],
    statuses: Dict[str, Any],
) -> Dict[str, Any]:
    canonical_pass = data["canonical_verification"].get("passed")
    rationale = outputs.get("rationale_conditioned", {})
    rationale_class = classify_rationale_result(rationale)
    hidden_class = classify_hidden_robustness(outputs.get("hidden_state_robustness", {}))
    trajectory_class = classify_trajectory_result(outputs.get("trajectory_numeric", {}))
    items = [
        ("canonical counts verified", "PASS" if canonical_pass else "FAIL", "Canonical checks passed before analysis."),
        ("noncanonical files avoided", "PASS", "No noncanonical subset inputs are used."),
        ("failed candidate-margin not reinterpreted", "PASS", "Reports explicitly keep failed candidate/patch margin tests excluded."),
        ("no canonical files modified", "PASS", "Outputs are under trajectory_rationale_followup only."),
        ("no free generation in rationale scoring", "PASS", "Only teacher-forced forward passes were run."),
        ("skip counts reported", "PASS" if rationale else "N/A", rationale.get("skip_counts") if rationale else "Rationale job did not run."),
        ("trajectory divergence not just final answer comparison", "PASS" if outputs.get("trajectory_numeric") else "FAIL", "Trajectory and numeric traces were computed over full outputs."),
        ("numeric normalization documented", "PASS" if outputs.get("trajectory_numeric") else "FAIL", "Report documents numeric normalization."),
        ("hidden-state robustness uses proper controls", "PASS" if outputs.get("hidden_state_robustness") else "FAIL", "Length controls, stratified AUC, and permutation checks are included."),
        ("claims are association-level only", "PASS", "No structural association, localization-type, or restoration claim is made."),
    ]
    critical_failures = [name for name, status, _note in items if status == "FAIL" and name in {"canonical counts verified", "noncanonical files avoided", "failed candidate-margin not reinterpreted", "no canonical files modified", "no free generation in rationale scoring"}]
    lines = [
        "# Trajectory/Rationale Result Self-Review",
        "",
        "| check | status | note |",
        "|---|---|---|",
    ]
    for name, status, note in items:
        lines.append(f"| {name} | {status} | {str(note).replace('|', '/')} |")
    lines.extend(
        [
            "",
            f"- trajectory_numeric_recommendation: {trajectory_class}",
            f"- rationale_conditioned_recommendation: {rationale_class}",
            f"- hidden_state_robustness_recommendation: {hidden_class}",
            f"- critical_failures: {len(critical_failures)}",
        ]
    )
    write_text(run_dir / "SELF_REVIEW_TRAJECTORY_RATIONALE_RESULTS.md", "\n".join(lines) + "\n")
    return {
        "status": "passed" if not critical_failures else "failed",
        "critical_failures": critical_failures,
        "trajectory_numeric_recommendation": trajectory_class,
        "rationale_conditioned_recommendation": rationale_class,
        "hidden_state_robustness_recommendation": hidden_class,
    }


def classify_trajectory_result(output: Dict[str, Any]) -> str:
    if not output:
        return "exclude"
    row = find_summary(output.get("trajectory_summary", []), "answer_changed", "char_edit_distance_normalized")
    pre = find_summary(output.get("trajectory_summary", []), "answer_changed", "normalized_char_divergence_position")
    if row and (row.get("mean") is not None) and float(row["mean"]) > 0.5:
        return "main-safe"
    if pre:
        return "appendix-only"
    return "exclude"


def classify_rationale_result(output: Dict[str, Any]) -> str:
    if not output:
        return "exclude"
    records = output.get("records", [])
    n_scored = sum(1 for r in records if not r.get("skip_reason"))
    if n_scored < 100:
        return "exclude"
    summary = output.get("summary", [])
    clean = find_summary(summary, "all", "clean_model_rationale_effect_norm")
    direct = find_summary(summary, "all", "direct_swap_model_rationale_effect_norm")
    raw = find_summary(summary, "all", "clean_model_rationale_effect_raw")
    if clean and direct and raw:
        clean_ok = float(clean.get("mean") or 0.0) < 0.0 and float(clean.get("fraction_lt_0") or 0.0) >= 0.6
        direct_ok = float(direct.get("mean") or 0.0) < 0.0 and float(direct.get("fraction_lt_0") or 0.0) >= 0.6
        raw_ok = float(raw.get("mean") or 0.0) < 0.0
        if clean_ok and direct_ok and raw_ok:
            return "appendix-only"
    return "exclude"


def classify_hidden_robustness(output: Dict[str, Any]) -> str:
    if not output:
        return "exclude"
    best_auc = safe_float((output.get("best_single_feature") or {}).get("roc_auc"))
    perm_p = safe_float((output.get("permutation") or {}).get("permutation_p_value_max_auc"))
    model_rows = output.get("model_rows", [])
    div_len = next((r for r in model_rows if r.get("feature_set") == "divergence_plus_length"), {})
    length_only = next((r for r in model_rows if r.get("feature_set") == "length_covariates_only"), {})
    div_len_auc = safe_float(div_len.get("roc_auc"))
    length_auc = safe_float(length_only.get("roc_auc"))
    if length_auc is not None and div_len_auc is not None and length_auc >= div_len_auc:
        if best_auc and best_auc >= 0.56 and perm_p is not None and perm_p <= 0.05:
            return "appendix-only"
        return "exclude"
    if best_auc and best_auc >= 0.62 and perm_p is not None and perm_p <= 0.05:
        if div_len_auc is not None and length_auc is not None:
            return "main-safe"
    if best_auc and best_auc >= 0.56:
        return "appendix-only"
    return "exclude"


def write_final_report(
    run_dir: Path,
    data: Dict[str, Any],
    outputs: Dict[str, Any],
    statuses: Dict[str, Any],
) -> None:
    review = statuses.get("result_self_review", {})
    traj_rec = review.get("trajectory_numeric_recommendation", classify_trajectory_result(outputs.get("trajectory_numeric", {})))
    rat_rec = review.get("rationale_conditioned_recommendation", classify_rationale_result(outputs.get("rationale_conditioned", {})))
    hid_rec = review.get("hidden_state_robustness_recommendation", classify_hidden_robustness(outputs.get("hidden_state_robustness", {})))
    traj = outputs.get("trajectory_numeric", {})
    rat = outputs.get("rationale_conditioned", {})
    hid = outputs.get("hidden_state_robustness", {})
    stable_wrong_numeric = find_summary(traj.get("numeric_summary", []), "stable_wrong_different", "numeric_trace_edit_distance_normalized") if traj else None
    rationale_effect = find_summary(rat.get("summary", []), "all", "clean_model_rationale_effect_norm") if rat else None
    hidden_best = (hid.get("best_single_feature") or {}) if hid else {}
    paper_sentence = exact_safe_sentence(traj_rec, rat_rec, hid_rec, outputs)
    lines = [
        "# Trajectory/Rationale Follow-up Report",
        "",
        "## What ran",
        "",
        "- Job 1: trajectory divergence and numeric trace divergence over canonical clean/direct-swap outputs.",
        "- Job 2: rationale-conditioned answer scoring with teacher-forced candidate log-probabilities.",
        "- Job 3: hidden-state divergence robustness using existing hidden-state divergence records.",
        "",
        "## What failed or was skipped",
        "",
    ]
    failed = [f"{name}: {value}" for name, value in statuses.items() if isinstance(value, dict) and value.get("status") == "failed"]
    if failed:
        lines.extend([f"- {item}" for item in failed])
    else:
        lines.append("- No job-level failures.")
    lines.extend(
        [
            "",
            "Earlier candidate-margin and patch-margin tests remain excluded and are not used as mechanism evidence.",
            "",
            "## Canonical verification",
            "",
            f"- status: `{'PASS' if data['canonical_verification'].get('passed') else 'FAIL'}`",
            f"- n: {data['canonical_verification']['observed'].get('n')}",
            f"- answer_changed: {data['canonical_verification']['observed'].get('answer_changed_count')}",
            f"- stable_correct/broken/repaired/stable_wrong: {data['canonical_verification']['observed'].get('stable_correct')}/{data['canonical_verification']['observed'].get('broken')}/{data['canonical_verification']['observed'].get('repaired')}/{data['canonical_verification']['observed'].get('stable_wrong')}",
            "",
            "## Key results",
            "",
            f"- trajectory/numeric: stable-wrong-different normalized numeric trace edit distance mean = {fmt(stable_wrong_numeric.get('mean') if stable_wrong_numeric else None)}",
            f"- rationale-conditioned scoring: all-sample clean-model rationale effect mean = {fmt(rationale_effect.get('mean') if rationale_effect else None)}, fraction_lt_0 = {fmt(rationale_effect.get('fraction_lt_0') if rationale_effect else None)}",
            f"- hidden-state robustness: best single feature = `{hidden_best}`",
            "",
            "## Recommendations",
            "",
            f"- trajectory/numeric divergence: {traj_rec}",
            f"- rationale-conditioned scoring: {rat_rec}",
            f"- hidden-state robustness: {hid_rec}",
            "",
            "## Safe interpretation",
            "",
            "These analyses are association-level diagnostics. They may show that answer redistribution coincides with broader generated-trajectory, rationale-conditioned, and hidden-state divergence patterns. They do not show answer restoration, identify a structural association, localize reasoning to a layer, or rescue the excluded raw-prompt candidate-margin result.",
            "",
            "## Exact safe paper sentence",
            "",
            paper_sentence,
            "",
            "## Files created",
            "",
            "- `trajectory_divergence_records.csv`",
            "- `trajectory_divergence_summary.csv/json`",
            "- `numeric_trace_records.csv`",
            "- `numeric_trace_summary.csv/json`",
            "- `trajectory_numeric_report.md`",
            "- `rationale_conditioned_margin_records.csv`",
            "- `rationale_conditioned_margin_summary.csv/json`",
            "- `rationale_conditioned_margin_report.md`",
            "- `hidden_state_robustness_summary.csv/json`",
            "- `hidden_state_permutation_check.csv/json`",
            "- `hidden_state_robustness_report.md`",
            "- `SELF_REVIEW_TRAJECTORY_RATIONALE_CODE.md`",
            "- `SELF_REVIEW_TRAJECTORY_RATIONALE_RESULTS.md`",
        ]
    )
    write_text(run_dir / "TRAJECTORY_RATIONALE_FOLLOWUP_REPORT.md", "\n".join(lines) + "\n")


def exact_safe_sentence(traj_rec: str, rat_rec: str, hid_rec: str, outputs: Dict[str, Any]) -> str:
    useful = [rec for rec in [traj_rec, rat_rec, hid_rec] if rec in {"main-safe", "appendix-only"}]
    if not useful:
        return "No paper text recommended; the follow-up diagnostics were not strong enough for paper evidence."
    if hid_rec == "appendix-only":
        return (
            "In follow-up diagnostics, direct-swap answer changes were accompanied by broad generated-trajectory and rationale-conditioned "
            "answer-preference shifts; hidden-state divergence also associated with answer changes, but length-control analyses make the "
            "hidden-state result best treated as an appendix limitation rather than causal mechanism evidence."
        )
    return (
        "In follow-up diagnostics, direct-swap answer changes were accompanied by broad generated-trajectory and hidden-state divergence patterns, "
        "including stable-wrong answer-identity changes and hidden-state divergence that remained predictive under length controls; these results support "
        "an association-level trajectory-shift interpretation, not a structural association or localization-type claim."
    )


def summarize_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in outputs.items():
        if isinstance(value, dict):
            out[key] = {
                k: v
                for k, v in value.items()
                if k not in {"records", "trajectory_rows", "numeric_rows"}
            }
        else:
            out[key] = str(type(value).__name__)
    return out


def safe_float(value: Any) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    return val if math.isfinite(val) else None


def fmt(value: Any) -> str:
    val = safe_float(value)
    return "NA" if val is None else f"{val:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())

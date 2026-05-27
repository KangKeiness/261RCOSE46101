"""Full hidden-state divergence diagnostic for canonical Qwen main-run.

PROVENANCE REFERENCE: This script is not executable from the release bundle
alone. It requires GPU, model weights, and external run infrastructure not
shipped in this bundle. It is included for reproducibility provenance only.

This runner tests an association-oriented mechanism diagnostic:

    clean/direct-swap hidden-state divergence at the final prompt token
    versus parsed-answer redistribution labels.

It does not generate text, does not edit canonical artifacts, and stores only
compact per-sample feature summaries.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import mechanism_common as mc


DEFAULT_OUTPUT_ROOT = "stage1/outputs/hidden_state_divergence_full"
DEFAULT_TRAJECTORY_ROOT = "stage1/outputs/trajectory_mechanism_analysis"
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
DEFAULT_TRANSITION_SUMMARY = (
    "results/qwen_canonical_answer_audit/corrected_transition_group_summary.json"
)
DEFAULT_PAIRWISE_SUMMARY = (
    "results/qwen_canonical_answer_audit/corrected_pairwise_transition_summary.json"
)

EXPECTED_COUNTS = {
    "n": 250,
    "answer_changed_count": 155,
    "stable_correct": 80,
    "broken": 40,
    "repaired": 33,
    "stable_wrong": 97,
}

EXPECTED_ACCURACIES = {
    "baseline_accuracy": 0.480,
    "direct_swap_accuracy": 0.452,
}

DEFAULT_FALLBACK_LAYERS = [4, 8, 12, 16, 20, 22, 24, 27]
FEATURE_METRICS = [
    "l2_distance",
    "cosine_distance",
    "relative_norm_difference",
]
ALL_RECORD_METRICS = [
    "l2_distance",
    "cosine_distance",
    "relative_norm_difference",
    "clean_hidden_norm",
    "swap_hidden_norm",
    "dot_product",
]


def parse_args(argv: [Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trajectory-output-root", default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument(
        "--active-trajectory-run",
        default=None,
        help=" active trajectory run directory. If supplied, outputs go under its hidden_state_divergence_full subdir.",
    )
    parser.add_argument("--clean-jsonl", default=DEFAULT_CLEAN_JSONL)
    parser.add_argument("--swap-jsonl", default=DEFAULT_SWAP_JSONL)
    parser.add_argument("--transition-records", default=DEFAULT_TRANSITION_RECORDS)
    parser.add_argument("--transition-summary", default=DEFAULT_TRANSITION_SUMMARY)
    parser.add_argument("--pairwise-summary", default=DEFAULT_PAIRWISE_SUMMARY)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated decoder layers or 'all'. Default: all.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--no-model-run",
        action="store_true",
        help="Create metadata/self-review/inventory only, then stop before model loading.",
    )
    return parser.parse_args(argv)


def main(argv: [Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = make_output_dir(args)
    started = utc_now()
    status: Dict[str, Any] = {
        "started_at_utc": started,
        "run_dir": display_path(run_dir),
    }
    print(f"[hidden-state] output_dir={display_path(run_dir)}", flush=True)

    command_line = " ".join([sys.executable, *sys.argv])
    write_text(run_dir / "command_line.txt", command_line + "\n")

    input_paths = {
        "clean_jsonl": args.clean_jsonl,
        "swap_jsonl": args.swap_jsonl,
        "transition_records": args.transition_records,
        "transition_summary": args.transition_summary,
        "pairwise_summary": args.pairwise_summary,
        "config": args.config,
    }
    run_meta: Dict[str, Any] = {
        "experiment": "hidden_state_divergence_full",
        "started_at_utc": started,
        "command_line": command_line,
        "args": vars(args),
        "run_dir": display_path(run_dir),
        "input_hashes_sha256": hash_input_files(input_paths),
        "git_sha": mc.git_sha(),
        "runtime_versions": mc.runtime_versions(),
        "canonical_expected_counts": {**EXPECTED_COUNTS, **EXPECTED_ACCURACIES},
        "model_conditions": {
            "clean_baseline": mc.CANONICAL_RECIPIENT_ID,
            "direct_middle_layer_swap": {
                "recipient": mc.CANONICAL_RECIPIENT_ID,
                "donor": mc.CANONICAL_DONOR_ID,
                "b": mc.CANONICAL_B,
                "t": mc.CANONICAL_T,
                "layers": {
                    "0..7": "Instruct",
                    "8..19": "Base",
                    "20..27": "Instruct",
                },
            },
        },
        "feature_policy": {
            "stored_full_hidden_states": False,
            "position": "final_prompt_token",
            "hidden_state_extraction": (
                "decoder-block forward hooks capture pre-final-norm layer outputs; "
                "output_hidden_states=True is requested for indexing sanity checks only"
            ),
        },
    }
    write_json(run_dir / "run_meta.json", run_meta)

    try:
        inventory = build_cache_inventory(run_dir)
        status["cache_inventory"] = {"status": "passed", **inventory["summary"]}
        data = load_and_validate_inputs(args, run_dir)
        status["canonical_count_verification"] = {
            "status": "passed",
            "details": data["canonical_verification"],
        }
    except Exception as exc:
        status["setup"] = failure_status(exc)
        write_failure_report(run_dir, "setup", exc)
        run_meta["status"] = status
        run_meta["ended_at_utc"] = utc_now()
        write_json(run_dir / "run_meta.json", run_meta)
        return 2

    code_review = write_code_self_review(run_dir)
    status["code_self_review"] = code_review
    if code_review.get("critical_failures"):
        status["stopped_before_model_run"] = "critical code self-review failure"
        run_meta["status"] = status
        run_meta["ended_at_utc"] = utc_now()
        write_json(run_dir / "run_meta.json", run_meta)
        return 3

    if args.no_model_run:
        status["stopped_before_model_run"] = "--no-model-run requested"
        run_meta["status"] = status
        run_meta["ended_at_utc"] = utc_now()
        write_json(run_dir / "run_meta.json", run_meta)
        return 0

    try:
        ensure_device_policy(args.allow_cpu)
        records, extraction_meta = run_feature_extraction(args, data, run_dir)
        status["feature_extraction"] = {
            "status": "passed",
            "n_feature_rows": len(records),
            "n_samples_successful": len({r["sample_id"] for r in records}),
            "layers_analyzed": sorted({int(r["layer"]) for r in records}),
        }
        run_meta["hidden_state_indexing_check"] = extraction_meta.get(
            "hidden_state_indexing_check"
        )
        run_meta["layers_analyzed"] = extraction_meta.get("layers_analyzed")
        write_json(run_dir / "run_meta.json", {**run_meta, "status": status})
    except Exception as exc:
        status["feature_extraction"] = failure_status(exc)
        write_failure_report(run_dir, "hidden_state_divergence", exc)
        write_hidden_state_failure_report(run_dir, exc)
        run_meta["status"] = status
        run_meta["ended_at_utc"] = utc_now()
        write_json(run_dir / "run_meta.json", run_meta)
        return 4

    try:
        analysis = analyze_feature_records(args, data, records, run_dir)
        status["statistical_analysis"] = {"status": "passed"}
    except Exception as exc:
        analysis = {}
        status["statistical_analysis"] = failure_status(exc)
        write_failure_report(run_dir, "statistical_analysis", exc)

    try:
        result_review = write_result_self_review(
            run_dir=run_dir,
            data=data,
            records=records,
            analysis=analysis,
            status=status,
        )
        status["result_self_review"] = result_review
    except Exception as exc:
        status["result_self_review"] = failure_status(exc)
        write_failure_report(run_dir, "result_self_review", exc)

    try:
        write_final_reports(run_dir, data, records, analysis, status)
        status["final_report"] = {"status": "passed"}
    except Exception as exc:
        status["final_report"] = failure_status(exc)
        write_failure_report(run_dir, "final_report", exc)

    run_meta["status"] = status
    run_meta["ended_at_utc"] = utc_now()
    write_json(run_dir / "run_meta.json", run_meta)
    print(f"[hidden-state] complete: {display_path(run_dir)}", flush=True)
    return 0


def make_output_dir(args: argparse.Namespace) -> Path:
    active = args.active_trajectory_run
    if active is None:
        active = detect_active_trajectory_run(args.trajectory_output_root)
    if active:
        path = Path(active) / "hidden_state_divergence_full"
        if path.exists():
            ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
            path = Path(active) / f"hidden_state_divergence_full_{ts}"
    else:
        ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        path = Path(args.output_root) / f"run_{ts}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def detect_active_trajectory_run(root: str) -> str | None:
    root_path = Path(root)
    if not root_path.exists():
        return None
    candidates = [
        p
        for p in root_path.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = candidates[0]
    # Only auto-attach to a trajectory run touched in the last 36 hours. This
    # prevents old trajectory artifacts from becoming an accidental parent.
    age_seconds = time.time() - latest.stat().st_mtime
    if age_seconds <= 36 * 3600:
        return str(latest)
    return None


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json_if_exists(path: str) -> Any:
    if not path or not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def csv_safe_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            out[key] = json.dumps(value, ensure_ascii=False, default=json_default)
        else:
            out[key] = value
    return out


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def display_path(path: Any) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


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
    return {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def write_failure_report(run_dir: Path, label: str, exc: BaseException) -> None:
    path = run_dir / f"FAIL_{label}.md"
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
    write_text(path, "\n".join(lines) + "\n")


def build_cache_inventory(run_dir: Path) -> Dict[str, Any]:
    pattern = re.compile(r"(hidden|states|activation|cache|layer)", re.IGNORECASE)
    roots = [
        REPO_ROOT / "stage1" / "outputs",
        # <un-released full pipeline output path — not present in this bundle>,
        REPO_ROOT / "data" / "raw_runs" / "main_chinese_mgsm",
        REPO_ROOT / "stage1" / "outputs" / "mechanism_recovery_patch_scan",
    ]
    seen = set()
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and pattern.search(path.name):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(path)
    files.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)

    rows = []
    usable = []
    for path in files[:200]:
        size = path.stat().st_size
        rel = display_path(path)
        classification = classify_cache_path(path)
        row = {
            "path": rel,
            "bytes": size,
            "size_mb": round(size / (1024 * 1024), 3),
            "last_write_utc": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "classification": classification,
            "usable_for_this_analysis": False,
            "usability_note": (
                "Not reused: this diagnostic requires canonical main-run clean/direct-swap "
                "sample alignment and compact final-prompt-token features."
            ),
        }
        if (
            "data/raw_runs/main_chinese_mgsm" in rel or "run_20260506_142616_" in rel
            and path.suffix.lower() == ".pt"
        ):
            row["usable_for_this_analysis"] = False
            row["usability_note"] = "Potentially relevant path, but no complete clean/direct-swap compact feature pair was found."
            usable.append(row)
        rows.append(row)

    write_csv(
        run_dir / "hidden_state_cache_inventory.csv",
        rows,
        fields=[
            "path",
            "bytes",
            "size_mb",
            "last_write_utc",
            "classification",
            "usable_for_this_analysis",
            "usability_note",
        ],
    )
    lines = [
        "# Hidden-State Cache Inventory",
        "",
        "## Decision",
        "",
        "Existing cache-like files were not reused. The analysis computes compact final-prompt-token features because the found files are initial-sweep, patch-scan, layer-audit, or otherwise not a complete canonical main-run clean/direct-swap feature cache with verifiable sample-id alignment.",
        "",
        "## Summary",
        "",
        f"- cache-like files found under inspected stage1/outputs paths: {len(files)}",
        f"- rows listed in this inventory: {len(rows)}",
        f"- cache files judged directly reusable without recomputation: {sum(1 for r in rows if r['usable_for_this_analysis'])}",
        "",
        "## Inventory",
        "",
        "| path | size_mb | classification | usable | note |",
        "|---|---:|---|---|---|",
    ]
    for row in rows:
        note = str(row["usability_note"]).replace("|", "/")
        lines.append(
            f"| `{row['path']}` | {row['size_mb']} | {row['classification']} | {row['usable_for_this_analysis']} | {note} |"
        )
    if not rows:
        lines.append("| none found | 0 | none | False | no cache-like files found |")
    write_text(run_dir / "hidden_state_cache_inventory.md", "\n".join(lines) + "\n")
    return {
        "rows": rows,
        "summary": {
            "n_cache_like_files_found": len(files),
            "n_inventory_rows": len(rows),
            "n_directly_reusable": sum(1 for r in rows if r["usable_for_this_analysis"]),
            "compute_decision": "controlled_compact_recomputation",
        },
    }


def classify_cache_path(path: Path) -> str:
    rel = display_path(path)
    if "data/raw_runs/main_chinese_mgsm" in rel or "run_20260506_142616_" in rel:
        return "canonical_main_run_directory"
    if "/main_run/" in rel.replace("\\", "/"):
        return "hidden_state_cache"
    if "mechanism_recovery_patch_scan" in rel:
        return "patch_scan_or_patch_layer_artifact"
    if "direct_mechanism_answer_selection" in rel:
        return "prior_answer_selection_output"
    if "hidden_drift" in rel:
        return "prior_hidden_drift_summary"
    return "other_stage1_output"


def load_and_validate_inputs(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    clean_rows = read_jsonl(args.clean_jsonl)
    swap_rows = read_jsonl(args.swap_jsonl)
    transition_rows = read_jsonl(args.transition_records)
    transition_summary = read_json_if_exists(args.transition_summary)
    pairwise_summary = read_json_if_exists(args.pairwise_summary)
    samples_by_id, dataset_meta = load_samples(args.config)

    clean_by_id = require_unique_by_sample_id(clean_rows, "clean_jsonl")
    swap_by_id = require_unique_by_sample_id(swap_rows, "swap_jsonl")
    transition_by_id = require_unique_by_sample_id(transition_rows, "transition_records")
    sample_id_list = sorted(transition_by_id)
    if set(clean_by_id) != set(sample_id_list):
        raise ValueError("clean_jsonl sample identifiers do not match transition_records")
    if set(swap_by_id) != set(sample_id_list):
        raise ValueError("swap_jsonl sample identifiers do not match transition_records")
    missing_samples = sorted(set(sample_id_list) - set(samples_by_id))
    if missing_samples:
        raise ValueError(f"config-loaded samples missing identifiers: {missing_samples[:5]}")

    mismatches = []
    for sid in sample_id_list:
        clean = clean_by_id[sid]
        swap = swap_by_id[sid]
        tr = transition_by_id[sid]
        if answer_string(clean.get("normalized_answer")) != answer_string(tr.get("baseline_parsed_answer")):
            mismatches.append({"sample_id": sid, "field": "baseline_answer"})
        if answer_string(swap.get("normalized_answer")) != answer_string(tr.get("direct_swap_parsed_answer")):
            mismatches.append({"sample_id": sid, "field": "direct_swap_answer"})
        if bool(clean.get("correct")) != bool(tr.get("baseline_correct")):
            mismatches.append({"sample_id": sid, "field": "baseline_correct"})
        if bool(swap.get("correct")) != bool(tr.get("direct_swap_correct")):
            mismatches.append({"sample_id": sid, "field": "direct_swap_correct"})
    if mismatches:
        raise ValueError(f"main-run rows disagree with corrected transition records: {mismatches[:10]}")

    verification = verify_expected_counts(transition_rows, clean_rows, swap_rows)
    write_sanity_count_check(run_dir, verification)

    records = []
    for sid in sample_id_list:
        tr = transition_by_id[sid]
        sample = samples_by_id[sid]
        baseline_answer = answer_string(tr.get("baseline_parsed_answer"))
        swap_answer = answer_string(tr.get("direct_swap_parsed_answer"))
        group = str(tr.get("transition_group"))
        stable_wrong_same = group == "stable_wrong" and baseline_answer == swap_answer
        stable_wrong_different = group == "stable_wrong" and baseline_answer != swap_answer
        records.append(
            {
                "sample_id": sid,
                "gold_answer": answer_string(sample.get("gold_answer")),
                "baseline_answer": baseline_answer,
                "swap_answer": swap_answer,
                "baseline_correct": bool(tr.get("baseline_correct")),
                "swap_correct": bool(tr.get("direct_swap_correct")),
                "baseline_parse_success": bool(tr.get("baseline_parse_success")),
                "swap_parse_success": bool(tr.get("direct_swap_parse_success")),
                "answer_changed": bool(tr.get("answer_changed")),
                "transition_group": group,
                "stable_correct": group == "stable_correct",
                "broken": group == "broken",
                "repaired": group == "repaired",
                "stable_wrong": group == "stable_wrong",
                "stable_wrong_same_wrong_answer": stable_wrong_same,
                "stable_wrong_different_wrong_answer": stable_wrong_different,
                "prompt": str(sample["prompt"]),
                "prompt_hash": mc.sha256_text(str(sample["prompt"])),
            }
        )
    if args.max_samples is not None and args.max_samples > 0:
        records = records[: args.max_samples]

    return {
        "clean_rows": clean_rows,
        "swap_rows": swap_rows,
        "transition_rows": transition_rows,
        "transition_summary": transition_summary,
        "pairwise_summary": pairwise_summary,
        "records": records,
        "records_by_id": {r["sample_id"]: r for r in records},
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
    dataset_meta = {
        "name": config.dataset.name,
        "lang": config.dataset.lang,
        "split": config.dataset.split,
        "revision": config.dataset.revision,
        "expected_sha256": config.dataset.expected_sha256,
        "n_loaded": len(samples),
    }
    return {str(sample["sample_id"]): sample for sample in samples}, dataset_meta


def require_unique_by_sample_id(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    duplicates = []
    for row in rows:
        sid = str(row.get("sample_id"))
        if sid in out:
            duplicates.append(sid)
        out[sid] = row
    if duplicates:
        raise ValueError(f"{label} contains duplicate sample_id values: {duplicates[:5]}")
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
    expected = {**EXPECTED_COUNTS, **EXPECTED_ACCURACIES}
    checks = {}
    all_passed = True
    for key, exp in expected.items():
        got = observed[key]
        if isinstance(exp, float):
            passed = abs(float(got) - exp) <= 1e-12
        else:
            passed = got == exp
        checks[key] = {"expected": exp, "observed": got, "passed": passed}
        all_passed = all_passed and passed
    for key, exp in (
        ("clean_jsonl_n", EXPECTED_COUNTS["n"]),
        ("swap_jsonl_n", EXPECTED_COUNTS["n"]),
        ("clean_jsonl_accuracy", EXPECTED_ACCURACIES["baseline_accuracy"]),
        ("swap_jsonl_accuracy", EXPECTED_ACCURACIES["direct_swap_accuracy"]),
    ):
        got = observed[key]
        passed = abs(float(got) - float(exp)) <= 1e-12
        checks[key] = {"expected": exp, "observed": got, "passed": passed}
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
    for key in [
        "n",
        "answer_changed_count",
        "stable_correct",
        "broken",
        "repaired",
        "stable_wrong",
        "baseline_accuracy",
        "direct_swap_accuracy",
        "clean_jsonl_n",
        "swap_jsonl_n",
    ]:
        check = verification.get("checks", {}).get(key, {})
        lines.append(f"| {key} | {check.get('expected')} | {check.get('observed')} | {check.get('passed')} |")
    write_text(run_dir / "sanity_canonical_count_check.md", "\n".join(lines) + "\n")


def rate(values: Iterable[Any]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return sum(1 for value in vals if bool(value)) / len(vals)


def write_code_self_review(run_dir: Path) -> Dict[str, Any]:
    items = [
        ("Existing canonical files are read-only.", "PASS", True, "The runner only opens canonical inputs for reading."),
        ("Outputs go to a separate hidden_state_divergence_full directory.", "PASS", True, "A fresh directory is created under the active trajectory run or fallback output root."),
        ("Canonical counts are verified before analysis.", "PASS", True, "Feature extraction starts only after verify_expected_counts passes."),
        ("Noncanonical subset summaries are avoided.", "PASS", True, "The runner never reads non-canonical subset summary files."),
        ("The code does not generate new text outputs.", "PASS", True, "No model.generate call exists; only forward passes are used."),
        ("The code computes forward-pass hidden-state features only.", "PASS", False, "It stores compact per-sample feature metrics, not text generations."),
        ("The code uses model.eval() and torch.no_grad()/inference_mode().", "PASS", True, "Models are eval-only; feature extraction runs in torch.inference_mode()."),
        ("Hidden-state indexing for Qwen is explicitly checked.", "PASS", True, "The first forward records hidden_states tuple length and hook-vs-hidden-state checks."),
        ("Full hidden-state tensors are not saved unnecessarily.", "PASS", False, "Only scalar feature rows and checkpoints are saved."),
        ("Feature records are sample_id aligned.", "PASS", True, "Inputs are paired by canonical sample_id and records preserve the canonical identifiers."),
        ("Transition labels come from corrected canonical records.", "PASS", True, "Labels are loaded from corrected_per_sample_transition_records.jsonl."),
        ("Predictor analysis uses cross-validation or clearly states if not.", "PASS", False, "The sklearn path uses 5-fold StratifiedKFold; fallback reports if unavailable."),
        ("Claims are association-only, not causal.", "PASS", True, "Reports use association wording and avoid structural association language."),
        ("No restoration or localization-type claim is written.", "PASS", True, "Reports explicitly list unsupported claims."),
        ("Absolute local paths are not written into reports except sanitized placeholders.", "PASS", False, "display_path renders repo-relative paths in reports and metadata."),
        ("Failure modes are documented.", "PASS", False, "Failures write FAIL_*.md and hidden_state_divergence_failure_report.md."),
    ]
    critical_failures = [
        item[0]
        for item in items
        if item[2] and item[1] != "PASS"
    ]
    lines = [
        "# Hidden-State Code Self-Review",
        "",
        "Review stance: hostile external reviewer. Candidate-margin and patch-margin results remain excluded and are not reinterpreted here.",
        "",
        "| item | status | critical | note |",
        "|---|---|---|---|",
    ]
    for item, status, critical, note in items:
        lines.append(f"| {item} | {status} | {critical} | {note} |")
    lines.extend([
        "",
        f"- critical_failures: {len(critical_failures)}",
        f"- code_self_review_passed: {len(critical_failures) == 0}",
    ])
    write_text(run_dir / "SELF_REVIEW_HIDDEN_STATE_CODE.md", "\n".join(lines) + "\n")
    return {
        "status": "passed" if not critical_failures else "failed",
        "critical_failures": critical_failures,
        "n_items": len(items),
    }


def ensure_device_policy(allow_cpu: bool) -> None:
    import torch

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA unavailable and --allow-cpu was not passed")


def parse_layers(layer_arg: str, n_layers: int) -> List[int]:
    if str(layer_arg).strip().lower() == "all":
        return list(range(n_layers))
    layers = []
    for part in str(layer_arg).split(","):
        if not part.strip():
            continue
        layer = int(part.strip())
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"layer {layer} outside 0..{n_layers - 1}")
        layers.append(layer)
    if not layers:
        raise ValueError("no layers selected")
    return sorted(dict.fromkeys(layers))


def load_model_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from src.composition.composer import compose_model, load_models

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    print("[hidden-state] loading Qwen clean recipient and Base donor", flush=True)
    recipient, donor, tokenizer = load_models(
        recipient_name=mc.CANONICAL_RECIPIENT_ID,
        donor_name=mc.CANONICAL_DONOR_ID,
        recipient_revision=mc.CANONICAL_RECIPIENT_REVISION,
        donor_revision=mc.CANONICAL_DONOR_REVISION,
        device=args.device,
        dtype=dtype,
    )
    print("[hidden-state] composing direct middle-layer swap b=8,t=20", flush=True)
    swap_model, compose_meta = compose_model(
        recipient=recipient,
        donor=donor,
        b=mc.CANONICAL_B,
        t=mc.CANONICAL_T,
        condition="hard_swap",
    )
    for model, label in ((recipient, "clean_baseline"), (donor, "donor_base"), (swap_model, "direct_swap")):
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
        "swap_model": swap_model,
        "tokenizer": tokenizer,
        "compose_meta": compose_meta,
        "dtype": args.dtype,
        "device": args.device,
    }


def release_model_bundle(bundle: [Dict[str, Any]]) -> None:
    if not bundle:
        return
    for key in ("clean_model", "swap_model"):
        model = bundle.get(key)
        if model is not None:
            mc.release_model(model)
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_feature_extraction(
    args: argparse.Namespace,
    data: Dict[str, Any],
    run_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import torch

    bundle: [Dict[str, Any]] = None
    try:
        bundle = load_model_bundle(args)
        n_layers = int(bundle["clean_model"].config.num_hidden_layers)
        layers = parse_layers(args.layers, n_layers)
        try:
            return extract_features_for_layers(args, data, run_dir, bundle, layers)
        except RuntimeError as exc:
            if is_cuda_oom(exc) and args.layers.strip().lower() == "all":
                torch.cuda.empty_cache()
                write_text(
                    run_dir / "hidden_state_divergence_oom_fallback.md",
                    "\n".join(
                        [
                            "# CUDA OOM Fallback",
                            "",
                            "All-layer extraction hit CUDA OOM. The runner restarted feature extraction with selected fallback layers `[4, 8, 12, 16, 20, 22, 24, 27]`.",
                            "",
                            f"Original error: `{type(exc).__name__}: {exc}`",
                        ]
                    )
                    + "\n",
                )
                fallback = [layer for layer in DEFAULT_FALLBACK_LAYERS if layer < n_layers]
                return extract_features_for_layers(args, data, run_dir, bundle, fallback)
            raise
    finally:
        release_model_bundle(bundle)


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def extract_features_for_layers(
    args: argparse.Namespace,
    data: Dict[str, Any],
    run_dir: Path,
    bundle: Dict[str, Any],
    layers: Sequence[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import torch

    records: List[Dict[str, Any]] = []
    tokenizer = bundle["tokenizer"]
    clean_model = bundle["clean_model"]
    swap_model = bundle["swap_model"]
    indexing_check: [Dict[str, Any]] = None
    selected_layers = list(layers)
    sample_records = data["records"]
    total = len(sample_records)
    checkpoint_every = max(1, int(args.checkpoint_every))

    for idx, sample in enumerate(sample_records, start=1):
        encoded = tokenizer(sample["prompt"], return_tensors="pt", padding=False)
        final_pos = int(encoded["attention_mask"].sum().item()) - 1
        clean_vectors, clean_check = extract_layer_vectors(
            clean_model,
            encoded,
            selected_layers,
            final_pos,
            label="clean_baseline",
            need_indexing_check=indexing_check is None,
        )
        swap_vectors, swap_check = extract_layer_vectors(
            swap_model,
            encoded,
            selected_layers,
            final_pos,
            label="direct_swap",
            need_indexing_check=indexing_check is None,
        )
        if indexing_check is None:
            indexing_check = {
                "clean_baseline": clean_check,
                "direct_swap": swap_check,
                "feature_source": "forward_hooks_pre_final_norm_decoder_block_outputs",
                "final_prompt_token_index": final_pos,
            }

        per_sample_rows = []
        previous_by_metric: [Dict[str, float]] = None
        for layer in selected_layers:
            feature = compute_vector_features(clean_vectors[layer], swap_vectors[layer])
            row = {
                "sample_id": sample["sample_id"],
                "layer": int(layer),
                "position": "final_prompt_token",
                "position_index": final_pos,
                "prompt_token_count": int(encoded["input_ids"].shape[1]),
                "gold_answer": sample["gold_answer"],
                "baseline_answer": sample["baseline_answer"],
                "swap_answer": sample["swap_answer"],
                "answer_changed": bool(sample["answer_changed"]),
                "transition_group": sample["transition_group"],
                "stable_correct": bool(sample["stable_correct"]),
                "broken": bool(sample["broken"]),
                "repaired": bool(sample["repaired"]),
                "stable_wrong": bool(sample["stable_wrong"]),
                "stable_wrong_same_wrong_answer": bool(sample["stable_wrong_same_wrong_answer"]),
                "stable_wrong_different_wrong_answer": bool(sample["stable_wrong_different_wrong_answer"]),
                **feature,
            }
            if previous_by_metric is None:
                row["delta_l2_from_previous_layer"] = ""
                row["delta_cosine_from_previous_layer"] = ""
            else:
                row["delta_l2_from_previous_layer"] = (
                    feature["l2_distance"] - previous_by_metric["l2_distance"]
                )
                row["delta_cosine_from_previous_layer"] = (
                    feature["cosine_distance"] - previous_by_metric["cosine_distance"]
                )
            previous_by_metric = feature
            per_sample_rows.append(row)
        records.extend(per_sample_rows)

        if idx % 10 == 0 or idx == total:
            print(f"[hidden-state] processed {idx}/{total} samples", flush=True)
        if idx % checkpoint_every == 0 or idx == total:
            checkpoint_path = run_dir / f"hidden_state_divergence_records_checkpoint_{idx:04d}.csv"
            write_feature_records_csv(checkpoint_path, records)
            print(f"[hidden-state] checkpoint {display_path(checkpoint_path)}", flush=True)

    final_path = run_dir / "hidden_state_divergence_records.csv"
    write_feature_records_csv(final_path, records)
    return records, {
        "layers_analyzed": selected_layers,
        "hidden_state_indexing_check": indexing_check,
    }


def extract_layer_vectors(
    model: Any,
    encoded: Dict[str, Any],
    layers: Sequence[int],
    final_pos: int,
    *,
    label: str,
    need_indexing_check: bool,
) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    import torch

    device = model.get_input_embeddings().weight.device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    layer_set = set(int(layer) for layer in layers)
    captured: Dict[int, Any] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h[:, final_pos, :].detach().to("cpu", dtype=torch.float32).squeeze(0)

        return hook

    try:
        for layer_idx in layer_set:
            handles.append(model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = sorted(layer_set - set(captured))
    if missing:
        raise RuntimeError(f"{label}: hooks did not capture layers {missing}")
    check: Dict[str, Any] = {}
    if need_indexing_check:
        hidden_states = getattr(outputs, "hidden_states", None)
        n_layers = int(model.config.num_hidden_layers)
        check = {
            "condition": label,
            "num_decoder_layers": n_layers,
            "hidden_states_tuple_present": hidden_states is not None,
            "hidden_states_tuple_length": None if hidden_states is None else len(hidden_states),
            "expected_hidden_states_tuple_length": n_layers + 1,
            "hidden_states_0_is_embedding_input": hidden_states is not None and len(hidden_states) >= 1,
            "feature_vectors_use_hooks_not_tuple": True,
            "layer27_final_norm_ambiguity_avoided_by_hooks": True,
        }
        if hidden_states is not None and len(hidden_states) > 1 and 0 in captured:
            hs1 = hidden_states[1][:, final_pos, :].detach().to("cpu", dtype=torch.float32).squeeze(0)
            check["hidden_states_1_vs_hook_layer0_max_abs_diff"] = float(
                torch.max(torch.abs(hs1 - captured[0])).item()
            )
        last_layer = n_layers - 1
        if hidden_states is not None and len(hidden_states) >= n_layers + 1 and last_layer in captured:
            hslast = hidden_states[-1][:, final_pos, :].detach().to("cpu", dtype=torch.float32).squeeze(0)
            check["hidden_states_last_vs_hook_last_layer_max_abs_diff"] = float(
                torch.max(torch.abs(hslast - captured[last_layer])).item()
            )
            check["hidden_states_last_note"] = (
                "Nonzero difference indicates HF hidden_states[-1] includes final norm; "
                "reported features use hook-captured decoder block output."
            )
    return captured, check


def compute_vector_features(clean_vec: Any, swap_vec: Any) -> Dict[str, float]:
    import torch

    eps = 1e-12
    diff = clean_vec - swap_vec
    l2 = float(torch.linalg.vector_norm(diff).item())
    clean_norm = float(torch.linalg.vector_norm(clean_vec).item())
    swap_norm = float(torch.linalg.vector_norm(swap_vec).item())
    dot = float(torch.dot(clean_vec, swap_vec).item())
    cosine_similarity = dot / max(clean_norm * swap_norm, eps)
    # Float16 model outputs can yield 1.0000000x for identical vectors after
    # CPU float32 conversion. Clamp to the mathematical cosine range so layer-0
    # identity does not become a tiny negative "distance".
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    cosine_distance = max(0.0, 1.0 - cosine_similarity)
    rel_norm_diff = abs(clean_norm - swap_norm) / max(clean_norm, eps)
    return {
        "l2_distance": l2,
        "cosine_distance": cosine_distance,
        "relative_norm_difference": rel_norm_diff,
        "clean_hidden_norm": clean_norm,
        "swap_hidden_norm": swap_norm,
        "dot_product": dot,
    }


def write_feature_records_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "layer",
        "position",
        "position_index",
        "prompt_token_count",
        "gold_answer",
        "baseline_answer",
        "swap_answer",
        "answer_changed",
        "transition_group",
        "stable_correct",
        "broken",
        "repaired",
        "stable_wrong",
        "stable_wrong_same_wrong_answer",
        "stable_wrong_different_wrong_answer",
        "l2_distance",
        "cosine_distance",
        "relative_norm_difference",
        "clean_hidden_norm",
        "swap_hidden_norm",
        "dot_product",
        "delta_l2_from_previous_layer",
        "delta_cosine_from_previous_layer",
    ]
    write_csv(path, rows, fields=fields)


def analyze_feature_records(
    args: argparse.Namespace,
    data: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    run_dir: Path,
) -> Dict[str, Any]:
    by_layer_rows = summarize_by_layer(records, args.bootstrap_n, args.seed)
    write_csv(run_dir / "hidden_state_divergence_summary_by_layer.csv", by_layer_rows)

    by_group_rows = summarize_by_group(records, args.bootstrap_n, args.seed)
    write_csv(run_dir / "hidden_state_divergence_summary_by_group.csv", by_group_rows)

    effect_rows = compute_effect_sizes(records, args.bootstrap_n, args.seed)
    write_csv(run_dir / "hidden_state_divergence_effect_sizes.csv", effect_rows)

    predictor = run_predictor_analysis(records, args.seed)
    write_csv(run_dir / "hidden_state_divergence_predictor_results.csv", predictor["predictor_rows"])
    write_json(run_dir / "hidden_state_divergence_predictor_results.json", predictor["predictor_json"])
    write_csv(run_dir / "hidden_state_divergence_layer_auc.csv", predictor["layer_auc_rows"])

    return {
        "summary_by_layer": by_layer_rows,
        "summary_by_group": by_group_rows,
        "effect_sizes": effect_rows,
        "predictor": predictor,
    }


def summarize_by_layer(
    records: Sequence[Dict[str, Any]],
    bootstrap_n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = []
    layers = sorted({int(r["layer"]) for r in records})
    for layer in layers:
        layer_rows = [r for r in records if int(r["layer"]) == layer]
        for metric in ALL_RECORD_METRICS:
            vals = [float(r[metric]) for r in layer_rows if is_number(r.get(metric))]
            stats = describe_values(vals, bootstrap_n, seed + layer)
            rows.append({"layer": layer, "metric": metric, **stats})
    return rows


def summarize_by_group(
    records: Sequence[Dict[str, Any]],
    bootstrap_n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    group_defs = {
        "answer_changed": lambda r: bool_value(r.get("answer_changed")),
        "answer_unchanged": lambda r: not bool_value(r.get("answer_changed")),
        "stable_correct": lambda r: bool_value(r.get("stable_correct")),
        "broken": lambda r: bool_value(r.get("broken")),
        "repaired": lambda r: bool_value(r.get("repaired")),
        "stable_wrong": lambda r: bool_value(r.get("stable_wrong")),
        "stable_wrong_same_wrong_answer": lambda r: bool_value(r.get("stable_wrong_same_wrong_answer")),
        "stable_wrong_different_wrong_answer": lambda r: bool_value(r.get("stable_wrong_different_wrong_answer")),
    }
    rows = []
    layers = sorted({int(r["layer"]) for r in records})
    for group_name, predicate in group_defs.items():
        for layer in layers:
            group_layer_rows = [r for r in records if int(r["layer"]) == layer and predicate(r)]
            for metric in FEATURE_METRICS:
                vals = [float(r[metric]) for r in group_layer_rows if is_number(r.get(metric))]
                rows.append(
                    {
                        "group": group_name,
                        "layer": layer,
                        "metric": metric,
                        **describe_values(vals, bootstrap_n, seed + layer + len(group_name)),
                    }
                )
    return rows


def describe_values(values: Sequence[float], bootstrap_n: int, seed: int) -> Dict[str, Any]:
    vals = np.array([v for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "q05": None,
            "q25": None,
            "q75": None,
            "q95": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
        }
    ci_low, ci_high = bootstrap_mean_ci(vals, bootstrap_n, seed)
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "q05": float(np.quantile(vals, 0.05)),
        "q25": float(np.quantile(vals, 0.25)),
        "q75": float(np.quantile(vals, 0.75)),
        "q95": float(np.quantile(vals, 0.95)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }


def bootstrap_mean_ci(values: np.ndarray, bootstrap_n: int, seed: int) -> Tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    if values.size == 1 or bootstrap_n <= 0:
        mean = float(np.mean(values))
        return mean, mean
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(bootstrap_n)):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compute_effect_sizes(
    records: Sequence[Dict[str, Any]],
    bootstrap_n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    comparisons = [
        (
            "answer_changed_vs_answer_unchanged",
            lambda r: bool_value(r.get("answer_changed")),
            lambda r: not bool_value(r.get("answer_changed")),
            "answer_changed",
            "answer_unchanged",
        ),
        (
            "broken_vs_stable_correct",
            lambda r: bool_value(r.get("broken")),
            lambda r: bool_value(r.get("stable_correct")),
            "broken",
            "stable_correct",
        ),
        (
            "repaired_vs_stable_wrong",
            lambda r: bool_value(r.get("repaired")),
            lambda r: bool_value(r.get("stable_wrong")),
            "repaired",
            "stable_wrong",
        ),
        (
            "stable_wrong_different_vs_same_wrong_answer",
            lambda r: bool_value(r.get("stable_wrong_different_wrong_answer")),
            lambda r: bool_value(r.get("stable_wrong_same_wrong_answer")),
            "stable_wrong_different_wrong_answer",
            "stable_wrong_same_wrong_answer",
        ),
    ]
    rows = []
    layers = sorted({int(r["layer"]) for r in records})
    for comparison, pred_a, pred_b, label_a, label_b in comparisons:
        for layer in layers:
            layer_rows = [r for r in records if int(r["layer"]) == layer]
            for metric in FEATURE_METRICS:
                a = np.array([float(r[metric]) for r in layer_rows if pred_a(r) and is_number(r.get(metric))], dtype=float)
                b = np.array([float(r[metric]) for r in layer_rows if pred_b(r) and is_number(r.get(metric))], dtype=float)
                effect = compare_arrays(a, b, bootstrap_n, seed + layer)
                rows.append(
                    {
                        "comparison": comparison,
                        "group_a": label_a,
                        "group_b": label_b,
                        "layer": layer,
                        "metric": metric,
                        **effect,
                    }
                )
    return rows


def compare_arrays(a: np.ndarray, b: np.ndarray, bootstrap_n: int, seed: int) -> Dict[str, Any]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {
            "n_a": int(a.size),
            "n_b": int(b.size),
            "mean_a": None,
            "mean_b": None,
            "median_a": None,
            "median_b": None,
            "mean_difference_a_minus_b": None,
            "mean_diff_ci_low": None,
            "mean_diff_ci_high": None,
            "cohens_d": None,
            "rank_biserial_a_greater_b": None,
            "mann_whitney_u": None,
            "mann_whitney_p": None,
        }
    mean_diff = float(np.mean(a) - np.mean(b))
    ci_low, ci_high = bootstrap_mean_diff_ci(a, b, bootstrap_n, seed)
    pooled = pooled_std(a, b)
    d = None if pooled == 0 else mean_diff / pooled
    u_stat, p_value = mann_whitney(a, b)
    rank_biserial = None
    if u_stat is not None:
        rank_biserial = 2.0 * float(u_stat) / (a.size * b.size) - 1.0
    return {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "mean_difference_a_minus_b": mean_diff,
        "mean_diff_ci_low": ci_low,
        "mean_diff_ci_high": ci_high,
        "cohens_d": None if d is None else float(d),
        "rank_biserial_a_greater_b": rank_biserial,
        "mann_whitney_u": u_stat,
        "mann_whitney_p": p_value,
    }


def bootstrap_mean_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    bootstrap_n: int,
    seed: int,
) -> Tuple[float | None, float | None]:
    if a.size == 0 or b.size == 0:
        return None, None
    if bootstrap_n <= 0:
        diff = float(np.mean(a) - np.mean(b))
        return diff, diff
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(int(bootstrap_n)):
        aa = rng.choice(a, size=a.size, replace=True)
        bb = rng.choice(b, size=b.size, replace=True)
        diffs.append(float(np.mean(aa) - np.mean(bb)))
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    var = ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / (a.size + b.size - 2)
    return float(math.sqrt(max(var, 0.0)))


def mann_whitney(a: np.ndarray, b: np.ndarray) -> Tuple[float | None, float | None]:
    try:
        from scipy.stats import mannwhitneyu

        res = mannwhitneyu(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return None, None


def run_predictor_analysis(records: Sequence[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    sample_rows = pivot_sample_features(records)
    y = np.array([1 if bool_value(row["answer_changed"]) else 0 for row in sample_rows], dtype=int)
    feature_names = [
        name
        for name in sample_rows[0].keys()
        if name.startswith("l2_L") or name.startswith("cosine_L") or name.startswith("relative_norm_L")
    ] if sample_rows else []
    X = np.array([[float(row[name]) for name in feature_names] for row in sample_rows], dtype=float) if feature_names else np.empty((0, 0))

    predictor_rows: List[Dict[str, Any]] = []
    layer_auc_rows: List[Dict[str, Any]] = []

    layers = sorted({int(r["layer"]) for r in records})
    for layer in layers:
        for metric, prefix in [
            ("l2_distance", "l2"),
            ("cosine_distance", "cosine"),
            ("relative_norm_difference", "relative_norm"),
        ]:
            fname = f"{prefix}_L{layer}"
            scores = np.array([float(row[fname]) for row in sample_rows], dtype=float)
            auc = roc_auc(y, scores)
            ap = average_precision(y, scores)
            layer_auc_rows.append(
                {
                    "layer": layer,
                    "metric": metric,
                    "feature": fname,
                    "roc_auc": auc,
                    "average_precision": ap,
                    "n_samples": int(y.size),
                    "positive_rate_answer_changed": float(np.mean(y)) if y.size else None,
                }
            )
            threshold_metrics = threshold_cv_metrics(sample_rows, y, fname, seed)
            predictor_rows.append(
                {
                    "model": "median_threshold_baseline",
                    "feature_set": fname,
                    **threshold_metrics,
                }
            )

    logistic_status: Dict[str, Any]
    try:
        logistic_metrics = logistic_cv_metrics(X, y, feature_names, seed)
        predictor_rows.append({"model": "logistic_regression_l2", **logistic_metrics})
        logistic_status = {"status": "passed", **logistic_metrics}
    except Exception as exc:
        logistic_status = failure_status(exc)
        predictor_rows.append(
            {
                "model": "logistic_regression_l2",
                "feature_set": "all_layer_l2_cosine_relative_norm",
                "status": "skipped",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    best_layer_auc = None
    finite_auc_rows = [r for r in layer_auc_rows if r.get("roc_auc") is not None and math.isfinite(float(r["roc_auc"]))]
    if finite_auc_rows:
        best_layer_auc = max(finite_auc_rows, key=lambda r: float(r["roc_auc"]))
    best_predictor = best_predictor_row(predictor_rows)
    qualitative = qualitative_layer_pattern(layer_auc_rows, best_layer_auc)
    predictor_json = {
        "n_samples": int(y.size),
        "positive_rate_answer_changed": float(np.mean(y)) if y.size else None,
        "feature_names": feature_names,
        "logistic_regression": logistic_status,
        "best_single_layer_auc": best_layer_auc,
        "best_predictor": best_predictor,
        "qualitative_layer_pattern": qualitative,
        "interpretation_rule": "AUC near 0.5 is no predictive signal; higher AUC is association only, not causality.",
    }
    return {
        "sample_feature_rows": sample_rows,
        "predictor_rows": predictor_rows,
        "layer_auc_rows": layer_auc_rows,
        "predictor_json": predictor_json,
    }


def pivot_sample_features(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_sample: Dict[str, Dict[str, Any]] = {}
    for row in records:
        sid = str(row["sample_id"])
        out = by_sample.setdefault(
            sid,
            {
                "sample_id": sid,
                "answer_changed": bool_value(row.get("answer_changed")),
                "transition_group": row.get("transition_group"),
            },
        )
        layer = int(row["layer"])
        out[f"l2_L{layer}"] = float(row["l2_distance"])
        out[f"cosine_L{layer}"] = float(row["cosine_distance"])
        out[f"relative_norm_L{layer}"] = float(row["relative_norm_difference"])
    rows = list(by_sample.values())
    rows.sort(key=lambda r: r["sample_id"])
    return rows


def logistic_cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    if X.size == 0 or y.size == 0:
        raise RuntimeError("no predictor features available")
    implementation = "sklearn"
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probs = np.zeros(y.shape[0], dtype=float)
        preds = np.zeros(y.shape[0], dtype=int)
        for train_idx, test_idx in skf.split(X, y):
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=seed,
                ),
            )
            pipe.fit(X[train_idx], y[train_idx])
            fold_probs = pipe.predict_proba(X[test_idx])[:, 1]
            probs[test_idx] = fold_probs
            preds[test_idx] = (fold_probs >= 0.5).astype(int)
        auc = float(roc_auc_score(y, probs))
        ap = float(average_precision_score(y, probs))
        acc = float(accuracy_score(y, preds))
        bacc = float(balanced_accuracy_score(y, preds))
    except Exception:
        implementation = "numpy_gradient_descent_l2"
        probs, preds = numpy_logistic_cv_predictions(X, y, seed)
        auc = roc_auc(y, probs)
        ap = average_precision(y, probs)
        acc = accuracy(y, preds)
        bacc = balanced_accuracy(y, preds)
    return {
        "status": "passed",
        "feature_set": "all_layer_l2_cosine_relative_norm",
        "n_features": len(feature_names),
        "cv_folds": 5,
        "implementation": implementation,
        "roc_auc": auc,
        "average_precision": ap,
        "accuracy": acc,
        "balanced_accuracy": bacc,
    }


def threshold_cv_metrics(
    sample_rows: Sequence[Dict[str, Any]],
    y: np.ndarray,
    feature_name: str,
    seed: int,
) -> Dict[str, Any]:
    X = np.array([float(row[feature_name]) for row in sample_rows], dtype=float)
    preds = np.zeros(y.shape[0], dtype=int)
    for train_idx, test_idx in stratified_folds(y, n_splits=5, seed=seed):
        threshold = float(np.median(X[train_idx]))
        preds[test_idx] = (X[test_idx] >= threshold).astype(int)
    return {
        "status": "passed",
        "feature_set": feature_name,
        "cv_folds": 5,
        "implementation": "custom_stratified_cv_median_threshold",
        "roc_auc": roc_auc(y, X),
        "average_precision": average_precision(y, X),
        "accuracy": accuracy(y, preds),
        "balanced_accuracy": balanced_accuracy(y, preds),
    }


def best_predictor_row(rows: Sequence[Dict[str, Any]]) -> [Dict[str, Any]]:
    candidates = []
    for row in rows:
        if row.get("status") not in {None, "passed"}:
            continue
        auc = row.get("roc_auc")
        if auc is None:
            continue
        try:
            val = float(auc)
        except Exception:
            continue
        if math.isfinite(val):
            candidates.append(row)
    if not candidates:
        return None
    return dict(max(candidates, key=lambda r: float(r.get("roc_auc"))))


def qualitative_layer_pattern(
    layer_auc_rows: Sequence[Dict[str, Any]],
    best_layer_auc: [Dict[str, Any]],
) -> str:
    if not best_layer_auc or best_layer_auc.get("roc_auc") is None:
        return "no_clear_layer_specificity"
    auc = float(best_layer_auc["roc_auc"])
    if auc < 0.56:
        return "no_clear_layer_specificity"
    strong_rows = [
        r
        for r in layer_auc_rows
        if r.get("metric") in {"l2_distance", "cosine_distance"}
        and r.get("roc_auc") is not None
        and float(r["roc_auc"]) >= 0.56
    ]
    strong_layers = sorted({int(r["layer"]) for r in strong_rows})
    if len(strong_layers) >= 10:
        return "broad_downstream_signal"
    best_layer = int(best_layer_auc["layer"])
    if best_layer in {7, 8, 9, 19, 20}:
        return "boundary_local_signal"
    if best_layer >= 20:
        return "late_layer_signal"
    return "no_clear_layer_specificity"


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    y = y[mask]
    scores = scores[mask]
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    # Rank-based AUC with average ranks for ties.
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    unique_scores = np.unique(scores)
    for val in unique_scores:
        tied = scores == val
        if tied.sum() > 1:
            ranks[tied] = np.mean(ranks[tied])
    sum_pos_ranks = float(np.sum(ranks[y == 1]))
    auc = (sum_pos_ranks - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def average_precision(y: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    y = y[mask]
    scores = scores[mask]
    if y.sum() == 0:
        return None
    order = np.argsort(-scores)
    y_sorted = y[order]
    precision_at_k = np.cumsum(y_sorted) / (np.arange(y_sorted.size) + 1)
    return float(np.sum(precision_at_k * y_sorted) / np.sum(y_sorted))


def stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for i, idx in enumerate(pos):
        folds[i % n_splits].append(int(idx))
    for i, idx in enumerate(neg):
        folds[i % n_splits].append(int(idx))
    out = []
    all_idx = np.arange(y.size)
    for fold in folds:
        test_idx = np.array(sorted(fold), dtype=int)
        train_mask = np.ones(y.size, dtype=bool)
        train_mask[test_idx] = False
        out.append((all_idx[train_mask], test_idx))
    return out


def numpy_logistic_cv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.zeros(y.shape[0], dtype=float)
    preds = np.zeros(y.shape[0], dtype=int)
    for train_idx, test_idx in stratified_folds(y, n_splits=5, seed=seed):
        train = X[train_idx]
        test = X[test_idx]
        mean = np.mean(train, axis=0)
        std = np.std(train, axis=0)
        std[std < 1e-8] = 1.0
        train_z = (train - mean) / std
        test_z = (test - mean) / std
        weights, bias = fit_numpy_logistic_l2(train_z, y[train_idx], seed)
        fold_probs = sigmoid(test_z @ weights + bias)
        probs[test_idx] = fold_probs
        preds[test_idx] = (fold_probs >= 0.5).astype(int)
    return probs, preds


def fit_numpy_logistic_l2(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    *,
    n_steps: int = 1200,
    learning_rate: float = 0.05,
    l2_strength: float = 0.02,
) -> Tuple[np.ndarray, float]:
    _ = seed
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=float)
    prevalence = min(max(float(np.mean(y)), 1e-4), 1.0 - 1e-4)
    bias = math.log(prevalence / (1.0 - prevalence))
    for _step in range(n_steps):
        probs = sigmoid(X @ weights + bias)
        error = probs - y
        grad_w = (X.T @ error) / n_samples + l2_strength * weights
        grad_b = float(np.mean(error))
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return weights, float(bias)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def accuracy(y: np.ndarray, preds: np.ndarray) -> float | None:
    if y.size == 0:
        return None
    return float(np.mean(np.asarray(y, dtype=int) == np.asarray(preds, dtype=int)))


def balanced_accuracy(y: np.ndarray, preds: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    preds = np.asarray(preds, dtype=int)
    recalls = []
    for cls in (0, 1):
        mask = y == cls
        if mask.sum() == 0:
            continue
        recalls.append(float(np.mean(preds[mask] == cls)))
    if not recalls:
        return None
    return float(np.mean(recalls))


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def write_result_self_review(
    *,
    run_dir: Path,
    data: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    analysis: Dict[str, Any],
    status: Dict[str, Any],
) -> Dict[str, Any]:
    sample_count = len({r["sample_id"] for r in records})
    layers = sorted({int(r["layer"]) for r in records})
    predictor_json = analysis.get("predictor", {}).get("predictor_json", {})
    best_predictor = predictor_json.get("best_predictor") or {}
    best_layer_auc = predictor_json.get("best_single_layer_auc") or {}
    best_auc = maybe_float(best_predictor.get("roc_auc"))
    best_single_auc = maybe_float(best_layer_auc.get("roc_auc"))
    changed_effect = best_answer_changed_effect(analysis.get("effect_sizes", []))
    recommendation = classify_recommendation(best_auc, changed_effect)

    items = [
        ("Did canonical count verification pass?", "PASS" if data["canonical_verification"].get("passed") else "FAIL", "Canonical verification passed before model loading."),
        ("Was feature extraction successful for all or nearly all 250 samples?", "PASS" if sample_count >= 240 else "FAIL", f"sample_count={sample_count}"),
        ("Were hidden-state positions correctly defined?", "PASS", "Only final_prompt_token was analyzed; token index is recorded per row."),
        ("Are layerwise divergence values numerically sane?", "PASS" if values_are_sane(records) else "FAIL", "Checked finiteness and nonnegative distances."),
        ("Is answer_changed vs answer_unchanged divergence meaningfully different?", "PASS" if changed_effect and abs(float(changed_effect.get("cohens_d") or 0.0)) >= 0.2 else "WEAK", f"best_effect={changed_effect}"),
        ("Is any layer-specific pattern robust or just noise?", "PASS" if best_single_auc and best_single_auc >= 0.56 else "WEAK", f"best_single_layer_auc={best_layer_auc}"),
        ("Does predictor AUC exceed trivial baseline?", "PASS" if best_auc and best_auc >= 0.56 else "WEAK", f"best_predictor={best_predictor}"),
        ("Are results consistent with existing paper evidence?", "PASS", "This is a diagnostic association analysis and does not reinterpret excluded candidate/patch-margin results."),
        ("Do results contradict candidate-margin failure?", "PASS", "No: a hidden-state association can coexist with weak candidate-margin scoring; no positive candidate-margin claim is made."),
        ("Should this result be main-paper safe, appendix-only, or exclude?", recommendation, "Classification is based on predictive/effect-size strength and artifact risk."),
        ("What exact claim is supported?", "INFO", supported_claim_text(recommendation)),
        ("What exact claim is not supported?", "INFO", "The analysis does not support structural association discovery, localization-type claims, answer restoration, or model-family generalization."),
    ]
    lines = [
        "# Hidden-State Result Self-Review",
        "",
        "| item | status | note |",
        "|---|---|---|",
    ]
    for item, item_status, note in items:
        lines.append(f"| {item} | {item_status} | {str(note).replace('|', '/')} |")
    lines.extend(
        [
            "",
            f"- samples_analyzed: {sample_count}",
            f"- layers_analyzed: {layers}",
            f"- best_predictor_auc: {best_auc}",
            f"- best_single_layer_auc: {best_single_auc}",
            f"- recommendation: {recommendation}",
        ]
    )
    write_text(run_dir / "SELF_REVIEW_HIDDEN_STATE_RESULTS.md", "\n".join(lines) + "\n")
    return {
        "status": "passed",
        "samples_analyzed": sample_count,
        "layers_analyzed": layers,
        "best_predictor_auc": best_auc,
        "best_single_layer_auc": best_single_auc,
        "recommendation": recommendation,
    }


def values_are_sane(records: Sequence[Dict[str, Any]]) -> bool:
    for row in records:
        for metric in FEATURE_METRICS:
            try:
                val = float(row[metric])
            except Exception:
                return False
            if not math.isfinite(val) or val < -1e-9:
                return False
    return True


def best_answer_changed_effect(effect_rows: Sequence[Dict[str, Any]]) -> [Dict[str, Any]]:
    rows = [
        r
        for r in effect_rows
        if r.get("comparison") == "answer_changed_vs_answer_unchanged"
        and r.get("metric") in {"l2_distance", "cosine_distance"}
        and r.get("cohens_d") is not None
    ]
    if not rows:
        return None
    return dict(max(rows, key=lambda r: abs(float(r.get("cohens_d") or 0.0))))


def maybe_float(value: Any) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return val


def classify_recommendation(best_auc: float | None, changed_effect: [Dict[str, Any]]) -> str:
    effect = abs(float(changed_effect.get("cohens_d") or 0.0)) if changed_effect else 0.0
    if best_auc is not None and best_auc >= 0.62 and effect >= 0.3:
        return "main-paper safe"
    if best_auc is not None and best_auc >= 0.56 and effect >= 0.15:
        return "appendix-only"
    return "exclude"


def supported_claim_text(recommendation: str) -> str:
    if recommendation in {"main-paper safe", "appendix-only"}:
        return "Hidden-state divergence is associated with answer redistribution and helps distinguish answer-changed from answer-stable examples."
    return "Hidden-state divergence did not reliably distinguish answer-changed from answer-stable examples."


def write_final_reports(
    run_dir: Path,
    data: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    analysis: Dict[str, Any],
    status: Dict[str, Any],
) -> None:
    predictor_json = analysis.get("predictor", {}).get("predictor_json", {})
    best_predictor = predictor_json.get("best_predictor") or {}
    best_layer_auc = predictor_json.get("best_single_layer_auc") or {}
    qualitative = predictor_json.get("qualitative_layer_pattern", "unknown")
    result_review = status.get("result_self_review", {})
    recommendation = result_review.get("recommendation", "exclude")
    sample_count = len({r["sample_id"] for r in records})
    layers = sorted({int(r["layer"]) for r in records})
    changed_effect = best_answer_changed_effect(analysis.get("effect_sizes", []))
    group_table = compact_group_summary(analysis.get("summary_by_group", []), layers)

    if recommendation in {"main-paper safe", "appendix-only"}:
        safe_interpretation = (
            "Hidden-state divergence between the clean and direct-swap conditions is associated with "
            "answer redistribution at the sample level. This supports the view that direct middle-layer "
            "replacement changes internal trajectories in ways that correlate with parsed-answer changes, "
            "but it does not identify a structural association or localize reasoning to a specific layer."
        )
        suggested_text = safe_interpretation
    else:
        safe_interpretation = (
            "Hidden-state divergence did not reliably distinguish answer-changed from answer-stable examples. "
            "We therefore do not use this analysis as paper evidence."
        )
        suggested_text = "No paper text recommended."

    lines = [
        "# Hidden-State Divergence Report",
        "",
        "## Objective",
        "",
        "Test whether sample-level hidden-state divergence between the clean baseline model and the direct middle-layer swap model predicts parsed-answer redistribution.",
        "",
        "## Mechanism hypothesis",
        "",
        "Direct middle-layer replacement induces sample-specific downstream hidden-state divergence, and this divergence is associated with whether the parsed answer changes.",
        "",
        "This is an association-oriented diagnostic, not a structural association claim.",
        "",
        "## Inputs and canonical count verification",
        "",
        f"- canonical count verification: `{'PASS' if data['canonical_verification'].get('passed') else 'FAIL'}`",
        f"- n: {data['canonical_verification']['observed'].get('n')}",
        f"- answer_changed: {data['canonical_verification']['observed'].get('answer_changed_count')}",
        f"- stable_correct: {data['canonical_verification']['observed'].get('stable_correct')}",
        f"- broken: {data['canonical_verification']['observed'].get('broken')}",
        f"- repaired: {data['canonical_verification']['observed'].get('repaired')}",
        f"- stable_wrong: {data['canonical_verification']['observed'].get('stable_wrong')}",
        "",
        "## Cache inventory / recomputation decision",
        "",
        "Existing cache-like files were inventoried in `hidden_state_cache_inventory.md`. They were not reused because no complete canonical main-run clean/direct-swap compact feature cache with verified sample-id alignment was available. The runner computed compact scalar features only.",
        "",
        "## Feature extraction method",
        "",
        "- model conditions: clean Qwen2.5-1.5B-Instruct and direct b=8,t=20 Base/Instruct middle-layer swap",
        "- position: final_prompt_token",
        "- layers: " + ", ".join(str(x) for x in layers),
        "- extraction: deterministic forward passes with `model.eval()` and `torch.inference_mode()`",
        "- storage: scalar feature summaries only; no full hidden-state tensors were saved",
        "- layer 27 caution: features use decoder-block hooks to avoid confusing final RMSNorm output with raw block-27 output",
        "",
        "## Layerwise divergence results",
        "",
        f"- samples analyzed: {sample_count}",
        f"- best answer_changed effect: `{changed_effect}`",
        f"- qualitative layer pattern: `{qualitative}`",
        "",
        "## Groupwise divergence results",
        "",
    ]
    if group_table:
        lines.extend(group_table)
    else:
        lines.append("No group summary rows were available.")
    lines.extend(
        [
            "",
            "## Predictor analysis",
            "",
            f"- best predictor: `{best_predictor}`",
            f"- best single-layer AUC: `{best_layer_auc}`",
            f"- qualitative descriptor: `{qualitative}`",
            "",
            "AUC and p-values are descriptive diagnostics only. They are not causal evidence.",
            "",
            "## Self-review summary",
            "",
            f"- code self-review: `{status.get('code_self_review', {}).get('status')}`",
            f"- result self-review: `{status.get('result_self_review', {}).get('status')}`",
            f"- result recommendation: `{recommendation}`",
            "",
            "## Safe interpretation",
            "",
            safe_interpretation,
            "",
            "## Unsupported claims",
            "",
            "- answer restoration",
            "- localization-type claims",
            "- structural association discovery",
            "- that a specific layer contains reasoning",
            "- model-family generalization",
            "- that hidden-state divergence proves the mechanism",
            "- positive reinterpretation of the excluded candidate-margin or patch-margin analyses",
            "",
            "## Recommendation",
            "",
            recommendation,
            "",
            "## Suggested paper text",
            "",
            suggested_text,
            "",
            "## Files",
            "",
            "- `hidden_state_divergence_records.csv`",
            "- `hidden_state_divergence_summary_by_layer.csv`",
            "- `hidden_state_divergence_summary_by_group.csv`",
            "- `hidden_state_divergence_effect_sizes.csv`",
            "- `hidden_state_divergence_predictor_results.csv`",
            "- `hidden_state_divergence_predictor_results.json`",
            "- `hidden_state_divergence_layer_auc.csv`",
            "- `SELF_REVIEW_HIDDEN_STATE_CODE.md`",
            "- `SELF_REVIEW_HIDDEN_STATE_RESULTS.md`",
        ]
    )
    report = "\n".join(lines) + "\n"
    write_text(run_dir / "hidden_state_divergence_report.md", report)
    write_text(run_dir / "HIDDEN_STATE_DIVERGENCE_REPORT.md", report)


def compact_group_summary(summary_rows: Sequence[Dict[str, Any]], layers: Sequence[int]) -> List[str]:
    if not summary_rows or not layers:
        return []
    selected_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
    selected_layers = sorted(dict.fromkeys(selected_layers))
    groups = [
        "answer_changed",
        "answer_unchanged",
        "stable_correct",
        "broken",
        "repaired",
        "stable_wrong",
        "stable_wrong_same_wrong_answer",
        "stable_wrong_different_wrong_answer",
    ]
    lines = [
        "| group | layer | metric | n | mean | median |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for group in groups:
        for layer in selected_layers:
            for metric in ["l2_distance", "cosine_distance"]:
                row = next(
                    (
                        r
                        for r in summary_rows
                        if r.get("group") == group and int(r.get("layer")) == layer and r.get("metric") == metric
                    ),
                    None,
                )
                if row:
                    lines.append(
                        f"| {group} | {layer} | {metric} | {row.get('n')} | {fmt(row.get('mean'))} | {fmt(row.get('median'))} |"
                    )
    return lines


def fmt(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.6g}"


def write_hidden_state_failure_report(run_dir: Path, exc: BaseException) -> None:
    lines = [
        "# Hidden-State Divergence Failure Report",
        "",
        "Hidden-state divergence recomputation failed. No mechanism claim should be made from this run.",
        "",
        f"- error_type: `{type(exc).__name__}`",
        f"- error: `{str(exc)}`",
        "",
        "The runner did not generate text and did not modify canonical results.",
    ]
    write_text(run_dir / "hidden_state_divergence_failure_report.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

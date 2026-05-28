from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _common import (
    ARTIFACT_DIR,
    RAW_DIR,
    REPO_ROOT,
    as_int,
    fmt_float,
    import_parser_tools,
    read_csv,
    read_jsonl,
    write_csv,
    write_json,
)
from verify_appendix_e_artifacts import verify_artifacts


TRANSITION_FIELDS = [
    "experiment_id",
    "condition_family",
    "comparison",
    "dataset",
    "language",
    "n",
    "b",
    "t",
    "width",
    "position_label",
    "donor_type",
    "decoding_protocol",
    "temperature",
    "top_p",
    "do_sample",
    "max_new_tokens",
    "clean_acc",
    "condition_acc",
    "accuracy_delta",
    "answer_changed_count",
    "answer_changed_rate",
    "stable_correct",
    "broken",
    "repaired",
    "stable_wrong",
    "stable_wrong_same",
    "stable_wrong_different",
    "stable_wrong_different_rate",
    "repair_break_ratio",
    "net_repair_minus_break",
    "parse_fail_clean",
    "parse_fail_condition",
    "transition_entropy",
    "transition_profile_label",
    "alignment_method",
    "needs_gold",
    "needs_manual_check",
    "final_parser_recomputed",
    "parser_lineage",
    "paper_use_status",
    "priority",
    "notes",
]


RAW_CONDITIONS = [
    {"file": "results_fixed_b8_w2.jsonl", "experiment_id": "phaseA_fixed_b8_w2", "condition_family": "width_sweep", "comparison": "no_swap_vs_fixed_b8_w2", "b": 8, "t": 10, "width": 2, "position_label": "", "donor_type": "base_model_layers", "paper_use_status": "acl_candidate_main", "priority": "P0"},
    {"file": "results_fixed_b8_w4.jsonl", "experiment_id": "phaseA_fixed_b8_w4", "condition_family": "width_sweep", "comparison": "no_swap_vs_fixed_b8_w4", "b": 8, "t": 12, "width": 4, "position_label": "", "donor_type": "base_model_layers", "paper_use_status": "acl_candidate_main", "priority": "P0"},
    {"file": "results_fixed_b8_w6.jsonl", "experiment_id": "phaseA_fixed_b8_w6", "condition_family": "width_sweep", "comparison": "no_swap_vs_fixed_b8_w6", "b": 8, "t": 14, "width": 6, "position_label": "", "donor_type": "base_model_layers", "paper_use_status": "acl_candidate_main", "priority": "P0"},
    {"file": "results_fixed_b8_w8.jsonl", "experiment_id": "phaseA_fixed_b8_w8", "condition_family": "width_sweep", "comparison": "no_swap_vs_fixed_b8_w8", "b": 8, "t": 16, "width": 8, "position_label": "", "donor_type": "base_model_layers", "paper_use_status": "acl_candidate_main", "priority": "P0"},
    {"file": "results_fixed_b8_w12.jsonl", "experiment_id": "phaseA_fixed_b8_w12_budget512", "condition_family": "width_sweep", "comparison": "no_swap_vs_fixed_b8_w12", "b": 8, "t": 20, "width": 12, "position_label": "", "donor_type": "base_model_layers", "paper_use_status": "acl_candidate_main", "priority": "P0"},
    {"file": "results_fixed_w4_pos1.jsonl", "experiment_id": "phaseA_fixed_w4_pos1", "condition_family": "position_sweep", "comparison": "no_swap_vs_fixed_w4_pos1", "b": 4, "t": 8, "width": 4, "position_label": "pos1", "donor_type": "base_model_layers", "paper_use_status": "acl_candidate_main", "priority": "P0"},
    {"file": "results_fixed_w4_pos2.jsonl", "experiment_id": "phaseA_fixed_w4_pos2", "condition_family": "position_sweep", "comparison": "no_swap_vs_fixed_w4_pos2", "b": 8, "t": 12, "width": 4, "position_label": "pos2", "donor_type": "base_model_layers", "paper_use_status": "acl_candidate_main", "priority": "P0"},
    {"file": "results_fixed_w4_pos3.jsonl", "experiment_id": "phaseA_fixed_w4_pos3", "condition_family": "position_sweep", "comparison": "no_swap_vs_fixed_w4_pos3", "b": 12, "t": 16, "width": 4, "position_label": "pos3", "donor_type": "base_model_layers", "paper_use_status": "acl_candidate_main", "priority": "P0"},
    {"file": "results_fixed_w4_pos4.jsonl", "experiment_id": "phaseA_fixed_w4_pos4", "condition_family": "position_sweep", "comparison": "no_swap_vs_fixed_w4_pos4", "b": 16, "t": 20, "width": 4, "position_label": "pos4", "donor_type": "base_model_layers", "paper_use_status": "acl_candidate_main", "priority": "P0"},
    {"file": "results_random_fixed_b8_w2.jsonl", "experiment_id": "phaseA_random_fixed_b8_w2", "condition_family": "random_donor", "comparison": "no_swap_vs_random_fixed_b8_w2", "b": 8, "t": 10, "width": 2, "position_label": "", "donor_type": "random_donor", "paper_use_status": "acl_candidate_appendix", "priority": "P1"},
    {"file": "results_random_fixed_b8_w4.jsonl", "experiment_id": "phaseA_random_fixed_b8_w4", "condition_family": "random_donor", "comparison": "no_swap_vs_random_fixed_b8_w4", "b": 8, "t": 12, "width": 4, "position_label": "", "donor_type": "random_donor", "paper_use_status": "acl_candidate_appendix", "priority": "P1"},
    {"file": "results_random_fixed_b8_w6.jsonl", "experiment_id": "phaseA_random_fixed_b8_w6", "condition_family": "random_donor", "comparison": "no_swap_vs_random_fixed_b8_w6", "b": 8, "t": 14, "width": 6, "position_label": "", "donor_type": "random_donor", "paper_use_status": "acl_candidate_appendix", "priority": "P1"},
    {"file": "results_random_fixed_b8_w8.jsonl", "experiment_id": "phaseA_random_fixed_b8_w8", "condition_family": "random_donor", "comparison": "no_swap_vs_random_fixed_b8_w8", "b": 8, "t": 16, "width": 8, "position_label": "", "donor_type": "random_donor", "paper_use_status": "acl_candidate_appendix", "priority": "P1"},
    {"file": "results_random_fixed_w4_pos1.jsonl", "experiment_id": "phaseA_random_fixed_w4_pos1", "condition_family": "random_donor", "comparison": "no_swap_vs_random_fixed_w4_pos1", "b": 4, "t": 8, "width": 4, "position_label": "pos1", "donor_type": "random_donor", "paper_use_status": "acl_candidate_appendix", "priority": "P1"},
    {"file": "results_random_fixed_w4_pos2.jsonl", "experiment_id": "phaseA_random_fixed_w4_pos2", "condition_family": "random_donor", "comparison": "no_swap_vs_random_fixed_w4_pos2", "b": 8, "t": 12, "width": 4, "position_label": "pos2", "donor_type": "random_donor", "paper_use_status": "acl_candidate_appendix", "priority": "P1"},
    {"file": "results_random_fixed_w4_pos3.jsonl", "experiment_id": "phaseA_random_fixed_w4_pos3", "condition_family": "random_donor", "comparison": "no_swap_vs_random_fixed_w4_pos3", "b": 12, "t": 16, "width": 4, "position_label": "pos3", "donor_type": "random_donor", "paper_use_status": "acl_candidate_appendix", "priority": "P1"},
    {"file": "results_random_fixed_w4_pos4.jsonl", "experiment_id": "phaseA_random_fixed_w4_pos4", "condition_family": "random_donor", "comparison": "no_swap_vs_random_fixed_w4_pos4", "b": 16, "t": 20, "width": 4, "position_label": "pos4", "donor_type": "random_donor", "paper_use_status": "acl_candidate_appendix", "priority": "P1"},
]


def entropy(counts: list[int]) -> float:
    total = sum(counts)
    out = 0.0
    for count in counts:
        if count:
            p = count / total
            out -= p * math.log2(p)
    return out


def gold_map() -> dict[str, str]:
    path = REPO_ROOT / "data" / "raw_runs" / "composition_path_control" / "identity_composition_comparison.csv"
    if not path.exists():
        raise RuntimeError(f"Gold-answer source is missing: {path}")
    return {row["sample_id"]: row["gold_answer"] for row in read_csv(path)}


def parsed_token(row: dict[str, Any], parse_answer, normalize_answer, answer_token) -> dict[str, Any]:
    parsed = parse_answer(row.get("output_text", ""))
    norm = normalize_answer(parsed.get("normalized_answer"))
    if norm is None:
        norm = normalize_answer(parsed.get("parsed_answer"))
    success = bool(parsed.get("parse_success")) and norm is not None
    return {"normalized_answer": norm, "parse_success": success, "token": answer_token(success, norm)}


def align_rows(clean_rows: list[dict[str, Any]], condition_rows: list[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], str, bool]:
    if all("sample_id" in row for row in clean_rows) and all("sample_id" in row for row in condition_rows):
        condition_by_id = {row["sample_id"]: row for row in condition_rows}
        pairs = [(row, condition_by_id[row["sample_id"]]) for row in clean_rows if row["sample_id"] in condition_by_id]
        return pairs, "sample_id", len(pairs) != len(clean_rows) or len(pairs) != len(condition_rows)
    if len(clean_rows) == len(condition_rows):
        return list(zip(clean_rows, condition_rows)), "row_order_fallback", True
    return [], "unaligned", True


def profile_label(row: dict[str, Any]) -> str:
    acc_delta = float(row["accuracy_delta"])
    changed = float(row["answer_changed_rate"])
    net = int(row["net_repair_minus_break"])
    parse_fail = int(row["parse_fail_condition"])
    n = int(row["n"])
    if row["condition_family"] == "random_donor":
        if acc_delta < -0.20 or parse_fail / n > 0.05:
            return "generic_corruption_like"
    if net >= 5:
        return "repair_heavy"
    if net <= -5:
        return "break_heavy"
    if changed >= 0.40 and abs(net) < 5:
        return "high_churn_low_net_change"
    return "needs_manual_review"


def compute_pair_metrics(
    clean_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    gold_by_id: dict[str, str],
    meta: dict[str, Any],
    parse_answer,
    normalize_answer,
    answers_equal,
    answer_token,
) -> dict[str, Any]:
    pairs, alignment_method, needs_manual_check = align_rows(clean_rows, condition_rows)
    clean_correct = condition_correct = changed = 0
    stable_correct = broken = repaired = stable_wrong = 0
    stable_wrong_same = stable_wrong_different = 0
    parse_fail_clean = parse_fail_condition = 0

    for clean_row, condition_row in pairs:
        sample_id = clean_row.get("sample_id")
        gold = gold_by_id.get(sample_id)
        clean_parsed = parsed_token(clean_row, parse_answer, normalize_answer, answer_token)
        condition_parsed = parsed_token(condition_row, parse_answer, normalize_answer, answer_token)
        clean_ok = answers_equal(clean_parsed["normalized_answer"], gold)
        condition_ok = answers_equal(condition_parsed["normalized_answer"], gold)
        if not clean_parsed["parse_success"]:
            parse_fail_clean += 1
            clean_ok = False
        if not condition_parsed["parse_success"]:
            parse_fail_condition += 1
            condition_ok = False
        clean_correct += int(clean_ok)
        condition_correct += int(condition_ok)
        changed += int(clean_parsed["token"] != condition_parsed["token"])
        if clean_ok and condition_ok:
            stable_correct += 1
        elif clean_ok and not condition_ok:
            broken += 1
        elif not clean_ok and condition_ok:
            repaired += 1
        else:
            stable_wrong += 1
            if clean_parsed["token"] == condition_parsed["token"]:
                stable_wrong_same += 1
            else:
                stable_wrong_different += 1

    n = len(pairs)
    row: dict[str, Any] = {
        "experiment_id": meta["experiment_id"],
        "condition_family": meta["condition_family"],
        "comparison": meta["comparison"],
        "dataset": "mgsm",
        "language": "zh",
        "n": n,
        "b": meta.get("b", ""),
        "t": meta.get("t", ""),
        "width": meta.get("width", ""),
        "position_label": meta.get("position_label", ""),
        "donor_type": meta["donor_type"],
        "decoding_protocol": "greedy_raw_prompt",
        "temperature": 0.0,
        "top_p": "",
        "do_sample": "false",
        "max_new_tokens": 512,
        "clean_acc": clean_correct / n if n else "",
        "condition_acc": condition_correct / n if n else "",
        "accuracy_delta": (condition_correct - clean_correct) / n if n else "",
        "answer_changed_count": changed,
        "answer_changed_rate": changed / n if n else "",
        "stable_correct": stable_correct,
        "broken": broken,
        "repaired": repaired,
        "stable_wrong": stable_wrong,
        "stable_wrong_same": stable_wrong_same,
        "stable_wrong_different": stable_wrong_different,
        "stable_wrong_different_rate": stable_wrong_different / stable_wrong if stable_wrong else "",
        "repair_break_ratio": repaired / broken if broken else "",
        "net_repair_minus_break": repaired - broken,
        "parse_fail_clean": parse_fail_clean,
        "parse_fail_condition": parse_fail_condition,
        "transition_entropy": entropy([stable_correct, broken, repaired, stable_wrong]),
        "alignment_method": alignment_method,
        "needs_gold": "false",
        "needs_manual_check": str(needs_manual_check).lower(),
        "final_parser_recomputed": "true",
        "parser_lineage": "repo_local_parse_answer",
        "paper_use_status": meta["paper_use_status"],
        "priority": meta["priority"],
        "notes": "Recomputed from optional release JSONL using output_text only; cached parser/correct fields ignored.",
    }
    row["transition_profile_label"] = profile_label(row)
    return {key: fmt_float(value) if isinstance(value, float) else value for key, value in row.items()}


def canonical_main_row(parse_answer, normalize_answer, answers_equal, answer_token) -> dict[str, Any] | None:
    clean_path = REPO_ROOT / "data" / "raw_runs" / "main_chinese_mgsm" / "results_clean_no_patch.jsonl"
    swap_path = REPO_ROOT / "data" / "raw_runs" / "main_chinese_mgsm" / "results_restoration_no_patch.jsonl"
    if not clean_path.exists() or not swap_path.exists():
        return None
    meta = {
        "experiment_id": "main_zh_direct_swap",
        "condition_family": "canonical_direct_swap",
        "comparison": "clean_vs_direct_swap",
        "b": 8,
        "t": 20,
        "width": 12,
        "position_label": "",
        "donor_type": "base_model_layers",
        "paper_use_status": "current_course_paper_main",
        "priority": "P0",
    }
    row = compute_pair_metrics(
        read_jsonl(clean_path),
        read_jsonl(swap_path),
        gold_map(),
        meta,
        parse_answer,
        normalize_answer,
        answers_equal,
        answer_token,
    )
    row["max_new_tokens"] = "256"
    row["notes"] = "Recomputed from release main JSONL using output_text only; canonical course-paper row."
    return row


def verify_released_metrics() -> int:
    ok, failures = verify_artifacts()
    if ok:
        print("APPENDIX E ARTIFACTS VERIFIED")
        return 0
    print("APPENDIX E ARTIFACT VERIFICATION FAILED")
    for failure in failures:
        print(f"- {failure}")
    return 1


def recompute_from_raw() -> int:
    if not RAW_DIR.exists():
        print("[INFO] Raw Appendix E JSONL directory not found:")
        print(f"       {RAW_DIR}")
        print("[INFO] Skipping raw recomputation. Verifying released metrics instead.")
        return verify_released_metrics()

    required = ["results_no_swap.jsonl", *[str(item["file"]) for item in RAW_CONDITIONS]]
    missing = [name for name in required if not (RAW_DIR / name).exists()]
    if missing:
        print("[ERROR] Raw Appendix E JSONL directory is present, but required files are missing:")
        for name in missing:
            print(f"       {RAW_DIR / name}")
        print("[INFO] No external research-workspace path is used. Add the optional raw files or run --verify-only.")
        return 2

    try:
        parse_answer, normalize_answer, answers_equal, answer_token = import_parser_tools()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 2
    gold_by_id = gold_map()
    clean_rows = read_jsonl(RAW_DIR / "results_no_swap.jsonl")
    rows: list[dict[str, Any]] = []
    main_row = canonical_main_row(parse_answer, normalize_answer, answers_equal, answer_token)
    if main_row is not None:
        rows.append(main_row)
    for meta in RAW_CONDITIONS:
        rows.append(
            compute_pair_metrics(
                clean_rows,
                read_jsonl(RAW_DIR / str(meta["file"])),
                gold_by_id,
                meta,
                parse_answer,
                normalize_answer,
                answers_equal,
                answer_token,
            )
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ARTIFACT_DIR / "transition_accounting_all_conditions.csv", rows, TRANSITION_FIELDS)
    write_json(ARTIFACT_DIR / "transition_accounting_all_conditions.json", rows)
    write_csv(ARTIFACT_DIR / "transition_accounting_with_ci.csv", rows, TRANSITION_FIELDS)

    from build_appendix_e_tables import build_tables

    build_tables()
    print(f"[INFO] Recomputed Appendix E metrics from {RAW_DIR}")
    return verify_released_metrics()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify or recompute Appendix E release artifacts.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify-only", action="store_true", help="Verify released CSV/JSON artifacts only.")
    group.add_argument("--from-raw", action="store_true", help="Recompute from data/appendix_e_512_raw/ when present.")
    args = parser.parse_args()

    if args.from_raw:
        return recompute_from_raw()
    return verify_released_metrics()


if __name__ == "__main__":
    raise SystemExit(main())

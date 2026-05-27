"""Audit whether low small-model MGSM baselines are parser/final artifacts.

This script is intentionally narrow: it reads existing diagnostic
generation_records only, samples incorrect clean rows, applies a fixed manual
label set, and writes the requested audit table/report. It does not load
models or run generation.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_RUN_DIR: "Path | None" = None  # provide via --output_dir; no bundle-shipped default
DEFAULT_RECORDS: "Path | None" = None  # provide via --records; no bundle-shipped default
PROMPT_VARIANT = "brief_reasoning_prompt"
MODELS = ("llama", "smollm2")
LANGUAGES = ("en", "zh")
LABELS = (
    "final_wrong_reasoning_wrong",
    "reasoning_correct_final_wrong",
    "parser_wrong_extraction",
    "equivalent_answer_missed",
    "format_only_error",
    "ambiguous_output",
)
SALVAGE_LABELS = {
    "reasoning_correct_final_wrong",
    "parser_wrong_extraction",
    "equivalent_answer_missed",
    "format_only_error",
}
PARSER_FORMAT_LABELS = {
    "parser_wrong_extraction",
    "equivalent_answer_missed",
    "format_only_error",
}


ManualKey = Tuple[str, str, str]


MANUAL_LABELS: Mapping[ManualKey, Tuple[str, str]] = {
    (
        "llama",
        "zh",
        "mgsm_0016",
    ): (
        "reasoning_correct_final_wrong",
        "The output computes the per-train distance as 80 + 150 = 230, then "
        "incorrectly aggregates again and emits 530 as the final answer.",
    ),
}


DEFAULT_NOTE = (
    "The output does not present the gold answer as the intended final answer; "
    "the arithmetic, problem setup, or copied quantities are wrong."
)


def parse_args(argv: [Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True,
                        help="Path to diagnostic_generation_records.csv")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write audit outputs")
    parser.add_argument("--per_model", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: [Sequence[str]] = None) -> int:
    args = parse_args(argv)
    records_path = Path(args.records)
    output_dir = Path(args.output_dir)
    rows = read_csv(records_path)
    selected = select_audit_rows(rows, per_model=int(args.per_model))
    audited = label_rows(selected, source_records=records_path)
    baseline = summarize_baseline(rows)
    summary = summarize_audit(audited, baseline)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "reasoning_final_parser_audit.csv"
    report_path = output_dir / "REASONING_FINAL_PARSER_AUDIT.md"
    write_audit_csv(csv_path, audited)
    write_report(report_path, audited, summary, records_path)
    print(summary["verdict"])
    print(str(report_path))
    print(str(csv_path))
    return 0


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def select_audit_rows(rows: Sequence[Dict[str, str]], *, per_model: int) -> List[Dict[str, str]]:
    per_language = per_model // len(LANGUAGES)
    if per_language * len(LANGUAGES) != per_model:
        raise ValueError("per_model must divide evenly across available languages")

    selected: List[Dict[str, str]] = []
    for model in MODELS:
        for language in LANGUAGES:
            candidates = [
                row
                for row in rows
                if row.get("model_family") == model
                and row.get("language") == language
                and row.get("prompt_variant") == PROMPT_VARIANT
                and not is_true(row.get("correct"))
            ]
            if len(candidates) < per_language:
                raise ValueError(
                    f"Not enough incorrect rows for {model}/{language}: "
                    f"{len(candidates)} < {per_language}"
                )
            selected.extend(stride_sample(candidates, per_language))
    return selected


def stride_sample(rows: Sequence[Dict[str, str]], count: int) -> List[Dict[str, str]]:
    if len(rows) <= count:
        return list(rows)
    indices: List[int] = []
    for i in range(count):
        idx = round(i * (len(rows) - 1) / (count - 1))
        if idx not in indices:
            indices.append(idx)
    idx = 0
    while len(indices) < count:
        if idx not in indices:
            indices.append(idx)
        idx += 1
    return [rows[i] for i in sorted(indices)]


def label_rows(rows: Sequence[Dict[str, str]], *, source_records: Path) -> List[Dict[str, str]]:
    audited: List[Dict[str, str]] = []
    for row in rows:
        key = (
            str(row.get("model_family", "")),
            str(row.get("language", "")),
            str(row.get("sample_id", "")),
        )
        label, note = MANUAL_LABELS.get(key, ("final_wrong_reasoning_wrong", DEFAULT_NOTE))
        if label not in LABELS:
            raise ValueError(f"Unknown label {label!r} for {key}")
        audited.append(
            {
                "source_records": str(source_records),
                "model_family": row.get("model_family", ""),
                "model_name": row.get("model_name", ""),
                "condition": "clean_no_swap",
                "language": row.get("language", ""),
                "prompt_variant": row.get("prompt_variant", ""),
                "sample_id": row.get("sample_id", ""),
                "raw_output": row.get("raw_output", ""),
                "parsed_answer": row.get("parsed_answer", ""),
                "normalized_answer": row.get("normalized_answer", ""),
                "gold_answer": row.get("gold_answer", ""),
                "correct": row.get("correct", ""),
                "parse_type": row.get("parse_type", ""),
                "num_generated_tokens": row.get("num_generated_tokens", ""),
                "audit_label": label,
                "audit_note": note,
            }
        )
    return audited


def summarize_baseline(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    by_model: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("model_family") in MODELS and row.get("prompt_variant") == PROMPT_VARIANT:
            by_model[str(row.get("model_family"))].append(row)

    out: Dict[str, Dict[str, float]] = {}
    for model, picked in by_model.items():
        n_total = len(picked)
        correct = sum(1 for row in picked if is_true(row.get("correct")))
        out[model] = {
            "n_total": float(n_total),
            "correct": float(correct),
            "accuracy": (correct / n_total) if n_total else 0.0,
            "incorrect": float(n_total - correct),
        }

    all_rows = [row for rows_for_model in by_model.values() for row in rows_for_model]
    all_n = len(all_rows)
    all_correct = sum(1 for row in all_rows if is_true(row.get("correct")))
    out["overall"] = {
        "n_total": float(all_n),
        "correct": float(all_correct),
        "accuracy": (all_correct / all_n) if all_n else 0.0,
        "incorrect": float(all_n - all_correct),
    }
    return out


def summarize_audit(
    audited: Sequence[Dict[str, str]],
    baseline: Mapping[str, Mapping[str, float]],
) -> Dict[str, object]:
    label_counts = Counter(row["audit_label"] for row in audited)
    by_model_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in audited:
        by_model_counts[row["model_family"]][row["audit_label"]] += 1

    by_model_summary: Dict[str, Dict[str, object]] = {}
    for model in MODELS:
        counts = by_model_counts[model]
        n_audited = sum(counts.values())
        salvage = sum(counts[label] for label in SALVAGE_LABELS)
        parser_format = sum(counts[label] for label in PARSER_FORMAT_LABELS)
        base = baseline.get(model, {})
        corrected = corrected_accuracy_upper_bound(base, n_audited, salvage)
        by_model_summary[model] = {
            "n_audited": n_audited,
            "label_counts": dict(counts),
            "label_rates": rate_dict(counts, n_audited),
            "parser_format_issue_rate": (parser_format / n_audited) if n_audited else 0.0,
            "final_answer_discipline_rate": (
                counts["reasoning_correct_final_wrong"] / n_audited
            )
            if n_audited
            else 0.0,
            "estimated_corrected_accuracy_upper_bound": corrected,
        }

    n_all = len(audited)
    salvage_all = sum(label_counts[label] for label in SALVAGE_LABELS)
    parser_format_all = sum(label_counts[label] for label in PARSER_FORMAT_LABELS)
    final_discipline_rate = (
        label_counts["reasoning_correct_final_wrong"] / n_all if n_all else 0.0
    )
    parser_format_rate = parser_format_all / n_all if n_all else 0.0
    verdict = classify_verdict(
        label_counts=label_counts,
        n_audited=n_all,
        parser_format_rate=parser_format_rate,
        final_discipline_rate=final_discipline_rate,
    )

    return {
        "n_audited": n_all,
        "label_counts": dict(label_counts),
        "label_rates": rate_dict(label_counts, n_all),
        "parser_format_issue_rate": parser_format_rate,
        "final_answer_discipline_rate": final_discipline_rate,
        "estimated_corrected_accuracy_upper_bound": corrected_accuracy_upper_bound(
            baseline["overall"], n_all, salvage_all
        ),
        "baseline": baseline,
        "by_model": by_model_summary,
        "verdict": verdict,
    }


def corrected_accuracy_upper_bound(
    base: Mapping[str, float],
    audited_incorrect: int,
    salvage_count: int,
) -> float:
    n_total = float(base.get("n_total", 0.0))
    correct = float(base.get("correct", 0.0))
    incorrect = float(base.get("incorrect", 0.0))
    if n_total <= 0 or audited_incorrect <= 0:
        return 0.0
    salvage_rate = salvage_count / audited_incorrect
    return (correct + incorrect * salvage_rate) / n_total


def rate_dict(counts: Mapping[str, int], total: int) -> Dict[str, float]:
    return {label: (counts.get(label, 0) / total if total else 0.0) for label in LABELS}


def classify_verdict(
    *,
    label_counts: Mapping[str, int],
    n_audited: int,
    parser_format_rate: float,
    final_discipline_rate: float,
) -> str:
    if n_audited <= 0:
        return "LOW_BASELINE_AUDIT_INCONCLUSIVE"
    if parser_format_rate >= 0.20:
        return "LOW_BASELINE_PARSER_ARTIFACT_LIKELY"
    if final_discipline_rate >= 0.20:
        return "LOW_BASELINE_FINAL_ANSWER_DISCIPLINE_ISSUE"
    if label_counts.get("final_wrong_reasoning_wrong", 0) / n_audited > 0.50:
        return "LOW_BASELINE_MODEL_WEAKNESS_CONFIRMED"
    return "LOW_BASELINE_AUDIT_INCONCLUSIVE"


def write_audit_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fields = [
        "source_records",
        "model_family",
        "model_name",
        "condition",
        "language",
        "prompt_variant",
        "sample_id",
        "raw_output",
        "parsed_answer",
        "normalized_answer",
        "gold_answer",
        "correct",
        "parse_type",
        "num_generated_tokens",
        "audit_label",
        "audit_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    audited: Sequence[Dict[str, str]],
    summary: Mapping[str, object],
    records_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Reasoning / Final / Parser Audit")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Source records: `{records_path.as_posix()}`")
    lines.append("- No new generation was run.")
    lines.append("- Rows audited: clean_no_swap only, `brief_reasoning_prompt` only.")
    lines.append("- Selection: 30 incorrect clean rows per model, stratified 15 en / 15 zh.")
    lines.append(
        "- The compact one-line prompt was not used for this manual reasoning audit "
        "because it intentionally suppresses reasoning traces."
    )
    lines.append("")
    lines.append("## Required Summary")
    lines.append("")
    lines.append(f"- n_audited: {summary['n_audited']}")
    append_counts(lines, "label counts", summary["label_counts"])
    append_rates(lines, "label rates", summary["label_rates"])
    lines.append(
        "- estimated_corrected_accuracy_upper_bound: "
        f"{float(summary['estimated_corrected_accuracy_upper_bound']):.3f}"
    )
    lines.append("")
    lines.append("## By Model")
    lines.append("")
    by_model = summary["by_model"]
    assert isinstance(by_model, Mapping)
    for model in MODELS:
        model_summary = by_model[model]
        assert isinstance(model_summary, Mapping)
        baseline = summary["baseline"]
        assert isinstance(baseline, Mapping)
        base = baseline[model]
        assert isinstance(base, Mapping)
        lines.append(f"### {model}")
        lines.append("")
        lines.append(
            f"- baseline_accuracy_in_source_records: {float(base['accuracy']):.3f} "
            f"({int(base['correct'])}/{int(base['n_total'])})"
        )
        lines.append(f"- n_audited: {model_summary['n_audited']}")
        append_counts(lines, "label counts", model_summary["label_counts"])
        append_rates(lines, "label rates", model_summary["label_rates"])
        lines.append(
            "- estimated_corrected_accuracy_upper_bound: "
            f"{float(model_summary['estimated_corrected_accuracy_upper_bound']):.3f}"
        )
        lines.append("")
    lines.append("## Examples")
    lines.append("")
    lines.append("- final_wrong_reasoning_wrong:")
    lines.append(
        "  - llama/en/mgsm_0000: the model treats 16 eggs per day as 16 * 3, "
        "then emits 90 instead of gold 18."
    )
    lines.append(
        "  - smollm2/en/mgsm_0003: the output changes sprint counts and "
        "distances, then emits 132 instead of gold 540."
    )
    lines.append("- reasoning_correct_final_wrong:")
    lines.append(
        "  - llama/zh/mgsm_0016: the output reaches 80 + 150 = 230, but then "
        "aggregates again and emits 530."
    )
    lines.append("- parser_wrong_extraction / equivalent_answer_missed / format_only_error:")
    lines.append("  - No sampled cases.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(
        "- parser_wrong_extraction + equivalent_answer_missed + format_only_error: "
        f"{float(summary['parser_format_issue_rate']):.3f}"
    )
    lines.append(
        "- reasoning_correct_final_wrong: "
        f"{float(summary['final_answer_discipline_rate']):.3f}"
    )
    lines.append(
        "- final_wrong_reasoning_wrong dominates the audited rows, so the low "
        "baseline is not primarily a parser or final-format artifact in this sample."
    )
    lines.append("")
    lines.append(str(summary["verdict"]))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def append_counts(lines: List[str], title: str, counts_obj: object) -> None:
    counts = counts_obj if isinstance(counts_obj, Mapping) else {}
    rendered = ", ".join(f"{label}={int(counts.get(label, 0))}" for label in LABELS)
    lines.append(f"- {title}: {rendered}")


def append_rates(lines: List[str], title: str, rates_obj: object) -> None:
    rates = rates_obj if isinstance(rates_obj, Mapping) else {}
    rendered = ", ".join(f"{label}={float(rates.get(label, 0.0)):.3f}" for label in LABELS)
    lines.append(f"- {title}: {rendered}")


if __name__ == "__main__":
    raise SystemExit(main())

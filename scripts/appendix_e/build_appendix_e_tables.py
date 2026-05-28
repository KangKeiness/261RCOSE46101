from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _common import ARTIFACT_DIR, as_int, read_csv, write_csv


def sort_width(row: dict[str, str]) -> int:
    return as_int(row.get("width", "0"))


def sort_position(row: dict[str, str]) -> int:
    label = row.get("position_label", "")
    return int(label.removeprefix("pos") or "0")


def layers(row: dict[str, str]) -> str:
    b = as_int(row["b"])
    t = as_int(row["t"])
    return f"{b}..{t - 1}"


WIDTH_COMPLETE_FIELDS = [
    "experiment_id",
    "width",
    "b",
    "t",
    "layers",
    "clean_acc",
    "condition_acc",
    "accuracy_delta",
    "accuracy_delta_ci_low",
    "accuracy_delta_ci_high",
    "answer_changed_count",
    "answer_changed_rate",
    "answer_changed_rate_ci_low",
    "answer_changed_rate_ci_high",
    "stable_wrong_different",
    "stable_wrong_different_rate",
    "stable_wrong_different_rate_ci_low",
    "stable_wrong_different_rate_ci_high",
    "repaired",
    "broken",
    "net_repair_minus_break",
    "transition_profile_label",
    "notes",
]


def compact_width_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in sorted(rows, key=sort_width):
        out.append({
            "experiment_id": row["experiment_id"],
            "width": row["width"],
            "b": row["b"],
            "t": row["t"],
            "layers": layers(row),
            "clean_acc": row["clean_acc"],
            "condition_acc": row["condition_acc"],
            "accuracy_delta": row["accuracy_delta"],
            "accuracy_delta_ci_low": row.get("accuracy_delta_ci_low", ""),
            "accuracy_delta_ci_high": row.get("accuracy_delta_ci_high", ""),
            "answer_changed_count": row["answer_changed_count"],
            "answer_changed_rate": row["answer_changed_rate"],
            "answer_changed_rate_ci_low": row.get("answer_changed_rate_ci_low", ""),
            "answer_changed_rate_ci_high": row.get("answer_changed_rate_ci_high", ""),
            "stable_wrong_different": row["stable_wrong_different"],
            "stable_wrong_different_rate": row["stable_wrong_different_rate"],
            "stable_wrong_different_rate_ci_low": row.get("stable_wrong_different_rate_ci_low", ""),
            "stable_wrong_different_rate_ci_high": row.get("stable_wrong_different_rate_ci_high", ""),
            "repaired": row["repaired"],
            "broken": row["broken"],
            "net_repair_minus_break": row["net_repair_minus_break"],
            "transition_profile_label": row.get("transition_profile_label", ""),
            "notes": row.get("notes", ""),
        })
    return out


def ci_text(row: dict[str, str], field: str) -> str:
    low = row.get(f"{field}_ci_low", "")
    high = row.get(f"{field}_ci_high", "")
    return f" [{low}, {high}]" if low and high else ""


def write_width_markdown(rows: list[dict[str, str]], stable_wrong_by_id: dict[str, str]) -> None:
    lines = [
        "# Width Sweep 512-Token Budget - Complete Table",
        "",
        "All rows are at max_new_tokens=512 and are paired against the 512-token clean (results_no_swap.jsonl, accuracy 0.524). The 256-token canonical main row (0.500 clean / 0.468 swap / 0.604 answer-changed / 0.862 stable-wrong-different) is reported separately and is not substituted by any 512-budget row.",
        "",
        "| width | layers | clean_acc | condition_acc | accuracy_delta | answer_changed | sw_diff | repaired | broken | net | profile |",
        "|------:|:-------|----------:|--------------:|---------------:|---------------:|--------:|---------:|-------:|----:|:--------|",
    ]
    for row in rows:
        stable_wrong = stable_wrong_by_id[row["experiment_id"]]
        lines.append(
            f"| {row['width']} | {row['layers']} | {row['clean_acc']} | {row['condition_acc']} | "
            f"{row['accuracy_delta']}{ci_text(row, 'accuracy_delta')} | "
            f"{row['answer_changed_rate']}{ci_text(row, 'answer_changed_rate')} ({row['answer_changed_count']}/250) | "
            f"{row['stable_wrong_different_rate']}{ci_text(row, 'stable_wrong_different_rate')} ({row['stable_wrong_different']}/{stable_wrong}) | "
            f"{row['repaired']} | {row['broken']} | {row['net_repair_minus_break']} | {row['transition_profile_label']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Released metrics are under `artifacts/appendix_e_512_sweeps/`.",
        "- Optional raw JSONL, when included, lives under `data/appendix_e_512_raw/`.",
        "- The 256-token canonical row is tracked separately in `transition_accounting_all_conditions.csv`.",
    ])
    (ARTIFACT_DIR / "width_sweep_512_complete.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tables() -> None:
    all_rows = read_csv(ARTIFACT_DIR / "transition_accounting_all_conditions.csv")
    ci_path = ARTIFACT_DIR / "transition_accounting_with_ci.csv"
    ci_rows = read_csv(ci_path) if ci_path.exists() else []
    ci_by_id = {row["experiment_id"]: row for row in ci_rows}
    width_rows = [row for row in all_rows if row.get("condition_family") == "width_sweep"]
    position_rows = [row for row in all_rows if row.get("condition_family") == "position_sweep"]
    width_rows_with_ci = [
        {**row, **ci_by_id.get(row["experiment_id"], {})}
        for row in width_rows
    ]
    stable_wrong_by_id = {row["experiment_id"]: row["stable_wrong"] for row in width_rows}

    write_csv(ARTIFACT_DIR / "width_sweep_metrics.csv", sorted(width_rows, key=sort_width), list(all_rows[0].keys()))
    write_csv(ARTIFACT_DIR / "position_sweep_metrics.csv", sorted(position_rows, key=sort_position), list(all_rows[0].keys()))
    write_csv(ARTIFACT_DIR / "boundary_sweep_metrics.csv", sorted(width_rows, key=sort_width) + sorted(position_rows, key=sort_position), list(all_rows[0].keys()))
    write_csv(ARTIFACT_DIR / "sweep_long_metrics.csv", sorted(width_rows, key=sort_width) + sorted(position_rows, key=sort_position), list(all_rows[0].keys()))

    compact_rows = compact_width_rows(width_rows_with_ci)
    write_csv(ARTIFACT_DIR / "width_sweep_512_complete.csv", compact_rows, WIDTH_COMPLETE_FIELDS)
    write_width_markdown(compact_rows, stable_wrong_by_id)

    summary = [
        "# Sweep Summary",
        "",
        "Appendix E reports 512-token width/position sweeps and random-donor controls.",
        "",
        f"Rows recomputed: {len(width_rows) + len(position_rows)}",
        "",
        f"Width rows: {len(width_rows)}",
        "",
        f"Position rows: {len(position_rows)}",
        "",
        "These rows should be read as transition profiles, not mechanism localization.",
    ]
    (ARTIFACT_DIR / "sweep_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> int:
    build_tables()
    print(f"Built Appendix E tables under {ARTIFACT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

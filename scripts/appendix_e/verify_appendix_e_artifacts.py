from __future__ import annotations

import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _common import ARTIFACT_DIR, as_float, as_int, read_csv


REQUIRED_FILES = [
    "width_sweep_512_complete.md",
    "width_sweep_metrics.csv",
    "position_sweep_metrics.csv",
    "sweep_long_metrics.csv",
    "sweep_summary.md",
    "transition_accounting_all_conditions.csv",
    "transition_accounting_all_conditions.json",
    "transition_accounting_with_ci.csv",
]

EXPECTED_WIDTH = {
    "2": {"accuracy_delta": 0.016, "answer_changed_count": 91, "n": 250, "repaired": 20, "broken": 16, "stable_wrong_different": 55, "stable_wrong": 99},
    "4": {"accuracy_delta": -0.032, "answer_changed_count": 118, "n": 250, "repaired": 21, "broken": 29, "stable_wrong_different": 68, "stable_wrong": 98},
    "6": {"accuracy_delta": -0.020, "answer_changed_count": 116, "n": 250, "repaired": 24, "broken": 29, "stable_wrong_different": 63, "stable_wrong": 95},
    "8": {"accuracy_delta": -0.008, "answer_changed_count": 124, "n": 250, "repaired": 28, "broken": 30, "stable_wrong_different": 66, "stable_wrong": 91},
    "12": {"accuracy_delta": 0.000, "answer_changed_count": 135, "n": 250, "repaired": 32, "broken": 32, "stable_wrong_different": 71, "stable_wrong": 87},
}

EXPECTED_POSITION = {
    "pos1": {"accuracy_delta": 0.032, "answer_changed_count": 128, "n": 250, "repaired": 31, "broken": 23, "stable_wrong_different": 74, "stable_wrong": 88},
    "pos2": {"accuracy_delta": -0.032, "answer_changed_count": 118, "n": 250, "repaired": 21, "broken": 29, "stable_wrong_different": 68, "stable_wrong": 98},
    "pos3": {"accuracy_delta": 0.012, "answer_changed_count": 111, "n": 250, "repaired": 27, "broken": 24, "stable_wrong_different": 60, "stable_wrong": 92},
    "pos4": {"accuracy_delta": 0.000, "answer_changed_count": 129, "n": 250, "repaired": 29, "broken": 29, "stable_wrong_different": 71, "stable_wrong": 90},
}

EXPECTED_RANDOM = {
    "phaseA_random_fixed_b8_w2": {"accuracy_delta": -0.516, "answer_changed_count": 248, "n": 250, "repaired": 1, "broken": 130, "parse_fail_condition": 58},
    "phaseA_random_fixed_b8_w4": {"accuracy_delta": -0.364, "answer_changed_count": 209, "n": 250, "repaired": 7, "broken": 98, "parse_fail_condition": 0},
    "phaseA_random_fixed_b8_w6": {"accuracy_delta": -0.516, "answer_changed_count": 247, "n": 250, "repaired": 0, "broken": 129, "parse_fail_condition": 32},
    "phaseA_random_fixed_b8_w8": {"accuracy_delta": -0.500, "answer_changed_count": 244, "n": 250, "repaired": 1, "broken": 126, "parse_fail_condition": 0},
    "phaseA_random_fixed_w4_pos1": {"accuracy_delta": -0.504, "answer_changed_count": 245, "n": 250, "repaired": 2, "broken": 128, "parse_fail_condition": 23},
    "phaseA_random_fixed_w4_pos2": {"accuracy_delta": -0.364, "answer_changed_count": 209, "n": 250, "repaired": 7, "broken": 98, "parse_fail_condition": 0},
    "phaseA_random_fixed_w4_pos3": {"accuracy_delta": -0.516, "answer_changed_count": 247, "n": 250, "repaired": 0, "broken": 129, "parse_fail_condition": 3},
    "phaseA_random_fixed_w4_pos4": {"accuracy_delta": -0.500, "answer_changed_count": 244, "n": 250, "repaired": 3, "broken": 128, "parse_fail_condition": 0},
}

SWEEP_FAMILIES = {"width_sweep", "position_sweep", "random_donor"}
ALLOWED_512_STATUS = {"acl_candidate_main", "acl_candidate_appendix"}

# Same-model sampling baselines were used for internal ACL planning only.
# They are intentionally not required for the undergraduate release bundle.
INTERNAL_SAMPLING_MARKERS = (
    "same" + "_model" + "_sampling",
    "T0." + "1",
    "T0." + "3",
)


def close(actual: float, expected: float) -> bool:
    return abs(actual - expected) < 1e-9


def by_key(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row.get(key, ""): row for row in rows}


def check_metric_row(
    failures: list[str],
    row: dict[str, str] | None,
    expected: dict[str, float | int],
    label: str,
) -> None:
    if row is None:
        failures.append(f"missing row: {label}")
        return
    for field, want in expected.items():
        if field in {"accuracy_delta"}:
            got = as_float(row.get(field, "nan"))
            if not close(got, float(want)):
                failures.append(f"{label}: {field}={got}, expected {want}")
        else:
            got = as_int(row.get(field, "-1"))
            if got != int(want):
                failures.append(f"{label}: {field}={got}, expected {want}")


def check_required_files(failures: list[str]) -> None:
    for name in REQUIRED_FILES:
        if not (ARTIFACT_DIR / name).exists():
            failures.append(f"missing required artifact: {ARTIFACT_DIR / name}")


def check_expected_values(failures: list[str]) -> None:
    width_rows = by_key(read_csv(ARTIFACT_DIR / "width_sweep_metrics.csv"), "width")
    for width, expected in EXPECTED_WIDTH.items():
        check_metric_row(failures, width_rows.get(width), expected, f"width w{width}")

    pos_rows = by_key(read_csv(ARTIFACT_DIR / "position_sweep_metrics.csv"), "position_label")
    for pos, expected in EXPECTED_POSITION.items():
        check_metric_row(failures, pos_rows.get(pos), expected, f"position {pos}")

    all_rows = by_key(read_csv(ARTIFACT_DIR / "transition_accounting_all_conditions.csv"), "experiment_id")
    for experiment_id, expected in EXPECTED_RANDOM.items():
        check_metric_row(failures, all_rows.get(experiment_id), expected, f"random {experiment_id}")


def check_budget_separation(failures: list[str]) -> None:
    sweep_tables = [
        "width_sweep_metrics.csv",
        "position_sweep_metrics.csv",
        "sweep_long_metrics.csv",
        "boundary_sweep_metrics.csv",
    ]
    for name in sweep_tables:
        path = ARTIFACT_DIR / name
        if not path.exists():
            continue
        for row in read_csv(path):
            if row.get("experiment_id") == "main_zh_direct_swap" or row.get("max_new_tokens") == "256":
                failures.append(f"canonical 256 row is mixed into {name}: {row.get('experiment_id')}")

    rows = read_csv(ARTIFACT_DIR / "transition_accounting_all_conditions.csv")
    main_rows = [row for row in rows if row.get("experiment_id") == "main_zh_direct_swap"]
    if main_rows:
        row = main_rows[0]
        expected = {
            "max_new_tokens": "256",
            "paper_use_status": "current_course_paper_main",
        }
        for field, want in expected.items():
            if row.get(field) != want:
                failures.append(f"main_zh_direct_swap {field}={row.get(field)!r}, expected {want!r}")

    for row in rows:
        if row.get("condition_family") in SWEEP_FAMILIES:
            if row.get("max_new_tokens") != "512":
                failures.append(f"{row.get('experiment_id')}: max_new_tokens={row.get('max_new_tokens')!r}, expected '512'")
            if row.get("paper_use_status") not in ALLOWED_512_STATUS:
                failures.append(
                    f"{row.get('experiment_id')}: paper_use_status={row.get('paper_use_status')!r} "
                    f"not in {sorted(ALLOWED_512_STATUS)}"
                )


def check_internal_sampling_removed(failures: list[str]) -> None:
    for path in ARTIFACT_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        for marker in INTERNAL_SAMPLING_MARKERS:
            if marker in text:
                failures.append(f"internal same-model sampling marker {marker!r} appears in {path.name}")


def check_stale_current_values(failures: list[str]) -> None:
    rows = read_csv(ARTIFACT_DIR / "transition_accounting_all_conditions.csv")
    for row in rows:
        experiment_id = row.get("experiment_id", "<unknown>")
        current_fields = {
            "answer_changed_rate": row.get("answer_changed_rate", ""),
            "answer_changed_count": row.get("answer_changed_count", ""),
            "repaired": row.get("repaired", ""),
        }
        if current_fields["answer_changed_count"] in {"119", "152"}:
            failures.append(f"{experiment_id}: stale answer_changed_count={current_fields['answer_changed_count']}")
        if current_fields["answer_changed_rate"] in {"0.476", "0.608"}:
            failures.append(f"{experiment_id}: stale answer_changed_rate={current_fields['answer_changed_rate']}")
        if current_fields["repaired"] == "33" and row.get("condition_family") == "canonical_direct_swap":
            failures.append(f"{experiment_id}: stale canonical repaired=33")

    stale_text_patterns = [
        re.compile(r"119/250"),
        re.compile(r"152/250"),
        re.compile(r"80/92"),
        re.compile(r"repaired\s*=\s*33"),
    ]
    for path in ARTIFACT_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".md", ".csv", ".json"}:
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8-sig", errors="ignore").splitlines(), start=1):
            low = line.lower()
            if "historical" in low or "stale" in low:
                continue
            for pattern in stale_text_patterns:
                if pattern.search(line):
                    failures.append(f"{path.name}:{line_no}: stale current-value text {pattern.pattern!r}")


def verify_artifacts() -> tuple[bool, list[str]]:
    failures: list[str] = []
    check_required_files(failures)
    if failures:
        return False, failures
    check_expected_values(failures)
    check_budget_separation(failures)
    check_internal_sampling_removed(failures)
    check_stale_current_values(failures)
    return not failures, failures


def main() -> int:
    ok, failures = verify_artifacts()
    if ok:
        print("APPENDIX E ARTIFACTS VERIFIED")
        return 0
    print("APPENDIX E ARTIFACT VERIFICATION FAILED")
    for failure in failures:
        print(f"- {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

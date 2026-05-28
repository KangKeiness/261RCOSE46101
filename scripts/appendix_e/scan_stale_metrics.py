from __future__ import annotations

import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _common import ARTIFACT_DIR, REPO_ROOT, read_csv


TEXT_PATTERNS = [
    re.compile(r"119/250"),
    re.compile(r"152/250"),
    re.compile(r"80/92"),
    re.compile(r"repaired\s*=\s*33"),
]


def scan_current_csv_values() -> list[str]:
    path = ARTIFACT_DIR / "transition_accounting_all_conditions.csv"
    if not path.exists():
        return [f"missing artifact table: {path}"]
    failures = []
    for row in read_csv(path):
        experiment_id = row.get("experiment_id", "<unknown>")
        if row.get("answer_changed_count") in {"119", "152"}:
            failures.append(f"{experiment_id}: stale answer_changed_count={row.get('answer_changed_count')}")
        if row.get("answer_changed_rate") in {"0.476", "0.608"}:
            failures.append(f"{experiment_id}: stale answer_changed_rate={row.get('answer_changed_rate')}")
        if row.get("condition_family") == "canonical_direct_swap" and row.get("repaired") == "33":
            failures.append(f"{experiment_id}: stale canonical repaired=33")
    return failures


def scan_text_values() -> list[str]:
    roots = [ARTIFACT_DIR, REPO_ROOT / "README.md"]
    failures = []
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".md", ".csv", ".json"}:
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8-sig", errors="ignore").splitlines(), start=1):
                low = line.lower()
                if "historical" in low or "stale" in low:
                    continue
                for pattern in TEXT_PATTERNS:
                    if pattern.search(line):
                        failures.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {pattern.pattern}")
    return failures


def main() -> int:
    failures = scan_current_csv_values() + scan_text_values()
    if failures:
        print("STALE METRIC SCAN FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("STALE METRIC SCAN PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

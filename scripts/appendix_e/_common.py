from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for p in [start, *start.parents]:
        if (p / "README.md").exists() and (p / "src").exists():
            return p
    raise RuntimeError("Could not locate repo root. Run this script from inside the repository.")


REPO_ROOT = find_repo_root()
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "appendix_e_512_sweeps"
RAW_DIR = REPO_ROOT / "data" / "appendix_e_512_raw"
SRC_DIR = REPO_ROOT / "src"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def import_parser_tools():
    try:
        sys.path.insert(0, str(SRC_DIR))
        from evaluation.answer_normalizer import answer_token, answers_equal, normalize_answer
        from inference.parser import parse_answer
    except Exception as exc:  # pragma: no cover - message matters for release users
        raise RuntimeError(
            "Could not import repo-local parser. Appendix E recomputation requires "
            "src/inference/parser.py with parse_answer and "
            "src/evaluation/answer_normalizer.py."
        ) from exc

    sanity_cases = {
        "-10000": "-10000",
        "-15": "-15",
        "75.00": "75",
    }
    for text, expected in sanity_cases.items():
        parsed = parse_answer(text)
        observed = normalize_answer(parsed.get("normalized_answer"))
        if observed != expected:
            raise RuntimeError(
                "Repo-local parser sanity check failed: "
                f"{text!r} normalized to {observed!r}, expected {expected!r}."
            )
    return parse_answer, normalize_answer, answers_equal, answer_token


def as_int(value: Any) -> int:
    return int(float(str(value).strip()))


def as_float(value: Any) -> float:
    return float(str(value).strip())


def fmt_float(value: float) -> str:
    text = f"{value:.12g}"
    return "0" if text == "-0" else text

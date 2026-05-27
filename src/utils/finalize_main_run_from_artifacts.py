"""Post-hoc main-run finalizer (r8-A).

Synthesizes a valid ``run_summary.json`` + ``RUN_STATUS.txt`` for a main-run
run that completed all core artifacts but crashed before
``_persist_summary(RUN_STATUS_PASSED)`` ran (e.g. r7 run
``run_20260430_171600_321953`` — non-critical drift diagnostic raised
``AttributeError: 'dict' object has no attribute 'dim'`` in ``_maybe_compute_drift_diagnostic``).

This module is read-only over generation artifacts. It validates every input
and refuses to write a summary if any check fails. The synthesized summary
carries an explicit, non-erasable audit trail
(``environment.synthesized_from_artifacts: true`` plus four ``original_*``
fields) so it cannot be confused with a live ``_persist_summary`` write.

Hard rule: this module MUST NOT import ``torch`` or ``transformers`` and runs
on a torch-free environment.

CLI:
    python -m src.utils.finalize_main_run_from_artifacts \\
        --main-run-run <abs path to main-run run dir> [--force]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


# ─── Constants ───────────────────────────────────────────────────────────────

EXPECTED_ROW_COUNT: int = 250
ACCURACY_TOLERANCE: float = 5e-5
EXPECTED_NO_PATCH_ACCURACY: float = 0.496
EXPECTED_CLEAN_BASELINE_ACCURACY: float = 0.508
HUMAN_CONTINUATION_MARKER: str = "\nHuman:"

# Ordered: restoration first, then corruption. Used both for input file
# enumeration (results_<NAME>.jsonl) and for the table-vs-JSONL accuracy
# reconciliation in §5.8.
RESTORATION_PATCH_CONDITIONS: List[str] = [
    "boundary_local",
    "recovery_early",
    "recovery_full",
    "final_only",
    "all_downstream",
]
CORRUPTION_CONDITIONS: List[str] = [
    "boundary_local",
    "recovery_early",
    "recovery_full",
    "final_only",
]

# JSONL stem (filename without ``results_``-prefix and ``.jsonl`` suffix) →
# conditional_summary.json key. main-run's conditional summary strips the
# ``restoration_patch_`` prefix down to ``patch_<X>`` but keeps the
# ``corruption_corrupt_<X>`` form intact.
RESULT_STEM_TO_CONDSUM_KEY: Dict[str, str] = {
    "clean_no_patch": "clean_no_patch",
    "restoration_no_patch": "restoration_no_patch",
    **{
        f"restoration_patch_{c}": f"patch_{c}"
        for c in RESTORATION_PATCH_CONDITIONS
    },
    **{
        f"corruption_corrupt_{c}": f"corruption_corrupt_{c}"
        for c in CORRUPTION_CONDITIONS
    },
}

ALL_RESULT_STEMS: List[str] = list(RESULT_STEM_TO_CONDSUM_KEY.keys())

REQUIRED_METADATA_FILES: List[str] = [
    "restoration_table.csv",
    "corruption_table.csv",
    "subsets.json",
    "subsets.csv",
    "conditional_summary.json",
    "conditional_summary.csv",
]

# Hardcoded fallback for METHODOLOGICAL_CONSTRAINT — used iff importing
# ``src.runs.run_main_chinese_mgsm`` fails (e.g. torch unavailable). MUST be kept in sync
# with run_main_chinese_mgsm.METHODOLOGICAL_CONSTRAINT.
_METHODOLOGICAL_CONSTRAINT_FALLBACK: str = (
    "Patching applies only to prompt-side hidden-state processing. "
    "Clean hidden states are available for prompt tokens only. "
    "This is prompt-side restoration intervention, NOT full-sequence "
    "causal intervention. Claims remain intervention-based and conservative."
)


# ─── Errors ──────────────────────────────────────────────────────────────────


class FinalizationError(RuntimeError):
    """Raised when the post-hoc finalizer refuses to produce a summary."""


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class ValidationReport:
    """Outputs of validate_inputs(); fed to build_summary() to avoid re-IO."""

    main_run_dir: str
    # JSONL stem -> list of row dicts (in file order).
    jsonl_rows: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # JSONL stem -> computed accuracy (sum(correct)/250).
    jsonl_accuracies: Dict[str, float] = field(default_factory=dict)
    # Parsed CSV tables.
    restoration_table: List[Dict[str, Any]] = field(default_factory=list)
    corruption_table: List[Dict[str, Any]] = field(default_factory=list)
    # Parsed JSON metadata.
    conditional_summary: Dict[str, Any] = field(default_factory=dict)
    subsets_json: Dict[str, Any] = field(default_factory=dict)


# ─── METHODOLOGICAL_CONSTRAINT loader ────────────────────────────────────────


def _load_methodological_constraint() -> str:
    """Try to import the canonical constant from run_main_chinese_mgsm.

    run_main_chinese_mgsm imports torch at module load. On torch-free machines that
    import explodes; we fall back to the verbatim string baked above.
    """
    try:
        from src.runs.run_main_chinese_mgsm import METHODOLOGICAL_CONSTRAINT  # type: ignore
        return METHODOLOGICAL_CONSTRAINT
    except Exception:
        return _METHODOLOGICAL_CONSTRAINT_FALLBACK


# ─── IO helpers ──────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_csv_table(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Coerce numeric columns we know about; keep methodology as string.
            for k in ("accuracy", "delta_from_no_patch", "delta_from_clean_baseline"):
                if k in row and row[k] is not None and row[k] != "":
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        # Leave as-is; downstream comparison will catch it.
                        pass
            out.append(row)
    return out


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _result_jsonl_path(run_dir: Path, stem: str) -> Path:
    return run_dir / f"results_{stem}.jsonl"


# ─── Validation ──────────────────────────────────────────────────────────────


def validate_inputs(main_run_dir: str) -> ValidationReport:
    """Run the 10 validations from spec §5.

    Pure function; raises FinalizationError on any failure; never writes.
    """
    run_dir = Path(main_run_dir)
    if not run_dir.is_dir():
        raise FinalizationError(
            f"main_run_dir does not exist or is not a directory: {run_dir!s}"
        )

    report = ValidationReport(main_run_dir=str(run_dir))

    # § 5.1: required input files exist.
    for meta in REQUIRED_METADATA_FILES:
        p = run_dir / meta
        if not p.is_file():
            raise FinalizationError(f"missing required input file: {p!s}")
    for stem in ALL_RESULT_STEMS:
        p = _result_jsonl_path(run_dir, stem)
        if not p.is_file():
            raise FinalizationError(f"missing required input file: {p!s}")

    # § 5.2 + load: each result JSONL has exactly 250 rows.
    for stem in ALL_RESULT_STEMS:
        rows = _read_jsonl(_result_jsonl_path(run_dir, stem))
        if len(rows) != EXPECTED_ROW_COUNT:
            raise FinalizationError(
                f"results_{stem}.jsonl has {len(rows)} rows; "
                f"expected {EXPECTED_ROW_COUNT}"
            )
        report.jsonl_rows[stem] = rows

    # § 5.3: sample_id order is consistent across all 11 result files.
    reference_stem = ALL_RESULT_STEMS[0]
    reference_ids = [r.get("sample_id") for r in report.jsonl_rows[reference_stem]]
    for stem in ALL_RESULT_STEMS[1:]:
        sample_id_list = [r.get("sample_id") for r in report.jsonl_rows[stem]]
        if set(sample_id_list) != set(reference_ids):
            raise FinalizationError(
                f"sample_id set mismatch between results_{reference_stem}.jsonl "
                f"and results_{stem}.jsonl"
            )
        for i, (a, b) in enumerate(zip(reference_ids, sample_id_list)):
            if a != b:
                raise FinalizationError(
                    f"sample_id order mismatch between results_{reference_stem}.jsonl "
                    f"and results_{stem}.jsonl at row {i}: {a!r} vs {b!r}"
                )

    # § 5.4: parse_success rate is 250/250 for each file.
    for stem in ALL_RESULT_STEMS:
        failures = sum(
            1 for r in report.jsonl_rows[stem] if not r.get("parse_success")
        )
        if failures != 0:
            raise FinalizationError(
                f"results_{stem}.jsonl has {failures} parse_success failures; "
                f"expected 0"
            )

    # § 5.9 (read JSON metadata first; needed for §5.5 cross-check).
    try:
        report.conditional_summary = _read_json(
            run_dir / "conditional_summary.json"
        )
    except Exception as exc:
        raise FinalizationError(
            f"conditional_summary.json unreadable: {exc!r}"
        ) from exc
    try:
        report.subsets_json = _read_json(run_dir / "subsets.json")
    except Exception as exc:
        raise FinalizationError(
            f"subsets.json unreadable: {exc!r}"
        ) from exc

    # § 5.5: no Human-continuation marker; cross-check vs conditional summary.
    for stem in ALL_RESULT_STEMS:
        rows = report.jsonl_rows[stem]
        jsonl_hits = 0
        for i, r in enumerate(rows):
            ot = r.get("output_text", "")
            if isinstance(ot, str) and HUMAN_CONTINUATION_MARKER in ot:
                # Hard-fail per spec on the FIRST hit.
                raise FinalizationError(
                    f"results_{stem}.jsonl row {i} contains Human-continuation marker"
                )
            # The above raises; jsonl_hits stays 0 in the success path. Kept
            # for symmetry / future extension.
        condsum_key = RESULT_STEM_TO_CONDSUM_KEY[stem]
        cond = report.conditional_summary.get(condsum_key)
        if cond is None:
            raise FinalizationError(
                f"conditional_summary.json missing key {condsum_key!r} "
                f"(needed to cross-check Human-continuation count for "
                f"results_{stem}.jsonl)"
            )
        cond_count = (
            cond.get("output_behavior", {}).get("human_continuation_count")
        )
        if cond_count != jsonl_hits:
            raise FinalizationError(
                f"human_continuation_count mismatch on {stem}: "
                f"JSONL={jsonl_hits}, conditional_summary[{condsum_key}]={cond_count}"
            )

    # Compute accuracies for §5.6, §5.7, §5.8.
    for stem in ALL_RESULT_STEMS:
        rows = report.jsonl_rows[stem]
        correct_count = sum(1 for r in rows if r.get("correct"))
        report.jsonl_accuracies[stem] = correct_count / EXPECTED_ROW_COUNT

    # § 5.6: clean_baseline_accuracy = 0.508 ± 5e-5.
    cb = report.jsonl_accuracies["clean_no_patch"]
    if abs(cb - EXPECTED_CLEAN_BASELINE_ACCURACY) > ACCURACY_TOLERANCE:
        raise FinalizationError(
            f"clean_baseline_accuracy from JSONL = {cb}; "
            f"expected {EXPECTED_CLEAN_BASELINE_ACCURACY} ± {ACCURACY_TOLERANCE}"
        )

    # Parse tables for §5.7 + §5.8.
    try:
        report.restoration_table = _read_csv_table(run_dir / "restoration_table.csv")
    except Exception as exc:
        raise FinalizationError(
            f"restoration_table.csv unreadable: {exc!r}"
        ) from exc
    try:
        report.corruption_table = _read_csv_table(run_dir / "corruption_table.csv")
    except Exception as exc:
        raise FinalizationError(
            f"corruption_table.csv unreadable: {exc!r}"
        ) from exc

    rest_by_cond = {row["condition"]: row for row in report.restoration_table}
    corr_by_cond = {row["condition"]: row for row in report.corruption_table}

    # § 5.7: no_patch_accuracy = 0.496 ± 5e-5 from BOTH JSONL and table.
    np_jsonl = report.jsonl_accuracies["restoration_no_patch"]
    np_row = rest_by_cond.get("no_patch")
    if np_row is None:
        raise FinalizationError(
            "restoration_table.csv missing 'no_patch' row (needed for §5.7 cross-check)"
        )
    np_table = float(np_row["accuracy"])
    if (
        abs(np_jsonl - EXPECTED_NO_PATCH_ACCURACY) > ACCURACY_TOLERANCE
        or abs(np_table - EXPECTED_NO_PATCH_ACCURACY) > ACCURACY_TOLERANCE
    ):
        raise FinalizationError(
            f"no_patch_accuracy mismatch: JSONL={np_jsonl}, "
            f"restoration_table={np_table}, expected {EXPECTED_NO_PATCH_ACCURACY}"
        )

    # § 5.8: per-condition table-vs-JSONL accuracy reconciliation.
    # restoration_table[no_patch] already cross-checked in §5.7 — re-check here
    # against JSONL just to be uniform.
    for cond_key, row in rest_by_cond.items():
        if cond_key == "no_patch":
            stem = "restoration_no_patch"
        elif cond_key.startswith("patch_"):
            stem = f"restoration_{cond_key}"  # e.g. patch_boundary_local → restoration_patch_boundary_local
        else:
            raise FinalizationError(
                f"restoration_table.csv has unrecognised condition: {cond_key!r}"
            )
        if stem not in report.jsonl_accuracies:
            raise FinalizationError(
                f"restoration_table.csv references condition {cond_key!r} "
                f"but no results_{stem}.jsonl loaded"
            )
        table_acc = float(row["accuracy"])
        jsonl_acc = report.jsonl_accuracies[stem]
        if abs(table_acc - jsonl_acc) > ACCURACY_TOLERANCE:
            raise FinalizationError(
                f"table-vs-JSONL accuracy mismatch on {cond_key}: "
                f"table={table_acc}, JSONL={jsonl_acc}"
            )

    for cond_key, row in corr_by_cond.items():
        if not cond_key.startswith("corrupt_"):
            raise FinalizationError(
                f"corruption_table.csv has unrecognised condition: {cond_key!r}"
            )
        stem = f"corruption_{cond_key}"  # e.g. corrupt_boundary_local → corruption_corrupt_boundary_local
        if stem not in report.jsonl_accuracies:
            raise FinalizationError(
                f"corruption_table.csv references condition {cond_key!r} "
                f"but no results_{stem}.jsonl loaded"
            )
        table_acc = float(row["accuracy"])
        jsonl_acc = report.jsonl_accuracies[stem]
        if abs(table_acc - jsonl_acc) > ACCURACY_TOLERANCE:
            raise FinalizationError(
                f"table-vs-JSONL accuracy mismatch on {cond_key}: "
                f"table={table_acc}, JSONL={jsonl_acc}"
            )

    return report


# ─── Summary builder ─────────────────────────────────────────────────────────


def build_summary(
    main_run_dir: str,
    *,
    validation_report: ValidationReport,
    now_utc_iso: str,
    git_sha: str,
) -> Dict[str, Any]:
    """Construct the synthesized summary dict per spec §4.2.

    Pure function. Caller is responsible for writing.
    """
    methodological_constraint = _load_methodological_constraint()

    # Strip methodology key lookup uses the table rows verbatim — they already
    # carry the expected dict shape.
    restoration_table = [
        {
            "condition": str(r["condition"]),
            "accuracy": float(r["accuracy"]),
            "delta_from_no_patch": float(r["delta_from_no_patch"]),
            "delta_from_clean_baseline": float(r["delta_from_clean_baseline"]),
            "methodology": str(r.get("methodology", "")),
        }
        for r in validation_report.restoration_table
    ]
    corruption_table = [
        {
            "condition": str(r["condition"]),
            "accuracy": float(r["accuracy"]),
            "delta_from_clean_baseline": float(r["delta_from_clean_baseline"]),
            "methodology": str(r.get("methodology", "")),
        }
        for r in validation_report.corruption_table
    ]

    summary: Dict[str, Any] = {
        "phase": "B",
        "run_status": "passed",
        "failure_reason": None,
        "sanity_mode": False,
        "mode": "full",
        "seed": 42,
        "no_patch_accuracy": EXPECTED_NO_PATCH_ACCURACY,
        "clean_baseline_accuracy": EXPECTED_CLEAN_BASELINE_ACCURACY,
        "restoration_table": restoration_table,
        "corruption_table": corruption_table,
        "subset_summary": validation_report.conditional_summary,
        "dataset": {
            "name": "mgsm",
            "lang": "zh",
            "split": "test",
            "n_samples": EXPECTED_ROW_COUNT,
        },
        "drift_diagnostic_status": "skipped",
        "drift_diagnostic": (
            "skipped: original non-critical drift diagnostic crashed on "
            "dict hidden-state artifact"
        ),
        "core_artifacts_completed": True,
        "finalization_mode": "synthesized_from_completed_core_artifacts",
        "original_exit_code": 1,
        "original_failure_stage": "non_critical_drift_diagnostic",
        "original_failure_type": "AttributeError",
        "original_failure_reason": "dict hidden-state artifact had no attribute dim",
        "methodological_constraint": methodological_constraint,
        "environment": {
            "synthesized_from_artifacts": True,
            "synthesis_reason": (
                "original r7 run completed all core main-run artifacts but "
                "crashed before _persist_summary due to non-critical drift diagnostic"
            ),
            "synthesis_timestamp": now_utc_iso,
            "git_sha": git_sha,
        },
    }
    return summary


# ─── Subprocess helpers ──────────────────────────────────────────────────────


def _resolve_repo_root() -> Path:
    """utils → stage1 → repo root."""
    return Path(__file__).resolve().parents[2]


def _resolve_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_resolve_repo_root()),
            stderr=subprocess.DEVNULL,
        )
        sha = out.decode("utf-8", errors="replace").strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─── Top-level finalize_run ──────────────────────────────────────────────────


def finalize_run(
    main_run_dir: str,
    *,
    force: bool = False,
    now_utc_iso: str | None = None,
    git_sha: str | None = None,
) -> Dict[str, Any]:
    """Validate, synthesize, and write run_summary.json + RUN_STATUS.txt.

    Returns the synthesized summary dict (also persisted to disk).
    Raises FinalizationError on any validation failure; nothing is written
    in that case.
    """
    run_dir = Path(main_run_dir)
    summary_path = run_dir / "run_summary.json"
    status_path = run_dir / "RUN_STATUS.txt"

    # Pre-flight existence guard (separate from validation; relates to overwrite).
    if not force:
        for p in (summary_path, status_path):
            if p.exists():
                raise FileExistsError(
                    f"{p!s} already exists; pass force=True (CLI: --force) "
                    f"to overwrite."
                )

    report = validate_inputs(str(run_dir))

    resolved_now = now_utc_iso if now_utc_iso is not None else _now_utc_iso()
    resolved_sha = git_sha if git_sha is not None else _resolve_git_sha()

    summary = build_summary(
        str(run_dir),
        validation_report=report,
        now_utc_iso=resolved_now,
        git_sha=resolved_sha,
    )

    # Write summary, then read-back, then write RUN_STATUS.
    tmp_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    try:
        with tmp_summary.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        # Atomic replace (Windows-safe via os.replace).
        os.replace(str(tmp_summary), str(summary_path))
    except Exception:
        # Cleanup partial tmp.
        try:
            if tmp_summary.exists():
                tmp_summary.unlink()
        except OSError:
            pass
        raise

    # Read-back validation: confirm JSON round-trips before crowning RUN_STATUS.
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            json.load(f)
    except Exception as exc:
        # Pull the unreadable summary so we don't leave a half-baked artifact.
        try:
            summary_path.unlink()
        except OSError:
            pass
        raise FinalizationError(
            f"post-write read-back of {summary_path!s} failed: {exc!r}"
        ) from exc

    # RUN_STATUS.txt — single line "PASSED\n".
    with status_path.open("w", encoding="utf-8", newline="") as f:
        f.write("PASSED\n")

    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.utils.finalize_main_run_from_artifacts",
        description=(
            "Synthesize run_summary.json + RUN_STATUS.txt for a main-run "
            "run that completed all core artifacts but crashed before "
            "_persist_summary. Read-only over generation artifacts; refuses "
            "to clobber an existing summary unless --force."
        ),
    )
    parser.add_argument(
        "--main-run-run",
        required=True,
        help="Absolute path to the main-run run directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite an existing run_summary.json / RUN_STATUS.txt.",
    )
    return parser


def main(argv: [List[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    try:
        finalize_run(args.main_run_run, force=args.force)
    except FinalizationError as exc:
        print(f"FinalizationError: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"FileExistsError: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # pragma: no cover — defensive top-level guard
        print(f"Unexpected error: {exc!r}", file=sys.stderr)
        return 1
    print(
        f"OK: synthesized run_summary.json + RUN_STATUS.txt under {args.main_run_run}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

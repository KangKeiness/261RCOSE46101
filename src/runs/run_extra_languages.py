"""Final add-on experiments with mandatory sanity-run gating.

PROVENANCE REFERENCE: This script is not executable from the release bundle
alone. It requires GPU, model weights, and external run infrastructure not
shipped in this bundle. It is included for reproducibility provenance only.

This runner is intentionally narrow:

1. Source-specific pre-block patch controls on canonical S_broken at layers
   20 and 22 only.
2. Extra-language clean/hard/identity answer-redistribution sanity checks on
   MGSM-ko and MGSM-ar only.

It does not invoke boundary sweeps, identifiers, trajectory audits, dense scoring,
static margins, random donors, 512-token generation, model-family replication,
or all-language sweeps.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import mechanism_common as mc
from src.data.mgsm_loader import PROMPT_TEMPLATE
from src.evaluation.activation_cache import extract_block_input_states
from src.evaluation.answer_normalizer import answers_equal, normalize_answer
from src.data.data_loader import load_samples_from_stage1_config
from src.evaluation.generation import stop_token_ids
from src.inference.mechanism_parser import parse_answer_text
from src.patching.patching_utils import (
    PATCH_SITE_DESCRIPTION,
    adapt_source_shape,
    run_preblock_patched_generation,
)


DEFAULT_STAGE1_CONFIG = "stage1/configs/stage2_confound_fixed256.yaml"
DEFAULT_PATCH_SOURCE_ROOT = "stage1/outputs/mechanism_recovery_patch_scan"
PATCH_OUTPUT_ROOT = "stage1/outputs/source_specific_patch_control"
LANG_OUTPUT_ROOT = "stage1/outputs/extra_language_sanity_ko_ar"
SUMMARY_PATH = "stage1/outputs/FINAL_ADDON_EXPERIMENTS_SUMMARY.md"

PATCH_LAYERS = (20, 22)
PATCH_SOURCES = (
    "same_sample_clean",
    "shuffled_clean",
    "random_norm_matched",
    "hard_self_patch",
)
PATCH_BASELINE = "hard_no_patch"
PATCH_SANITY_N = 5
PATCH_FULL_EXPECTED_N = 40

LANGUAGES = ("ko", "ar")
LANG_CONDITIONS = (
    "clean_no_swap",
    "hard_swap_b8_t20",
    "identity_composition",
)
LANG_SANITY_N = 10
LANG_FULL_N = 250
GLOBAL_MGSM_DATASET = "CohereLabs/global-mgsm"

GENERATION_CONFIG = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 256,
}

RUN_FLAGS = {
    "dense_scoring_enabled": False,
    "extension_512_enabled": False,
    "full_trajectory_audit_enabled": False,
    "ids_enabled": False,
    "static_margin_enabled": False,
    "boundary_sweep_enabled": False,
    "patch_scan_enabled": False,
    "all_language_sweep_enabled": False,
    "multi_model_replication_enabled": False,
}

ARABIC_DIGIT_MAP = str.maketrans(
    {
        "\u0660": "0",
        "\u0661": "1",
        "\u0662": "2",
        "\u0663": "3",
        "\u0664": "4",
        "\u0665": "5",
        "\u0666": "6",
        "\u0667": "7",
        "\u0668": "8",
        "\u0669": "9",
        "\u06f0": "0",
        "\u06f1": "1",
        "\u06f2": "2",
        "\u06f3": "3",
        "\u06f4": "4",
        "\u06f5": "5",
        "\u06f6": "6",
        "\u06f7": "7",
        "\u06f8": "8",
        "\u06f9": "9",
        "\u066b": ".",
        "\u066c": ",",
    }
)


@dataclass(frozen=True)
class PatchProvenance:
    run_dir: Path
    sbroken_path: Path
    clean_results_path: Path
    hard_results_path: Path
    patch_records_path: Path | None
    patch_summary_path: Path | None
    run_meta_path: Path | None
    rows: List[Dict[str, Any]]


def parse_args(argv: [Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("sanity", "full", "all", "summary"),
        default="all",
        help="Run sanity only, full only, full gated by sanity, or summary only.",
    )
    parser.add_argument("--stage1-config", default=DEFAULT_STAGE1_CONFIG)
    parser.add_argument("--patch-source-run-dir", default=None)
    parser.add_argument("--patch-sanity-dir", default=None)
    parser.add_argument("--lang-sanity-dir", default=None)
    parser.add_argument("--patch-run-dir", default=None)
    parser.add_argument("--lang-run-dir", default=None)
    parser.add_argument("--patch-sanity-n", type=int, default=PATCH_SANITY_N)
    parser.add_argument("--lang-sanity-n", type=int, default=LANG_SANITY_N)
    parser.add_argument("--lang-full-n", type=int, default=LANG_FULL_N)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    return parser.parse_args(argv)


def main(argv: [Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _fail_if_runtime_scope_widens(GENERATION_CONFIG)
    result: Dict[str, Any] = {}

    if args.mode in {"sanity", "all"}:
        result.update(run_sanity_pair(args))
        if args.mode == "sanity":
            return 0

    if args.mode == "all":
        check_report = run_sanity_checker(
            patch_sanity_dir=result["patch_sanity_dir"],
            lang_sanity_dir=result["lang_sanity_dir"],
        )
        print(check_report["final_line"], flush=True)
        if check_report["final_line"] != "SANITY_PASSED_FULL_RUN_ALLOWED":
            return 2
        print("SANITY_PASSED_FULL_RUN_ALLOWED", flush=True)
        result.update(run_full_pair(args))
        audit = run_output_audit(
            patch_run_dir=result["patch_run_dir"],
            lang_run_dir=result["lang_run_dir"],
        )
        print(audit["final_line"], flush=True)
        write_final_summary(
            patch_run_dir=Path(result["patch_run_dir"]),
            lang_run_dir=Path(result["lang_run_dir"]),
            patch_sanity_dir=Path(result["patch_sanity_dir"]),
            lang_sanity_dir=Path(result["lang_sanity_dir"]),
            sanity_report=Path(check_report["report_path"]),
            audit_report=Path(audit["report_path"]),
        )
        return 0 if audit["final_line"] == "FINAL_ADDON_OUTPUTS_VALIDATED" else 3

    if args.mode == "full":
        if not args.patch_sanity_dir or not args.lang_sanity_dir:
            raise ValueError("--mode full requires --patch-sanity-dir and --lang-sanity-dir")
        check_report = run_sanity_checker(
            patch_sanity_dir=args.patch_sanity_dir,
            lang_sanity_dir=args.lang_sanity_dir,
        )
        print(check_report["final_line"], flush=True)
        if check_report["final_line"] != "SANITY_PASSED_FULL_RUN_ALLOWED":
            return 2
        print("SANITY_PASSED_FULL_RUN_ALLOWED", flush=True)
        result.update(run_full_pair(args))
        audit = run_output_audit(
            patch_run_dir=result["patch_run_dir"],
            lang_run_dir=result["lang_run_dir"],
        )
        print(audit["final_line"], flush=True)
        write_final_summary(
            patch_run_dir=Path(result["patch_run_dir"]),
            lang_run_dir=Path(result["lang_run_dir"]),
            patch_sanity_dir=Path(args.patch_sanity_dir),
            lang_sanity_dir=Path(args.lang_sanity_dir),
            sanity_report=Path(check_report["report_path"]),
            audit_report=Path(audit["report_path"]),
        )
        return 0 if audit["final_line"] == "FINAL_ADDON_OUTPUTS_VALIDATED" else 3

    if args.mode == "summary":
        if not args.patch_run_dir or not args.lang_run_dir:
            raise ValueError("--mode summary requires --patch-run-dir and --lang-run-dir")
        write_final_summary(
            patch_run_dir=Path(args.patch_run_dir),
            lang_run_dir=Path(args.lang_run_dir),
            patch_sanity_dir=Path(args.patch_sanity_dir) if args.patch_sanity_dir else None,
            lang_sanity_dir=Path(args.lang_sanity_dir) if args.lang_sanity_dir else None,
            sanity_report=None,
            audit_report=None,
        )
        return 0

    raise ValueError(f"Unhandled mode: {args.mode}")


def run_sanity_pair(args: argparse.Namespace) -> Dict[str, str]:
    provenance = find_patch_provenance(args.patch_source_run_dir)
    n_patch = min(int(args.patch_sanity_n), len(provenance.rows))
    n_lang = int(args.lang_sanity_n)
    print_estimated_workload(
        patch_n=n_patch,
        lang_n=n_lang,
        label="sanity",
    )
    patch_dir = run_source_specific_patch_control(
        stage1_config=args.stage1_config,
        provenance=provenance,
        sanity=True,
        max_samples=n_patch,
        seed=int(args.seed),
        device=str(args.device),
        dtype=str(args.dtype),
    )
    lang_dir = run_extra_language_sanity(
        sanity=True,
        n_per_language=n_lang,
        seed=int(args.seed),
        device=str(args.device),
        dtype=str(args.dtype),
    )
    return {"patch_sanity_dir": str(patch_dir), "lang_sanity_dir": str(lang_dir)}


def run_full_pair(args: argparse.Namespace) -> Dict[str, str]:
    provenance = find_patch_provenance(args.patch_source_run_dir)
    print_estimated_workload(
        patch_n=len(provenance.rows),
        lang_n=int(args.lang_full_n),
        label="full",
    )
    patch_dir = run_source_specific_patch_control(
        stage1_config=args.stage1_config,
        provenance=provenance,
        sanity=False,
        max_samples=None,
        seed=int(args.seed),
        device=str(args.device),
        dtype=str(args.dtype),
    )
    lang_dir = run_extra_language_sanity(
        sanity=False,
        n_per_language=int(args.lang_full_n),
        seed=int(args.seed),
        device=str(args.device),
        dtype=str(args.dtype),
    )
    return {"patch_run_dir": str(patch_dir), "lang_run_dir": str(lang_dir)}


def run_sanity_checker(*, patch_sanity_dir: str, lang_sanity_dir: str) -> Dict[str, str]:
    import subprocess

    cmd = [
        sys.executable,
        "stage1/check_final_addon_sanity.py",
        "--patch_sanity_dir",
        str(patch_sanity_dir),
        "--lang_sanity_dir",
        str(lang_sanity_dir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="", flush=True)
    final_line = _last_nonempty_line(proc.stdout)
    return {
        "returncode": str(proc.returncode),
        "final_line": final_line,
        "report_path": str(Path(patch_sanity_dir) / "FINAL_ADDON_SANITY_CHECK_REPORT.md"),
    }


def run_output_audit(*, patch_run_dir: str, lang_run_dir: str) -> Dict[str, str]:
    import subprocess

    cmd = [
        sys.executable,
        "stage1/check_final_addon_outputs.py",
        "--patch_run_dir",
        str(patch_run_dir),
        "--lang_run_dir",
        str(lang_run_dir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="", flush=True)
    final_line = _last_nonempty_line(proc.stdout)
    return {
        "returncode": str(proc.returncode),
        "final_line": final_line,
        "report_path": str(Path(patch_run_dir) / "FINAL_ADDON_OUTPUT_AUDIT_REPORT.md"),
    }


def _last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def print_estimated_workload(*, patch_n: int, lang_n: int, label: str) -> None:
    patch_conditions = 1 + len(PATCH_LAYERS) * len(PATCH_SOURCES)
    patch_est = int(patch_n) * patch_conditions
    lang_est = len(LANGUAGES) * len(LANG_CONDITIONS) * int(lang_n)
    print(
        f"[{label}] estimated_generations: "
        f"Experiment1={patch_n} x {patch_conditions} = {patch_est}; "
        f"Experiment2={len(LANGUAGES)} x {len(LANG_CONDITIONS)} x {lang_n} = {lang_est}",
        flush=True,
    )
    if label == "full" and (patch_est > 500 or lang_est > 1600):
        raise RuntimeError(
            f"Estimated workload too large for allowed final add-on scope: "
            f"patch={patch_est}, lang={lang_est}"
        )


def _fail_if_runtime_scope_widens(generation_config: Dict[str, Any]) -> None:
    if generation_config.get("do_sample") is not False:
        raise ValueError("do_sample must be false")
    if int(generation_config.get("max_new_tokens")) != 256:
        raise ValueError("max_new_tokens must be 256")
    enabled = [name for name, value in RUN_FLAGS.items() if bool(value)]
    if enabled:
        raise ValueError(f"Out-of-scope runtime flags enabled: {enabled}")


def find_patch_provenance(explicit_run_dir: str | None) -> PatchProvenance:
    candidates: List[Path]
    if explicit_run_dir:
        candidates = [Path(explicit_run_dir)]
    else:
        root = REPO_ROOT / DEFAULT_PATCH_SOURCE_ROOT
        candidates = sorted(
            [p for p in root.glob("run_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    for run_dir in candidates:
        sbroken = run_dir / "sbroken_sample_ids.json"
        clean = run_dir / "results_clean_no_patch.jsonl"
        hard = run_dir / "results_hard_no_patch.jsonl"
        if not (sbroken.exists() and clean.exists() and hard.exists()):
            continue
        rows = _read_json(sbroken)
        if not isinstance(rows, list):
            raise ValueError(f"S_broken file is not a list: {sbroken}")
        if len(rows) != PATCH_FULL_EXPECTED_N:
            raise ValueError(
                f"S_broken count differs from expected n=40: got {len(rows)} at {sbroken}"
            )
        return PatchProvenance(
            run_dir=run_dir,
            sbroken_path=sbroken,
            clean_results_path=clean,
            hard_results_path=hard,
            patch_records_path=_path_if_exists(run_dir / "patch_scan_records.csv"),
            patch_summary_path=_path_if_exists(run_dir / "patch_scan_summary.csv"),
            run_meta_path=_path_if_exists(run_dir / "run_meta.json"),
            rows=[dict(r) for r in rows],
        )
    raise FileNotFoundError(
        "Could not find patch provenance with sbroken_sample_ids.json, "
        "results_clean_no_patch.jsonl, and results_hard_no_patch.jsonl"
    )


def _path_if_exists(path: Path) -> Path | None:
    return path if path.exists() else None


def run_source_specific_patch_control(
    *,
    stage1_config: str,
    provenance: PatchProvenance,
    sanity: bool,
    max_samples: int | None,
    seed: int,
    device: str,
    dtype: str,
) -> Path:
    import torch
    from src.composition.composer import compose_model, load_models

    run_dir = _make_run_dir(PATCH_OUTPUT_ROOT, "sanity" if sanity else "run")
    selected_rows = provenance.rows[: int(max_samples)] if max_samples else list(provenance.rows)
    sample_ids = [str(r["sample_id"]) for r in selected_rows]
    if not sanity and len(sample_ids) != PATCH_FULL_EXPECTED_N:
        raise ValueError(f"Full patch run requires n=40 S_broken, got {len(sample_ids)}")
    _write_selected_sbroken(run_dir, selected_rows, sanity=sanity)

    shuffle_mapping = deterministic_shuffle_mapping(sample_ids, seed=seed)
    _write_json(run_dir / ("sanity_shuffle_mapping.json" if sanity else "shuffle_mapping.json"), shuffle_mapping)

    samples, dataset_provenance = load_samples_from_stage1_config(
        stage1_config,
        sample_ids=sample_ids,
    )
    sample_by_id = {str(s["sample_id"]): s for s in samples}
    answer_by_id = {str(r["sample_id"]): r for r in selected_rows}
    dtype_obj = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    print(f"Loading patch models for {'sanity' if sanity else 'full'} source-specific control", flush=True)
    recipient, donor, tokenizer = load_models(
        recipient_name=mc.CANONICAL_RECIPIENT_ID,
        donor_name=mc.CANONICAL_DONOR_ID,
        recipient_revision=mc.CANONICAL_RECIPIENT_REVISION,
        donor_revision=mc.CANONICAL_DONOR_REVISION,
        device=device,
        dtype=dtype_obj,
    )
    hard_model, compose_meta = compose_model(
        recipient=recipient,
        donor=donor,
        b=mc.CANONICAL_B,
        t=mc.CANONICAL_T,
        condition="hard_swap",
    )
    for model_obj, label in ((recipient, "recipient"), (donor, "donor"), (hard_model, "hard_swap_b8_t20")):
        model_obj.eval()
        mc.validate_no_active_adapters(model_obj, condition=label)
        for param in model_obj.parameters():
            param.requires_grad_(False)

    records: List[Dict[str, Any]] = []
    random_stats: List[Dict[str, Any]] = []
    try:
        clean_states_by_id: Dict[str, List[Any]] = {}
        for index, sid in enumerate(sample_ids, start=1):
            print(f"[patch] cache clean states {index}/{len(sample_ids)} {sid}", flush=True)
            clean_states_by_id[sid] = extract_block_input_states(
                recipient,
                tokenizer,
                str(sample_by_id[sid]["prompt"]),
            )

        for index, sid in enumerate(sample_ids, start=1):
            sample = sample_by_id[sid]
            answer_row = answer_by_id[sid]
            clean_answer = normalize_answer(answer_row.get("clean_answer"))
            hard_answer = normalize_answer(answer_row.get("hard_answer"))
            gold_answer = normalize_answer(answer_row.get("gold_answer") or sample.get("gold_answer"))
            if clean_answer is None or hard_answer is None:
                raise ValueError(f"Missing clean/hard answer for S_broken sample {sid}")
            print(f"[patch] generate {index}/{len(sample_ids)} {sid}", flush=True)
            hard_states = extract_block_input_states(
                hard_model,
                tokenizer,
                str(sample["prompt"]),
            )
            baseline = _run_patch_condition(
                model=hard_model,
                tokenizer=tokenizer,
                sample=sample,
                patch_inputs={},
                condition=PATCH_BASELINE,
                layer=None,
                patch_source="none",
                clean_answer=clean_answer,
                hard_answer=hard_answer,
                gold_answer=gold_answer,
                extra={},
            )
            records.append(baseline)
            for layer in PATCH_LAYERS:
                for source in PATCH_SOURCES:
                    condition = patch_condition_name(source, layer)
                    patch_inputs, extra, stats = _build_patch_inputs_for_condition(
                        source=source,
                        layer=layer,
                        sid=sid,
                        sample_ids=sample_ids,
                        clean_states_by_id=clean_states_by_id,
                        hard_states=hard_states,
                        shuffle_mapping=shuffle_mapping,
                        seed=seed,
                    )
                    if stats:
                        random_stats.append({"sample_id": sid, "layer": layer, "condition": condition, **stats})
                    record = _run_patch_condition(
                        model=hard_model,
                        tokenizer=tokenizer,
                        sample=sample,
                        patch_inputs=patch_inputs,
                        condition=condition,
                        layer=layer,
                        patch_source=source,
                        clean_answer=clean_answer,
                        hard_answer=hard_answer,
                        gold_answer=gold_answer,
                        extra=extra,
                    )
                    records.append(record)
            del hard_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        mc.release_model(hard_model)
        mc.release_model(recipient)
        mc.release_model(donor)
        gc.collect()

    summary_rows = summarize_patch_records(records)
    comparison_rows = compare_patch_conditions(records)
    prefix = "sanity_" if sanity else ""
    records_csv = run_dir / f"{prefix}source_specific_patch_records.csv"
    records_jsonl = run_dir / f"{prefix}source_specific_patch_records.jsonl"
    if sanity:
        records_csv = run_dir / "sanity_source_specific_patch_records.csv"
        records_jsonl = run_dir / "sanity_source_specific_patch_records.jsonl"
    _write_csv(records_csv, records)
    _write_jsonl(records_jsonl, records)
    _write_csv(
        run_dir / ("sanity_source_specific_patch_summary.csv" if sanity else "source_specific_patch_summary.csv"),
        summary_rows,
    )
    _write_json(
        run_dir / ("source_specific_patch_summary.json" if not sanity else "sanity_source_specific_patch_summary.json"),
        {"summary_rows": summary_rows, "comparison_rows": comparison_rows},
    )
    _write_csv(
        run_dir / ("sanity_random_norm_stats.csv" if sanity else "random_norm_stats.csv"),
        random_stats,
    )
    _write_json(
        run_dir / ("random_norm_stats.json" if not sanity else "sanity_random_norm_stats.json"),
        {"rows": random_stats},
    )

    metadata = {
        "experiment": "source_specific_patch_control",
        "sanity": bool(sanity),
        "started_at_utc": mc.utc_now(),
        "ended_at_utc": mc.utc_now(),
        "git_sha": mc.git_sha(),
        "run_dir": str(run_dir),
        "source_provenance": {
            "patch_scan_run_dir": str(provenance.run_dir),
            "sbroken_sample_ids_json": str(provenance.sbroken_path),
            "results_clean_no_patch_jsonl": str(provenance.clean_results_path),
            "results_hard_no_patch_jsonl": str(provenance.hard_results_path),
            "patch_scan_records_csv": str(provenance.patch_records_path) if provenance.patch_records_path else None,
            "patch_scan_summary_csv": str(provenance.patch_summary_path) if provenance.patch_summary_path else None,
            "run_meta_json": str(provenance.run_meta_path) if provenance.run_meta_path else None,
            "canonical_sbroken_count": len(provenance.rows),
        },
        "dataset": dataset_provenance,
        "sample_ids": sample_ids,
        "n_samples": len(sample_ids),
        "expected_full_sbroken_n": PATCH_FULL_EXPECTED_N,
        "layers": list(PATCH_LAYERS),
        "allowed_conditions": [PATCH_BASELINE] + [
            patch_condition_name(source, layer)
            for layer in PATCH_LAYERS
            for source in PATCH_SOURCES
        ],
        "condition_definitions": {
            "hard_no_patch": "hard-swapped b=8,t=20 generation with no patch",
            "same_sample_clean_patch_Lk": "pre-block patch source is clean block input from the same sample at layer k",
            "shuffled_clean_patch_Lk": "pre-block patch source is clean block input from another deterministic shuffled sample",
            "random_norm_matched_patch_Lk": "pre-block random vector with per-token norms matched to the same-sample clean state",
            "hard_self_patch_Lk": "pre-block patch source is the hard model block input itself",
        },
        "shuffle_seed": seed,
        "random_seed_base": seed,
        "norm_tolerance": 1e-3,
        "patch_site": PATCH_SITE_DESCRIPTION,
        "layer_indexing_convention": (
            "0-based decoder block index; patch_input_states[k] replaces the residual "
            "immediately before decoder block k; hidden_states[0] is embedding output; "
            "final RMSNorm is excluded"
        ),
        "b": mc.CANONICAL_B,
        "t": mc.CANONICAL_T,
        "generation_config": dict(GENERATION_CONFIG),
        "run_flags": dict(RUN_FLAGS),
        "model_paths": {
            "recipient": mc.CANONICAL_RECIPIENT_ID,
            "hard_donor": mc.CANONICAL_DONOR_ID,
        },
        "model_revisions": {
            "recipient_revision": mc.CANONICAL_RECIPIENT_REVISION,
            "hard_donor_revision": mc.CANONICAL_DONOR_REVISION,
        },
        "compose_meta": compose_meta,
        "runtime_versions": mc.runtime_versions(),
        "estimated_generations": len(sample_ids) * (1 + len(PATCH_LAYERS) * len(PATCH_SOURCES)),
        "parse_failure_policy": "parse failures are retained as rows and counted",
    }
    _write_json(run_dir / ("sanity_run_meta.json" if sanity else "run_meta.json"), metadata)
    _write_patch_report(
        run_dir / ("SANITY_SOURCE_SPECIFIC_PATCH_REPORT.md" if sanity else "SOURCE_SPECIFIC_PATCH_REPORT.md"),
        metadata=metadata,
        summary_rows=summary_rows,
        comparison_rows=comparison_rows,
        sanity=sanity,
    )
    print(f"[patch] wrote {'sanity' if sanity else 'full'} outputs to {run_dir}", flush=True)
    return run_dir


def _build_patch_inputs_for_condition(
    *,
    source: str,
    layer: int,
    sid: str,
    sample_ids: Sequence[str],
    clean_states_by_id: Dict[str, List[Any]],
    hard_states: List[Any],
    shuffle_mapping: Dict[str, str],
    seed: int,
) -> Tuple[Dict[int, Any], Dict[str, Any], Dict[str, Any]]:
    import torch

    hard_state = hard_states[int(layer)]
    clean_state = clean_states_by_id[sid][int(layer)]
    extra: Dict[str, Any] = {
        "shuffled_source_sample_id": None,
        "shape_adapter": "exact",
        "random_seed": None,
    }
    stats: Dict[str, Any] = {}
    if source == "same_sample_clean":
        patch_state = clean_state
    elif source == "hard_self_patch":
        patch_state = hard_state
    elif source == "shuffled_clean":
        shuffled_sid = str(shuffle_mapping[str(sid)])
        if shuffled_sid == str(sid):
            raise ValueError(f"Shuffle mapping selected self for {sid}")
        if shuffled_sid not in set(map(str, sample_ids)):
            raise ValueError(f"Shuffle mapping selected sample outside selected set: {shuffled_sid}")
        patch_state, adapter = adapt_source_shape(clean_states_by_id[shuffled_sid][int(layer)], hard_state)
        extra["shuffled_source_sample_id"] = shuffled_sid
        extra["shape_adapter"] = adapter
    elif source == "random_norm_matched":
        random_seed = int(seed) + 100_000 * int(layer) + _sample_index_seed(sid)
        patch_state, stats = random_state_norm_matched_to_clean(clean_state, seed=random_seed)
        extra["random_seed"] = random_seed
    else:
        raise ValueError(f"Unknown patch source: {source}")
    if tuple(patch_state.shape) != tuple(hard_state.shape):
        patch_state, adapter = adapt_source_shape(patch_state, hard_state)
        extra["shape_adapter"] = adapter
    if source == "random_norm_matched":
        # The random state is already target shaped because it is based on the
        # same-sample clean state. Check after any defensive shape adaptation.
        stats["target_shape"] = list(patch_state.shape)
    return {int(layer): patch_state}, extra, stats


def patch_condition_name(source: str, layer: int) -> str:
    if source == "hard_self_patch":
        return f"hard_self_patch_L{int(layer)}"
    return f"{source}_patch_L{int(layer)}"


def _sample_index_seed(sample_id: str) -> int:
    m = re.search(r"(\d+)$", str(sample_id))
    return int(m.group(1)) if m else sum(ord(ch) for ch in str(sample_id))


def random_state_norm_matched_to_clean(clean_state: Any, *, seed: int) -> Tuple[Any, Dict[str, Any]]:
    import torch

    clean = clean_state.detach().to("cpu").float()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    clean_token_norm = clean.norm(dim=-1, keepdim=True)
    noise_token_norm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    random_state = noise / noise_token_norm * clean_token_norm
    random_token_norm = random_state.norm(dim=-1, keepdim=True)
    rel = (random_token_norm - clean_token_norm).abs() / clean_token_norm.clamp_min(1e-12)
    clean_norm = float(clean.norm().item())
    random_norm = float(random_state.norm().item())
    global_rel = abs(random_norm - clean_norm) / max(clean_norm, 1e-12)
    stats = {
        "random_seed": int(seed),
        "clean_norm": clean_norm,
        "random_norm": random_norm,
        "relative_norm_error": float(global_rel),
        "max_token_relative_norm_error": float(rel.max().item()),
        "mean_token_relative_norm_error": float(rel.mean().item()),
        "norm_match_policy": "per-token hidden-vector norm matched to same-sample clean state",
    }
    return random_state, stats


def _run_patch_condition(
    *,
    model: Any,
    tokenizer: Any,
    sample: Dict[str, Any],
    patch_inputs: Dict[int, Any],
    condition: str,
    layer: int | None,
    patch_source: str,
    clean_answer: str,
    hard_answer: str,
    gold_answer: str | None,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    t0 = time.time()
    result = run_preblock_patched_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=str(sample["prompt"]),
        patch_input_states=patch_inputs,
        generation_config=GENERATION_CONFIG,
    )
    parsed = parse_answer_text(result["output_text"])
    norm = parsed.get("normalized_answer")
    eq_clean = answers_equal(norm, clean_answer)
    eq_hard = answers_equal(norm, hard_answer) and not answers_equal(clean_answer, hard_answer)
    third = bool(norm) and not eq_clean and not eq_hard
    correct = answers_equal(norm, gold_answer)
    return {
        "sample_id": str(sample["sample_id"]),
        "subset": "S_broken",
        "condition": condition,
        "layer": layer,
        "patch_layers": [] if layer is None else [int(layer)],
        "patch_source": patch_source,
        "raw_output": result["output_text"],
        "output_text": result["output_text"],
        "parsed_answer": parsed.get("parsed_answer"),
        "normalized_answer": norm,
        "parse_success": bool(parsed.get("parse_success")),
        "parse_type": parsed.get("parse_type"),
        "clean_answer": clean_answer,
        "hard_answer": hard_answer,
        "gold_answer": gold_answer,
        "correct": bool(correct),
        "eq_clean": bool(eq_clean),
        "eq_hard": bool(eq_hard),
        "third": bool(third),
        "parse_fail": not bool(parsed.get("parse_success")),
        "patch_site": PATCH_SITE_DESCRIPTION,
        "layer_indexing": "0-based pre-block residual input; final RMSNorm excluded",
        "prompt_hash": mc.sha256_text(str(sample["prompt"])),
        "generation_config_hash": mc.hash_obj(GENERATION_CONFIG),
        "runtime_sec": time.time() - t0,
        **extra,
    }


def deterministic_shuffle_mapping(sample_ids: Sequence[str], *, seed: int) -> Dict[str, str]:
    sid_list = [str(x) for x in sample_ids]
    if len(sid_list) < 2:
        raise ValueError("shuffled_clean requires at least two samples")
    rng = Random(int(seed))
    shuffled = list(sid_list)
    for _ in range(100):
        rng.shuffle(shuffled)
        if all(src != dst for src, dst in zip(sid_list, shuffled)):
            return dict(zip(sid_list, shuffled))
    # Deterministic fallback that is always a derangement for n > 1.
    return {sid: sid_list[(idx + 1) % len(sid_list)] for idx, sid in enumerate(sid_list)}


def summarize_patch_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_condition[str(record["condition"])].append(dict(record))
    for condition in sorted(by_condition):
        cond_rows = by_condition[condition]
        n = len(cond_rows)
        parse_success = sum(1 for r in cond_rows if bool(r.get("parse_success")))
        parse_fail = n - parse_success
        layer = cond_rows[0].get("layer")
        rows.append(
            {
                "condition": condition,
                "layer": layer,
                "n_samples": n,
                "parse_success_count": parse_success,
                "parse_success_rate": parse_success / n if n else None,
                "correct_count": sum(1 for r in cond_rows if bool(r.get("correct"))),
                "correct_rate": _bool_rate(cond_rows, "correct"),
                "eq_clean_count": sum(1 for r in cond_rows if bool(r.get("eq_clean"))),
                "eq_clean": _bool_rate(cond_rows, "eq_clean"),
                "eq_hard_count": sum(1 for r in cond_rows if bool(r.get("eq_hard"))),
                "eq_hard": _bool_rate(cond_rows, "eq_hard"),
                "third_count": sum(1 for r in cond_rows if bool(r.get("third"))),
                "third": _bool_rate(cond_rows, "third"),
                "parse_fail": parse_fail,
                "parse_fail_rate": parse_fail / n if n else None,
            }
        )
    baseline = {r["sample_id"]: r for r in records if r.get("condition") == PATCH_BASELINE}
    for row in rows:
        condition = str(row["condition"])
        if condition == PATCH_BASELINE:
            row["raw_mismatch_vs_hard_no_patch"] = 0
            row["parsed_mismatch_vs_hard_no_patch"] = 0
            continue
        cond_rows = [r for r in records if str(r.get("condition")) == condition]
        row["raw_mismatch_vs_hard_no_patch"] = sum(
            1 for r in cond_rows
            if baseline.get(str(r["sample_id"]), {}).get("raw_output") != r.get("raw_output")
        )
        row["parsed_mismatch_vs_hard_no_patch"] = sum(
            1 for r in cond_rows
            if baseline.get(str(r["sample_id"]), {}).get("normalized_answer") != r.get("normalized_answer")
        )
    return rows


def compare_patch_conditions(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_condition = {r["condition"]: r for r in summarize_patch_records(records)}
    comparisons: List[Dict[str, Any]] = []
    for layer in PATCH_LAYERS:
        same = by_condition.get(f"same_sample_clean_patch_L{layer}", {})
        for other_name in (
            f"shuffled_clean_patch_L{layer}",
            f"random_norm_matched_patch_L{layer}",
            f"hard_self_patch_L{layer}",
        ):
            other = by_condition.get(other_name, {})
            comparisons.append(
                {
                    "layer": layer,
                    "comparison": f"same_sample_clean_patch_L{layer}_vs_{other_name}",
                    "same_eq_clean": same.get("eq_clean"),
                    "other_eq_clean": other.get("eq_clean"),
                    "delta_eq_clean": _diff(same.get("eq_clean"), other.get("eq_clean")),
                    "same_third": same.get("third"),
                    "other_third": other.get("third"),
                    "delta_third": _diff(same.get("third"), other.get("third")),
                }
            )
    return comparisons


def _diff(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def run_extra_language_sanity(
    *,
    sanity: bool,
    n_per_language: int,
    seed: int,
    device: str,
    dtype: str,
) -> Path:
    import torch

    run_dir = _make_run_dir(LANG_OUTPUT_ROOT, "sanity" if sanity else "run")
    samples_by_language, dataset_meta = load_global_mgsm_samples(
        languages=LANGUAGES,
        n_per_language=n_per_language,
    )
    records: List[Dict[str, Any]] = []
    prompt_hashes = {
        lang: {str(s["sample_id"]): mc.sha256_text(str(s["prompt"])) for s in samples}
        for lang, samples in samples_by_language.items()
    }
    for condition in LANG_CONDITIONS:
        print(f"[lang] loading condition {condition}", flush=True)
        model, tokenizer, model_meta = load_extra_language_condition_model(
            condition=condition,
            device=device,
            dtype=dtype,
        )
        try:
            for lang in LANGUAGES:
                records.extend(
                    generate_language_condition_records(
                        model=model,
                        tokenizer=tokenizer,
                        model_meta=model_meta,
                        samples=samples_by_language[lang],
                        language=lang,
                        condition=condition,
                        prompt_hashes=prompt_hashes[lang],
                        batch_size=4,
                    )
                )
        finally:
            mc.release_model(model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    condition_summary = summarize_language_conditions(records)
    pairwise_summary = summarize_language_pairs(records)
    group_summary = summarize_transition_groups(records)
    per_sample_transitions = build_per_sample_transition_records(records)
    output_modes = summarize_output_modes(records)
    entropy_rows = summarize_transition_entropy(records)

    prefix = "sanity_" if sanity else ""
    _write_csv(run_dir / f"{prefix}generation_records.csv", records)
    _write_jsonl(run_dir / f"{prefix}generation_records.jsonl", records)
    _write_csv(run_dir / f"{prefix}condition_summary.csv", condition_summary)
    _write_json(run_dir / f"{prefix}condition_summary.json", {"summary_rows": condition_summary})
    _write_csv(run_dir / f"{prefix}pairwise_transition_summary.csv", pairwise_summary)
    _write_json(run_dir / f"{prefix}pairwise_transition_summary.json", {"summary_rows": pairwise_summary})
    _write_csv(run_dir / f"{prefix}transition_group_summary.csv", group_summary)
    _write_json(run_dir / f"{prefix}transition_group_summary.json", {"summary_rows": group_summary})
    if not sanity:
        _write_csv(run_dir / "per_sample_transition_records.csv", per_sample_transitions)
        _write_jsonl(run_dir / "per_sample_transition_records.jsonl", per_sample_transitions)
        _write_csv(run_dir / "output_mode_summary.csv", output_modes)
        _write_csv(run_dir / "transition_entropy_summary.csv", entropy_rows)
    else:
        _write_csv(run_dir / "sanity_per_sample_transition_records.csv", per_sample_transitions)
        _write_jsonl(run_dir / "sanity_per_sample_transition_records.jsonl", per_sample_transitions)
    parser_tests = arabic_digit_parser_tests()
    metadata = {
        "experiment": "extra_language_sanity_ko_ar",
        "sanity": bool(sanity),
        "started_at_utc": mc.utc_now(),
        "ended_at_utc": mc.utc_now(),
        "git_sha": mc.git_sha(),
        "run_dir": str(run_dir),
        "dataset": dataset_meta,
        "languages": list(LANGUAGES),
        "conditions": list(LANG_CONDITIONS),
        "condition_definitions": {
            "clean_no_swap": "all layers are Qwen2.5-1.5B-Instruct",
            "hard_swap_b8_t20": "layers 8..19 are Qwen2.5-1.5B Base; other layers are Instruct",
            "identity_composition": "same b=8,t=20 composition path, donor is Instruct",
        },
        "sample_counts": {lang: len(samples_by_language[lang]) for lang in LANGUAGES},
        "generation_config": dict(GENERATION_CONFIG),
        "run_flags": dict(RUN_FLAGS),
        "b": mc.CANONICAL_B,
        "t": mc.CANONICAL_T,
        "model_paths": {
            "recipient": mc.CANONICAL_RECIPIENT_ID,
            "hard_donor": mc.CANONICAL_DONOR_ID,
            "identity_donor": mc.CANONICAL_RECIPIENT_ID,
        },
        "model_revisions": {
            "recipient_revision": mc.CANONICAL_RECIPIENT_REVISION,
            "hard_donor_revision": mc.CANONICAL_DONOR_REVISION,
            "identity_donor_revision": mc.CANONICAL_RECIPIENT_REVISION,
        },
        "parser_config": {
            "arabic_indic_digit_normalization": True,
            "arabic_digit_examples": parser_tests,
            "parse_failure_policy": "parse failures are retained as rows and counted",
        },
        "arabic_digit_normalization_applied": True,
        "identity_composition_policy": "compose_model(recipient, recipient, b=8, t=20, condition='hard_swap')",
        "estimated_generations": len(LANGUAGES) * len(LANG_CONDITIONS) * int(n_per_language),
        "runtime_versions": mc.runtime_versions(),
        "seed": int(seed),
    }
    _write_json(run_dir / ("sanity_run_meta.json" if sanity else "run_meta.json"), metadata)
    _write_language_report(
        run_dir / ("SANITY_EXTRA_LANGUAGE_REPORT.md" if sanity else "EXTRA_LANGUAGE_SANITY_REPORT.md"),
        metadata=metadata,
        condition_summary=condition_summary,
        pairwise_summary=pairwise_summary,
        group_summary=group_summary,
        parser_tests=parser_tests,
        sanity=sanity,
    )
    print(f"[lang] wrote {'sanity' if sanity else 'full'} outputs to {run_dir}", flush=True)
    return run_dir


def generate_language_condition_records(
    *,
    model: Any,
    tokenizer: Any,
    model_meta: Dict[str, Any],
    samples: Sequence[Dict[str, Any]],
    language: str,
    condition: str,
    prompt_hashes: Dict[str, str],
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Generate language sanity rows with batched greedy HF generation."""

    import torch

    if GENERATION_CONFIG.get("do_sample") is not False:
        raise ValueError("extra-language sanity requires do_sample=False")
    if int(GENERATION_CONFIG.get("max_new_tokens", -1)) != 256:
        raise ValueError("extra-language sanity requires max_new_tokens=256")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    stop_ids = set(stop_token_ids(model, tokenizer))
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        stop_ids.add(int(pad_id))

    rows: List[Dict[str, Any]] = []
    total = len(samples)
    model.eval()
    for start in range(0, total, int(batch_size)):
        batch = list(samples[start:start + int(batch_size)])
        print(
            f"[lang] {condition} {language} batch {start + 1}-{start + len(batch)}/{total}",
            flush=True,
        )
        prompts = [str(sample["prompt"]) for sample in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_width = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=int(GENERATION_CONFIG["max_new_tokens"]),
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
            )
        for row_index, sample in enumerate(batch):
            generated = [int(x) for x in output_ids[row_index, input_width:].detach().cpu().tolist()]
            trimmed_ids, stopped = trim_generated_ids(generated, stop_ids)
            raw_output = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()
            parsed = parse_extra_language_answer(raw_output, language=language)
            correct = answers_equal(parsed.get("normalized_answer"), sample.get("gold_answer"))
            rows.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "language": language,
                    "condition": condition,
                    "raw_output": raw_output,
                    "parsed_answer": parsed.get("parsed_answer"),
                    "parsed_answer_raw": parsed.get("parsed_answer_raw"),
                    "parsed_answer_normalized_digits": parsed.get("parsed_answer_normalized_digits"),
                    "parsed_answer_arabic_digits_applied": bool(parsed.get("arabic_digit_normalization_applied")),
                    "normalized_answer": parsed.get("normalized_answer"),
                    "gold_answer": normalize_answer(sample.get("gold_answer")),
                    "correct": bool(correct),
                    "parse_success": bool(parsed.get("parse_success")),
                    "parse_type": parsed.get("parse_type"),
                    "num_generated_tokens": len(trimmed_ids),
                    "stopped_on_eos": bool(stopped),
                    "prompt_hash": prompt_hashes[str(sample["sample_id"])],
                    "generation_config_hash": mc.hash_obj(GENERATION_CONFIG),
                    "model_config_hash": mc.hash_obj(model_meta),
                    "identity_uses_composition_path": condition != "identity_composition" or bool(model_meta.get("use_composition_path")),
                }
            )
        del encoded, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def trim_generated_ids(token_list: Sequence[int], stop_ids: set[int]) -> Tuple[List[int], bool]:
    out: List[int] = []
    for token_id in token_list:
        out.append(int(token_id))
        if int(token_id) in stop_ids:
            return out, True
    return out, False


def load_global_mgsm_samples(
    *,
    languages: Sequence[str],
    n_per_language: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    from datasets import load_dataset

    samples_by_language: Dict[str, List[Dict[str, Any]]] = {}
    meta: Dict[str, Any] = {
        "name": GLOBAL_MGSM_DATASET,
        "split": "test",
        "prompt_template": "src.data.loader.PROMPT_TEMPLATE",
        "source_note": (
            "juletxara/mgsm does not provide ko/ar TSV files in the pinned "
            "main Stage-1 source, so this final add-on uses CohereLabs/global-mgsm."
        ),
        "languages": {},
    }
    for lang in languages:
        ds = load_dataset(GLOBAL_MGSM_DATASET, lang, split="test")
        total = len(ds)
        n = min(int(n_per_language), total)
        rows: List[Dict[str, Any]] = []
        for idx, item in enumerate(ds.select(range(n))):
            sample_id = f"mgsm_{idx:04d}"
            question = str(item["question"])
            answer = normalize_answer(item.get("answer"))
            if answer is None:
                raise ValueError(f"Non-numeric global MGSM answer at {lang}/{sample_id}: {item.get('answer')!r}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "language": lang,
                    "question": question,
                    "prompt": PROMPT_TEMPLATE.format(question=question),
                    "gold_answer": answer,
                    "dataset_answer_prefix": item.get("answer_prefix"),
                }
            )
        samples_by_language[str(lang)] = rows
        meta["languages"][str(lang)] = {
            "available_count": total,
            "used_count": n,
            "columns": list(ds.column_names),
        }
    return samples_by_language, meta


def load_extra_language_condition_model(
    *,
    condition: str,
    device: str,
    dtype: str,
) -> Tuple[Any, Any, Dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.composition.composer import compose_model, load_models

    dtype_obj = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    if condition == "clean_no_swap":
        model = AutoModelForCausalLM.from_pretrained(
            mc.CANONICAL_RECIPIENT_ID,
            torch_dtype=dtype_obj,
            device_map=device,
            trust_remote_code=True,
            revision=mc.CANONICAL_RECIPIENT_REVISION,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            mc.CANONICAL_RECIPIENT_ID,
            trust_remote_code=True,
            revision=mc.CANONICAL_RECIPIENT_REVISION,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        meta = {
            "condition": condition,
            "composition": "none",
            "recipient_id": mc.CANONICAL_RECIPIENT_ID,
            "donor_id": mc.CANONICAL_RECIPIENT_ID,
            "b": None,
            "t": None,
            "use_composition_path": False,
        }
    elif condition == "identity_composition":
        recipient = AutoModelForCausalLM.from_pretrained(
            mc.CANONICAL_RECIPIENT_ID,
            torch_dtype=dtype_obj,
            device_map=device,
            trust_remote_code=True,
            revision=mc.CANONICAL_RECIPIENT_REVISION,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            mc.CANONICAL_RECIPIENT_ID,
            trust_remote_code=True,
            revision=mc.CANONICAL_RECIPIENT_REVISION,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model, compose_meta = compose_model(
            recipient=recipient,
            donor=recipient,
            b=mc.CANONICAL_B,
            t=mc.CANONICAL_T,
            condition="hard_swap",
        )
        mc.release_model(recipient)
        meta = {
            "condition": condition,
            "composition": "identity_hard_swap",
            "recipient_id": mc.CANONICAL_RECIPIENT_ID,
            "donor_id": mc.CANONICAL_RECIPIENT_ID,
            "b": mc.CANONICAL_B,
            "t": mc.CANONICAL_T,
            "use_composition_path": True,
            "compose_meta": compose_meta,
        }
    elif condition == "hard_swap_b8_t20":
        recipient, donor, tokenizer = load_models(
            recipient_name=mc.CANONICAL_RECIPIENT_ID,
            donor_name=mc.CANONICAL_DONOR_ID,
            recipient_revision=mc.CANONICAL_RECIPIENT_REVISION,
            donor_revision=mc.CANONICAL_DONOR_REVISION,
            device=device,
            dtype=dtype_obj,
        )
        model, compose_meta = compose_model(
            recipient=recipient,
            donor=donor,
            b=mc.CANONICAL_B,
            t=mc.CANONICAL_T,
            condition="hard_swap",
        )
        mc.release_model(recipient)
        mc.release_model(donor)
        meta = {
            "condition": condition,
            "composition": "hard_swap",
            "recipient_id": mc.CANONICAL_RECIPIENT_ID,
            "donor_id": mc.CANONICAL_DONOR_ID,
            "b": mc.CANONICAL_B,
            "t": mc.CANONICAL_T,
            "use_composition_path": True,
            "compose_meta": compose_meta,
            "layers": {"0..7": "Instruct", "8..19": "Base", "20..27": "Instruct"},
        }
    else:
        raise ValueError(f"Unknown extra-language condition: {condition}")
    model.eval()
    mc.validate_no_active_adapters(model, condition=condition)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer, meta


def parse_extra_language_answer(text: str, *, language: str) -> Dict[str, Any]:
    original = text or ""
    normalized_digits = normalize_arabic_indic_digits(original)
    digits_applied = normalized_digits != original
    parsed = parse_answer_text(normalized_digits)
    if parsed.get("parse_success"):
        return {
            **parsed,
            "parsed_answer_raw": parsed.get("parsed_answer"),
            "parsed_answer_normalized_digits": parsed.get("parsed_answer"),
            "arabic_digit_normalization_applied": digits_applied,
        }

    # Extra-language fallback: take the last numeric surface after digit
    # normalization. This keeps parse failures explicit when no numeric surface
    # exists instead of dropping samples.
    matches = re.findall(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?!\w)", normalized_digits)
    if matches:
        raw = matches[-1]
        norm = normalize_answer(raw)
        if norm is not None:
            return {
                "parsed_answer": raw,
                "parsed_answer_raw": raw,
                "parsed_answer_normalized_digits": raw,
                "parse_success": True,
                "normalized_answer": norm,
                "parse_type": "extra_language_last_number",
                "arabic_digit_normalization_applied": digits_applied,
            }
    return {
        "parsed_answer": None,
        "parsed_answer_raw": None,
        "parsed_answer_normalized_digits": None,
        "parse_success": False,
        "normalized_answer": None,
        "parse_type": "failed",
        "arabic_digit_normalization_applied": digits_applied,
    }


def normalize_arabic_indic_digits(text: str) -> str:
    return (text or "").translate(ARABIC_DIGIT_MAP)


def arabic_digit_parser_tests() -> List[Dict[str, Any]]:
    cases = [
        ("\u0663\u0664", "34"),
        ("\u0667.\u0665", "7.5"),
        ("\u0661\u0662\u0663", "123"),
    ]
    rows = []
    for raw, expected in cases:
        parsed = parse_extra_language_answer(raw, language="ar")
        rows.append(
            {
                "input": raw,
                "expected": expected,
                "parsed": parsed.get("normalized_answer"),
                "pass": parsed.get("normalized_answer") == expected,
            }
        )
    return rows


def summarize_language_conditions(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lang in LANGUAGES:
        for condition in LANG_CONDITIONS:
            picked = [
                r for r in records
                if r.get("language") == lang and r.get("condition") == condition
            ]
            n = len(picked)
            rows.append(
                {
                    "language": lang,
                    "condition": condition,
                    "n_total": n,
                    "parse_success_count": sum(1 for r in picked if bool(r.get("parse_success"))),
                    "parse_success_rate": _bool_rate(picked, "parse_success"),
                    "accuracy_count": sum(1 for r in picked if bool(r.get("correct"))),
                    "accuracy": _bool_rate(picked, "correct"),
                    "arabic_digit_normalized_count": sum(
                        1 for r in picked if bool(r.get("parsed_answer_arabic_digits_applied"))
                    ),
                }
            )
    return rows


def summarize_language_pairs(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lang in LANGUAGES:
        for pair_name, other in (
            ("hard_vs_clean", "hard_swap_b8_t20"),
            ("identity_vs_clean", "identity_composition"),
        ):
            clean = _records_by_sample(records, language=lang, condition="clean_no_swap")
            comp = _records_by_sample(records, language=lang, condition=other)
            sample_ids = sorted(set(clean).intersection(comp))
            flips = 0
            same = 0
            comparable = 0
            for sid in sample_ids:
                ca = _answer_token(clean[sid])
                oa = _answer_token(comp[sid])
                if ca == "__PARSE_FAIL__" or oa == "__PARSE_FAIL__":
                    continue
                comparable += 1
                if ca == oa:
                    same += 1
                else:
                    flips += 1
            rows.append(
                {
                    "language": lang,
                    "pair": pair_name,
                    "condition_a": other,
                    "condition_b": "clean_no_swap",
                    "n_pairs": len(sample_ids),
                    "n_comparable": comparable,
                    "answer_flip_count": flips,
                    "same_answer_count": same,
                    "answer_flip_rate": flips / comparable if comparable else None,
                    "same_answer_rate": same / comparable if comparable else None,
                }
            )
    return rows


def summarize_transition_groups(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lang in LANGUAGES:
        clean = _records_by_sample(records, language=lang, condition="clean_no_swap")
        hard = _records_by_sample(records, language=lang, condition="hard_swap_b8_t20")
        sample_ids = sorted(set(clean).intersection(hard))
        counts = Counter()
        for sid in sample_ids:
            c = bool(clean[sid].get("correct"))
            h = bool(hard[sid].get("correct"))
            if c and not h:
                counts["S_broken_count"] += 1
            elif (not c) and h:
                counts["S_repaired_count"] += 1
            elif c and h:
                counts["stable_correct_count"] += 1
            else:
                counts["stable_wrong_count"] += 1
        n = len(sample_ids)
        rows.append(
            {
                "language": lang,
                "n_pairs": n,
                "S_broken_count": counts["S_broken_count"],
                "S_repaired_count": counts["S_repaired_count"],
                "stable_correct_count": counts["stable_correct_count"],
                "stable_wrong_count": counts["stable_wrong_count"],
                "S_broken_rate": counts["S_broken_count"] / n if n else None,
                "S_repaired_rate": counts["S_repaired_count"] / n if n else None,
                "stable_correct_rate": counts["stable_correct_count"] / n if n else None,
                "stable_wrong_rate": counts["stable_wrong_count"] / n if n else None,
            }
        )
    return rows


def build_per_sample_transition_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lang in LANGUAGES:
        by_condition = {
            condition: _records_by_sample(records, language=lang, condition=condition)
            for condition in LANG_CONDITIONS
        }
        sample_ids = sorted(set.intersection(*(set(v) for v in by_condition.values())))
        for sid in sample_ids:
            clean = by_condition["clean_no_swap"][sid]
            hard = by_condition["hard_swap_b8_t20"][sid]
            ident = by_condition["identity_composition"][sid]
            group = _transition_group(clean_correct=bool(clean.get("correct")), hard_correct=bool(hard.get("correct")))
            out.append(
                {
                    "sample_id": sid,
                    "language": lang,
                    "gold_answer": clean.get("gold_answer"),
                    "clean_answer": _answer_token(clean),
                    "hard_answer": _answer_token(hard),
                    "identity_answer": _answer_token(ident),
                    "clean_correct": bool(clean.get("correct")),
                    "hard_correct": bool(hard.get("correct")),
                    "identity_correct": bool(ident.get("correct")),
                    "hard_vs_clean_flip": _answer_token(clean) != _answer_token(hard),
                    "identity_vs_clean_flip": _answer_token(clean) != _answer_token(ident),
                    "transition_group": group,
                    "clean_parse_success": bool(clean.get("parse_success")),
                    "hard_parse_success": bool(hard.get("parse_success")),
                    "identity_parse_success": bool(ident.get("parse_success")),
                }
            )
    return out


def summarize_output_modes(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lang in LANGUAGES:
        for condition in LANG_CONDITIONS:
            picked = [
                _answer_token(r) for r in records
                if r.get("language") == lang and r.get("condition") == condition
            ]
            counts = Counter(picked)
            total = sum(counts.values())
            for answer, count in counts.most_common(20):
                rows.append(
                    {
                        "language": lang,
                        "condition": condition,
                        "answer": answer,
                        "count": count,
                        "rate": count / total if total else None,
                    }
                )
    return rows


def summarize_transition_entropy(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lang in LANGUAGES:
        for condition in LANG_CONDITIONS:
            answers = [
                _answer_token(r) for r in records
                if r.get("language") == lang and r.get("condition") == condition
            ]
            counts = Counter(answers)
            total = sum(counts.values())
            entropy = 0.0
            for count in counts.values():
                p = count / total if total else 0.0
                if p > 0:
                    entropy -= p * math.log2(p)
            rows.append(
                {
                    "language": lang,
                    "condition": condition,
                    "n": total,
                    "unique_answer_count": len(counts),
                    "transition_entropy": entropy,
                }
            )
    return rows


def _transition_group(*, clean_correct: bool, hard_correct: bool) -> str:
    if clean_correct and not hard_correct:
        return "S_broken"
    if (not clean_correct) and hard_correct:
        return "S_repaired"
    if clean_correct and hard_correct:
        return "stable_correct"
    return "stable_wrong"


def _records_by_sample(
    records: Sequence[Dict[str, Any]],
    *,
    language: str,
    condition: str,
) -> Dict[str, Dict[str, Any]]:
    return {
        str(r["sample_id"]): dict(r)
        for r in records
        if r.get("language") == language and r.get("condition") == condition
    }


def _answer_token(record: Dict[str, Any]) -> str:
    if not bool(record.get("parse_success")):
        return "__PARSE_FAIL__"
    norm = normalize_answer(record.get("normalized_answer"))
    return norm if norm is not None else "__PARSE_FAIL__"


def write_final_summary(
    *,
    patch_run_dir: Path,
    lang_run_dir: Path,
    patch_sanity_dir: Path | None,
    lang_sanity_dir: Path | None,
    sanity_report: Path | None,
    audit_report: Path | None,
) -> Path:
    if sanity_report is None and patch_sanity_dir is not None:
        inferred = patch_sanity_dir / "FINAL_ADDON_SANITY_CHECK_REPORT.md"
        sanity_report = inferred if inferred.exists() else None
    if audit_report is None:
        inferred = patch_run_dir / "FINAL_ADDON_OUTPUT_AUDIT_REPORT.md"
        audit_report = inferred if inferred.exists() else None
    patch_summary = _read_summary_rows(patch_run_dir / "source_specific_patch_summary.json")
    patch_comparisons = _read_json(patch_run_dir / "source_specific_patch_summary.json").get("comparison_rows", [])
    lang_condition = _read_summary_rows(lang_run_dir / "condition_summary.json")
    lang_pairwise = _read_summary_rows(lang_run_dir / "pairwise_transition_summary.json")
    lang_groups = _read_summary_rows(lang_run_dir / "transition_group_summary.json")
    patch_verdict = classify_patch_verdict(patch_summary, patch_comparisons)
    lang_verdicts = classify_languages(lang_condition, lang_pairwise, lang_groups)
    final_token = final_addon_token(patch_verdict, lang_verdicts)

    lines = [
        "# Final Add-on Experiments Summary",
        "",
        "## Run Directories",
        f"- source-specific patch sanity: `{patch_sanity_dir}`",
        f"- extra-language sanity: `{lang_sanity_dir}`",
        f"- source-specific patch full: `{patch_run_dir}`",
        f"- extra-language full: `{lang_run_dir}`",
        f"- sanity checker report: `{sanity_report}`",
        f"- final audit report: `{audit_report}`",
        "",
        "Full runs used the same P0-relevant config that passed sanity: layers 20/22 only, greedy max_new_tokens=256, no dense scoring, no 512 extension, and no trajectory/identifiers/static-margin paths.",
        "",
        "## Summary Table",
        "",
        "| Experiment | Result | Paper placement | Claim impact | Risk / caveat |",
        "|---|---|---|---|---|",
        (
            f"| Source-specific patch control | {patch_verdict['label']} | "
            f"{patch_verdict['placement']} | {patch_verdict['claim']} | {patch_verdict['caveat']} |"
        ),
    ]
    for lang in LANGUAGES:
        verdict = lang_verdicts[lang]
        lines.append(
            f"| MGSM-{lang} answer redistribution sanity | {verdict['label']} | "
            f"{verdict['placement']} | {verdict['claim']} | {verdict['caveat']} |"
        )
    lines.extend(
        [
            "",
            "## Updated Claim Wording",
            "- Recovery-zone patches show source-sensitive clean-aligned perturbational leverage only if same-sample clean patches clearly beat shuffled and random controls.",
            "- The answer-redistribution phenomenon qualitatively appears in additional MGSM languages only for languages classified as usable or appendix-only here.",
            "- These add-ons are limited sanity/add-on audits and should not be used for broader mechanism or cross-lingual claims.",
            "",
            final_token,
        ]
    )
    out = REPO_ROOT / SUMMARY_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] wrote {out}", flush=True)
    print(final_token, flush=True)
    return out


def classify_patch_verdict(
    summary_rows: Sequence[Dict[str, Any]],
    comparison_rows: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    by_condition = {str(r["condition"]): r for r in summary_rows}
    hard_self_bad = False
    for layer in PATCH_LAYERS:
        hs = by_condition.get(f"hard_self_patch_L{layer}", {})
        if int(float(hs.get("parsed_mismatch_vs_hard_no_patch") or 0)) > 1:
            hard_self_bad = True
    if hard_self_bad:
        return {
            "label": "PATCH_CONTROL_FAILED_NEEDS_FIX",
            "placement": "omit",
            "claim": "no claim upgrade",
            "caveat": "hard_self_patch differs from hard_no_patch",
        }
    supports = []
    distribution = []
    for layer in PATCH_LAYERS:
        same = by_condition.get(f"same_sample_clean_patch_L{layer}", {}).get("eq_clean")
        shuffled = by_condition.get(f"shuffled_clean_patch_L{layer}", {}).get("eq_clean")
        random_rate = by_condition.get(f"random_norm_matched_patch_L{layer}", {}).get("eq_clean")
        if same is None or shuffled is None or random_rate is None:
            continue
        same_f = float(same)
        shuffled_f = float(shuffled)
        random_f = float(random_rate)
        if same_f >= shuffled_f + 0.10 and same_f >= random_f + 0.10:
            supports.append(layer)
        if abs(same_f - shuffled_f) <= 0.05 and same_f >= random_f + 0.10:
            distribution.append(layer)
    if supports:
        return {
            "label": "PATCH_SOURCE_SPECIFICITY_SUPPORTED",
            "placement": "main text or appendix, depending on space",
            "claim": "source-sensitive clean-aligned perturbational leverage",
            "caveat": "third-answer perturbation remains part of the interpretation",
        }
    if distribution:
        return {
            "label": "PATCH_DISTRIBUTIONAL_CLEAN_LEVERAGE",
            "placement": "appendix",
            "claim": "clean-distribution compatibility rather than sample-specific restoration",
            "caveat": "same-sample and shuffled controls are similar",
        }
    return {
        "label": "PATCH_GENERIC_PERTURBATION",
        "placement": "appendix or omit",
        "claim": "patching indicates perturbational sensitivity",
        "caveat": "controls are too similar for source-specific wording",
    }


def classify_languages(
    condition_summary: Sequence[Dict[str, Any]],
    pairwise_summary: Sequence[Dict[str, Any]],
    group_summary: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    cond = {(r["language"], r["condition"]): r for r in condition_summary}
    pairs = {(r["language"], r["pair"]): r for r in pairwise_summary}
    groups = {r["language"]: r for r in group_summary}
    out: Dict[str, Dict[str, str]] = {}
    for lang in LANGUAGES:
        clean = cond.get((lang, "clean_no_swap"), {})
        hard = cond.get((lang, "hard_swap_b8_t20"), {})
        ident = cond.get((lang, "identity_composition"), {})
        hard_pair = pairs.get((lang, "hard_vs_clean"), {})
        ident_pair = pairs.get((lang, "identity_vs_clean"), {})
        grp = groups.get(lang, {})
        parse_min = min(
            float(clean.get("parse_success_rate") or 0.0),
            float(hard.get("parse_success_rate") or 0.0),
            float(ident.get("parse_success_rate") or 0.0),
        )
        identity_flip = float(ident_pair.get("answer_flip_rate") or 0.0)
        hard_flip = float(hard_pair.get("answer_flip_rate") or 0.0)
        clean_acc = float(clean.get("accuracy") or 0.0)
        hard_acc = float(hard.get("accuracy") or 0.0)
        has_transition = (
            int(float(grp.get("S_broken_count") or 0)) > 0
            or int(float(grp.get("S_repaired_count") or 0)) > 0
        )
        catastrophic = hard_acc < max(0.05, clean_acc - 0.60)
        if parse_min < 0.90 or identity_flip > 0.05 or catastrophic:
            out[lang] = {
                "label": "excluded",
                "placement": "omit",
                "claim": "no language sanity claim",
                "caveat": f"parse_min={parse_min:.3f}, identity_flip={identity_flip:.3f}, hard_acc={hard_acc:.3f}",
            }
        elif parse_min >= 0.95 and identity_flip <= 0.02 and hard_flip >= 0.30 and has_transition:
            out[lang] = {
                "label": "main-text usable",
                "placement": "main text sanity or appendix",
                "claim": "qualitative additional-language sanity check",
                "caveat": f"hard_flip={hard_flip:.3f}; not full cross-lingual generalization",
            }
        else:
            out[lang] = {
                "label": "appendix-only",
                "placement": "appendix",
                "claim": "weak qualitative sanity evidence only",
                "caveat": f"parse_min={parse_min:.3f}, hard_flip={hard_flip:.3f}, identity_flip={identity_flip:.3f}",
            }
    return out


def final_addon_token(patch_verdict: Dict[str, str], lang_verdicts: Dict[str, Dict[str, str]]) -> str:
    if patch_verdict["label"] == "PATCH_CONTROL_FAILED_NEEDS_FIX":
        return "FINAL_ADDONS_FAILED_NEEDS_REVIEW"
    if any(v["label"] == "main-text usable" for v in lang_verdicts.values()):
        return "FINAL_ADDONS_COMPLETE_MAIN_USABLE"
    if any(v["label"] == "appendix-only" for v in lang_verdicts.values()):
        return "FINAL_ADDONS_COMPLETE_APPENDIX_ONLY"
    return "FINAL_ADDONS_FAILED_NEEDS_REVIEW"


def _write_patch_report(
    path: Path,
    *,
    metadata: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    comparison_rows: Sequence[Dict[str, Any]],
    sanity: bool,
) -> None:
    verdict = "SANITY_SOURCE_SPECIFIC_PATCH_PASSED"
    if not sanity:
        verdict = classify_patch_verdict(summary_rows, comparison_rows)["label"]
    lines = [
        "# Source-Specific Patch Control Report" if not sanity else "# Source-Specific Patch Sanity Report",
        "",
        "## Executive Summary",
        "This run tests whether same-sample clean pre-block patches at layers 20/22 are stronger than shuffled-clean, random norm-matched, and hard-self controls.",
        "",
        "## S_broken Provenance",
        f"- source run: `{metadata['source_provenance']['patch_scan_run_dir']}`",
        f"- S_broken file: `{metadata['source_provenance']['sbroken_sample_ids_json']}`",
        f"- canonical S_broken count: {metadata['source_provenance']['canonical_sbroken_count']}",
        f"- selected count: {metadata['n_samples']}",
        "",
        "## Patch Site And Layer Indexing",
        f"- {metadata['patch_site']}",
        f"- {metadata['layer_indexing_convention']}",
        "",
        "## Condition Definitions",
        "- hard_no_patch: hard-swapped b=8,t=20 generation with no patch.",
        "- same_sample_clean_patch_Lk: same-sample clean block input at layer k.",
        "- shuffled_clean_patch_Lk: deterministic other-sample clean block input.",
        "- random_norm_matched_patch_Lk: random state with per-token norm matched to the same-sample clean state.",
        "- hard_self_patch_Lk: hard block input patched back into the hard model.",
        "",
        "## Layer 20 Results",
        *_summary_lines_for_layer(summary_rows, 20),
        "",
        "## Layer 22 Results",
        *_summary_lines_for_layer(summary_rows, 22),
        "",
        "## Same Vs Shuffled Vs Random Comparison",
        *_generic_markdown_rows(comparison_rows),
        "",
        "## hard_self_patch Sanity Check",
        *_hard_self_lines(summary_rows),
        "",
        "## Claim-Safe Interpretation",
        "Use source-sensitive clean-aligned perturbational leverage only if same-sample clean clearly beats shuffled and random controls. Avoid claims beyond output redirection.",
        "",
        "## Patch Claim Recommendation",
        verdict,
        "",
        verdict,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_language_report(
    path: Path,
    *,
    metadata: Dict[str, Any],
    condition_summary: Sequence[Dict[str, Any]],
    pairwise_summary: Sequence[Dict[str, Any]],
    group_summary: Sequence[Dict[str, Any]],
    parser_tests: Sequence[Dict[str, Any]],
    sanity: bool,
) -> None:
    lang_verdicts = classify_languages(condition_summary, pairwise_summary, group_summary)
    if sanity:
        final = "SANITY_EXTRA_LANGUAGE_PASSED"
    else:
        usable = [lang for lang, v in lang_verdicts.items() if v["label"] == "main-text usable"]
        appendix = [lang for lang, v in lang_verdicts.items() if v["label"] == "appendix-only"]
        if len(usable) == 2:
            final = "EXTRA_LANG_SANITY_BOTH_USABLE"
        elif len(usable) == 1:
            final = "EXTRA_LANG_SANITY_ONE_USABLE"
        elif appendix:
            final = "EXTRA_LANG_SANITY_APPENDIX_ONLY"
        else:
            final = "EXTRA_LANG_SANITY_FAILED"
    lines = [
        "# Extra-Language Sanity Report" if not sanity else "# Extra-Language Sanity Run Report",
        "",
        "## Executive Summary",
        "This run checks whether clean/hard answer redistribution qualitatively appears on MGSM-ko and MGSM-ar under the canonical b=8,t=20 hard swap.",
        "",
        "## Dataset / Language Details",
        f"- dataset: `{metadata['dataset']['name']}`",
        f"- split: `{metadata['dataset']['split']}`",
        f"- sample counts: `{metadata['sample_counts']}`",
        "",
        "## Parser Normalization Notes",
        "- Arabic-Indic and Eastern Arabic-Indic digits are normalized before numeric extraction.",
        *_generic_markdown_rows(parser_tests),
        "",
        "## Condition Definitions",
        "- clean_no_swap: all layers are Qwen2.5-1.5B-Instruct.",
        "- hard_swap_b8_t20: layers 8..19 are Qwen2.5-1.5B Base.",
        "- identity_composition: same composition path as hard, with Instruct donor.",
        "",
        "## ko Results",
        *_language_lines("ko", condition_summary, pairwise_summary, group_summary, lang_verdicts),
        "",
        "## ar Results",
        *_language_lines("ar", condition_summary, pairwise_summary, group_summary, lang_verdicts),
        "",
        "## Identity Control Check",
        "identity_composition uses compose_model(recipient, recipient, b=8, t=20, condition='hard_swap'); it does not bypass the composition wrapper.",
        "",
        "## Hard-vs-Clean Redistribution Check",
        *_generic_markdown_rows(pairwise_summary),
        "",
        "## Claim-Safe Interpretation",
        "This is a qualitative additional-language sanity check only. It does not establish full cross-lingual generalization.",
        "",
        final,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_lines_for_layer(summary_rows: Sequence[Dict[str, Any]], layer: int) -> List[str]:
    picked = [
        r for r in summary_rows
        if r.get("condition") == PATCH_BASELINE or int(float(r.get("layer") or -1)) == int(layer)
    ]
    return _generic_markdown_rows(picked)


def _hard_self_lines(summary_rows: Sequence[Dict[str, Any]]) -> List[str]:
    picked = [
        r for r in summary_rows
        if str(r.get("condition")).startswith("hard_self_patch")
    ]
    return _generic_markdown_rows(picked)


def _language_lines(
    language: str,
    condition_summary: Sequence[Dict[str, Any]],
    pairwise_summary: Sequence[Dict[str, Any]],
    group_summary: Sequence[Dict[str, Any]],
    verdicts: Dict[str, Dict[str, str]],
) -> List[str]:
    rows: List[Dict[str, Any]] = []
    rows.extend([r for r in condition_summary if r.get("language") == language])
    rows.extend([r for r in pairwise_summary if r.get("language") == language])
    rows.extend([r for r in group_summary if r.get("language") == language])
    rows.append({"language": language, "classification": verdicts[language]["label"], "caveat": verdicts[language]["caveat"]})
    return _generic_markdown_rows(rows)


def _generic_markdown_rows(rows: Sequence[Dict[str, Any]]) -> List[str]:
    out = []
    for row in rows:
        compact = ", ".join(
            f"{key}={value}" for key, value in row.items()
            if value not in (None, "", [])
        )
        out.append(f"- {compact}")
    return out or ["- No rows available."]


def _bool_rate(rows: Sequence[Dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if bool(r.get(field))) / len(rows)


def _make_run_dir(root: str, prefix: str) -> Path:
    import datetime as dt

    out_root = REPO_ROOT / root
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = out_root / f"{prefix}_{stamp}"
    suffix = 0
    while path.exists():
        suffix += 1
        path = out_root / f"{prefix}_{stamp}_{suffix}"
    path.mkdir(parents=True)
    return path


def _write_selected_sbroken(run_dir: Path, rows: Sequence[Dict[str, Any]], *, sanity: bool) -> None:
    name = "sanity_sbroken_sample_ids.json" if sanity else "sbroken_sample_ids.json"
    _write_json(run_dir / name, list(rows))


def _read_summary_rows(path: Path) -> List[Dict[str, Any]]:
    obj = _read_json(path)
    return [dict(r) for r in obj.get("summary_rows", [])]


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(v) for k, v in row.items()})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


if __name__ == "__main__":
    raise SystemExit(main())

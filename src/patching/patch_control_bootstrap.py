#!/usr/bin/env python3
"""
Compute paired bootstrap confidence intervals for patch-control match-baseline differences.

Input: source_specific_patch_records.jsonl (default) or a zip archive containing it.
Output: CSV with point estimates and percentile bootstrap 95% CIs.

The bootstrap unit is sample_id. For each layer and comparison, this script computes
mean(match_baseline under condition A - match_baseline under condition B) over S_broken
sample identifiers, then resamples sample identifiers with replacement.

Column mapping: the JSONL field ``eq_clean`` records whether the patched output matched
the clean baseline answer under the final normalized-answer evaluator. This column is
loaded directly — no renaming is applied.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

_BUNDLE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = str(
    _BUNDLE_ROOT
    / "data"
    / "raw_runs"
    / "source_specific_patch_control"
    / "source_specific_patch_records.jsonl"
)
DEFAULT_OUTPUT = str(_BUNDLE_ROOT / "artifacts" / "patch_control" / "bootstrap_ci.csv")
DEFAULT_S_BROKEN_IDS = _BUNDLE_ROOT / "artifacts" / "patch_control" / "s_broken_ids.txt"
DEFAULT_CANONICAL_RECORDS = _BUNDLE_ROOT / "artifacts" / "patch_control" / "patch_control_records.jsonl"
DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 20260517

# Comparisons reported in the paper: source-specific same-sample baseline-state patch
# versus shuffled baseline-state and norm-matched random controls.
COMPARISONS = [
    (20, "same_sample_clean_patch_L20", "shuffled_clean_patch_L20", "same_sample_minus_shuffled"),
    (20, "same_sample_clean_patch_L20", "random_norm_matched_patch_L20", "same_sample_minus_random"),
    (22, "same_sample_clean_patch_L22", "shuffled_clean_patch_L22", "same_sample_minus_shuffled"),
    (22, "same_sample_clean_patch_L22", "random_norm_matched_patch_L22", "same_sample_minus_random"),
]


def _read_jsonl_lines(path: Path, member: str | None = None) -> list[str]:
    if member is None:
        return path.read_text(encoding="utf-8").splitlines()
    with zipfile.ZipFile(path) as zf:
        return zf.read(member).decode("utf-8").splitlines()


def load_records(input_path: str) -> pd.DataFrame:
    """Load patch-control records from a .jsonl file or a zip containing it."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            matches = [name for name in zf.namelist() if name.endswith("source_specific_patch_records.jsonl")]
            if not matches:
                raise FileNotFoundError("No source_specific_patch_records.jsonl found inside zip")
            # Prefer top-level file if present.
            member = "source_specific_patch_records.jsonl" if "source_specific_patch_records.jsonl" in matches else matches[0]
        lines = _read_jsonl_lines(path, member)
    else:
        lines = _read_jsonl_lines(path)

    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        raise ValueError("No records loaded")
    return pd.DataFrame(rows)


def load_canonical_s_broken_ids(path: Path = DEFAULT_S_BROKEN_IDS) -> set[str]:
    """Load the canonical n=39 S_broken sample identifiers used in the paper."""
    if not path.exists():
        raise FileNotFoundError(f"Canonical S_broken list not found: {path}")
    sample_ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if len(sample_ids) != 39:
        raise ValueError(f"Expected 39 canonical S_broken sample identifiers, found {len(sample_ids)}")
    return sample_ids


def subset_to_canonical_s_broken(df: pd.DataFrame, sample_ids: set[str]) -> pd.DataFrame:
    """Keep only shipped canonical S_broken records before strict validation."""
    if "sample_id" not in df.columns or "subset" not in df.columns:
        return df
    return df[(df["subset"] == "S_broken") & (df["sample_id"].isin(sample_ids))].copy()


def select_canonical_s_broken_records(df: pd.DataFrame, sample_ids: set[str]) -> pd.DataFrame:
    """Return the canonical n=39 patch-control records used for the paper CI."""
    subset = subset_to_canonical_s_broken(df, sample_ids)
    if "sample_id" in subset.columns and len(set(subset["sample_id"])) == 39:
        return subset
    canonical = load_records(str(DEFAULT_CANONICAL_RECORDS))
    return subset_to_canonical_s_broken(canonical, sample_ids)


def validate_records(df: pd.DataFrame) -> None:
    required = {"sample_id", "condition", "eq_clean", "subset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "S_broken" not in set(df["subset"]):
        raise ValueError("No S_broken records found")

    sb = df[df["subset"] == "S_broken"]
    sample_ids = set(sb["sample_id"])
    if len(sample_ids) != 39:
        raise ValueError(f"Expected 39 S_broken sample identifiers, found {len(sample_ids)}")

    conds = set(sb["condition"])
    needed = {a for _, a, _, _ in COMPARISONS} | {b for _, _, b, _ in COMPARISONS}
    missing_conds = needed - conds
    if missing_conds:
        raise ValueError(f"Missing comparison conditions: {sorted(missing_conds)}")


def _bca_interval(values: np.ndarray, boot_means: np.ndarray, point: float) -> tuple[float, float]:
    """Return BCa 95% interval for a mean statistic over paired values."""
    if np.all(values == values[0]):
        return point, point
    n = len(values)
    if n < 2:
        return point, point

    prop_less = float(np.mean(boot_means < point))
    eps = 1.0 / (2.0 * len(boot_means))
    prop_less = min(max(prop_less, eps), 1.0 - eps)
    z0 = float(norm.ppf(prop_less))

    total = float(values.sum())
    jack = np.array([(total - values[i]) / (n - 1) for i in range(n)], dtype=float)
    jack_mean = float(jack.mean())
    centered = jack_mean - jack
    denom = 6.0 * float(np.sum(centered ** 2) ** 1.5)
    accel = 0.0 if denom == 0.0 else float(np.sum(centered ** 3) / denom)

    adjusted = []
    for alpha in (0.025, 0.975):
        z_alpha = float(norm.ppf(alpha))
        denom_adj = 1.0 - accel * (z0 + z_alpha)
        if denom_adj == 0.0:
            adjusted.append(alpha)
        else:
            adjusted.append(float(norm.cdf(z0 + ((z0 + z_alpha) / denom_adj))))
    lo, hi = np.quantile(boot_means, adjusted)
    return float(lo), float(hi)


def paired_bootstrap(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    method: str = "percentile",
) -> tuple[float, float, float]:
    """Return point estimate and percentile 95% CI for paired sample-level values."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("No values to bootstrap")
    point = float(values.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    if method == "percentile":
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
    elif method == "bca":
        lo, hi = _bca_interval(values, boot_means, point)
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")
    return point, float(lo), float(hi)


def compute_ci(df: pd.DataFrame, n_boot: int, seed: int, method: str = "percentile") -> pd.DataFrame:
    sb = df[df["subset"] == "S_broken"].copy()
    sb["match_baseline"] = sb["eq_clean"].astype(bool).astype(int)

    sample_ids = sorted(sb["sample_id"].unique())
    pivot = sb.pivot_table(
        index="sample_id",
        columns="condition",
        values="match_baseline",
        aggfunc="first",
    ).reindex(sample_ids)

    rows = []
    for layer, same_cond, control_cond, comparison in COMPARISONS:
        values = (pivot[same_cond] - pivot[control_cond]).to_numpy(dtype=float)
        point, lo, hi = paired_bootstrap(values, n_boot=n_boot, seed=seed, method=method)
        rows.append({
            "layer": layer,
            "comparison": comparison,
            "same_sample_condition": same_cond,
            "control_condition": control_cond,
            "n_sample_ids": len(values),
            "metric": "match_baseline_rate_difference",
            "point_estimate": point,
            "ci_low": lo,
            "ci_high": hi,
            "ci_level": 0.95,
            "bootstrap_method": f"paired_nonparametric_{method}",
            "bootstrap_unit": "sample_id",
            "n_bootstrap": n_boot,
            "seed": seed,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to patch_control.zip or source_specific_patch_records.jsonl")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--method", choices=["percentile", "bca"], default="percentile")
    args = parser.parse_args()

    df = load_records(args.input)
    sample_ids = load_canonical_s_broken_ids()
    df = select_canonical_s_broken_records(df, sample_ids)
    validate_records(df)
    out = compute_ci(df, n_boot=args.n_bootstrap, seed=args.seed, method=args.method)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, lineterminator="\n")
    print(out.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

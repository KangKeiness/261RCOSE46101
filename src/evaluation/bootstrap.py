"""Bootstrap utilities that resample sample identifiers, not token rows."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def bootstrap_mean_ci_by_sample_id(
    rows: Sequence[Dict[str, Any]],
    value_fn: Callable[[Sequence[Dict[str, Any]]], float | None],
    *,
    sample_id_field: str = "sample_id",
    n_resamples: int = 1000,
    seed: int = 13,
    ci: float = 0.95,
) -> Tuple[float | None, float | None, float | None]:
    """Bootstrap a statistic over sample identifiers with multiplicity preserved."""

    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = str(row.get(sample_id_field))
        if sid:
            by_sample[sid].append(row)
    sample_ids = list(by_sample)
    if not sample_ids:
        return None, None, None
    point = value_fn(rows)
    if point is None or n_resamples <= 0:
        return point, None, None
    rng = random.Random(seed)
    stats: List[float] = []
    for _ in range(int(n_resamples)):
        sample = [rng.choice(sample_ids) for _ in sample_ids]
        sampled_rows: List[Dict[str, Any]] = []
        for sid in sample:
            sampled_rows.extend(by_sample[sid])
        stat = value_fn(sampled_rows)
        if stat is not None:
            stats.append(float(stat))
    if not stats:
        return point, None, None
    stats.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = max(0, min(len(stats) - 1, int(alpha * len(stats))))
    hi_idx = max(0, min(len(stats) - 1, int((1.0 - alpha) * len(stats)) - 1))
    return float(point), float(stats[lo_idx]), float(stats[hi_idx])


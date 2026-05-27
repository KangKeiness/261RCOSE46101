"""Answer normalization shared by mechanism harness modes."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any


_TRAILING_PUNCT = ".,;:!?\u3002\uff0c\uff1b\uff1a"
_CURRENCY_RE = re.compile(r"[$\u20ac\u00a3\u00a5\u20a9]")


def normalize_answer(value: Any) -> str | None:
    """Normalize a scalar numeric answer without using gold-answer context."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.strip(_TRAILING_PUNCT)
    text = _CURRENCY_RE.sub("", text)
    text = text.replace(",", "").replace(" ", "")
    text = text.replace("\u2212", "-")
    if text.endswith("%"):
        text = text[:-1]
    if "/" in text:
        return text or None
    try:
        dec = Decimal(text)
    except InvalidOperation:
        return None
    if not dec.is_finite():
        return None
    if dec == dec.to_integral_value():
        return str(int(dec))
    out = format(dec.normalize(), "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return "0" if out == "-0" else out


def answers_equal(left: Any, right: Any) -> bool:
    """Return normalized scalar equality."""

    left_norm = normalize_answer(left)
    right_norm = normalize_answer(right)
    if left_norm is None or right_norm is None:
        return False
    return left_norm == right_norm


def answer_token(parse_success: bool, normalized_answer: Any) -> str:
    """Stable token for candidate grouping; parse failures are explicit."""

    if not parse_success:
        return "__PARSE_FAIL__"
    norm = normalize_answer(normalized_answer)
    return "__PARSE_FAIL__" if norm is None else norm


def extract_numeric_surfaces(text: str) -> list[tuple[int, int, str]]:
    """Return numeric-looking spans in text."""

    number_re = re.compile(
        r"(?<![\w.])[-+]?\d[\d,\s]*(?:\.\d+)?(?:\s*%|/\d+)?(?!\w)"
    )
    return [(m.start(), m.end(), m.group(0)) for m in number_re.finditer(text or "")]


def finite_float(value: Any) -> float | None:
    """Coerce a finite float or return None."""

    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None

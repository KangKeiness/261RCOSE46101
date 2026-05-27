"""Parser wrapper for the unified mechanism harness.

Note: this module is a provenance reference used by src/runs/*.py (GPU
generation scripts). It is not part of the release verification path
(src/evaluation/, src/figures/, src/patching/).
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Dict

from src.inference import parser as stage1_parser
from src.evaluation.answer_normalizer import normalize_answer


def parse_answer_text(text: str) -> Dict[str, Any]:
    """Parse generated text with the project parser and normalize consistently."""

    parsed = stage1_parser.parse_answer(text or "")
    norm = normalize_answer(parsed.get("normalized_answer") or parsed.get("parsed_answer"))
    return {
        "parsed_answer": parsed.get("parsed_answer"),
        "parse_success": bool(parsed.get("parse_success")) and norm is not None,
        "normalized_answer": norm,
        "parse_type": parsed.get("parse_type"),
    }


def parser_config_hash() -> str:
    """Hash parser source for run metadata."""

    try:
        source = inspect.getsource(stage1_parser.parse_answer)
        source += inspect.getsource(stage1_parser._normalize_number)
    except Exception:
        source = repr(stage1_parser.parse_answer)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


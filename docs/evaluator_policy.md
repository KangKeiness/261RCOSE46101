# Evaluator Policy

This document describes the normalized-answer evaluator used for all
correctness judgements and answer-identity comparisons in this release bundle.
The same policy is applied to every language (Chinese, Korean, Arabic).

---

## Core rule: decimal-equivalent normalization

Two answer strings are considered equal if and only if they represent the same
number after decimal-equivalent normalization. The normalization steps are:

1. Strip leading/trailing whitespace.
2. Remove a leading currency symbol (`$`, `¥`, `€`, etc.) if present.
3. Remove thousands-separator commas (e.g. `1,000` becomes `1000`).
4. Replace a Unicode minus sign (U+2212, `−`) with an ASCII hyphen-minus (`-`).
5. Parse the resulting string as a decimal number.
6. Two strings are equal if and only if both parse successfully and both
   produce the same decimal value.

---

## Parse failures

If a model output cannot be parsed as a number after normalization (for
example, a blank output, an unparseable string, or no extracted answer), it is
retained in the dataset and counted as **incorrect** in all accuracy
computations. In identity comparisons (e.g. "did the answer change?"), a parse
failure is treated as **not equal** to any other answer, including another
parse failure.

---

## Fraction equivalence: intentionally not supported

Fraction strings (e.g. `5/2`) are not treated as equal to their decimal
equivalents (e.g. `2.5`). If a model outputs `5/2` and the gold answer is
`2.5`, the evaluator marks them as unequal. This is a deliberate policy choice
to avoid ambiguity in fraction detection.

---

## Sanity test cases

The following nine cases were used to validate the evaluator implementation.
All nine passed.

| # | Left | Right | Equal? | Notes |
|---|---|---|---|---|
| 1 | `75.00` | `75` | Yes | Trailing zeros normalized |
| 2 | `28` | `28.00` | Yes | Integer vs. decimal |
| 3 | `4` | `4.0` | Yes | Integer vs. decimal |
| 4 | `1,000` | `1000` | Yes | Thousands separator stripped |
| 5 | `$1,000.00` | `1000` | Yes | Currency prefix stripped |
| 6 | `-3.0` | `-3` | Yes | Negative decimal |
| 7 | `−3.0` | `-3` | Yes | Unicode minus normalized to ASCII |
| 8 | `75.00` | `76` | No | Genuinely different values |
| 9 | (parse failure) | `75` | No | Parse failure is unequal to any value |

---

## Scope

This evaluator is used in:

- `src/evaluation/evaluator.py` — per-sample correctness and identity scoring.
- `src/evaluation/compute_main_metrics.py` — aggregate metric computation.
- `src/evaluation/verify_paper_numbers.py` — post-hoc CSV verification.

It is not a claim about model reasoning; it is a string-comparison rule applied
uniformly to model-generated and gold answer strings.

---

## Signed Final-Answer Parsing

The parser preserves ASCII hyphen-minus and Unicode minus signs when they occur
inside final-answer markers such as `The answer is -10000`, `答案是 -15`, or
`答案为：−15`. The fallback parser also supports standalone signed numbers, but
treats a sign immediately following another number as subtraction so unfinished
arithmetic such as `500 - 250` continues to yield `250`, not `-250`.

The Chinese MGSM raw JSONL files were re-derived under this parser. This moved
two direct-swap false positives (`mgsm_0129`, `mgsm_0164`) from correct to
incorrect and also made the clean and direct-swap answer for `mgsm_0164` the
same signed wrong answer under a single consistent parser.

---

## KR/AR Parser Fix Caveat

The signed-final-answer parser fix was validated against the Chinese MGSM raw
outputs (`data/raw_runs/main_chinese_mgsm/results_*.jsonl`). Korean and Arabic
raw `output_text` strings are not shipped in this bundle; only per-sample
transition records are shipped. Korean and Arabic gold answers are all
non-negative in the shipped transition records, which limits the possible
impact of a sign-parse error to false-positive cases where a model produced a
negative number whose absolute value matched a positive gold answer. Such cases
cannot be directly audited from this bundle, so Appendix A Korean and Arabic
numbers remain as originally computed.

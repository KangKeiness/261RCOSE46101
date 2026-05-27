# 261R0136COSE34102 — Answer Redistribution under Middle-Layer Replacement

This repository contains the artifact bundle for the Korea University COSE461
final project on answer redistribution under direct middle-layer replacement.

---

## Main claim

Direct middle-layer replacement produces only a modest aggregate accuracy change in this setting while extensively reassigning individual parsed answers. On Chinese MGSM (n=250, deterministic greedy decoding), clean and
direct-swap accuracy agree to within 3.2 percentage points, yet 60.4% of
individual samples change their parsed answer. Korean and Arabic sanity checks
(n=250 each) replicate the same redistribution pattern at similar or higher rates.

No causal localization or functional-improvement claims are made; these
results describe answer-identity transitions under a single replacement
configuration.

---

## Main result table

<!-- BEGIN AUTO-GENERATED TABLE -->
<!-- generated from artifacts/main/main_metrics.csv and artifacts/main/bootstrap_ci.csv -->
<!-- render with: python scripts/render_readme_main_table.py -->

**Chinese MGSM (n=250) — main result**

| Metric | Value | 95% CI |
|---|---|---|
| Clean accuracy | 0.500 (125/250) | [0.440, 0.564] |
| Direct-swap accuracy | 0.468 (117/250) | [0.408, 0.532] |
| Accuracy delta | -0.032 | [-0.096, 0.032] |
| Parsed-answer change rate | 0.604 (151/250) | [0.540, 0.664] |

**Transition breakdown**

| Category | Count |
|---|---|
| Stable-correct | 86 |
| Broken (correct → wrong) | 39 |
| Repaired (wrong → correct) | 31 |
| Stable-wrong, same answer | 13 |
| Stable-wrong, different answer | 81 |
| Stable-wrong-different rate | 0.862 (81/94) | [0.787, 0.929] |

Bootstrap CIs: paired nonparametric percentile, 10000 resamples, seed 20260517.

<!-- END AUTO-GENERATED TABLE -->

---

## Quick reproduction

```bash
# Verify every paper-cited number from the pre-computed CSVs
bash scripts/verify_paper_numbers.sh
# Must print: PAPER NUMBERS VERIFIED

# Run the test suite
pytest tests/
# Must pass with no failures
```

---

## Directory structure

```
261R0136COSE34102/
  src/
    composition/       # Layer-composition utilities
    intervention/      # Patch application
    inference/         # Runner, parser, final-parser audit
    evaluation/        # Evaluator, bootstrap, answer normalizer,
                       #   paper_constants, verify_paper_numbers,
                       #   compute_main_metrics
    patching/          # Patch-control bootstrap CI
    figures/           # Figure generation (CSV-driven)
    data/              # MGSM data loader
    runs/              # Run entry-points (main, extra languages,
                       #   trajectory diagnostics, hidden-state appendix)
    utils/
    configs/           # YAML configs (main.yaml, main_canonical.yaml)
  scripts/             # Bash wrappers for each src entry-point
  artifacts/
    main/              # main_metrics.csv, bootstrap_ci.csv
    extra_language/    # Korean and Arabic metrics and CIs
    patch_control/     # Patch-control bootstrap CIs, S_broken sample list
    diagnostics/       # Behavioral diagnostics CSV
    hidden_state_appendix/
    repetition_robustness/
    figures/           # Rendered PDF/PNG (gitignored; reproducible)
  data/
    raw_runs/          # Per-sample JSONL/CSV records (copied verbatim)
  docs/
    artifact_lineage.md
    evaluator_policy.md
    reproduction_guide.md
  tests/
    test_paper_number_invariants.py
  paper/               # Reserved for PDF/TeX/bib (inserted by author)
  requirements.txt
  LICENSE
  .gitignore
```

---

## Evaluator policy

All correctness and answer-identity comparisons use decimal-equivalent numeric
normalization: trailing zeros, thousands separators, currency prefixes, and
Unicode minus are stripped before comparison; parse failures are retained and
counted as incorrect. Fraction equivalence (`5/2` = `2.5`) is intentionally
**not** supported.

Full policy and sanity test table: [docs/evaluator_policy.md](docs/evaluator_policy.md)

---

## Lineage

Every paper-cited number traces to a CSV under `artifacts/` and a raw-run
input under `data/raw_runs/`. Full lineage per cluster:
[docs/artifact_lineage.md](docs/artifact_lineage.md)

Legacy internal directory names (e.g. `run_20260506_142616_fixed256_full`)
may appear inside `data/raw_runs/` paths; these are immutable run identifiers
and are acknowledged in `docs/artifact_lineage.md`.

---

## Limitations

The experiment uses a single model pair (Qwen2.5-1.5B-Instruct recipient and
Qwen2.5-1.5B base donor), a single layer-range configuration (layers 8–19 of
28), and a single evaluation benchmark (MGSM). Results may not generalize to
other model families, layer ranges, or benchmarks. Decoding is deterministic
greedy only; stochastic decoding is out of scope. The findings are descriptive
(answer-identity transitions under replacement) and make no claims about
causality, functional localization, or improvement of model capabilities.

---

## Citation and course note

This is the artifact bundle for the COSE461 final project submission, Team 8,
Korea University, 2026. Paper PDF and TeX inserted under `paper/` by
the author.

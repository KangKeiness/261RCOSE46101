"""Generate paper figures from CSV artifacts.

All paper-cited numbers are loaded from CSVs under artifacts/ and verified
against paper_constants.py before any figure is drawn.

Usage:
    python -m src.figures.generate_paper_figures
    python -m src.figures.generate_paper_figures --artifacts-root /path/to/artifacts
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import re
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image


ARTIFACTS_ROOT = pathlib.Path(__file__).resolve().parents[2] / "artifacts"
MAIN_METRICS_CSV = ARTIFACTS_ROOT / "main" / "main_metrics.csv"
DIAGNOSTICS_CSV = ARTIFACTS_ROOT / "diagnostics" / "diagnostics.csv"
OUT_DIR = ARTIFACTS_ROOT / "figures"


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "axes.linewidth": 0.7,
    }
)


def load_transitions(main_metrics_csv: pathlib.Path) -> dict[str, int]:
    """Load transition counts from main_metrics.csv (long-form metric,value).

    Returns a dict with keys: n, baseline_correct, baseline_wrong,
    direct_swap_correct, direct_swap_wrong, stable_correct, broken,
    repaired, stable_wrong, same_wrong, different_wrong.
    """
    metrics: dict[str, str] = {}
    with main_metrics_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            metrics[row["metric"]] = row["value"]

    n = int(metrics["n"])
    clean_correct = int(metrics["clean_correct"])
    direct_swap_correct = int(metrics["direct_swap_correct"])
    return {
        "n": n,
        "baseline_correct": clean_correct,
        "baseline_wrong": n - clean_correct,
        "direct_swap_correct": direct_swap_correct,
        "direct_swap_wrong": n - direct_swap_correct,
        "stable_correct": int(metrics["stable_correct"]),
        "broken": int(metrics["S_broken"]),
        "repaired": int(metrics["S_repaired"]),
        "stable_wrong": int(metrics["stable_wrong"]),
        "same_wrong": int(metrics["stable_wrong_same"]),
        "different_wrong": int(metrics["stable_wrong_different"]),
    }


def load_margin_values(diagnostics_csv: pathlib.Path) -> np.ndarray:
    """Load rationale-conditioned scoring means from diagnostics.csv.

    Returns shape (2, 2) array:
      [0, 0] = clean_model_clean_rationale (Baseline model, Baseline rationale)
      [0, 1] = clean_model_swap_rationale (Baseline model, Direct-swap rationale)
      [1, 0] = direct_swap_model_clean_rationale (Direct-swap model, Baseline rationale)
      [1, 1] = direct_swap_model_swap_rationale (Direct-swap model, Direct-swap rationale)
    """
    patterns = [
        (re.compile(r"^clean_model_clean_rationale_.*margin", re.IGNORECASE), (0, 0)),
        (re.compile(r"^clean_model_swap_rationale_.*margin", re.IGNORECASE), (0, 1)),
        (re.compile(r"^direct_swap_model_clean_rationale_.*margin", re.IGNORECASE), (1, 0)),
        (re.compile(r"^direct_swap_model_swap_rationale_.*margin", re.IGNORECASE), (1, 1)),
    ]
    rows: list[dict[str, str]] = []
    with diagnostics_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    result = np.zeros((2, 2))
    for pat, (i, j) in patterns:
        matching = [r for r in rows
                    if r["diagnostic"] == "rationale_conditioned_scoring" and pat.match(r["metric"])]
        if len(matching) != 1:
            raise ValueError(
                f"Pattern {pat.pattern!r} matched {len(matching)} rows in {diagnostics_csv}; "
                "expected exactly 1."
            )
        result[i, j] = float(matching[0]["value"])

    return result


def assert_against_paper_constants(T: dict[str, int], M: np.ndarray) -> None:
    """Assert loaded CSV values match paper_constants before drawing."""
    from src.evaluation.paper_constants import (
        DIAG_RAT_BASELINE_BASELINE_MEAN_RAW,
        DIAG_RAT_BASELINE_SWAP_MEAN_RAW,
        DIAG_RAT_SWAP_BASELINE_MEAN_RAW,
        DIAG_RAT_SWAP_SWAP_MEAN_RAW,
        MAIN_ZH_CLEAN_CORRECT,
        MAIN_ZH_DIRECT_SWAP_CORRECT,
        MAIN_ZH_N,
        MAIN_ZH_S_BROKEN,
        MAIN_ZH_S_REPAIRED,
        MAIN_ZH_STABLE_CORRECT,
        MAIN_ZH_STABLE_WRONG,
        MAIN_ZH_STABLE_WRONG_DIFFERENT,
        MAIN_ZH_STABLE_WRONG_SAME,
    )

    assert T["n"] == MAIN_ZH_N, f"n mismatch: {T['n']} vs {MAIN_ZH_N}"
    assert T["baseline_correct"] == MAIN_ZH_CLEAN_CORRECT, (
        f"baseline_correct mismatch: {T['baseline_correct']} vs {MAIN_ZH_CLEAN_CORRECT}"
    )
    assert T["direct_swap_correct"] == MAIN_ZH_DIRECT_SWAP_CORRECT, (
        f"direct_swap_correct mismatch: {T['direct_swap_correct']} vs {MAIN_ZH_DIRECT_SWAP_CORRECT}"
    )
    assert T["stable_correct"] == MAIN_ZH_STABLE_CORRECT
    assert T["broken"] == MAIN_ZH_S_BROKEN
    assert T["repaired"] == MAIN_ZH_S_REPAIRED
    assert T["stable_wrong"] == MAIN_ZH_STABLE_WRONG
    assert T["same_wrong"] == MAIN_ZH_STABLE_WRONG_SAME
    assert T["different_wrong"] == MAIN_ZH_STABLE_WRONG_DIFFERENT

    assert math.isclose(M[0, 0], DIAG_RAT_BASELINE_BASELINE_MEAN_RAW, abs_tol=1e-9), (
        f"M[0,0] mismatch: {M[0,0]} vs {DIAG_RAT_BASELINE_BASELINE_MEAN_RAW}"
    )
    assert math.isclose(M[0, 1], DIAG_RAT_BASELINE_SWAP_MEAN_RAW, abs_tol=1e-9), (
        f"M[0,1] mismatch: {M[0,1]} vs {DIAG_RAT_BASELINE_SWAP_MEAN_RAW}"
    )
    assert math.isclose(M[1, 0], DIAG_RAT_SWAP_BASELINE_MEAN_RAW, abs_tol=1e-9), (
        f"M[1,0] mismatch: {M[1,0]} vs {DIAG_RAT_SWAP_BASELINE_MEAN_RAW}"
    )
    assert math.isclose(M[1, 1], DIAG_RAT_SWAP_SWAP_MEAN_RAW, abs_tol=1e-9), (
        f"M[1,1] mismatch: {M[1,1]} vs {DIAG_RAT_SWAP_SWAP_MEAN_RAW}"
    )


def save_figure(fig: plt.Figure, stem: str, out_dir: pathlib.Path) -> None:
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(
        png_path,
        bbox_inches="tight",
        pad_inches=0.035,
        facecolor="white",
        dpi=300,
    )
    plt.close(fig)


def make_heatmap(margin_values: np.ndarray, out_dir: pathlib.Path) -> None:
    vmax = float(np.max(np.abs(margin_values)))
    cmap = colors.LinearSegmentedColormap.from_list(
        "muted_margin",
        ["#4f77a3", "#f7f7f4", "#b45f5c"],
    )
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    im = ax.imshow(margin_values, cmap=cmap, norm=norm, aspect="equal")

    ax.set_title("Rationale-conditioned scoring", pad=6, fontweight="regular")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline\nrationale", "Direct-swap\nrationale"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Baseline\nmodel", "Direct-swap\nmodel"])

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0, pad=5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(margin_values.shape[0]):
        for col in range(margin_values.shape[1]):
            value = margin_values[row, col]
            ax.text(
                col,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="semibold",
                color="#222222",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.045)
    cbar.set_label("Margin for baseline answer", rotation=270, labelpad=11)
    cbar.ax.tick_params(labelsize=7, length=2.5)
    cbar.outline.set_linewidth(0.5)

    save_figure(fig, "rationale_conditioned_heatmap", out_dir)


def _box(ax, cx, cy, w, h, label, facecolor, edgecolor="#6a6a6a", fontsize=6.45):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.75,
        zorder=4,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#222222",
        linespacing=1.05,
        zorder=5,
    )
    return {"cx": cx, "cy": cy, "w": w, "h": h}


def _pt_right(box, offset=0.0):
    return (box["cx"] + box["w"] / 2, box["cy"] + offset)


def _pt_left(box, offset=0.0):
    return (box["cx"] - box["w"] / 2, box["cy"] + offset)


def _arrow_width(count: int, baseline_correct: int) -> float:
    return 0.75 + 2.85 * (count / float(baseline_correct))


def _arrow(ax, start, end, count, baseline_correct, color="#9a9a9a", rad=0.0, alpha=0.72, zorder=2):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=_arrow_width(count, baseline_correct),
        color=color,
        alpha=alpha,
        shrinkA=2.5,
        shrinkB=2.5,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arrow)


def make_transition_flow_simple(T: dict[str, int], out_dir: pathlib.Path) -> None:
    bc = T["baseline_correct"]
    bw = T["baseline_wrong"]
    dc = T["direct_swap_correct"]
    dw = T["direct_swap_wrong"]
    sc = T["stable_correct"]
    br = T["broken"]
    rp = T["repaired"]
    sw = T["stable_wrong"]
    same = T["same_wrong"]
    diff = T["different_wrong"]
    n = T["n"]

    fig, ax = plt.subplots(figsize=(3.95, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors_by_group = {
        "neutral": "#f3f3ef",
        "correct": "#dceade",
        "broken": "#edd9d6",
        "repaired": "#dbe5f0",
        "wrong": "#eee4cf",
        "refine": "#f4ebd9",
        "outcome": "#eef0f2",
    }

    ax.text(0.09, 0.965, "Baseline", ha="center", va="center", fontsize=6.8, color="#333333")
    ax.text(0.365, 0.965, "Transition", ha="center", va="center", fontsize=6.8, color="#333333")
    ax.text(0.635, 0.965, "Direct-swap", ha="center", va="center", fontsize=6.8, color="#333333")
    ax.text(
        0.86,
        0.965,
        "Stable-wrong\nrefinement",
        ha="center",
        va="center",
        fontsize=6.4,
        linespacing=1.0,
        color="#333333",
    )

    boxes = {
        "bc": _box(ax, 0.09, 0.755, 0.155, 0.105, f"Correct\n{bc}", colors_by_group["neutral"]),
        "bw": _box(ax, 0.09, 0.345, 0.155, 0.105, f"Wrong\n{bw}", colors_by_group["neutral"]),
        "sc": _box(ax, 0.365, 0.835, 0.19, 0.092, f"Stable\ncorrect\n{sc}", colors_by_group["correct"]),
        "br": _box(ax, 0.365, 0.650, 0.19, 0.092, f"Broken\n{br}", colors_by_group["broken"]),
        "rp": _box(ax, 0.365, 0.470, 0.19, 0.092, f"Repaired\n{rp}", colors_by_group["repaired"]),
        "sw": _box(ax, 0.365, 0.285, 0.19, 0.092, f"Stable\nwrong\n{sw}", colors_by_group["wrong"]),
        "dc": _box(ax, 0.635, 0.715, 0.18, 0.105, f"Correct\n{dc}", colors_by_group["outcome"]),
        "dw": _box(ax, 0.635, 0.420, 0.18, 0.105, f"Wrong\n{dw}", colors_by_group["outcome"]),
        "same": _box(ax, 0.86, 0.250, 0.185, 0.086, f"Same wrong\n{same}",
                     colors_by_group["refine"], fontsize=6.15),
        "diff": _box(ax, 0.86, 0.105, 0.185, 0.086, f"Different\nwrong\n{diff}",
                     colors_by_group["refine"], fontsize=6.0),
    }

    _arrow(ax, _pt_right(boxes["bc"], 0.022), _pt_left(boxes["sc"]), sc, bc, "#93aa93", rad=0.03)
    _arrow(ax, _pt_right(boxes["bc"], -0.022), _pt_left(boxes["br"]), br, bc, "#b99390", rad=-0.02)
    _arrow(ax, _pt_right(boxes["bw"], 0.022), _pt_left(boxes["rp"]), rp, bc, "#8fa3ba", rad=0.03)
    _arrow(ax, _pt_right(boxes["bw"], -0.022), _pt_left(boxes["sw"]), sw, bc, "#b8a77e", rad=-0.02)

    _arrow(ax, _pt_right(boxes["sc"]), _pt_left(boxes["dc"], 0.024), sc, bc, "#93aa93", rad=-0.03)
    _arrow(ax, _pt_right(boxes["rp"]), _pt_left(boxes["dc"], -0.024), rp, bc, "#8fa3ba", rad=0.10)
    _arrow(ax, _pt_right(boxes["br"]), _pt_left(boxes["dw"], 0.024), br, bc, "#b99390", rad=-0.03)
    _arrow(ax, _pt_right(boxes["sw"], 0.026), _pt_left(boxes["dw"], -0.024), sw, bc, "#b8a77e", rad=0.05)

    _arrow(ax, _pt_right(boxes["sw"], -0.006), _pt_left(boxes["same"]), same, bc, "#b8a77e",
           rad=-0.06, alpha=0.78)
    _arrow(ax, _pt_right(boxes["sw"], -0.032), _pt_left(boxes["diff"]), diff, bc, "#b8a77e",
           rad=-0.10, alpha=0.78)

    ax.text(0.013, 0.025, f"n = {n}", ha="left", va="bottom", fontsize=6.8, color="#555555")

    save_figure(fig, "transition_flow_simple", out_dir)


def _ribbon(ax, x0, y0, x1, y1, width, color, alpha=0.58, zorder=1):
    dx = x1 - x0
    c1x = x0 + 0.48 * dx
    c2x = x1 - 0.48 * dx
    half = width / 2

    verts = [
        (x0, y0 + half),
        (c1x, y0 + half),
        (c2x, y1 + half),
        (x1, y1 + half),
        (x1, y1 - half),
        (c2x, y1 - half),
        (c1x, y0 - half),
        (x0, y0 - half),
        (x0, y0 + half),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(
        MplPath(verts, codes),
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)


def _sankey_node(ax, key, x, y, count, label, color, *, width=0.128, height=None, fontsize=None):
    scale = 0.00218
    height = max(0.052, count * scale) if height is None else height
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.007,rounding_size=0.010",
        facecolor=color,
        edgecolor="#666666",
        linewidth=0.65,
        zorder=3,
    )
    ax.add_patch(patch)
    if fontsize is None:
        fontsize = 5.65 if count >= 30 else 5.35
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.0,
        color="#202020",
        zorder=4,
    )
    return {"key": key, "x": x, "y": y, "w": width, "h": height, "count": count}


def make_transition_sankey(T: dict[str, int], out_dir: pathlib.Path) -> None:
    bc = T["baseline_correct"]
    bw = T["baseline_wrong"]
    dc = T["direct_swap_correct"]
    dw = T["direct_swap_wrong"]
    sc = T["stable_correct"]
    br = T["broken"]
    rp = T["repaired"]
    sw = T["stable_wrong"]
    same = T["same_wrong"]
    diff = T["different_wrong"]
    n = T["n"]

    fig, ax = plt.subplots(figsize=(4.35, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    scale = 0.00218
    nodes = {
        "bc": _sankey_node(ax, "bc", 0.08, 0.76, bc, f"Baseline\ncorrect\n{bc}", "#f3f3ef"),
        "bw": _sankey_node(ax, "bw", 0.08, 0.32, bw, f"Baseline\nwrong\n{bw}", "#f3f3ef"),
        "sc": _sankey_node(ax, "sc", 0.34, 0.81, sc, f"Stable\ncorrect\n{sc}", "#dceade"),
        "br": _sankey_node(ax, "br", 0.34, 0.60, br, f"Broken\n{br}", "#edd9d6"),
        "rp": _sankey_node(ax, "rp", 0.34, 0.43, rp, f"Repaired\n{rp}", "#dbe5f0"),
        "sw": _sankey_node(ax, "sw", 0.34, 0.22, sw, f"Stable\nwrong\n{sw}", "#eee4cf"),
        "same": _sankey_node(ax, "same", 0.62, 0.32, same, f"Same\nwrong\n{same}", "#f4ebd9",
                              width=0.145, height=0.088, fontsize=5.15),
        "diff": _sankey_node(ax, "diff", 0.62, 0.145, diff, f"Different\nwrong\n{diff}", "#f4ebd9",
                              width=0.128, height=0.176, fontsize=5.25),
        "dc": _sankey_node(ax, "dc", 0.90, 0.70, dc, f"Direct-swap\ncorrect\n{dc}", "#eef0f2"),
        "dw": _sankey_node(ax, "dw", 0.90, 0.28, dw, f"Direct-swap\nwrong\n{dw}", "#eef0f2"),
    }

    def right(key):
        node = nodes[key]
        return node["x"] + node["w"] / 2

    def left(key):
        node = nodes[key]
        return node["x"] - node["w"] / 2

    def width(count):
        return max(0.024, count * scale)

    bc_top = nodes["bc"]["y"] + nodes["bc"]["h"] / 2
    bc_to_sc = bc_top - width(sc) / 2
    bc_to_br = nodes["bc"]["y"] - nodes["bc"]["h"] / 2 + width(br) / 2

    bw_top = nodes["bw"]["y"] + nodes["bw"]["h"] / 2
    bw_to_rp = bw_top - width(rp) / 2
    bw_to_sw = nodes["bw"]["y"] - nodes["bw"]["h"] / 2 + width(sw) / 2

    dc_top = nodes["dc"]["y"] + nodes["dc"]["h"] / 2
    dc_from_sc = dc_top - width(sc) / 2
    dc_from_rp = nodes["dc"]["y"] - nodes["dc"]["h"] / 2 + width(rp) / 2

    dw_top = nodes["dw"]["y"] + nodes["dw"]["h"] / 2
    dw_from_br = dw_top - width(br) / 2
    dw_from_same = dw_top - width(br) - width(same) / 2
    dw_from_diff = nodes["dw"]["y"] - nodes["dw"]["h"] / 2 + width(diff) / 2

    sw_top = nodes["sw"]["y"] + nodes["sw"]["h"] / 2
    sw_to_same = sw_top - width(same) / 2
    sw_to_diff = nodes["sw"]["y"] - nodes["sw"]["h"] / 2 + width(diff) / 2

    _ribbon(ax, right("bc"), bc_to_sc, left("sc"), nodes["sc"]["y"], width(sc), "#93aa93")
    _ribbon(ax, right("bc"), bc_to_br, left("br"), nodes["br"]["y"], width(br), "#b99390")
    _ribbon(ax, right("bw"), bw_to_rp, left("rp"), nodes["rp"]["y"], width(rp), "#8fa3ba")
    _ribbon(ax, right("bw"), bw_to_sw, left("sw"), nodes["sw"]["y"], width(sw), "#b8a77e")

    _ribbon(ax, right("sc"), nodes["sc"]["y"], left("dc"), dc_from_sc, width(sc), "#93aa93")
    _ribbon(ax, right("rp"), nodes["rp"]["y"], left("dc"), dc_from_rp, width(rp), "#8fa3ba")
    _ribbon(ax, right("br"), nodes["br"]["y"], left("dw"), dw_from_br, width(br), "#b99390")
    _ribbon(ax, right("sw"), sw_to_same, left("same"), nodes["same"]["y"], width(same), "#b8a77e",
            alpha=0.62)
    _ribbon(ax, right("sw"), sw_to_diff, left("diff"), nodes["diff"]["y"], width(diff), "#b8a77e",
            alpha=0.62)
    _ribbon(ax, right("same"), nodes["same"]["y"], left("dw"), dw_from_same, width(same), "#b8a77e",
            alpha=0.52)
    _ribbon(ax, right("diff"), nodes["diff"]["y"], left("dw"), dw_from_diff, width(diff), "#b8a77e",
            alpha=0.52)

    ax.text(0.08, 0.965, "Baseline", ha="center", va="center", fontsize=6.8, color="#333333")
    ax.text(0.34, 0.965, "Transition", ha="center", va="center", fontsize=6.8, color="#333333")
    ax.text(0.62, 0.965, "Stable-wrong\nidentity", ha="center", va="center",
            fontsize=6.35, linespacing=1.0, color="#333333")
    ax.text(0.90, 0.965, "Direct-swap", ha="center", va="center", fontsize=6.8, color="#333333")
    ax.text(0.015, 0.025, f"n = {n}", ha="left", va="bottom", fontsize=6.8, color="#555555")

    save_figure(fig, "transition_sankey", out_dir)


def write_latex_snippets(out_dir: pathlib.Path) -> None:
    snippets = r"""
% Candidate figure snippets for the answer redistribution paper.
% Paths assume this directory is kept as artifacts/figures/ at the repo root.

% Heatmap replacement for Table 3.
\begin{figure}[htbp]
\centering
\includegraphics[width=0.78\columnwidth]{artifacts/figures/rationale_conditioned_heatmap.pdf}
\caption{Rationale-conditioned final-answer margins on answer-changed examples. Positive values indicate preference for the baseline answer, while negative values indicate preference for the direct-swap answer. Margins track the rationale source more strongly than the model condition.}
\label{fig:rationale_conditioned}
\end{figure}

% Simple flow replacement candidate for Table 1.
\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{artifacts/figures/transition_flow_simple.pdf}
\caption{Answer redistribution under direct middle-layer replacement on Chinese MGSM. Broken and repaired examples partially offset in aggregate accuracy, while most stable-wrong examples move to different incorrect answers.}
\label{fig:transition_flow}
\end{figure}

% Experimental Sankey-style alternative for Table 1.
\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{artifacts/figures/transition_sankey.pdf}
\caption{Sankey-style view of answer redistribution under direct middle-layer replacement. Flow widths indicate counts of examples moving between baseline correctness, transition groups, stable-wrong answer identity, and direct-swap correctness.}
\label{fig:transition_sankey}
\end{figure}
"""
    (out_dir / "figure_latex_snippets.tex").write_text(dedent(snippets).lstrip(), encoding="utf-8")


def write_readme(sankey_ok: bool, out_dir: pathlib.Path,
                 ci_strings_from_constants: dict[str, str]) -> None:
    from src.evaluation.paper_constants import (
        DIAG_RAT_BASELINE_BASELINE_CI95_HIGH_ROUNDED,
        DIAG_RAT_BASELINE_BASELINE_CI95_LOW_ROUNDED,
        DIAG_RAT_BASELINE_SWAP_CI95_HIGH_ROUNDED,
        DIAG_RAT_BASELINE_SWAP_CI95_LOW_ROUNDED,
        DIAG_RAT_SWAP_BASELINE_CI95_HIGH_ROUNDED,
        DIAG_RAT_SWAP_BASELINE_CI95_LOW_ROUNDED,
        DIAG_RAT_SWAP_SWAP_CI95_HIGH_ROUNDED,
        DIAG_RAT_SWAP_SWAP_CI95_LOW_ROUNDED,
    )
    status_line = (
        "The Sankey-style flow was generated as `transition_sankey.pdf/png`."
        if sankey_ok
        else "Sankey-style generation failed; see `transition_sankey_STATUS.txt`."
    )
    readme = f"""
# Generated Paper Figures

Standalone candidate figures for the COSE461 final project paper on answer redistribution under direct middle-layer replacement.

## Files

- `rationale_conditioned_heatmap.pdf/png`: 2x2 rationale-conditioned scoring heatmap. Safest replacement for Table 3.
- `transition_flow_simple.pdf/png`: static transition decomposition flow. Alternative replacement candidate for Table 1.
- `transition_sankey.pdf/png`: experimental Sankey-style transition flow. {status_line}
- `figure_latex_snippets.tex`: LaTeX snippets and replacement notes.
- `generate_paper_figures.py`: regeneration script.

## Input values used

### Rationale-conditioned scoring

Rows are model condition; columns are rationale source. Values are mean margin for the baseline answer.
All values loaded from `artifacts/diagnostics/diagnostics.csv` and verified against paper_constants.py.

Supplemental CIs recorded for paper text or appendix:

| Model condition | Rationale source | CI |
| --- | --- | --- |
| Baseline model | Baseline rationale | [{DIAG_RAT_BASELINE_BASELINE_CI95_LOW_ROUNDED}, {DIAG_RAT_BASELINE_BASELINE_CI95_HIGH_ROUNDED}] |
| Baseline model | Direct-swap rationale | [{DIAG_RAT_BASELINE_SWAP_CI95_LOW_ROUNDED}, {DIAG_RAT_BASELINE_SWAP_CI95_HIGH_ROUNDED}] |
| Direct-swap model | Baseline rationale | [{DIAG_RAT_SWAP_BASELINE_CI95_LOW_ROUNDED}, {DIAG_RAT_SWAP_BASELINE_CI95_HIGH_ROUNDED}] |
| Direct-swap model | Direct-swap rationale | [{DIAG_RAT_SWAP_SWAP_CI95_LOW_ROUNDED}, {DIAG_RAT_SWAP_SWAP_CI95_HIGH_ROUNDED}] |

### Transition decomposition

Chinese MGSM. All values loaded from `artifacts/main/main_metrics.csv` and verified against paper_constants.py.

## How to regenerate

```bash
bash scripts/reproduce_figures.sh
```

The script requires `matplotlib`, `numpy`, and Pillow.

## Recommended figure choice

- Safest replacement: heatmap for Table 3.
- Alternative replacement: simple flow for Table 1.
- Experimental alternative: Sankey-style flow.

## Warnings

- Do not include both an old table and its replacement figure in the main paper unless explicitly needed.
- Replacing Table 3 loses explicit CI display unless CIs are mentioned in text or appendix.
- Replacing Table 1 with a flow diagram should preserve all transition counts in labels.
- Do not change the paper numbers or introduce new analysis when swapping figures into the paper.
"""
    (out_dir / "README.md").write_text(dedent(readme).lstrip(), encoding="utf-8")


def self_check(sankey_ok: bool, out_dir: pathlib.Path) -> list[str]:
    expected = [
        "rationale_conditioned_heatmap.pdf",
        "rationale_conditioned_heatmap.png",
        "transition_flow_simple.pdf",
        "transition_flow_simple.png",
        "figure_latex_snippets.tex",
        "README.md",
    ]
    if sankey_ok:
        expected.extend(["transition_sankey.pdf", "transition_sankey.png"])
    else:
        expected.append("transition_sankey_STATUS.txt")

    notes: list[str] = []
    for filename in expected:
        path = out_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing expected output: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"Expected nonzero file size for {path}")
        notes.append(f"{filename}: {size} bytes")

    for filename in [
        "rationale_conditioned_heatmap.png",
        "transition_flow_simple.png",
        "transition_sankey.png" if sankey_ok else None,
    ]:
        if filename is None:
            continue
        path = out_dir / filename
        with Image.open(path) as image:
            w, h = image.size
        if w < 600 or h < 450:
            raise RuntimeError(f"PNG dimensions look too small for {path}: {w}x{h}")
        if w > 2200 or h > 1800:
            raise RuntimeError(f"PNG dimensions look unexpectedly large for {path}: {w}x{h}")
        notes.append(f"{filename}: {w}x{h} px")

    for filename in [
        "rationale_conditioned_heatmap.pdf",
        "transition_flow_simple.pdf",
        "transition_sankey.pdf" if sankey_ok else None,
    ]:
        if filename is None:
            continue
        path = out_dir / filename
        if path.stat().st_size < 1000:
            raise RuntimeError(f"PDF size looks too small for {path}")

    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper figures from CSV artifacts.")
    parser.add_argument(
        "--artifacts-root",
        type=pathlib.Path,
        default=None,
        help="Path to artifacts/ directory. Defaults to <repo_root>/artifacts.",
    )
    args = parser.parse_args()

    artifacts_root = args.artifacts_root or ARTIFACTS_ROOT
    main_metrics_csv = artifacts_root / "main" / "main_metrics.csv"
    diagnostics_csv = artifacts_root / "diagnostics" / "diagnostics.csv"
    out_dir = artifacts_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    T = load_transitions(main_metrics_csv)
    M = load_margin_values(diagnostics_csv)
    assert_against_paper_constants(T, M)

    make_heatmap(M, out_dir)
    make_transition_flow_simple(T, out_dir)

    sankey_ok = True
    status_path = out_dir / "transition_sankey_STATUS.txt"
    if status_path.exists():
        status_path.unlink()
    try:
        make_transition_sankey(T, out_dir)
    except Exception as exc:
        sankey_ok = False
        status_path.write_text(
            dedent(
                f"""
                Sankey-style figure generation failed.

                Error:
                {type(exc).__name__}: {exc}

                To regenerate, install the plotting dependencies:

                    python -m pip install matplotlib numpy pillow

                Then rerun:

                    bash scripts/reproduce_figures.sh
                """
            ).lstrip(),
            encoding="utf-8",
        )

    write_latex_snippets(out_dir)
    write_readme(sankey_ok, out_dir, {})
    check_notes = self_check(sankey_ok, out_dir)

    print("Self-check:")
    for note in check_notes:
        print(f"- {note}")
    print()
    print("Generated figures:")
    print("- rationale_conditioned_heatmap.pdf/png")
    print("- transition_flow_simple.pdf/png")
    if sankey_ok:
        print("- transition_sankey.pdf/png")
    else:
        print("- transition_sankey_STATUS.txt")
    print("Written to:", out_dir)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

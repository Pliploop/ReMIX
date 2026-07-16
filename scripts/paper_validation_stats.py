#!/usr/bin/env python
"""Generate the paper's *validation* figures/tables (LLM-as-judge + cross-LLM agreement).

Reuses the Streamlit admin agreement layer (`human_validation_app._agreement_rows`,
`_admin_question_rows`, ...) so the paper numbers match the app exactly, and the
figure styling from `paper_data_stats` (Okabe-Ito palette, sans-serif, low-opacity
bars with black contours). One publication-grade plot per PDF into `paper/figures/`,
LaTeX tables into `paper/tables/`.

Inputs are the frozen per-judge rating files written by `run_llm_validation_judge.sh`:
each dataset supplies a Qwen and a Gemma full-validation-slice file, judged on the
identical item set with the identical rubric.

Example:
  PYTHONPATH=src python scripts/paper_validation_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402

from jamendo_instruct.demo.human_validation_app import (  # noqa: E402
    _admin_question_rows,
    _agreement_rows,
    _item_mean_scores,
    _read_jsonl_records,
)
from jamendo_instruct.demo.validation_rubric import RATING_QUESTIONS  # noqa: E402

from paper_data_stats import (  # noqa: E402
    BAR,
    BLUE,
    ORANGE,
    QUESTION_SHORT,
    _bar_labels,
    _new,
    nice,
    setup_style,
)

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "paper" / "figures"
TAB_DIR = REPO / "paper" / "tables"

# Judge display names (short, for legends/columns).
QWEN = "Qwen3.6-27B"
GEMMA = "Gemma-4-31B"

# Frozen per-judge rating files: (dataset label, pretty name, {judge: path}).
DATASETS: List[Dict[str, Any]] = [
    {
        "label": "music4all",
        "pretty": "Music4All",
        "validation_dir": Path(
            "/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1"
            "/instructions_axis_focused_5/validation"
        ),
        "judges": {
            QWEN: "llm_ratings.jsonl",
            GEMMA: "llm_ratings_gemma_full.jsonl",
        },
    },
    {
        "label": "mtg_jamendo",
        "pretty": "MTG-Jamendo",
        "validation_dir": Path(
            "/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1"
            "/instructions_axis_focused_5/validation"
        ),
        "judges": {
            QWEN: "llm_ratings_qwen_full.jsonl",
            GEMMA: "llm_ratings_gemma_full.jsonl",
        },
    },
]

QUESTION_ORDER = [str(q["id"]) for q in RATING_QUESTIONS]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_ratings(ds: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for judge, fname in ds["judges"].items():
        out[judge] = _read_jsonl_records(ds["validation_dir"] / fname)
    return out


def accept_rate(records: Sequence[Dict[str, Any]], qid: str) -> float | None:
    """Fraction of parsed, applicable ratings with score >= 4 for one question."""
    scored = []
    for r in records:
        a = dict((r.get("answers", {}) or {}).get(qid, {}) or {})
        if a.get("cannot_judge") or a.get("not_applicable"):
            continue
        s = a.get("score")
        if isinstance(s, int):
            scored.append(s)
    if not scored:
        return None
    return sum(1 for s in scored if s >= 4) / len(scored)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_accept_by_question(ds: Dict[str, Any], ratings: Dict[str, List[Dict[str, Any]]]) -> None:
    """Grouped horizontal bars: per-question accept rate, one bar per judge."""
    qids = QUESTION_ORDER[::-1]
    labels = [QUESTION_SHORT.get(q, nice(q)) for q in qids]
    judges = list(ds["judges"])
    colors = {judges[0]: BLUE, judges[1]: ORANGE}
    y = np.arange(len(qids))
    h = 0.38
    fig, ax = _new(6.6, 4.4)
    for i, judge in enumerate(judges):
        vals = [accept_rate(ratings[judge], q) or 0.0 for q in qids]
        off = (i - 0.5) * h
        b = ax.barh(y + off, vals, height=h, color=colors[judge], label=judge, **BAR)
        _bar_labels(ax, b, vals, fmt="{:.0%}", horizontal=True)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Acceptance rate  (score $\\geq 4$)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / f"{ds['label']}_val_accept_by_question.pdf")
    plt.close(fig)


def fig_rubric_dist(ds: Dict[str, Any], judge: str, records: Sequence[Dict[str, Any]]) -> None:
    """Stacked 1-5 score distribution per question for one judge (RdYlGn)."""
    rows = {r["question_id"]: r for r in _admin_question_rows(records)}
    qids = QUESTION_ORDER[::-1]
    labels = [QUESTION_SHORT.get(q, nice(q)) for q in qids]
    colors = plt.get_cmap("RdYlGn")(np.linspace(0.12, 0.88, 5))
    fig, ax = _new(6.6, 4.2)
    left = np.zeros(len(qids))
    for s in range(1, 6):
        vals = np.array([float(rows[q][f"score_{s}"]) for q in qids])
        ax.barh(labels, vals, left=left, color=colors[s - 1], edgecolor="black",
                linewidth=0.5, label=f"{s}")
        left += vals
    ax.set_xlabel("Number of ratings")
    ax.legend(title="Score (1–5)", ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / f"{ds['label']}_val_rubric_{_judge_slug(judge)}.pdf")
    plt.close(fig)


def fig_agreement_ac1(agreements: Dict[str, List[Dict[str, Any]]]) -> None:
    """Per-question Gwet AC1 on the accept decision, grouped bars per dataset."""
    qids = QUESTION_ORDER[::-1]
    labels = [QUESTION_SHORT.get(q, nice(q)) for q in qids]
    dss = list(agreements)
    colors = {dss[0]: BLUE, dss[1]: ORANGE}
    y = np.arange(len(qids))
    h = 0.38
    fig, ax = _new(6.6, 4.4)
    for i, ds_pretty in enumerate(dss):
        row_by_q = {r["question_id"]: r for r in agreements[ds_pretty]}
        vals = [float(row_by_q[q].get("accept_ac1") or 0.0) for q in qids]
        off = (i - 0.5) * h
        b = ax.barh(y + off, vals, height=h, color=colors[ds_pretty], label=ds_pretty, **BAR)
        _bar_labels(ax, b, vals, fmt="{:.2f}", horizontal=True)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Cross-judge agreement  (Gwet's AC1 on accept)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / "remix_val_agreement_ac1.pdf")
    plt.close(fig)


def _paired_scores(ratings: Dict[str, List[Dict[str, Any]]], judges: Sequence[str], qid: str):
    """Per-item integer scores for one question, aligned across the two judges."""
    left = _item_mean_scores(ratings[judges[0]], qid)
    right = _item_mean_scores(ratings[judges[1]], qid)
    keys = sorted(set(left) & set(right))
    xs = np.array([left[k] for k in keys])
    ys = np.array([right[k] for k in keys])
    return xs, ys


def fig_joint_scatter(ds: Dict[str, Any], ratings: Dict[str, List[Dict[str, Any]]],
                      qid: str = "overall_validity") -> None:
    """Bubble scatter of the two judges' scores on one question: marker area and
    color encode how many items sit at each (Qwen, Gemma) integer cell. The honest
    form of a scatterplot for discrete 1-5 ratings; the dashed diagonal is exact
    agreement, and off-diagonal mass shows the direction of disagreement."""
    judges = list(ds["judges"])
    xs, ys = _paired_scores(ratings, judges, qid)
    if len(xs) == 0:
        return
    n = len(xs)
    grid = np.zeros((5, 5), dtype=float)  # [qwen-1, gemma-1]
    for x, y in zip(xs, ys):
        grid[int(round(x)) - 1, int(round(y)) - 1] += 1
    gx, gy = np.meshgrid(np.arange(1, 6), np.arange(1, 6), indexing="ij")
    mask = grid > 0
    counts = grid[mask]
    frac = counts / n
    sizes = 90 + 1450 * (counts / counts.max())

    fig, ax = _new(4.9, 4.3)
    ax.plot([0.5, 5.5], [0.5, 5.5], ls="--", lw=1.0, color="#999", zorder=0)
    sc = ax.scatter(gx[mask], gy[mask], s=sizes, c=counts, cmap="Blues",
                    edgecolor="black", linewidth=0.8, alpha=0.9, zorder=2)
    for x, y, f in zip(gx[mask], gy[mask], frac):
        if f >= 0.02:
            ax.annotate(f"{100 * f:.0f}%", (x, y), ha="center", va="center",
                        fontsize=8, color="black" if f < 0.35 else "white", zorder=3)
    exact = float(np.mean(xs == ys))
    within1 = float(np.mean(np.abs(xs - ys) <= 1))
    ax.text(0.03, 0.97, f"exact {100 * exact:.0f}%\n$\\pm$1 {100 * within1:.0f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.set_xlabel(f"{judges[0]}  score")
    ax.set_ylabel(f"{judges[1]}  score")
    ax.set_aspect("equal")
    ax.grid(True, color="#eeeeee")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Items")
    fig.savefig(FIG_DIR / f"{ds['label']}_val_joint_{qid}.pdf")
    plt.close(fig)


def fig_mean_dumbbell(ds: Dict[str, Any], agreement_rows: List[Dict[str, Any]]) -> None:
    """Dumbbell of per-question mean score for the two judges, sorted by Qwen mean.
    Makes the per-question leniency gap (bias) legible across the rubric."""
    judges = list(ds["judges"])
    lslug, rslug = _judge_slug(judges[0]), _judge_slug(judges[1])
    rows = [r for r in agreement_rows if r.get("n_items")]
    rows.sort(key=lambda r: r.get(f"{lslug}_mean", 0.0))
    labels = [QUESTION_SHORT.get(r["question_id"], nice(r["question_id"])) for r in rows]
    y = np.arange(len(rows))
    lm = [r[f"{lslug}_mean"] for r in rows]
    rm = [r[f"{rslug}_mean"] for r in rows]
    fig, ax = _new(6.2, 4.2)
    for yi, a, b in zip(y, lm, rm):
        ax.plot([a, b], [yi, yi], color="#bbbbbb", lw=2.0, zorder=1)
    ax.scatter(lm, y, s=70, color=BLUE, edgecolor="black", linewidth=0.7, label=judges[0], zorder=2)
    ax.scatter(rm, y, s=70, color=ORANGE, edgecolor="black", linewidth=0.7, label=judges[1], zorder=2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean rubric score  (1–5)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / f"{ds['label']}_val_mean_dumbbell.pdf")
    plt.close(fig)


def _judge_slug(judge: str) -> str:
    return judge.split("-")[0].lower().replace(".", "")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float | None) -> str:
    return "--" if x is None else f"{100 * x:.1f}"


def _fmt(x: float | None, nd: int = 2) -> str:
    return "--" if x is None else f"{x:+.{nd}f}" if nd == 2 and x is not None else f"{x:.{nd}f}"


def table_judge_accept(all_ratings: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
    """Per-question acceptance rate (%) for each dataset x judge."""
    cols = []  # (dataset pretty, judge)
    for ds in DATASETS:
        for judge in ds["judges"]:
            cols.append((ds["pretty"], judge))
    header_top = " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{ds}}}"
                                    for ds in [d["pretty"] for d in DATASETS]) + " \\\\"
    header_judges = "Question & " + " & ".join(j for _, j in cols) + " \\\\"

    lines = [
        "% Auto-generated by scripts/paper_validation_stats.py",
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        header_top,
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        header_judges,
        "\\midrule",
    ]
    for qid in QUESTION_ORDER:
        cells = []
        for ds in DATASETS:
            for judge in ds["judges"]:
                cells.append(_fmt_pct(accept_rate(all_ratings[ds["pretty"]][judge], qid)))
        name = QUESTION_SHORT.get(qid, nice(qid))
        row = f"{name} & " + " & ".join(cells) + " \\\\"
        if qid == "overall_validity":
            lines.append("\\midrule")
            row = f"\\textbf{{{name}}} & " + " & ".join(f"\\textbf{{{c}}}" for c in cells) + " \\\\"
        lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB_DIR / "val_judge_accept.tex").write_text("\n".join(lines) + "\n")


def table_cross_agreement(agreements: Dict[str, List[Dict[str, Any]]]) -> None:
    """Per-question cross-judge agreement: accept-agree %, AC1, QWK, Spearman, bias."""
    dss = list(agreements)
    metrics = [("Agr.\\%", "accept_agree_rate", "pct"), ("AC1", "accept_ac1", "f2"),
               ("QWK", "quadratic_kappa", "f2"), ("$\\rho$", "spearman_r", "f2")]
    ncols = len(dss) * len(metrics)
    header_top = " & " + " & ".join(f"\\multicolumn{{{len(metrics)}}}{{c}}{{{ds}}}" for ds in dss) + " \\\\"
    cmids = []
    start = 2
    for _ in dss:
        cmids.append(f"\\cmidrule(lr){{{start}-{start + len(metrics) - 1}}}")
        start += len(metrics)
    header_metrics = "Question & " + " & ".join(m[0] for _ in dss for m in metrics) + " \\\\"

    lines = [
        "% Auto-generated by scripts/paper_validation_stats.py",
        "\\begin{tabular}{l" + "c" * ncols + "}",
        "\\toprule",
        header_top,
        "".join(cmids),
        header_metrics,
        "\\midrule",
    ]
    rows_by_ds = {ds: {r["question_id"]: r for r in agreements[ds]} for ds in dss}
    for qid in QUESTION_ORDER:
        cells = []
        for ds in dss:
            r = rows_by_ds[ds].get(qid, {})
            for _, key, kind in metrics:
                v = r.get(key)
                if kind == "pct":
                    cells.append(_fmt_pct(v))
                elif kind == "s2":
                    cells.append("--" if v is None else f"{v:+.2f}")
                else:
                    cells.append("--" if v is None else f"{v:.2f}")
        name = QUESTION_SHORT.get(qid, nice(qid))
        if qid == "overall_validity":
            lines.append("\\midrule")
            lines.append(f"\\textbf{{{name}}} & " + " & ".join(f"\\textbf{{{c}}}" for c in cells) + " \\\\")
        else:
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB_DIR / "val_cross_agreement.tex").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    console = Console()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    all_ratings: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    agreements: Dict[str, List[Dict[str, Any]]] = {}

    for ds in DATASETS:
        ratings = load_ratings(ds)
        all_ratings[ds["pretty"]] = ratings
        judges = list(ds["judges"])
        n = {j: len(ratings[j]) for j in judges}
        console.print(f"[bold]{ds['pretty']}[/bold]: " + ", ".join(f"{j}={n[j]}" for j in judges))

        # Cross-LLM agreement (Qwen as left/reference, Gemma as right).
        agreements[ds["pretty"]] = _agreement_rows(
            ratings[judges[0]], ratings[judges[1]],
            left_label=_judge_slug(judges[0]), right_label=_judge_slug(judges[1]),
        )

        # Per-dataset figures.
        fig_accept_by_question(ds, ratings)
        fig_joint_scatter(ds, ratings, "overall_validity")
        fig_mean_dumbbell(ds, agreements[ds["pretty"]])
        for judge in judges:
            fig_rubric_dist(ds, judge, ratings[judge])

        # Console summary: overall accept + overall-validity agreement.
        ov = next(r for r in agreements[ds["pretty"]] if r["question_id"] == "overall_validity")
        for judge in judges:
            console.print(f"  accept[{judge}] overall = {_fmt_pct(accept_rate(ratings[judge], 'overall_validity'))}%")
        console.print(f"  overall-validity agreement: agree={_fmt_pct(ov.get('accept_agree_rate'))}%, "
                      f"AC1={ov.get('accept_ac1')}, QWK={ov.get('quadratic_kappa')}, "
                      f"rho={ov.get('spearman_r')}, bias={ov.get('mean_diff')}, n={ov.get('n_items')}")

    fig_agreement_ac1(agreements)
    table_judge_accept(all_ratings)
    table_cross_agreement(agreements)
    console.print(f"[green]Wrote figures to {FIG_DIR} and tables to {TAB_DIR}[/green]")


if __name__ == "__main__":
    main()

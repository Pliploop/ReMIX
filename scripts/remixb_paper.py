#!/usr/bin/env python
"""ReMIX-B results -> paper table + figures. Re-run whenever a baseline lands.

  python scripts/remixb_paper.py                       # results/remix_b/music4all -> paper/
  python scripts/remixb_paper.py --results-dir results/remix_b/music4all --out paper

Reads every <baseline>.json in --results-dir (written by run_remix_b.py; use
`--as NAME` there to add a ReMIX-C checkpoint as its own row) and writes
  paper/tables/remixb_results.tex        grouped table, best non-oracle score in bold
  paper/figures/remixb_cost_quality.pdf  nDCG@10 vs query-time TFLOP, Pareto frontier
  paper/figures/remixb_precision_recall.pdf  nDCG@10 vs R@100, dot area ~ log TFLOP
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import EDGE, FS_SMALL, GREY, HALF, INK, apply  # noqa: E402
from remixb_table import GROUPS, write_table  # noqa: E402

FAMILY_COLOR = {"Reference": GREY, "Seed only": "#56B4E9", "Instruction only": "#009E73",
                "Composed, no training": "#0072B2", "LLM-assisted": "#D55E00", "Trained on ReMIX": "#CC79A7"}
FLOOR = 1e-2  # TFLOP shown for baselines with (almost) no query-time compute
SHORT = {"random": "Random", "seed_audio_nn": "Seed audio", "instruction_text": "Instruction",
         "late_fusion": "Late fusion", "llm_caption_rewrite": "LLM rewrite", "hybrid_score_fusion": "Hybrid fusion",
         "llm_pointwise_rerank": "LLM rerank", "remix_c": "ReMIX-C (filtered)",
         "remix_c_unfiltered": "ReMIX-C", "bm25": "BM25", "analogy_steering": "Analogy"}
SKIP = {"target_caption_oracle", "remix_c_untrained"}  # oracle is annotated; untrained sits on random


def load(results_dir: Path) -> dict:
    out = {}
    for p in sorted(results_dir.glob("*.json")):
        if p.name != "results.json":
            d = json.loads(p.read_text())
            out[d["baseline"]] = d
    return out


def family(name: str) -> str:
    return next((g for g, rows in GROUPS if any(k == name for k, _ in rows)), "Composed, no training")


def _points(res):
    for name, d in res.items():
        if name in SKIP:
            continue
        m = d["metrics"]
        yield name, max(d.get("tflops", 0.0), FLOOR), m["nDCG@10"], m["R(rel=3)@100"], family(name)


def _style(ax):
    ax.grid(True)
    ax.tick_params(length=2)


# per-figure label offsets (points) for crowded neighbours; default (4, 3), left-aligned
OFFSETS = {
    "cost": {"remix_c_unfiltered": (-5, 4, "right"), "remix_c": (5, -7, "left"), "hybrid_score_fusion": (4, 3, "left"),
             "llm_caption_rewrite": (5, -6, "left"), "llm_pointwise_rerank": (-5, 0, "right")},
    "pr": {"remix_c_unfiltered": (5, 3, "left"), "remix_c": (5, -8, "left"), "llm_caption_rewrite": (-5, 3, "right"),
           "hybrid_score_fusion": (5, 0, "left"), "random": None},
}


def _label(ax, name, x, y, fig):
    off = OFFSETS[fig].get(name, (4, 3, "left"))
    if name in SHORT and off is not None:
        dx, dy, ha = off
        ax.annotate(SHORT[name], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=FS_SMALL,
                    color=INK, ha=ha, va="center")


def fig_cost_quality(res, path: Path) -> None:
    pts = sorted(_points(res), key=lambda p: p[1])
    fig, ax = plt.subplots(figsize=HALF)
    # Pareto frontier: best nDCG@10 reachable at or below each cost
    front, best = [], -1.0
    for name, x, y, *_ in pts:
        if y > best:
            front.append((x, y))
            best = y
    fx, fy = zip(*front)
    ax.step(list(fx) + [pts[-1][1] * 3], list(fy) + [fy[-1]], where="post", color="#999999", lw=0.8, ls="--",
            zorder=1, label="Pareto frontier")
    for name, x, y, _, fam in pts:
        star = name == "remix_c_unfiltered"
        ax.scatter(x, y, s=70 if star else 20, color=FAMILY_COLOR[fam], edgecolor=EDGE,
                   linewidth=0.5, zorder=4 if star else 3, marker="*" if star else "o")
        _label(ax, name, x, y, "cost")
    ax.set_xscale("log")
    ax.set_xlabel("Query-time compute (TFLOP, all queries)")
    ax.set_ylabel("nDCG@10")
    oracle = res.get("target_caption_oracle", {}).get("metrics", {}).get("nDCG@10")
    top = max(y for _, _, y, *_ in pts)
    ax.set_ylim(0, top * 1.25)
    if oracle:
        ax.annotate(f"text oracle: {oracle:.2f} ↑", xy=(0.02, 0.97), xycoords="axes fraction", fontsize=FS_SMALL,
                    color="#666666", va="top")
    ax.set_xlim(FLOOR / 2, pts[-1][1] * 3)
    _style(ax)
    fig.savefig(path)
    plt.close(fig)


def fig_precision_recall(res, path: Path) -> None:
    pts = list(_points(res))
    fig, ax = plt.subplots(figsize=HALF)
    for name, flops, y, r100, fam in pts:
        size = 10 + 9 * (np.log10(flops) - np.log10(FLOOR))          # area grows with log compute
        star = name == "remix_c_unfiltered"
        ax.scatter(r100, y, s=size * (2.5 if star else 1), color=FAMILY_COLOR[fam], edgecolor=EDGE, linewidth=0.5,
                   alpha=0.9, zorder=4 if star else 3, marker="*" if star else "o")
        _label(ax, name, r100, y, "pr")
    ax.set_xlabel("Recall@100 (grade ≥ 3)")
    ax.set_ylabel("nDCG@10")
    ax.set_xlim(0, max(p[3] for p in pts) * 1.3)
    ax.set_ylim(0, max(p[2] for p in pts) * 1.2)
    _style(ax)
    fams = [f for f in FAMILY_COLOR if any(p[4] == f for p in pts)]
    handles = [plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], markeredgecolor=EDGE,
                          markeredgewidth=0.5, markersize=4, label=f) for f in fams]
    ax.legend(handles=handles, loc="upper left", fontsize=FS_SMALL - 0.5, handletextpad=0.2, borderaxespad=0.2)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/remix_b/music4all")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    apply()
    res = load(Path(a.results_dir))
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    write_table({k: {"metrics": d["metrics"], "tflops": d.get("tflops", 0.0)} for k, d in res.items()},
                out / "tables" / "remixb_results.tex")
    fig_cost_quality(res, out / "figures" / "remixb_cost_quality.pdf")
    fig_precision_recall(res, out / "figures" / "remixb_precision_recall.pdf")
    print(f"{len(res)} baselines -> {out}/tables/remixb_results.tex, {out}/figures/remixb_*.pdf")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate paper data-exploration figures/tables for a Jamendo-Instruct run.

Reuses the Streamlit analysis layer (`chains_demo._analysis_step_rows`,
`_analysis_manifest_rows`, ...) as the wrangling core so the paper numbers match
the app exactly, caches the two heavy frames to parquet for fast re-runs, and
renders **one publication-grade plot per PDF** into `paper/figures/`.

Design: every figure is a single Axes (no multi-panel grids), no titles
(captions handle them), NeurIPS serif typography, colorblind-safe palette,
low-opacity fills with black contours, human-readable labels/legends.

Example:
  PYTHONPATH=src python scripts/paper_data_stats.py \
    --run-root /gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1 \
    --instructions-folder instructions_axis_focused_5 --label music4all
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from jamendo_instruct.demo.chains_demo import (  # noqa: E402
    _analysis_clip_ids,
    _analysis_manifest_rows,
    _analysis_step_rows,
    _load_dataset_for_streamlit,
)


# --------------------------------------------------------------------------- #
# Curated tag -> coarse-genre map (option i). Specific genres first so they win
# over the broad pop/rock buckets when a clip carries multiple tags.
# --------------------------------------------------------------------------- #
GENRE_KEYWORDS: List[tuple[str, List[str]]] = [
    ("Classical", ["classical", "orchestra", "symphony", "baroque", "opera", "chamber", "piano solo"]),
    ("Jazz", ["jazz", "swing", "bebop", "bossa"]),
    ("Blues", ["blues"]),
    ("Metal", ["metal", "metalcore", "death metal", "black metal", "thrash", "doom"]),
    ("Punk", ["punk", "hardcore"]),
    ("Hip-hop", ["hip hop", "hip-hop", "rap", "trap", "boom bap"]),
    ("Reggae", ["reggae", "ska", "dub", "dancehall"]),
    ("Electronic", ["electronic", "techno", "house", "trance", "edm", "dubstep", "drum and bass",
                    "dnb", "electro", "synth", "idm", "downtempo", "breakbeat", "chillwave"]),
    ("Folk", ["folk", "singer-songwriter", "acoustic", "americana"]),
    ("Country", ["country", "bluegrass"]),
    ("Latin", ["latin", "salsa", "tango", "flamenco", "bossa nova", "reggaeton", "cumbia"]),
    ("Funk", ["funk"]),
    ("Soul/R&B", ["soul", "r&b", "rnb", "motown", "gospel"]),
    ("Ambient", ["ambient", "drone", "new age", "meditation"]),
    ("World", ["world", "celtic", "african", "indian", "folklore", "ethnic"]),
    ("Pop", ["pop", "synthpop", "electropop", "indie pop", "dance pop", "k-pop"]),
    ("Rock", ["rock", "indie", "alternative", "grunge", "psychedelic", "garage"]),
    ("Experimental", ["experimental", "avant-garde", "noise"]),
]


def tags_to_genre(tags: Sequence[str]) -> str:
    lowered = [str(t).lower() for t in (tags or [])]
    for genre, keywords in GENRE_KEYWORDS:
        for kw in keywords:
            if any(kw in tag for tag in lowered):
                return genre
    return "Other"


# --------------------------------------------------------------------------- #
# Publication style + palette (Okabe-Ito, colorblind-safe)
# --------------------------------------------------------------------------- #
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, YELLOW, BLACK = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#111111",
)
QUAL = [BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, "#999999", YELLOW]
COOC_CMAP = "Greens"   # co-occurrence (sequential, non-viridis)
PROB = "Blues"         # probabilities / transition heatmaps
# Low-opacity fills with crisp black contours (applied to every bar/hist).
BAR = dict(alpha=0.82, edgecolor="black", linewidth=0.8)
HIST = dict(alpha=0.55, edgecolor="black", linewidth=0.8)


def setup_style() -> None:
    # NeurIPS body is Times; STIX is a bundled Times-like serif (no external font
    # needed) and also renders math nicely.
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "figure.dpi": 150,
        }
    )


# --------------------------------------------------------------------------- #
# Human-readable naming
# --------------------------------------------------------------------------- #
PRETTY: Dict[str, str] = {
    "vocal": "Vocal", "instrumental": "Instrumental", "unknown": "Unknown",
    "very fast": "Very fast", "fast": "Fast", "medium": "Medium", "slow": "Slow", "very slow": "Very slow",
    "easy": "Easy", "hard": "Hard",
    "genre": "Genre", "mood": "Mood", "instrument": "Instrument", "instrumentation": "Instrumentation",
    "tempo": "Tempo", "vocals": "Vocals", "speed": "Tempo", "energy": "Energy", "rhythm": "Rhythm",
    "production": "Production", "none": "None",
}
QUESTION_SHORT: Dict[str, str] = {
    "meaningful_change": "Meaningful change",
    "target_follows": "Target follows",
    "source_support": "Source described correctly",
    "source_compatible": "No false source claim",
    "conservation_supported": "Conservation kept",
    "edit_specificity": "Edit-specific",
    "clarity_actionability": "Clarity",
    "overall_validity": "Overall validity",
}


def nice(value: Any) -> str:
    s = str(value).strip()
    if s.lower() in PRETTY:
        return PRETTY[s.lower()]
    return " ".join(w.capitalize() if w.islower() else w for w in re.split(r"[_\-\s]+", s) if w) or s


# --------------------------------------------------------------------------- #
# Frame building (with parquet cache)
# --------------------------------------------------------------------------- #
def build_or_load_frames(args: argparse.Namespace, console: Console, progress: Progress, task: int):
    cache = Path(args.cache_dir) / args.label
    cache.mkdir(parents=True, exist_ok=True)
    steps_path = cache / "steps.parquet"
    corpus_path = cache / "corpus.parquet"

    if steps_path.exists() and corpus_path.exists() and not args.refresh:
        progress.update(task, description="Loading cached frames (parquet)")
        steps_df = pd.read_parquet(steps_path)
        progress.advance(task)
        corpus_df = pd.read_parquet(corpus_path)
        progress.advance(task)
        return steps_df, corpus_df

    progress.update(task, description="Loading dataset (streaming, ~1-2 min)")
    run_root = str(Path(args.run_root).expanduser())
    instr_jsonl = str(Path(run_root) / args.instructions_folder / "chain_step_instructions.jsonl")
    max_chains = None if int(args.max_chains) <= 0 else int(args.max_chains)
    dataset = _load_dataset_for_streamlit(run_root, None, None, instr_jsonl, 0, max_chains)
    progress.advance(task)

    progress.update(task, description="Building step + corpus frames")
    chains = list(dataset.chains)
    steps_df = pd.DataFrame(_analysis_step_rows(chains, dataset))
    corpus_df = pd.DataFrame(_analysis_manifest_rows(dataset, _analysis_clip_ids(chains)))
    corpus_df["genre"] = corpus_df["tags"].apply(tags_to_genre)
    steps_df.to_parquet(steps_path, index=False)
    corpus_df.to_parquet(corpus_path, index=False)
    progress.advance(task)
    console.print(f"[dim]cached frames -> {steps_path.parent}[/dim]")
    return steps_df, corpus_df


def derive(steps_df: pd.DataFrame, corpus_df: pd.DataFrame) -> Dict[str, Any]:
    genre_by_clip = dict(zip(corpus_df["clip_id"], corpus_df["genre"]))
    steps_unique = steps_df.drop_duplicates(subset=["chain_id", "turn_index"]).copy()
    steps_unique["source_genre"] = steps_unique["source_clip_id"].map(genre_by_clip).fillna("Other")
    steps_unique["target_genre"] = steps_unique["target_clip_id"].map(genre_by_clip).fillna("Other")
    instr = steps_df[steps_df["has_instruction"]].copy()
    return {"steps_unique": steps_unique, "instr": instr, "genre_by_clip": genre_by_clip}


# --------------------------------------------------------------------------- #
# Figure primitives (one Axes per figure, no titles)
# --------------------------------------------------------------------------- #
def _new(w: float = 5.2, h: float = 3.4):
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    return fig, ax


def _finish(fig, ax, ctx: Dict[str, Any], name: str, *, grid: str | None = "y") -> None:
    ax.grid(axis="x", visible=grid in ("x", "both"))
    ax.grid(axis="y", visible=grid in ("y", "both"))
    fig.savefig(ctx["figures_dir"] / f"{ctx['label']}_{name}.pdf")
    plt.close(fig)


def _bar_labels(ax, bars, values, *, fmt="{:,}", pad=3, horizontal=False) -> None:
    for b, v in zip(bars, values):
        if horizontal:
            ax.annotate(fmt.format(v), (b.get_width(), b.get_y() + b.get_height() / 2),
                        xytext=(pad, 0), textcoords="offset points", va="center", fontsize=9, color="#333")
        else:
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, pad), textcoords="offset points", ha="center", fontsize=9, color="#333")


def _abbrev(n: float) -> str:
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{int(n)}"


def _explode_counts(series: pd.Series) -> pd.Series:
    flat: List[str] = []
    for v in series:
        flat.extend(list(v) if v is not None else [])
    return pd.Series(flat).value_counts()


# --------------------------------------------------------------------------- #
# Corpus figures
# --------------------------------------------------------------------------- #
def fig_genre(ctx):
    gc = ctx["corpus_df"]["genre"].value_counts()
    other = int(gc.pop("Other")) if "Other" in gc.index else 0   # explicit unmatched bucket
    top = gc.head(9)
    other += int(gc.iloc[9:].sum())                              # + long tail -> single "Other"
    if other > 0:
        top = pd.concat([top, pd.Series({"Other": other})])
    labels = [nice(g) for g in top.index]
    sizes = top.values.astype(float)
    pct = 100.0 * sizes / sizes.sum()
    colors = [QUAL[i % len(QUAL)] for i in range(len(top))]
    fig, ax = _new(6.6, 5.6)
    wedges, _ = ax.pie(
        sizes,
        explode=[0.09] * len(sizes),
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops=dict(edgecolor="none", linewidth=0, alpha=0.92),
    )
    # Nice elbow ("bracket") callouts; each label box is tinted to its wedge colour.
    for w, lab, p, col in zip(wedges, labels, pct, colors):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        x, y = np.cos(ang), np.sin(ang)
        ha = "left" if x >= 0 else "right"
        ax.annotate(
            f"{lab}  ({p:.0f}%)",
            xy=(x, y),
            xytext=(1.4 * np.sign(x), 1.28 * y),
            ha=ha, va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.5", fc="white", ec=col, lw=1.2),
            arrowprops=dict(arrowstyle="-", color=col, lw=1.1,
                            connectionstyle=f"angle,angleA=0,angleB={np.rad2deg(ang):.0f}"),
        )
    ax.set(aspect="equal")
    _finish(fig, ax, ctx, "corpus_genre", grid=None)


def fig_vocals(ctx):
    vc = ctx["corpus_df"]["vocals"].value_counts()
    fig, ax = _new(4.2, 3.2)
    b = ax.bar([nice(x) for x in vc.index], vc.values, color=GREEN, width=0.62, **BAR)
    _bar_labels(ax, b, vc.values)
    ax.set_ylabel("Number of clips")
    _finish(fig, ax, ctx, "corpus_vocals")


def fig_speed(ctx):
    order = ["very slow", "slow", "medium", "fast", "very fast", "unknown"]
    sc = ctx["corpus_df"]["speed"].value_counts()
    sc = sc.reindex([o for o in order if o in sc.index]).dropna()
    fig, ax = _new(4.8, 3.2)
    b = ax.bar([nice(x) for x in sc.index], sc.values, color=ORANGE, width=0.62, **BAR)
    _bar_labels(ax, b, sc.values.astype(int))
    ax.set_ylabel("Number of clips")
    _finish(fig, ax, ctx, "corpus_tempo")


def fig_tags_per_clip(ctx):
    tc = ctx["corpus_df"]["tag_count"].dropna()
    fig, ax = _new(5.0, 3.4)
    ax.hist(tc, bins=range(0, int(tc.max()) + 2), color=PURPLE, **HIST)
    ax.set_xlabel("Tags per clip")
    ax.set_ylabel("Number of clips")
    _finish(fig, ax, ctx, "corpus_tags_per_clip")


def fig_coverage(ctx):
    c = ctx["corpus_df"]
    cov = {
        "Caption": c["has_caption"].mean(),
        "Tags": (c["tag_count"] > 0).mean(),
        "Vocal label": (c["vocals"] != "unknown").mean(),
        "Tempo label": (c["speed"] != "unknown").mean(),
        "Duration": c["duration_sec"].notna().mean(),
        "Lyrics": c["lyrics_status"].isin(["ok", "found", "available"]).mean(),
    }
    items = sorted(cov.items(), key=lambda kv: kv[1])
    fig, ax = _new(5.0, 3.2)
    b = ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE, **BAR)
    _bar_labels(ax, b, [v for _, v in items], fmt="{:.0%}", horizontal=True)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Fraction of clips")
    _finish(fig, ax, ctx, "corpus_coverage", grid="x")


def fig_caption_length(ctx):
    cw = ctx["corpus_df"]["caption_words"].dropna()
    fig, ax = _new(5.0, 3.4)
    ax.hist(cw, bins=40, color=SKY, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Caption length (words)")
    ax.set_ylabel("Number of clips")
    _finish(fig, ax, ctx, "corpus_caption_length")


# --------------------------------------------------------------------------- #
# Transition figures
# --------------------------------------------------------------------------- #
def fig_transition_score(ctx):
    su = ctx["steps_unique"]
    fig, ax = _new(5.4, 3.6)
    for i, (hard, g) in enumerate(sorted(su.groupby("hardness"), key=lambda kv: str(kv[0]))):
        ax.hist(g["transition_score"].dropna(), bins=40, histtype="step", linewidth=1.8,
                color=QUAL[i % len(QUAL)], label=nice(hard))
    ax.set_xlabel("Transition score")
    ax.set_ylabel("Number of steps")
    ax.legend(title="Difficulty")
    _finish(fig, ax, ctx, "trans_score")


def fig_genre_matrix(ctx):
    su = ctx["steps_unique"]
    top = su["source_genre"].value_counts().head(12).index.tolist()
    sub = su[su["source_genre"].isin(top) & su["target_genre"].isin(top)]
    mat = pd.crosstab(sub["source_genre"], sub["target_genre"], normalize="index").reindex(index=top, columns=top).fillna(0)
    fig, ax = _new(5.8, 5.2)
    im = ax.imshow(mat.values, cmap=PROB, vmin=0, aspect="auto")
    ax.set_xticks(range(len(top)), [nice(t) for t in top], rotation=45, ha="right")
    ax.set_yticks(range(len(top)), [nice(t) for t in top])
    ax.set_xlabel("Target genre")
    ax.set_ylabel("Source genre")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("P(target | source)")
    _finish(fig, ax, ctx, "trans_genre_matrix", grid=None)


def _transition_heatmap(ctx, src_col, tgt_col, order, name):
    su = ctx["steps_unique"]
    observed = {c for c in (set(su[src_col].dropna()) | set(su[tgt_col].dropna())) if str(c).strip()}
    cats = [c for c in order if c in observed] + sorted(observed - set(order))
    if not cats:
        return
    mat = pd.crosstab(su[src_col], su[tgt_col], normalize="index").reindex(index=cats, columns=cats).fillna(0)
    fig, ax = _new(4.8, 4.2)
    im = ax.imshow(mat.values, cmap=PROB, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cats)), [nice(c) for c in cats], rotation=30, ha="right")
    ax.set_yticks(range(len(cats)), [nice(c) for c in cats])
    for i in range(len(cats)):
        for j in range(len(cats)):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.55 else "#222")
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("P(target | source)")
    _finish(fig, ax, ctx, name, grid=None)


def fig_vocals_flips(ctx):
    _transition_heatmap(ctx, "source_vocals", "target_vocals", ["vocal", "instrumental", "unknown"], "trans_vocals_matrix")


def fig_speed_flips(ctx):
    _transition_heatmap(ctx, "source_speed", "target_speed",
                        ["very slow", "slow", "medium", "fast", "very fast", "unknown"], "trans_tempo_matrix")


def fig_tag_churn(ctx):
    su = ctx["steps_unique"]
    fig, ax = _new(5.2, 3.4)
    ax.hist(su["tags_added_count"], bins=range(0, 12), histtype="step", linewidth=1.8, color=GREEN, label="Added")
    ax.hist(su["tags_removed_count"], bins=range(0, 12), histtype="step", linewidth=1.8, color=VERM, label="Removed")
    ax.set_xlabel("Tags changed per step")
    ax.set_ylabel("Number of steps")
    ax.legend()
    _finish(fig, ax, ctx, "trans_tag_churn")


# --------------------------------------------------------------------------- #
# Recipe figures
# --------------------------------------------------------------------------- #
def fig_axes_change_preserve(ctx):
    instr = ctx["instr"]
    ch = _explode_counts(instr["change_axes"])
    pr = _explode_counts(instr["preservation_axes"])
    names = sorted(set(ch.index) | set(pr.index), key=lambda a: -(ch.get(a, 0) + pr.get(a, 0)))[:10][::-1]
    y = np.arange(len(names))
    fig, ax = _new(5.8, 4.0)
    ax.barh(y + 0.2, [ch.get(a, 0) for a in names], height=0.38, color=BLUE, label="Changed", **BAR)
    ax.barh(y - 0.2, [pr.get(a, 0) for a in names], height=0.38, color=ORANGE, label="Preserved", **BAR)
    ax.set_yticks(y, [nice(a) for a in names])
    ax.set_xlabel("Number of instructions")
    ax.legend()
    _finish(fig, ax, ctx, "recipe_axes", grid="x")


def fig_axis_cooccurrence(ctx):
    instr = ctx["instr"]
    top = _explode_counts(instr["change_axes"]).head(8).index.tolist()
    idx = {a: i for i, a in enumerate(top)}
    co = np.zeros((len(top), len(top)))
    for axlist in instr["change_axes"]:
        present = [a for a in (list(axlist) if axlist is not None else []) if a in idx]
        for a in present:
            for b in present:
                co[idx[a], idx[b]] += 1
    np.fill_diagonal(co, 0)  # self-co-occurrence is uninformative; drop it
    fig, ax = _new(5.2, 4.6)
    im = ax.imshow(co, cmap=COOC_CMAP, aspect="auto")
    ax.set_xticks(range(len(top)), [nice(a) for a in top], rotation=45, ha="right")
    ax.set_yticks(range(len(top)), [nice(a) for a in top])
    hi = co.max() if co.size else 1
    for i in range(len(top)):
        for j in range(len(top)):
            if i == j:
                continue
            ax.text(j, i, _abbrev(co[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if co[i, j] > hi * 0.55 else "#222")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Co-occurring instructions")
    _finish(fig, ax, ctx, "recipe_cooccurrence", grid=None)


def fig_num_axes(ctx):
    nc = ctx["instr"]["change_axis_count"].value_counts().sort_index()
    fig, ax = _new(4.8, 3.4)
    b = ax.bar(nc.index.astype(int), nc.values, color=GREEN, width=0.7, **BAR)
    _bar_labels(ax, b, nc.values)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Number of change axes")
    ax.set_ylabel("Number of instructions")
    _finish(fig, ax, ctx, "recipe_num_axes")


def fig_caption_only(ctx):
    r = ctx["instr"]["caption_only_change"].value_counts(normalize=True)
    labels = {True: "Caption-grounded", False: "Tag / metadata"}
    order = [k for k in [True, False] if k in r.index]
    fig, ax = _new(4.2, 3.4)
    b = ax.bar([labels[k] for k in order], [r[k] for k in order],
               color=[ORANGE, BLUE][: len(order)], width=0.6, **BAR)
    _bar_labels(ax, b, [r[k] for k in order], fmt="{:.0%}")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Fraction of instructions")
    _finish(fig, ax, ctx, "recipe_caption_only")


# --------------------------------------------------------------------------- #
# Instruction / variant figures
# --------------------------------------------------------------------------- #
def fig_instruction_length(ctx):
    words = ctx["instr"]["history_unaware_words"].dropna()
    fig, ax = _new(5.2, 3.4)
    ax.hist(words, bins=40, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Instruction length (words)")
    ax.set_ylabel("Number of instructions")
    _finish(fig, ax, ctx, "instr_length")


def fig_chain_length(ctx):
    clen = ctx["steps_unique"].groupby("chain_id")["turn_index"].nunique()
    vc = clen.value_counts().sort_index()
    fig, ax = _new(4.8, 3.4)
    b = ax.bar(vc.index.astype(int), vc.values, color=PURPLE, width=0.7, **BAR)
    _bar_labels(ax, b, vc.values)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Steps per chain")
    ax.set_ylabel("Number of chains")
    _finish(fig, ax, ctx, "chain_length")


# --------------------------------------------------------------------------- #
# Validation figures (LLM judge)
# --------------------------------------------------------------------------- #
def _llm_records(ctx):
    from jamendo_instruct.demo.human_validation_app import _llm_rating_records

    return _llm_rating_records(Path(ctx["run_root"]) / ctx["instructions_folder"] / "validation")


def fig_llm_rubric(ctx):
    from jamendo_instruct.demo.human_validation_app import _admin_question_rows

    records = _llm_records(ctx)
    if not records:
        return
    q = pd.DataFrame(_admin_question_rows(records))
    q["label"] = q["question_id"].map(lambda x: QUESTION_SHORT.get(x, nice(x)))
    q = q.iloc[::-1]
    score_colors = plt.get_cmap("RdYlGn")(np.linspace(0.12, 0.88, 5))  # 1=red ... 5=green
    fig, ax = _new(6.6, 4.2)
    left = np.zeros(len(q))
    for s in range(1, 6):
        vals = q[f"score_{s}"].to_numpy(dtype=float)
        ax.barh(q["label"], vals, left=left, color=score_colors[s - 1],
                edgecolor="black", linewidth=0.5, label=f"{s}")
        left += vals
    ax.set_xlabel("Number of ratings")
    ax.legend(title="Score (1–5)", ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _finish(fig, ax, ctx, "val_rubric_scores", grid="x")


def _llm_accept_frame(records, instr) -> pd.DataFrame:
    rows = []
    for r in records:
        ov = dict((r.get("answers", {}) or {}).get("overall_validity", {}) or {})
        s = ov.get("score")
        if isinstance(s, int):
            rows.append({"chain_id": str(r.get("chain_id", "")), "turn_index": int(r.get("turn_index", 0) or 0),
                         "variant_index": int(r.get("variant_index", 0) or 0), "accept": 1.0 if s >= 4 else 0.0})
    if not rows:
        return pd.DataFrame()
    keys = instr[["chain_id", "turn_index", "variant_index", "hardness", "change_axis_count", "caption_only_change"]]
    return pd.DataFrame(rows).merge(keys, on=["chain_id", "turn_index", "variant_index"], how="left").dropna(subset=["hardness"])


def fig_llm_accept_hardness(ctx):
    records = _llm_records(ctx)
    df = _llm_accept_frame(records, ctx["instr"]) if records else pd.DataFrame()
    if df.empty:
        return
    by = df.groupby("hardness")["accept"].mean().sort_values()
    fig, ax = _new(4.6, 3.2)
    b = ax.barh([nice(x) for x in by.index], by.values, color=BLUE, **BAR)
    _bar_labels(ax, b, by.values, fmt="{:.0%}", horizontal=True)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Acceptance rate  (overall validity ≥ 4)")
    _finish(fig, ax, ctx, "val_accept_hardness", grid="x")


def fig_llm_issue_tags(ctx):
    from jamendo_instruct.demo.human_validation_app import _issue_tag_rows

    records = _llm_records(ctx)
    if not records:
        return
    it = pd.DataFrame(_issue_tag_rows(records))
    if it.empty:
        return
    it = it.sort_values("count").tail(12)
    fig, ax = _new(6.0, 4.0)
    b = ax.barh([nice(x) for x in it["issue_tag"]], it["count"].values, color=VERM, **BAR)
    _bar_labels(ax, b, it["count"].values, horizontal=True)
    ax.set_xlabel("Number of flagged instructions")
    _finish(fig, ax, ctx, "val_issue_tags", grid="x")


def fig_leakage(ctx):
    corpus_df = ctx["corpus_df"]
    splits = [s for s in ["train", "validation", "test"] if s in set(corpus_df["split"])] or sorted(corpus_df["split"].unique())
    art = {s: set(corpus_df[corpus_df["split"] == s]["artist_id"]) - {""} for s in splits}
    mat = np.array([[len(art[a] & art[b]) / max(1, len(art[a])) for b in splits] for a in splits])
    fig, ax = _new(4.6, 4.0)
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(splits)), [nice(s) for s in splits])
    ax.set_yticks(range(len(splits)), [nice(s) for s in splits])
    for i in range(len(splits)):
        for j in range(len(splits)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] > 0.55 else "#222", fontsize=10)
    ax.set_xlabel("Also appears in")
    ax.set_ylabel("Artists from")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Overlap fraction")
    _finish(fig, ax, ctx, "split_leakage", grid=None)


# --------------------------------------------------------------------------- #
# Table 1 (csv + tex)
# --------------------------------------------------------------------------- #
def make_table1(ctx):
    corpus_df, su, instr = ctx["corpus_df"], ctx["steps_unique"], ctx["instr"]
    artists = corpus_df["artist_id"].replace("", np.nan).fillna(corpus_df["artist_name"])
    rows = [
        ("Chains", su["chain_id"].nunique()),
        ("Steps", len(su)),
        ("Instruction variants", len(instr)),
        ("Unique clips", corpus_df["clip_id"].nunique()),
        ("Unique tracks", corpus_df["track_id"].nunique()),
        ("Unique artists", int(artists.nunique())),
        ("Audio hours (referenced)", round(corpus_df["duration_sec"].dropna().sum() / 3600.0, 1)),
        ("Total instruction words", int(instr["history_unaware_words"].sum())),
        ("Mean steps / chain", round(len(su) / max(1, su["chain_id"].nunique()), 2)),
        ("Mean variants / step", round(len(instr) / max(1, len(su)), 2)),
    ]
    for split, g in su.groupby("split"):
        rows.append((f"Steps ({nice(split)})", len(g)))
    table = pd.DataFrame(rows, columns=["Metric", "Value"])
    table.to_csv(ctx["tables_dir"] / f"{ctx['label']}_overview.csv", index=False)
    (ctx["tables_dir"] / f"{ctx['label']}_overview.tex").write_text(
        table.to_latex(index=False, caption=f"{nice(ctx['label'])} dataset at a glance.",
                       label=f"tab:{ctx['label']}_overview"),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Registry + main
# --------------------------------------------------------------------------- #
def figure_registry(include_appendix: bool) -> List[tuple[str, Callable[[Dict[str, Any]], None]]]:
    main: List[tuple[str, Callable]] = [
        ("Table 1: overview", make_table1),
        ("Corpus: genre", fig_genre),
        ("Corpus: vocals", fig_vocals),
        ("Corpus: tempo", fig_speed),
        ("Corpus: tags/clip", fig_tags_per_clip),
        ("Corpus: coverage", fig_coverage),
        ("Transitions: score", fig_transition_score),
        ("Transitions: genre matrix", fig_genre_matrix),
        ("Transitions: vocal matrix", fig_vocals_flips),
        ("Transitions: tempo matrix", fig_speed_flips),
        ("Recipe: change vs preserve", fig_axes_change_preserve),
        ("Recipe: co-occurrence", fig_axis_cooccurrence),
        ("Recipe: edit complexity", fig_num_axes),
        ("Recipe: edit grounding", fig_caption_only),
        ("Validation: rubric scores", fig_llm_rubric),
        ("Validation: accept by difficulty", fig_llm_accept_hardness),
        ("Validation: issue tags", fig_llm_issue_tags),
    ]
    appendix: List[tuple[str, Callable]] = [
        ("Appendix: caption length", fig_caption_length),
        ("Appendix: tag churn", fig_tag_churn),
        ("Appendix: instruction length", fig_instruction_length),
        ("Appendix: chain length", fig_chain_length),
        ("Appendix: split leakage", fig_leakage),
    ]
    return main + (appendix if include_appendix else [])


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", default="/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1")
    p.add_argument("--instructions-folder", default="instructions_axis_focused_5")
    p.add_argument("--label", default="music4all", help="Prefix for output filenames.")
    p.add_argument("--out-dir", default="paper")
    p.add_argument("--cache-dir", default="paper/cache")
    p.add_argument("--max-chains", type=int, default=0, help="0 = all instructed chains.")
    p.add_argument("--refresh", action="store_true", help="Rebuild parquet cache from the dataset.")
    p.add_argument("--no-appendix", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    setup_style()
    console = Console()
    figures_dir = Path(args.out_dir) / "figures"
    tables_dir = Path(args.out_dir) / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    stages = figure_registry(not args.no_appendix)
    columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("Preparing data", total=2 + len(stages))
        steps_df, corpus_df = build_or_load_frames(args, console, progress, task)
        d = derive(steps_df, corpus_df)
        ctx: Dict[str, Any] = {
            "steps_df": steps_df,
            "corpus_df": corpus_df,
            "steps_unique": d["steps_unique"],
            "instr": d["instr"],
            "label": args.label,
            "run_root": str(Path(args.run_root).expanduser()),
            "instructions_folder": args.instructions_folder,
            "figures_dir": figures_dir,
            "tables_dir": tables_dir,
        }
        for name, fn in stages:
            progress.update(task, description=name)
            try:
                fn(ctx)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]  {name} failed: {exc.__class__.__name__}: {exc}[/red]")
            progress.advance(task)

    console.print(f"[green]Done.[/green] {len(stages)} outputs -> {figures_dir} / {tables_dir}")


if __name__ == "__main__":
    main()

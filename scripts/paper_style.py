"""Shared figure style for every paper plot (data, validation, relevance pool).

Figures are authored at their printed size, so LaTeX includes them unscaled and
the text prints at the sizes below. NeurIPS text width is 5.5 in.

  HALF  (2.65 in) -> \\begin{subfigure}{0.49\\linewidth} ... width=\\linewidth
  FULL  (5.5 in)  -> width=\\linewidth
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

TEXT_W = 5.5
HALF_W = 2.65
HALF = (HALF_W, HALF_W)          # square by default
HALF_TALL = HALF
THIRD_W = 1.8                      # three panels per row: \begin{subfigure}{0.325\linewidth}
THIRD = (THIRD_W, THIRD_W)
FULL = (TEXT_W, 2.1)

FS = 7.5          # base text
FS_SMALL = 6.5    # in-plot annotations (bar values, cell labels)

# Okabe-Ito (colour-blind safe). One colour per entity, everywhere in the paper.
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#8C8C8C",
)
INK = "#222222"
DATASET = {"music4all": BLUE, "mtg_jamendo": ORANGE}
JUDGE = {"Qwen3.6-27B": "#4C6EB1", "Gemma-4-31B": "#D1495B"}
# Relevance grades 6 (exact) .. 0 (non-relevant), as in the pool figures.
GRADE = {6: "#0A7D3E", 5: "#22C55E", 4: "#A7E32C", 3: "#F7B500", 2: "#FB6A0A", 1: "#E4231B", 0: "#CBD0D6"}
# Rubric scores 1 (bad) .. 5 (good).
SCORE = {1: "#C0392B", 2: "#EB8A5B", 3: "#E9D78C", 4: "#8CC68B", 5: "#2E8B57"}
SEQ = "Blues"      # probabilities / heatmaps

_INTER = Path.home() / ".local/share/fonts/inter"


def apply() -> None:
    for f in sorted(_INTER.glob("Inter-*.ttf")):
        font_manager.fontManager.addfont(str(f))
    family = "Inter" if any(_INTER.glob("Inter-*.ttf")) else "DejaVu Sans"
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS_SMALL + 0.5, "ytick.labelsize": FS_SMALL + 0.5,
        "legend.fontsize": FS_SMALL + 0.5, "legend.title_fontsize": FS,
        "legend.frameon": False, "legend.handlelength": 1.0, "legend.handleheight": 0.7,
        "legend.borderaxespad": 0.3, "legend.columnspacing": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "axes.edgecolor": "#444444", "axes.labelcolor": INK,
        "axes.labelpad": 2.5, "axes.titlepad": 3,
        "xtick.color": "#444444", "ytick.color": "#444444",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
        "lines.linewidth": 1.2, "patch.linewidth": 0.5,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
    })


# Thin dark-grey contour around every filled surface (bars, histograms, wedges, dots).
EDGE = "#333333"
BAR = dict(edgecolor=EDGE, linewidth=0.5)
HIST = dict(edgecolor=EDGE, linewidth=0.4)


# Printed size per figure, keyed by file stem; anything not listed keeps the size it was
# drawn at (HALF by default). This is the single place that encodes the paper layout.
_THIRD_PANELS = """
music4all_corpus_caption_length music4all_corpus_tags_per_clip music4all_corpus_tempo music4all_split_leakage
remix_trans_score remix_chain_length music4all_trans_tag_churn
music4all_trans_genre_matrix music4all_trans_vocals_matrix music4all_trans_tempo_matrix
remixc_curve_r10 remixc_curve_mrr remixc_curve_loss
remixc_overfit_r10 remixc_overfit_mrr remixc_overfit_loss
""".split()
PANEL = {stem: THIRD for stem in _THIRD_PANELS} | {
    "music4all_relpool_pool_composition_pie": (0.7 * TEXT_W, 1.75),   # placed at .7\linewidth
    "music4all_val_mean_dumbbell": (2.3, 2.3),                          # wrapfigures at .42\linewidth
    "music4all_relpool_failure_modes": (2.3, 2.3),
    "music4all_recipe_caption_only": (1.45, 1.7),      # narrow wrapfigure next to its paragraph
}


def savefig(fig, path) -> None:
    """Save at the printed size from PANEL (constrained layout re-flows at save time)."""
    size = PANEL.get(Path(path).stem)
    if size:
        fig.set_size_inches(*size)
    fig.savefig(path)

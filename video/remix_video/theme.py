"""Shared visual language for the ReMIX video.

Matches the paper's main figure (paper/figures/Remix Pipeline.png) and the
companion website (website/src/theme.js): white ground, five stage colours,
tinted cards with thin coloured borders, sans type.

No LaTeX anywhere. Tex/MathTex would need a TeX install and would render in
Computer Modern serif, which fights the sans identity. Text (Pango) is both
available and correct, and the one formula sets fine in Unicode.
"""

from __future__ import annotations

from manim import *

# --- stage palette -------------------------------------------------------- #
ENRICH = "#E23B34"
NEIGHBOUR = "#2E6FD6"
CHAIN = "#1FA347"
INSTRUCT = "#FB8B24"
VALIDATE = "#7B3FF2"

STAGE_COLORS = [ENRICH, NEIGHBOUR, CHAIN, INSTRUCT, VALIDATE]
STAGE_NAMES = [
    "Dataset Enrichment",
    "Neighbourhood Building",
    "Chain Sampling",
    "Instruction Generation",
    "Validation & Benchmark",
]

# --- neutrals ------------------------------------------------------------- #
INK = "#141414"
MUTED = "#71717A"
FAINT = "#D4D4D8"
HAIR = "#E4E4E7"
PAPER = "#FFFFFF"

# Inter, matching the companion website exactly (index.html loads it from rsms.me).
# Helvetica Neue was the ask, but it is proprietary Linotype/Apple and cannot be
# installed here; Inter is the neo-grotesque the site already renders in, so the
# video and the site now agree. Installed to ~/.local/share/fonts/inter.
FONT = "Inter"

# Type scale, in manim units.
T_TITLE = 0.72
T_HEAD = 0.46
T_BODY = 0.32
T_SMALL = 0.24
T_TINY = 0.19

# --- layout grid ---------------------------------------------------------- #
# Fixed bands, so text can never land on text. Every scene uses these instead of
# guessing offsets, which is what caused the overlaps in the first draft.
Y_HEADER = 2.55     # stage number + name
Y_STAGE_TOP = 1.85  # content may not go above this
Y_STAGE_BOT = -1.5  # ...nor below this
Y_EXPLAIN = -2.15   # the one explanatory sentence
Y_FIGURES = -2.95   # the numbers

# The rail of finished stages: padded from the top, tighter to the right.
RAIL_TOP = 3.55
RAIL_LEFT = -6.55


def tint(color: str, alpha: float = 0.10) -> str:
    """Blend a stage colour toward white -- the figure's tinted card fill."""
    return interpolate_color(ManimColor(PAPER), ManimColor(color), alpha).to_hex()


def txt(s: str, size: float = T_BODY, color: str = INK, weight: str = NORMAL) -> Text:
    return Text(s, font=FONT, color=color, weight=weight).scale(size)


def card(width: float, height: float, color: str, alpha: float = 0.08, radius: float = 0.16) -> RoundedRectangle:
    """The figure's card: tinted fill, thin coloured border, generous radius."""
    return RoundedRectangle(
        width=width,
        height=height,
        corner_radius=radius,
        fill_color=tint(color, alpha),
        fill_opacity=1,
        stroke_color=color,
        stroke_width=2.0,
    )


def arrow(start, end, color: str = INK, width: float = 3.0) -> Arrow:
    return Arrow(
        start, end,
        color=color,
        stroke_width=width,
        buff=0.0,
        max_tip_length_to_length_ratio=0.18,
        tip_length=0.16,
    )

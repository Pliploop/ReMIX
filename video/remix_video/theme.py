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
# Accents only. Structure (cards, lines, text) is neutral; a stage colour marks the
# one thing that matters in a frame -- the eyebrow, the walked path, the highlighted
# bar -- never every box. Refined, slightly softened hues (Radix scale 9).
ENRICH = "#E5484D"
NEIGHBOUR = "#3E63DD"
CHAIN = "#30A46C"
INSTRUCT = "#F76B15"
VALIDATE = "#8E4EC6"

STAGE_COLORS = [ENRICH, NEIGHBOUR, CHAIN, INSTRUCT, VALIDATE]
STAGE_NAMES = [
    "Dataset Enrichment",
    "Neighbourhood Building",
    "Chain Sampling",
    "Instruction Generation",
    "Validation & Benchmark",
]

# --- neutrals ------------------------------------------------------------- #
INK = "#18181B"       # primary text
INK2 = "#52525B"      # secondary text, labels
MUTED = "#8E8E96"     # tertiary text, connectors
FAINT = "#D4D4D8"
HAIR = "#E7E5E4"      # hairline borders
LINE = "#E7E5E4"
FILL = "#F4F4F3"      # neutral chip / track fill
PAPER = "#FFFFFF"     # surfaces (cards)
BG = "#FAFAF8"        # the ground: warm off-white, so white cards read as raised

# Inter, matching the companion website exactly (index.html loads it from rsms.me).
# Helvetica Neue was the ask, but it is proprietary Linotype/Apple and cannot be
# installed here; Inter is the neo-grotesque the site already renders in, so the
# video and the site now agree. Installed to ~/.local/share/fonts/inter.
FONT = "Inter"

# Type scale, in manim units.
T_TITLE = 0.72
T_HEAD = 0.46
T_BODY = 0.32
T_SMALL = 0.25
T_TINY = 0.205       # the floor: nothing on screen is smaller than a readable label

# --- layout grid ---------------------------------------------------------- #
# Fixed bands, so text can never land on text. Every scene uses these instead of
# guessing offsets, which is what caused the overlaps in the first draft.
Y_HEADER = 2.55     # stage number + name
Y_STAGE_TOP = 1.85  # content may not go above this
Y_STAGE_BOT = -1.5  # ...nor below this
Y_EXPLAIN = -2.15   # the one explanatory sentence
Y_FIGURES = -2.95   # the numbers

# The rail of finished stages. Pulled down clear of the frame edge so the patches
# read as floating cards rather than tabs stuck to the top.
RAIL_TOP = 3.05
RAIL_LEFT = -6.45


# --- motion ---------------------------------------------------------------- #
# One vocabulary of durations and easings, so every scene moves at the same tempo.
# Arrivals decelerate into place (ease-out); moves accelerate and settle (in-out);
# exits are quick. Holds are long enough to read one short sentence.
T_ENTER = 0.7
T_MOVE = 0.9
T_EXIT = 0.45
HOLD = 1.2          # after a beat
HOLD_READ = 2.0     # after the stage's sentence and figures
EASE_IN = rate_functions.ease_out_cubic
EASE_MOVE = rate_functions.ease_in_out_cubic
EASE_OUT = rate_functions.ease_in_quad
LAG = 0.12          # stagger between siblings entering together

# Hairlines: the figure's look at video scale. Thicker strokes read as clip art.
STROKE = 1.25
STROKE_ARROW = 1.8


def tint(color: str, alpha: float = 0.10) -> str:
    """Blend a stage colour toward white -- the figure's tinted card fill."""
    return interpolate_color(ManimColor(PAPER), ManimColor(color), alpha).to_hex()


def txt(s: str, size: float = T_BODY, color: str = INK, weight: str = NORMAL) -> Text:
    return Text(s, font=FONT, color=color, weight=weight).scale(size)


def card(width: float, height: float, color: str = INK, alpha: float = 0.0, radius: float = 0.12,
         accent: bool = False) -> RoundedRectangle:
    """A surface: white, hairline grey border. `color` is ignored unless `accent`
    (then the border takes the stage colour, softly) -- structure stays neutral so
    colour can mean something."""
    return RoundedRectangle(
        width=width,
        height=height,
        corner_radius=radius,
        fill_color=PAPER,
        fill_opacity=1,
        stroke_color=color if accent else LINE,
        stroke_width=STROKE,
        stroke_opacity=0.7 if accent else 1.0,
    )


def shadow(m: Mobject, depth: float = 1.0) -> VGroup:
    """A soft two-layer drop shadow under a rounded surface: what makes a white
    card sit on the off-white ground instead of being outlined on it."""
    layers = VGroup()
    for dy, grow, op in ((0.05, 0.06, 0.035), (0.02, 0.02, 0.05)):
        s = RoundedRectangle(
            width=m.width + grow, height=m.height + grow,
            corner_radius=getattr(m, "corner_radius", 0.12) + grow / 2,
            fill_color="#000000", fill_opacity=op * depth, stroke_width=0,
        ).move_to(m.get_center() + DOWN * dy * depth)
        layers.add(s)
    return layers


def eyebrow(text: str, color: str, size: float = T_TINY) -> MarkupText:
    """Small tracked capitals: the editorial label above a title."""
    return MarkupText(f'<span letter_spacing="2800">{text.upper()}</span>', font=FONT,
                      color=color, weight=SEMIBOLD).scale(size * 0.95)


def arrow(start, end, color: str = INK, width: float = STROKE_ARROW) -> Arrow:
    return Arrow(
        start, end,
        color=color,
        stroke_width=min(width, STROKE_ARROW),
        buff=0.0,
        max_tip_length_to_length_ratio=0.14,
        tip_length=0.12,
    )

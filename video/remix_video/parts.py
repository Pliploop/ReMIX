"""Scene furniture shared across stages: catalogue cylinders, latent-space
backdrops, JSON frames, and the judge logos.

Kept here so stage 2 and stage 3 can render *the same* graph, and so the JSON
frames in stage 4 look like one designed object rather than three.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
from manim import *

from .theme import FAINT, FONT, INK, MUTED, PAPER, T_TINY, tint, txt

ASSETS = Path(__file__).resolve().parents[1] / "assets"
QWEN_LOGO = ASSETS / "qwen-icon-logo-png_seeklogo-669128 (1).png"
GEMMA_LOGO = ASSETS / "gemma-color.png"


def music_note(color: str, scale: float = 1.0) -> VGroup:
    """A quaver. Says 'this cylinder is full of music', which the bare cylinder did not."""
    stem = Line(ORIGIN, UP * 0.34, color=color, stroke_width=2.6)
    head = Ellipse(width=0.16, height=0.12, fill_color=color, fill_opacity=1,
                   stroke_width=0).rotate(-0.35).move_to(stem.get_start() + LEFT * 0.055)
    flag = ArcBetweenPoints(
        stem.get_end(), stem.get_end() + RIGHT * 0.12 + DOWN * 0.17,
        angle=-TAU / 6, color=color, stroke_width=2.4,
    )
    return VGroup(stem, head, flag).scale(scale)


def catalogue(label: str, count: str, color: str, w: float = 1.5, h: float = 1.3) -> VGroup:
    """A cylinder that is visibly a music catalogue."""
    body = Rectangle(width=w, height=h, fill_color=tint(color, 0.1), fill_opacity=1, stroke_width=0)
    bot = Arc(radius=w / 2, start_angle=PI, angle=PI, color=color, stroke_width=2)
    bot.stretch(0.3, 1).move_to(body.get_bottom())
    left = Line(body.get_corner(UL), body.get_corner(DL), color=color, stroke_width=2)
    right = Line(body.get_corner(UR), body.get_corner(DR), color=color, stroke_width=2)
    top = Ellipse(width=w, height=w * 0.3, fill_color=tint(color, 0.24), fill_opacity=1,
                  stroke_color=color, stroke_width=2).move_to(body.get_top())

    note = music_note(color, 1.05).move_to(body.get_center() + DOWN * 0.04)
    shell = VGroup(body, left, right, bot, top, note)

    name = txt(label, T_TINY, INK, BOLD).next_to(shell, DOWN, buff=0.14)
    n = txt(count, T_TINY * 0.95, color, BOLD).next_to(name, DOWN, buff=0.05)
    return VGroup(shell, name, n)


def latent_backdrop(width: float = 8.4, height: float = 4.0, color: str = "#2E6FD6") -> VGroup:
    """A soft field behind the graph, so it reads as a latent space and not a
    flowchart floating on paper."""
    field = RoundedRectangle(
        width=width, height=height, corner_radius=0.3,
        fill_color=tint(color, 0.05), fill_opacity=1,
        stroke_color=color, stroke_width=1.2, stroke_opacity=0.25,
    )
    rng = random.Random(11)
    dust = VGroup(*[
        Dot(
            np.array([rng.uniform(-width / 2 + 0.3, width / 2 - 0.3),
                      rng.uniform(-height / 2 + 0.3, height / 2 - 0.3), 0]),
            radius=rng.uniform(0.012, 0.028),
            color=color,
            fill_opacity=rng.uniform(0.12, 0.3),
        )
        for _ in range(70)
    ])
    label = txt("latent space", T_TINY * 0.8, color).set_opacity(0.5)
    label.move_to(field.get_corner(DL) + RIGHT * 0.75 + UP * 0.22)
    return VGroup(field, dust, label)


# --- the graph shared by stages 2 and 3 ------------------------------------ #
GRAPH_PTS = [
    LEFT * 3.5 + UP * 0.75,
    LEFT * 1.85 + UP * 1.35,
    LEFT * 0.15 + UP * 0.95,
    LEFT * 2.7 + DOWN * 0.75,
    RIGHT * 1.5 + UP * 1.4,
    LEFT * 0.9 + DOWN * 1.25,
    RIGHT * 0.95 + DOWN * 0.35,
    RIGHT * 2.9 + UP * 0.35,
    RIGHT * 2.2 + DOWN * 1.3,
    LEFT * 3.15 + DOWN * 1.5,
]
GRAPH_EDGES = [
    (0, 1), (1, 2), (2, 4), (0, 3), (3, 5), (5, 6), (6, 7), (4, 7),
    (2, 6), (1, 3), (5, 8), (6, 8), (3, 9), (9, 5), (2, 7), (4, 2), (8, 7),
]
# The chain the walk picks out, over that same graph.
GRAPH_WALK = [(0, 3), (3, 5), (5, 6), (6, 7)]


def graph_nodes(color: str, radius: float = 0.14) -> VGroup:
    return VGroup(*[
        Circle(radius=radius, fill_color=PAPER, fill_opacity=1,
               stroke_color=color, stroke_width=2).move_to(p)
        for p in GRAPH_PTS
    ])


def graph_edges(color: str, width: float = 1.5, opacity: float = 0.55) -> VGroup:
    """Straight edges. The graph is the one place the first draft's curves made
    the layout unreadable once the node count went up."""
    return VGroup(*[
        Line(GRAPH_PTS[a], GRAPH_PTS[b], color=color, stroke_width=width,
             stroke_opacity=opacity)
        for a, b in GRAPH_EDGES
    ])


def edge_index(a: int, b: int) -> int:
    for i, (x, y) in enumerate(GRAPH_EDGES):
        if (x, y) == (a, b) or (x, y) == (b, a):
            return i
    raise KeyError(f"edge {a}-{b} is not in the graph")


# --- json ------------------------------------------------------------------ #
def json_frame(
    title: str,
    rows: Sequence[tuple[str, str]],
    color: str,
    width: float = 3.3,
    key_color: str | None = None,
) -> VGroup:
    """A metadata frame with real JSON formatting: coloured keys, quoted values,
    braces. The first draft dumped raw text and it looked like a log file."""
    key_color = key_color or color
    head = txt(title, T_TINY * 0.85, color, BOLD)

    lines = VGroup()
    lines.add(txt("{", T_TINY * 0.9, MUTED, weight=NORMAL))
    for k, v in rows:
        k_t = txt(f'"{k}"', T_TINY * 0.82, key_color, BOLD)
        c_t = txt(":", T_TINY * 0.82, MUTED)
        v_t = txt(f'"{_clip(v, 20)}"', T_TINY * 0.82, INK)
        row = VGroup(k_t, c_t, v_t).arrange(RIGHT, buff=0.05)
        row.shift(RIGHT * 0.18)
        lines.add(row)
    lines.add(txt("}", T_TINY * 0.9, MUTED, weight=NORMAL))
    lines.arrange(DOWN, buff=0.075, aligned_edge=LEFT)

    bg = RoundedRectangle(
        width=max(width, lines.width + 0.5), height=lines.height + head.height + 0.55,
        corner_radius=0.12,
        fill_color=PAPER, fill_opacity=1,
        stroke_color=color, stroke_width=1.6,
    )
    head.move_to(bg.get_top() + DOWN * 0.2)
    lines.next_to(head, DOWN, buff=0.14)
    lines.align_to(bg.get_left() + RIGHT * 0.24, LEFT)
    return VGroup(bg, head, lines)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def judge_logo(which: str, height: float = 0.42) -> Mobject:
    """The real brand mark if we have it; a neutral labelled chip if not."""
    path = QWEN_LOGO if which == "qwen" else GEMMA_LOGO
    if path.is_file():
        img = ImageMobject(str(path))
        img.height = height
        return img
    label = "Qwen3.6" if which == "qwen" else "Gemma 4"
    t = txt(label, T_TINY, INK, BOLD)
    bg = RoundedRectangle(width=t.width + 0.24, height=height, corner_radius=0.06,
                          fill_color=PAPER, fill_opacity=1,
                          stroke_color=FAINT, stroke_width=1.4)
    return VGroup(bg, t.move_to(bg.get_center()))

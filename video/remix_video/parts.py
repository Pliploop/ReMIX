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


ICONS = Path(__file__).resolve().parents[1] / "assets" / "icons"


def svg_icon(name: str, color: str, height: float = 0.4) -> VMobject:
    """A real icon from an SVG path, not hand-drawn strokes."""
    m = SVGMobject(str(ICONS / f"{name}.svg"))
    m.set_fill(color, opacity=1).set_stroke(width=0)
    m.height = height
    return m


def music_note(color: str, height: float = 0.5) -> VMobject:
    return svg_icon("music_note", color, height)


def person(color: str, height: float = 0.5) -> VMobject:
    return svg_icon("person", color, height)


def cylinder(color: str, w: float = 1.5, h: float = 1.15, ry: float = 0.19) -> VGroup:
    """An actual cylinder.

    The previous one stretched an Arc, which left the bottom curve out of line
    with the sides and a white gap between them. Built here as one closed
    silhouette -- down the left side, round the front of the base as a true
    half-ellipse, up the right side -- so the outline meets exactly.
    """
    hw = w / 2

    def base_pt(t: float) -> np.ndarray:
        return np.array([hw * np.cos(t), -h / 2 + ry * np.sin(t), 0.0])

    # Sample the ellipse explicitly. ParametricFunction.points are bezier control
    # points, not samples -- feeding them to set_points_as_corners mangles the
    # outline, which is where the white gap under the body came from.
    front_pts = [base_pt(t) for t in np.linspace(PI, TAU, 48)]
    back_pts = [base_pt(t) for t in np.linspace(0, PI, 48)]

    # Closed body: down the left wall, round the front of the base, up the right.
    silhouette = VMobject(
        fill_color=tint(color, 0.1), fill_opacity=1,
        stroke_color=color, stroke_width=2,
    )
    silhouette.set_points_as_corners(
        [np.array([-hw, h / 2, 0.0])] + front_pts + [np.array([hw, h / 2, 0.0])]
    )

    # The base's back half, drawn faint, is what makes it read as 3D.
    back = VMobject(stroke_color=color, stroke_width=1.2, stroke_opacity=0.35)
    back.set_points_as_corners(back_pts)

    top = Ellipse(width=w, height=ry * 2, fill_color=tint(color, 0.26), fill_opacity=1,
                  stroke_color=color, stroke_width=2)
    top.move_to(np.array([0.0, h / 2, 0.0]))

    return VGroup(silhouette, back, top)


def catalogue(label: str, count: str, color: str, w: float = 1.5, h: float = 1.15) -> VGroup:
    """A cylinder that is visibly a music catalogue."""
    shell = cylinder(color, w, h)
    note = music_note(color, 0.52).move_to(np.array([0.0, -0.08, 0.0]))
    body = VGroup(shell, note)

    name = txt(label, T_TINY, INK, BOLD).next_to(body, DOWN, buff=0.16)
    n = txt(count, T_TINY * 0.95, color, BOLD).next_to(name, DOWN, buff=0.05)
    return VGroup(body, name, n)


def latent_grid(width: float = 9.6, height: float = 3.9, color: str = "#2E6FD6",
                spacing: float = 0.62) -> VGroup:
    """A grid that fades out at the edges: a coordinate space, not a rectangle
    with dust in it. The fade is per-line opacity keyed to distance from centre,
    so the field has no hard border to bump against."""
    lines = VGroup()
    hw, hh = width / 2, height / 2

    n_v = int(hw / spacing)
    for i in range(-n_v, n_v + 1):
        x = i * spacing
        fade = max(0.0, 1.0 - (abs(x) / hw) ** 1.7)
        if fade <= 0.02:
            continue
        lines.add(Line([x, -hh, 0], [x, hh, 0], color=color,
                       stroke_width=1.0, stroke_opacity=0.3 * fade))

    n_h = int(hh / spacing)
    for j in range(-n_h, n_h + 1):
        y = j * spacing
        fade = max(0.0, 1.0 - (abs(y) / hh) ** 1.7)
        if fade <= 0.02:
            continue
        lines.add(Line([-hw, y, 0], [hw, y, 0], color=color,
                       stroke_width=1.0, stroke_opacity=0.3 * fade))

    label = txt("latent space", T_TINY * 0.8, color).set_opacity(0.45)
    label.move_to(np.array([-hw + 0.85, -hh + 0.3, 0.0]))
    return VGroup(lines, label)


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

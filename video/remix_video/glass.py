"""Glassy cards, organic links, and the stage panels that accumulate top-left.

Manim has no backdrop blur, so "glass" is faked the way print does it: a soft
shadow, a translucent white body over a faint colour wash, and a bright hairline
edge. At video scale it reads as glass.

Chain links are cubic beziers, never straight arrows -- a chain is a walk through
a similarity space, and a curve says that where a ruler-straight line says
"flowchart".
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from manim import *

from .components import Waveform, _chip, _clip
from .theme import FAINT, FONT, HAIR, INK, INSTRUCT, MUTED, PAPER, T_TINY, T_SMALL, tint, txt


class GlassCard(VGroup):
    """A track that looks like a player: glassy body, play button, waveform."""

    def __init__(
        self,
        title: str,
        artist: str,
        seed: int = 0,
        color: str = "#2E6FD6",
        tags: Sequence[str] = (),
        width: float = 3.3,
        energy: float = 1.0,
        playing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        height = 1.62 if not tags else 1.95

        # shadow -> wash -> body -> edge highlight
        shadow = RoundedRectangle(
            width=width, height=height, corner_radius=0.2,
            fill_color="#000000", fill_opacity=0.05, stroke_width=0,
        ).shift(DOWN * 0.055)
        wash = RoundedRectangle(
            width=width, height=height, corner_radius=0.2,
            fill_color=color, fill_opacity=0.12, stroke_width=0,
        )
        body = RoundedRectangle(
            width=width, height=height, corner_radius=0.2,
            fill_color=PAPER, fill_opacity=0.62,
            stroke_color=color, stroke_width=1.6, stroke_opacity=0.55,
        )
        gleam = RoundedRectangle(
            width=width - 0.22, height=height * 0.42, corner_radius=0.16,
            fill_color=PAPER, fill_opacity=0.30, stroke_width=0,
        ).move_to(body.get_top() + DOWN * (height * 0.24))

        self.plate = VGroup(shadow, wash, body, gleam)

        # play button
        self.button = VGroup(
            Circle(radius=0.19, fill_color=color, fill_opacity=1, stroke_width=0),
            Triangle(fill_color=PAPER, fill_opacity=1, stroke_width=0)
            .scale(0.075).rotate(-PI / 2).shift(RIGHT * 0.022),
        )
        if playing:
            self.button[1] = VGroup(
                RoundedRectangle(width=0.045, height=0.14, corner_radius=0.02,
                                 fill_color=PAPER, fill_opacity=1, stroke_width=0).shift(LEFT * 0.037),
                RoundedRectangle(width=0.045, height=0.14, corner_radius=0.02,
                                 fill_color=PAPER, fill_opacity=1, stroke_width=0).shift(RIGHT * 0.037),
            )
            self.button = VGroup(self.button[0], self.button[1])

        self.title = txt(_clip(title, 20), T_TINY * 1.18, INK, BOLD)
        self.artist = txt(_clip(artist, 22), T_TINY * 0.92, MUTED)
        head = VGroup(self.title, self.artist).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        top = VGroup(self.button, head).arrange(RIGHT, buff=0.16, aligned_edge=UP)

        self.wave = Waveform(seed=seed, n=26, width=width - 0.5, height=0.42,
                             color=color, energy=energy)

        stack = VGroup(top, self.wave)
        if tags:
            stack.add(VGroup(*[_chip(t, color) for t in tags[:3]]).arrange(RIGHT, buff=0.08))
        stack.arrange(DOWN, buff=0.14).move_to(body.get_center())

        self.add(self.plate, stack)

    def pulse(self, factor: float = 1.07, run_time: float = 0.5):
        return Succession(
            self.wave.animate(run_time=run_time / 2).scale(factor),
            self.wave.animate(run_time=run_time / 2).scale(1 / factor),
        )


def elbow_link(
    a,
    b,
    color: str = INSTRUCT,
    width: float = 2.6,
    radius: float = 0.16,
    tip: bool = True,
    mid: float | None = None,
) -> VGroup:
    """A three-segment orthogonal connector with rounded elbows: out, across, in.

    This is the paper figure's language -- diagram plumbing is elbowed, not
    curved. Curves are reserved for the graph and the chain, where the bend means
    something.

    Three segments, not two: a two-segment L runs to the target's x and then turns,
    so it arrives from underneath with the arrowhead pointing up. Boxes want to be
    entered from the side they face.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dx, dy = b[0] - a[0], b[1] - a[1]

    if abs(dy) < 1e-3:
        path = Line(a, b, color=color, stroke_width=width)
        g = VGroup(path)
        if tip:
            g.add(_tip_dir(b, np.array([np.sign(dx) or 1.0, 0.0, 0.0]), color))
        return g

    mx = a[0] + dx * (0.5 if mid is None else mid)
    c1 = np.array([mx, a[1], 0.0])
    c2 = np.array([mx, b[1], 0.0])
    r = min(radius, abs(mx - a[0]) * 0.9, abs(b[0] - mx) * 0.9, abs(dy) / 2)

    hdir = np.sign(dx) or 1.0
    vdir = np.sign(dy)

    # Arc sweep signs: the fillet must curve *into* the corner, centred on the
    # inside of the turn. With the signs inverted the arc bulges outward, adding
    # a visible kink at each end so a 3-segment elbow reads as 5 with hard
    # corners. Going right-then-up, the first fillet is centred at (mx-r, ay+r),
    # so it sweeps counter-clockwise: +PI/2.
    path = VMobject(color=color, stroke_width=width)
    path.set_points_as_corners([a, c1 - np.array([hdir * r, 0, 0])])
    path.append_points(
        ArcBetweenPoints(
            c1 - np.array([hdir * r, 0, 0]),
            c1 + np.array([0, vdir * r, 0]),
            angle=PI / 2 * hdir * vdir,
        ).points
    )
    path.add_points_as_corners([c2 - np.array([0, vdir * r, 0])])
    path.append_points(
        ArcBetweenPoints(
            c2 - np.array([0, vdir * r, 0]),
            c2 + np.array([hdir * r, 0, 0]),
            angle=-PI / 2 * hdir * vdir,
        ).points
    )
    path.add_points_as_corners([b])

    g = VGroup(path)
    if tip:
        g.add(_tip_dir(b, np.array([hdir, 0.0, 0.0]), color))
    return g


def _tip_dir(at, direction, color: str, size: float = 0.13) -> Triangle:
    angle = np.arctan2(direction[1], direction[0])
    return (
        Triangle(fill_color=color, fill_opacity=1, stroke_width=0)
        .scale(size)
        .rotate(angle - PI / 2)
        .move_to(at)
    )


def fork_link(
    source,
    targets,
    color: str = INSTRUCT,
    width: float = 2.2,
    radius: float = 0.14,
    mid: float = 0.45,
    tip: bool = True,
    reverse: bool = False,
) -> VGroup:
    """One source fanning to several targets (or several sources merging to one).

    A bus: a trunk out to a shared x, a spine down that x, and a spur into each
    target. Drawing N independent elbows instead makes them share the trunk and
    the turn, and their mirrored fillets overlap into a lens -- which is what
    looked like a bubble at every junction.

    reverse=True merges `targets` into `source` instead of fanning out.
    """
    source = np.array(source, dtype=float)
    targets = [np.array(t, dtype=float) for t in targets]

    mx = source[0] + (targets[0][0] - source[0]) * mid
    hdir = np.sign(targets[0][0] - source[0]) or 1.0

    g = VGroup()
    g.add(Line(source, np.array([mx, source[1], 0.0]), color=color, stroke_width=width))

    ys = [t[1] for t in targets] + [source[1]]
    g.add(Line(np.array([mx, min(ys), 0.0]), np.array([mx, max(ys), 0.0]),
               color=color, stroke_width=width))

    for t in targets:
        r = min(radius, abs(t[0] - mx) * 0.9, max(abs(t[1] - source[1]) / 2, 1e-3))
        vdir = np.sign(t[1] - source[1])
        if abs(t[1] - source[1]) < 1e-3 or r < 1e-3:
            g.add(Line(np.array([mx, t[1], 0.0]), t, color=color, stroke_width=width))
        else:
            corner = np.array([mx, t[1], 0.0])
            arc = ArcBetweenPoints(
                corner - np.array([0, vdir * r, 0]),
                corner + np.array([hdir * r, 0, 0]),
                angle=-PI / 2 * hdir * vdir,
            )
            arc.set_stroke(color, width)
            g.add(arc)
            g.add(Line(corner + np.array([hdir * r, 0, 0]), t, color=color, stroke_width=width))
        if tip:
            g.add(_tip_dir(t, np.array([hdir if not reverse else -hdir, 0.0, 0.0]), color))
    return g


def organic_link(
    a: np.ndarray,
    b: np.ndarray,
    color: str = INSTRUCT,
    width: float = 3.0,
    bow: float = 0.42,
    seed: int = 0,
    tip: bool = True,
) -> VMobject:
    """A curved link. Bows perpendicular to the run, sign varying with seed, so a
    graph of these looks grown rather than drawn with a ruler."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    d = b - a
    n = np.array([-d[1], d[0], 0.0])
    norm = np.linalg.norm(n)
    if norm > 1e-6:
        n = n / norm
    sign = 1.0 if (seed % 2 == 0) else -1.0
    rng = random.Random(seed)
    lift = n * bow * sign * rng.uniform(0.7, 1.25)

    c1 = a + d * 0.28 + lift
    c2 = a + d * 0.72 + lift

    curve = CubicBezier(a, c1, c2, b, color=color, stroke_width=width)
    if not tip:
        return curve
    g = VGroup(curve)
    g.add(_tip_at(curve, color))
    return g


def _tip_at(curve: VMobject, color: str, size: float = 0.15) -> Triangle:
    end = curve.point_from_proportion(1.0)
    just_before = curve.point_from_proportion(0.97)
    d = end - just_before
    angle = np.arctan2(d[1], d[0])
    return (
        Triangle(fill_color=color, fill_opacity=1, stroke_width=0)
        .scale(size)
        .rotate(angle - PI / 2)
        .move_to(end)
    )


class StatBadge(VGroup):
    """A number that earns its place: big figure, small label."""

    def __init__(self, value: str, label: str, color: str = INK, size: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        v = txt(value, size, color, BOLD)
        l = txt(label, T_TINY, MUTED)
        self.add(VGroup(v, l).arrange(DOWN, buff=0.09))


# One graph, shared by the stage 2 and stage 3 icons. Deliberately dense with
# crossing edges: a neighbourhood is overlapping, not a tidy tree.
GRAPH_ICON_PTS = [
    LEFT * 0.42 + UP * 0.06,
    LEFT * 0.16 + UP * 0.26,
    RIGHT * 0.16 + UP * 0.24,
    LEFT * 0.26 + DOWN * 0.22,
    RIGHT * 0.44 + UP * 0.02,
    RIGHT * 0.06 + DOWN * 0.28,
    RIGHT * 0.3 + DOWN * 0.1,
    LEFT * 0.02 + UP * 0.02,
]
GRAPH_ICON_EDGES = [
    (0, 1), (1, 2), (2, 4), (0, 3), (3, 5), (5, 6), (6, 4),
    (1, 7), (7, 2), (3, 7), (7, 6), (0, 7), (5, 7), (2, 6),
]


def panel_icon(n: int, color: str, scale: float = 1.0) -> VGroup:
    """A tiny sketch of what a stage does, in straight lines.

    Without these the rail is five empty boxes, and the final assemble -- where
    the rail flies to centre and becomes the paper's main figure -- has nothing
    to become. Straight, not curved: at icon size a bezier reads as a smudge.
    """
    g = VGroup()
    if n == 1:  # catalogue -> rows
        cyl = _mini_cylinder(color)
        rows = VGroup(*[
            RoundedRectangle(width=0.4, height=0.055, corner_radius=0.02,
                             fill_color=tint(color, 0.34), fill_opacity=1, stroke_width=0)
            for _ in range(4)
        ]).arrange(DOWN, buff=0.04)
        g.add(VGroup(cyl, rows).arrange(RIGHT, buff=0.2))
    elif n in (2, 3):
        # Stages 2 and 3 share one graph: stage 3 *is* stage 2 with a path picked
        # out. Drawing two different graphs implied the chain lived somewhere
        # else. Dense and overlapping, because that is what a neighbourhood is.
        pts = GRAPH_ICON_PTS
        walk = [(0, 3), (3, 6), (6, 4)]
        for a, b in GRAPH_ICON_EDGES:
            on_walk = (a, b) in walk or (b, a) in walk
            if n == 2:
                g.add(Line(pts[a], pts[b], color=color, stroke_width=1.0, stroke_opacity=0.6))
            elif on_walk:
                g.add(Line(pts[a], pts[b], color=color, stroke_width=2.4))
            else:
                g.add(Line(pts[a], pts[b], color=MUTED, stroke_width=0.8, stroke_opacity=0.22))
        walk_nodes = {i for e in walk for i in e}
        for i, p in enumerate(pts):
            if n == 3 and i not in walk_nodes:
                g.add(Dot(p, radius=0.03, color=MUTED, fill_opacity=0.3))
            else:
                g.add(Dot(p, radius=0.042, color=color))
    elif n == 4:  # instruction lines
        g.add(VGroup(*[
            RoundedRectangle(width=w, height=0.13, corner_radius=0.06,
                             fill_color=tint(color, 0.26), fill_opacity=1,
                             stroke_color=color, stroke_width=0.7)
            for w in (0.66, 0.5, 0.6)
        ]).arrange(DOWN, buff=0.07, aligned_edge=LEFT))
    else:  # rubric + gate
        bars = VGroup(*[
            VGroup(
                RoundedRectangle(width=0.32, height=0.05, corner_radius=0.02,
                                 fill_color=tint(color, 0.18), fill_opacity=1, stroke_width=0),
                RoundedRectangle(width=0.32 * f, height=0.05, corner_radius=0.02,
                                 fill_color=color, fill_opacity=1, stroke_width=0),
            )
            for f in (0.9, 0.7, 0.85)
        ])
        for b in bars:
            b[1].align_to(b[0], LEFT)
        bars.arrange(DOWN, buff=0.06)
        check = VGroup(
            Circle(radius=0.1, fill_color=tint("#1FA347", 0.2), fill_opacity=1,
                   stroke_color="#1FA347", stroke_width=1.2),
            Text("✓", font=FONT, color="#1FA347").scale(0.11),
        )
        g.add(VGroup(bars, check).arrange(RIGHT, buff=0.16))
    return g.scale(scale)


def _mini_cylinder(color: str, w: float = 0.28, h: float = 0.24) -> VGroup:
    body = Rectangle(width=w, height=h, fill_color=tint(color, 0.2), fill_opacity=1, stroke_width=0)
    top = Ellipse(width=w, height=w * 0.36, fill_color=tint(color, 0.42), fill_opacity=1,
                  stroke_color=color, stroke_width=0.9).move_to(body.get_top())
    bot = Arc(radius=w / 2, start_angle=PI, angle=PI, color=color, stroke_width=0.9)
    bot.stretch(0.36, 1).move_to(body.get_bottom())
    l = Line(body.get_corner(UL), body.get_corner(DL), color=color, stroke_width=0.9)
    r = Line(body.get_corner(UR), body.get_corner(DR), color=color, stroke_width=0.9)
    return VGroup(body, l, r, bot, top)


class StagePanel(VGroup):
    """A finished stage, shrunk into the accumulating rail at the top left.

    These assemble into the paper's main figure at the end, so they keep the
    figure's grammar. The label sits *outside* the patch: inside, it crowds the
    icon and the panel stops reading at rail size.
    """

    def __init__(
        self,
        n: int,
        name: str,
        color: str,
        width: float = 1.95,
        height: float = 1.05,
        label_size: float = 0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Square-ish with a generous radius: the rounded-square card of the
        # figure, not a wide tab.
        side = min(width, height)
        bg = RoundedRectangle(
            width=width, height=height, corner_radius=side * 0.22,
            fill_color=tint(color, 0.07), fill_opacity=1,
            stroke_color=color, stroke_width=1.3, stroke_opacity=0.8,
        )
        self.body = panel_icon(n, color, scale=min(1.0, height / 1.05))
        self.body.move_to(bg.get_center())

        label = txt(f"{n}. {name}", T_TINY * label_size, color, BOLD)
        if label.width > width + 0.3:
            label.set(width=width + 0.3)
        label.next_to(bg, DOWN, buff=0.1)

        self.add(bg, self.body, label)
        self.bg = bg
        self.label = label

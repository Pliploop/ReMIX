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
from .theme import FAINT, HAIR, INK, INSTRUCT, MUTED, PAPER, T_TINY, T_SMALL, tint, txt


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


class StagePanel(VGroup):
    """A finished stage, shrunk into the accumulating rail at the top left.

    These are what assemble into the paper's main figure at the end, so they keep
    the figure's grammar: numbered, colour-titled, tinted card.
    """

    def __init__(self, n: int, name: str, color: str, width: float = 2.05, height: float = 1.28, **kwargs):
        super().__init__(**kwargs)
        bg = RoundedRectangle(
            width=width, height=height, corner_radius=0.1,
            fill_color=tint(color, 0.09), fill_opacity=1,
            stroke_color=color, stroke_width=1.6,
        )
        label = txt(f"{n}. {name}", T_TINY * 0.8, color, BOLD)
        label.set(width=min(label.width, width - 0.2))
        label.move_to(bg.get_top() + DOWN * 0.16)
        self.body = VGroup()
        self.add(bg, label)
        self.bg = bg
        self.label = label

"""Reusable mobjects for the ReMIX video: waveforms, track cards, instruction
bubbles, and the stage rail that keeps the viewer oriented across scenes.

These are the video's vocabulary. Scenes compose them; nothing here knows about
any particular scene.
"""

from __future__ import annotations

import random
from typing import Sequence

from manim import *

from .theme import (
    FAINT, HAIR, INK, INSTRUCT, MUTED, PAPER, STAGE_COLORS, STAGE_NAMES,
    T_SMALL, T_TINY, card, tint, txt,
)


class Waveform(VGroup):
    """A stylised waveform. Deterministic per seed, so a track looks like itself
    every time it appears, and different tracks look different."""

    def __init__(
        self,
        seed: int = 0,
        n: int = 28,
        width: float = 2.4,
        height: float = 0.62,
        color: str = MUTED,
        energy: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        rng = random.Random(seed)
        bar_w = width / (n * 1.7)
        self.bars = VGroup()
        for i in range(n):
            # Envelope so it reads as audio rather than noise.
            env = 0.42 + 0.58 * abs(np.sin(i / n * np.pi * rng.uniform(1.4, 3.0) + seed))
            h = max(0.06, height * env * rng.uniform(0.35, 1.0) * energy)
            bar = RoundedRectangle(
                width=bar_w, height=h, corner_radius=bar_w / 2,
                fill_color=color, fill_opacity=1, stroke_width=0,
            )
            self.bars.add(bar)
        self.bars.arrange(RIGHT, buff=bar_w * 0.7)
        self.add(self.bars)

    def set_wave_color(self, color: str):
        for b in self.bars:
            b.set_fill(color)
        return self


class TrackCard(VGroup):
    """A track: title, artist, waveform, optional tag chips."""

    def __init__(
        self,
        title: str,
        artist: str,
        seed: int = 0,
        color: str = "#2E6FD6",
        tags: Sequence[str] = (),
        width: float = 3.5,
        energy: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        height = 2.05 if tags else 1.7
        self.bg = card(width, height, color, alpha=0.07)

        self.title = txt(_clip(title, 22), T_SMALL, INK, BOLD)
        self.artist = txt(_clip(artist, 24), T_TINY, MUTED)
        self.wave = Waveform(seed=seed, width=width - 0.7, color=color, energy=energy)

        body = VGroup(self.title, self.artist, self.wave).arrange(DOWN, buff=0.16)

        if tags:
            chips = VGroup(*[_chip(t, color) for t in tags[:3]]).arrange(RIGHT, buff=0.09)
            body.add(chips)
            body.arrange(DOWN, buff=0.15)

        body.move_to(self.bg.get_center())
        self.add(self.bg, body)

    def pulse(self):
        return Succession(
            self.wave.animate(run_time=0.25).scale(1.06),
            self.wave.animate(run_time=0.25).scale(1 / 1.06),
        )


def _chip(label: str, color: str) -> VGroup:
    t = txt(_clip(label, 14), T_TINY * 0.92, color)
    bg = RoundedRectangle(
        width=t.width + 0.22, height=t.height + 0.14, corner_radius=0.08,
        fill_color=tint(color, 0.16), fill_opacity=1, stroke_width=0,
    )
    return VGroup(bg, t.move_to(bg.get_center()))


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


class InstructionBubble(VGroup):
    """The instruction. This is the hero of the whole video, so it gets the
    orange of stage 4 and sits on the arrow between two tracks."""

    def __init__(self, text: str, width: float = 5.2, color: str = INSTRUCT, **kwargs):
        super().__init__(**kwargs)
        self.label = _wrapped(text, width - 0.5)
        self.bg = RoundedRectangle(
            width=max(width, self.label.width + 0.5),
            height=self.label.height + 0.45,
            corner_radius=0.14,
            fill_color=tint(color, 0.11), fill_opacity=1,
            stroke_color=color, stroke_width=2.0,
        )
        self.label.move_to(self.bg.get_center())
        self.add(self.bg, self.label)


def _wrapped(text: str, max_w: float, size: float = T_SMALL) -> VGroup:
    """Manim has no text wrapping; break on words and stack the lines."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        probe = f"{cur} {w}".strip()
        if txt(probe, size).width > max_w and cur:
            lines.append(cur)
            cur = w
        else:
            cur = probe
    if cur:
        lines.append(cur)
    return VGroup(*[txt(l, size) for l in lines]).arrange(DOWN, buff=0.1, aligned_edge=LEFT)


class StageRail(VGroup):
    """Five chips across the top, mirroring the website's rail.

    This is the video's spine: it tells the viewer which stage they are in and
    makes every scene transition legible instead of a hard cut.
    """

    def __init__(self, width: float = 11.4, **kwargs):
        super().__init__(**kwargs)
        self.chips = VGroup()
        w = width / 5 - 0.16
        for i, (name, color) in enumerate(zip(STAGE_NAMES, STAGE_COLORS)):
            bg = RoundedRectangle(
                width=w, height=0.42, corner_radius=0.1,
                fill_color=PAPER, fill_opacity=1,
                stroke_color=FAINT, stroke_width=1.4,
            )
            label = txt(f"{i + 1}. {name}", T_TINY * 0.92, MUTED).move_to(bg.get_center())
            self.chips.add(VGroup(bg, label))
        self.chips.arrange(RIGHT, buff=0.16)
        self.add(self.chips)

    def activate(self, i: int):
        """Light chip i, dim the rest. Returns an animation."""
        anims = []
        for j, chip in enumerate(self.chips):
            bg, label = chip
            if j == i:
                anims += [
                    bg.animate.set_fill(tint(STAGE_COLORS[j], 0.14)).set_stroke(STAGE_COLORS[j], 2.0),
                    label.animate.set_color(STAGE_COLORS[j]),
                ]
            else:
                anims += [
                    bg.animate.set_fill(PAPER).set_stroke(FAINT, 1.4),
                    label.animate.set_color(MUTED),
                ]
        return AnimationGroup(*anims, run_time=0.5)


class Logo(VGroup):
    """The mark: five bars in the stage colours, an instruction bubble, and an
    arrow looping back to re-shape the waveform."""

    def __init__(self, scale_factor: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        heights = [0.30, 0.50, 0.66, 0.44, 0.33]
        bars = VGroup(*[
            RoundedRectangle(
                width=0.15, height=h, corner_radius=0.07,
                fill_color=c, fill_opacity=1, stroke_width=0,
            )
            for h, c in zip(heights, STAGE_COLORS)
        ]).arrange(RIGHT, buff=0.07, aligned_edge=DOWN)

        bubble = RoundedRectangle(
            width=0.46, height=0.32, corner_radius=0.1,
            fill_color=PAPER, fill_opacity=1, stroke_color=INK, stroke_width=2.2,
        ).next_to(bars, UR, buff=0.06).shift(RIGHT * 0.06)

        loop = CurvedArrow(
            bubble.get_left() + LEFT * 0.02 + DOWN * 0.06,
            bars.get_left() + UP * 0.30,
            angle=-TAU / 7,
            color=INK,
            stroke_width=2.4,
            tip_length=0.14,
        )
        self.add(bars, bubble, loop)
        self.scale(scale_factor)


def caption(scene: Scene, text: str, at=DOWN * 3.1, size: float = T_SMALL) -> Text:
    """A caption at the bottom. The video is silent, so these carry the words."""
    t = txt(text, size, MUTED).move_to(at)
    return t


def title_card(text: str, color: str = INK, size: float = 0.56) -> VGroup:
    """A short statement, centred, with a coloured underline."""
    t = txt(text, size, INK, BOLD)
    rule = Line(LEFT, RIGHT, color=color, stroke_width=4).set_width(min(t.width, 4.2))
    rule.next_to(t, DOWN, buff=0.22)
    return VGroup(t, rule)

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
    # Centred: a quoted instruction ragged-right in a centred bubble looks like a
    # mistake rather than a choice.
    return VGroup(*[txt(l, size) for l in lines]).arrange(DOWN, buff=0.1)


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


def _arrow_tip(at, direction, color: str, size: float = 0.13) -> Triangle:
    angle = np.arctan2(direction[1], direction[0])
    return (
        Triangle(fill_color=color, fill_opacity=1, stroke_width=0)
        .scale(size)
        .rotate(angle - PI / 2)
        .move_to(at)
    )


class SpeechBubble(VGroup):
    """A bubble that reads as one: rounded body, a tail pointing down-left, and two
    lines of 'text'. Glassy -- a colour wash under a translucent body with a gleam
    across the top, the same trick the track cards use."""

    def __init__(self, w: float = 0.62, h: float = 0.42, color: str = INK,
                 wash: str = "#7B3FF2", **kwargs):
        super().__init__(**kwargs)
        shadow = RoundedRectangle(
            width=w, height=h, corner_radius=0.11,
            fill_color="#000000", fill_opacity=0.06, stroke_width=0,
        ).shift(DOWN * 0.03)
        tintwash = RoundedRectangle(
            width=w, height=h, corner_radius=0.11,
            fill_color=wash, fill_opacity=0.16, stroke_width=0,
        )
        body = RoundedRectangle(
            width=w, height=h, corner_radius=0.11,
            fill_color=PAPER, fill_opacity=0.66, stroke_color=color, stroke_width=2.2,
        )
        gleam = RoundedRectangle(
            width=w - 0.13, height=h * 0.4, corner_radius=0.08,
            fill_color=PAPER, fill_opacity=0.4, stroke_width=0,
        ).move_to(body.get_top() + DOWN * (h * 0.23))
        self.add(shadow, tintwash)
        # Tail: a triangle overlapping the body, with a white patch hiding the seam.
        tail = Polygon(
            body.get_bottom() + LEFT * w * 0.24 + UP * 0.01,
            body.get_bottom() + LEFT * w * 0.06 + UP * 0.01,
            body.get_bottom() + LEFT * w * 0.30 + DOWN * 0.16,
            fill_color=PAPER, fill_opacity=1, stroke_color=color, stroke_width=2.2,
        )
        seam = Line(
            body.get_bottom() + LEFT * w * 0.24 + UP * 0.012,
            body.get_bottom() + LEFT * w * 0.06 + UP * 0.012,
            color=PAPER, stroke_width=3.2,
        )
        lines = VGroup(
            Line(ORIGIN, RIGHT * w * 0.42, color=color, stroke_width=1.8),
            Line(ORIGIN, RIGHT * w * 0.26, color=color, stroke_width=1.8),
        ).arrange(DOWN, buff=0.09, aligned_edge=LEFT).move_to(body.get_center())
        self.add(body, gleam, tail, seam, lines)


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

        # A real loop: an almost-closed circular arrow *around* the waveform, with
        # the bubble sitting in its gap. The previous arc started at the bubble
        # and stopped in mid-air, so it read as a stray swoosh rather than a
        # cycle. A loop says "do it again", which is the whole idea.
        centre = bars.get_center()
        radius = max(bars.width, bars.height) * 0.78

        start_a = TAU * 0.07     # just below the bubble, on the right
        sweep = TAU * 0.78       # counter-clockwise, leaving a gap top-right
        loop = Arc(
            radius=radius, start_angle=start_a, angle=sweep,
            arc_center=centre, color=INK, stroke_width=2.4,
        )
        end_pt = loop.point_from_proportion(1.0)
        prev_pt = loop.point_from_proportion(0.985)
        tip = _arrow_tip(end_pt, end_pt - prev_pt, INK, 0.115)

        bubble = SpeechBubble(0.6, 0.4, INK)
        bubble.move_to(centre + np.array([np.cos(TAU * 0.075), np.sin(TAU * 0.075), 0]) * radius * 1.16)

        self.add(loop, tip, bars, bubble)
        self.scale(scale_factor)


def wordmark(size: float = 0.92) -> VGroup:
    """ReMIX, with the acronym letters carrying the stage colours."""
    letters = VGroup(
        txt("R", size, STAGE_COLORS[0], BOLD),
        txt("e", size, MUTED, BOLD),
        txt("M", size, STAGE_COLORS[1], BOLD),
        txt("I", size, STAGE_COLORS[3], BOLD),
        txt("X", size, STAGE_COLORS[4], BOLD),
    ).arrange(RIGHT, buff=0.045, aligned_edge=DOWN)
    return letters


def expansion(size: float = 0.26) -> VGroup:
    """Retrieval of Music with Instruction Xpression — initials coloured to match
    the wordmark, so the acronym is legible without a caption explaining it.

    Built word by word: manim collapses trailing spaces, so a single Text with
    "etrieval of " loses the gap and the line runs together.
    """
    def word(head: str | None, head_color: str, rest: str) -> VGroup:
        parts = []
        if head:
            parts.append(txt(head, size, head_color, BOLD))
        if rest:
            parts.append(txt(rest, size, MUTED, NORMAL))
        g = VGroup(*parts)
        if len(g) > 1:
            g.arrange(RIGHT, buff=0.012, aligned_edge=DOWN)
        return g

    words = VGroup(
        word("R", STAGE_COLORS[0], "etrieval"),
        word(None, MUTED, "of"),
        word("M", STAGE_COLORS[1], "usic"),
        word(None, MUTED, "with"),
        word("I", STAGE_COLORS[3], "nstruction"),
        word("X", STAGE_COLORS[4], "pression"),
    )
    words.arrange(RIGHT, buff=size * 0.42, aligned_edge=DOWN)
    return words


def _icon_paper(color: str) -> VGroup:
    page = RoundedRectangle(width=0.17, height=0.22, corner_radius=0.03,
                            fill_color=PAPER, fill_opacity=1, stroke_color=color, stroke_width=1.6)
    lines = VGroup(*[
        Line(ORIGIN, RIGHT * 0.09, color=color, stroke_width=1.2) for _ in range(3)
    ]).arrange(DOWN, buff=0.04).move_to(page.get_center())
    return VGroup(page, lines)


def _icon_dataset(color: str) -> VGroup:
    return _cyl_icon(color)


def _cyl_icon(color: str, w: float = 0.19, h: float = 0.16) -> VGroup:
    body = Rectangle(width=w, height=h, fill_color=PAPER, fill_opacity=1, stroke_width=0)
    top = Ellipse(width=w, height=w * 0.4, fill_color=PAPER, fill_opacity=1,
                  stroke_color=color, stroke_width=1.5).move_to(body.get_top())
    bot = Arc(radius=w / 2, start_angle=PI, angle=PI, color=color, stroke_width=1.5)
    bot.stretch(0.4, 1).move_to(body.get_bottom())
    l = Line(body.get_corner(UL), body.get_corner(DL), color=color, stroke_width=1.5)
    r = Line(body.get_corner(UR), body.get_corner(DR), color=color, stroke_width=1.5)
    return VGroup(body, l, r, bot, top)


def _icon_code(color: str) -> VGroup:
    left = VGroup(
        Line(RIGHT * 0.05 + UP * 0.09, LEFT * 0.05, color=color, stroke_width=1.8),
        Line(LEFT * 0.05, RIGHT * 0.05 + DOWN * 0.09, color=color, stroke_width=1.8),
    )
    right = left.copy().rotate(PI).shift(RIGHT * 0.19)
    return VGroup(left, right)


def link_pill(label: str, kind: str, color: str = INK) -> VGroup:
    """A sleek pill with an icon, matching the website's link buttons."""
    icon = {"paper": _icon_paper, "dataset": _icon_dataset, "code": _icon_code}[kind](color)
    t = txt(label, T_SMALL * 0.92, color)
    row = VGroup(icon, t).arrange(RIGHT, buff=0.13)
    bg = RoundedRectangle(
        width=row.width + 0.44, height=row.height + 0.26,
        corner_radius=(row.height + 0.26) / 2,
        fill_color=PAPER, fill_opacity=1, stroke_color=FAINT, stroke_width=1.6,
    )
    row.move_to(bg.get_center())
    return VGroup(bg, row)


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

"""Stage 2 — Neighbourhood Building.

Each clip is embedded twice: audio with MuQ-MuLan, text with EmbeddingGemma. The
two similarities are averaged, weak edges are pruned, and what is left is a graph
of transitions that are actually plausible.

The formula is set in Unicode, not LaTeX: no TeX install here, and Computer
Modern serif would fight the sans identity anyway.
"""

from __future__ import annotations

import random

from manim import *

from remix_video.facts import AUDIO_DIM, AUDIO_MODEL, TEXT_DIM, TEXT_MODEL
from remix_video.glass import GlassCard, organic_link
from remix_video.stagebase import StageScene, explain
from remix_video.theme import INK, MUTED, NEIGHBOUR, PAPER, T_TINY, card, tint, txt


def vector_strip(n: int = 7, color: str = NEIGHBOUR, seed: int = 0) -> VGroup:
    """The little embedding boxes from the paper figure."""
    rng = random.Random(seed)
    cells = VGroup(*[
        Square(side_length=0.17, fill_color=tint(color, rng.uniform(0.15, 0.62)),
               fill_opacity=1, stroke_color=color, stroke_width=0.9)
        for _ in range(n)
    ]).arrange(RIGHT, buff=0.03)
    dots = txt("…", T_TINY, MUTED).next_to(cells, RIGHT, buff=0.08)
    return VGroup(cells, dots)


def model_box(name: str, dim: int, color: str) -> VGroup:
    box = card(2.15, 0.66, color, alpha=0.13, radius=0.12)
    t = txt(name, T_TINY * 1.05, color, BOLD).move_to(box.get_center() + UP * 0.08)
    d = txt(f"{dim}-d", T_TINY * 0.85, MUTED).move_to(box.get_center() + DOWN * 0.15)
    return VGroup(box, t, d)


class Neighbourhood(StageScene):
    stage_index = 1

    def construct(self):
        rail, header = self.open_stage(upto=1)
        content = VGroup()

        line = explain("Every clip is embedded twice — how it sounds, and how it is described.")
        content.add(line)

        # --- two encoders --------------------------------------------------- #
        clip = GlassCard("Rubicon of Gits.", "Conway Hambone", seed=3, color=NEIGHBOUR,
                         width=2.7).scale(0.82).move_to(LEFT * 4.5 + UP * 0.15)

        audio_m = model_box(AUDIO_MODEL, AUDIO_DIM, NEIGHBOUR).move_to(LEFT * 1.35 + UP * 1.15)
        text_m = model_box(TEXT_MODEL, TEXT_DIM, NEIGHBOUR).move_to(LEFT * 1.35 + DOWN * 0.55)
        v1 = vector_strip(7, NEIGHBOUR, 1).next_to(audio_m, RIGHT, buff=0.3)
        v2 = vector_strip(7, NEIGHBOUR, 2).next_to(text_m, RIGHT, buff=0.3)

        feeds = VGroup(
            organic_link(clip.get_right() + RIGHT * 0.05 + UP * 0.2, audio_m.get_left() + LEFT * 0.05,
                         NEIGHBOUR, 2.2, bow=0.18, seed=0),
            organic_link(clip.get_right() + RIGHT * 0.05 + DOWN * 0.2, text_m.get_left() + LEFT * 0.05,
                         NEIGHBOUR, 2.2, bow=0.18, seed=1),
        )
        content.add(clip, audio_m, text_m, v1, v2, feeds)

        self.play(FadeIn(clip, shift=RIGHT * 0.2), FadeIn(line), run_time=0.8)
        self.play(
            *[Create(f) for f in feeds],
            FadeIn(audio_m, shift=UP * 0.1), FadeIn(text_m, shift=DOWN * 0.1),
            run_time=0.9,
        )
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.6) for c in v1[0]], lag_ratio=0.05),
            LaggedStart(*[FadeIn(c, scale=0.6) for c in v2[0]], lag_ratio=0.05),
            FadeIn(v1[1]), FadeIn(v2[1]),
            run_time=0.9,
        )
        self.wait(0.4)

        # --- the composite similarity ---------------------------------------- #
        fbox = card(3.5, 0.9, NEIGHBOUR, alpha=0.09, radius=0.12).move_to(RIGHT * 3.9 + UP * 0.3)
        # Subscripts via U+2090/U+209C, not the Mathematical Bold block (U+1D400+):
        # the sans font has no glyphs for those and renders tofu.
        formula = txt("s(A,B) = ½ (sₐ + sₜ)", 0.34, NEIGHBOUR, BOLD).move_to(fbox.get_center())
        content.add(fbox, formula)

        line2 = explain("Sound and description are averaged into one similarity.")
        content.add(line2)
        self.play(
            FadeIn(fbox), Write(formula),
            ReplacementTransform(line, line2),
            run_time=0.9,
        )
        self.wait(0.7)

        # --- the graph, grown then pruned ------------------------------------ #
        self.play(
            FadeOut(VGroup(clip, audio_m, text_m, v1, v2, feeds)),
            VGroup(fbox, formula).animate.scale(0.78).move_to(UP * 2.0 + RIGHT * 4.3),
            run_time=0.6,
        )

        rng = random.Random(7)
        pts = [
            LEFT * 3.6 + UP * 0.9, LEFT * 1.6 + UP * 1.5, RIGHT * 0.4 + UP * 0.7,
            LEFT * 2.8 + DOWN * 1.0, LEFT * 0.6 + DOWN * 0.6, RIGHT * 1.8 + DOWN * 1.4,
            RIGHT * 2.6 + UP * 1.4, RIGHT * 0.2 + DOWN * 1.8,
        ]
        nodes = VGroup(*[
            Circle(radius=0.15, fill_color=tint(NEIGHBOUR, 0.35), fill_opacity=1,
                   stroke_color=NEIGHBOUR, stroke_width=2).move_to(p)
            for p in pts
        ])
        keep_pairs = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (2, 6), (4, 7)]
        drop_pairs = [(1, 4), (2, 7), (3, 6), (5, 6)]

        kept = VGroup(*[
            organic_link(pts[a], pts[b], NEIGHBOUR, 2.0, bow=0.3, seed=i, tip=False)
            for i, (a, b) in enumerate(keep_pairs)
        ])
        dropped = VGroup(*[
            DashedVMobject(
                organic_link(pts[a], pts[b], MUTED, 1.4, bow=0.3, seed=i + 20, tip=False),
                num_dashes=18,
            )
            for i, (a, b) in enumerate(drop_pairs)
        ])
        content.add(nodes, kept, dropped)

        line3 = explain("Weak links are pruned. What is left is a graph of plausible moves.")
        content.add(line3)
        self.play(
            LaggedStart(*[GrowFromCenter(n) for n in nodes], lag_ratio=0.06),
            ReplacementTransform(line2, line3),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[Create(e) for e in [*kept, *dropped]], lag_ratio=0.05),
            run_time=1.1,
        )
        self.wait(0.4)
        # the prune
        self.play(FadeOut(dropped, scale=0.9), run_time=0.6)
        self.wait(0.9)

        self.close_stage(content, rail, header)

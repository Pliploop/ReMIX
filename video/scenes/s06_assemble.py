"""Assemble, then the logo.

The five panels have been accumulating at the top left for the whole film. Here
they fly back to centre and land as the paper's main figure -- the payoff the
rail has been setting up. Only then does the mark appear.
"""

from __future__ import annotations

from manim import *

from remix_video.components import Logo
from remix_video.facts import FIGURES, thousands
from remix_video.glass import StagePanel
from remix_video.stagebase import SLOT_H, SLOT_W, slot_position
from remix_video.theme import (
    INK, MUTED, PAPER, STAGE_COLORS, STAGE_NAMES, T_SMALL, T_TINY, txt,
)


class Assemble(Scene):
    def construct(self):
        self.camera.background_color = PAPER

        # the rail as the last stage left it
        rail = VGroup()
        for i in range(5):
            p = StagePanel(i + 1, STAGE_NAMES[i], STAGE_COLORS[i], SLOT_W, SLOT_H)
            p.move_to(slot_position(i))
            rail.add(p)
        self.add(rail)
        self.wait(0.5)

        # --- fly to centre and become the figure ---------------------------- #
        big = VGroup()
        for i in range(5):
            p = StagePanel(i + 1, STAGE_NAMES[i], STAGE_COLORS[i], 2.5, 3.1)
            big.add(p)
        big.arrange(RIGHT, buff=0.22).move_to(UP * 0.25)

        # thin black connectors, as in the figure
        links = VGroup(*[
            Arrow(big[i].get_right(), big[i + 1].get_left(), buff=0.03, color=INK,
                  stroke_width=3, max_tip_length_to_length_ratio=0.4, tip_length=0.1)
            for i in range(4)
        ])

        self.play(
            LaggedStart(*[ReplacementTransform(rail[i], big[i]) for i in range(5)], lag_ratio=0.09),
            run_time=1.4,
        )
        self.play(LaggedStart(*[GrowArrow(l) for l in links], lag_ratio=0.08), run_time=0.7)
        self.wait(0.6)

        # --- the funnel, in real numbers ------------------------------------- #
        funnel = VGroup(
            _fig(thousands(FIGURES["clips"]), "clips"),
            _arrowtxt(),
            _fig(thousands(FIGURES["chains"]), "chains"),
            _arrowtxt(),
            _fig(thousands(FIGURES["variants"]), "instructions"),
        ).arrange(RIGHT, buff=0.42).move_to(DOWN * 2.45)
        self.play(FadeIn(funnel, shift=UP * 0.2), run_time=0.8)
        self.wait(1.4)

        # --- dissolve into the mark ------------------------------------------ #
        self.play(FadeOut(funnel), FadeOut(links), run_time=0.5)

        logo = Logo(scale_factor=1.6).move_to(UP * 0.6)
        self.play(
            LaggedStart(*[FadeOut(p, scale=0.7) for p in big], lag_ratio=0.05),
            run_time=0.7,
        )
        self.play(FadeIn(logo, scale=0.85), run_time=0.7)

        name = txt("ReMIX", 0.92, INK, BOLD).next_to(logo, DOWN, buff=0.42)
        sub = txt("Multi-turn, compositional music retrieval", T_SMALL, MUTED).next_to(name, DOWN, buff=0.2)
        self.play(Write(name), run_time=0.6)
        self.play(FadeIn(sub, shift=UP * 0.1), run_time=0.4)

        links_txt = txt("paper · dataset · code · pliploop.github.io/ReMIX", T_TINY, MUTED)
        links_txt.next_to(sub, DOWN, buff=0.5)
        self.play(FadeIn(links_txt), run_time=0.5)
        self.wait(1.8)
        self.play(FadeOut(VGroup(logo, name, sub, links_txt)), run_time=0.8)


def _fig(value: str, label: str) -> VGroup:
    v = txt(value, 0.46, INK, BOLD)
    l = txt(label, T_TINY, MUTED)
    return VGroup(v, l).arrange(DOWN, buff=0.08)


def _arrowtxt() -> Text:
    return txt("→", 0.4, MUTED)

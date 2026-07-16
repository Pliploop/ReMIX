"""Assemble, then the logo.

The five panels have been accumulating in the rail for the whole film. Here they
fly back to centre and land as the paper's main figure -- the payoff the rail has
been setting up. It doubles as the summary, so this is where the figures live.
Only then does the mark appear.
"""

from __future__ import annotations

from manim import *

from remix_video.components import Logo, expansion, link_pill, wordmark
from remix_video.facts import CATALOGUE_TOTAL, FIGURES, thousands
from remix_video.glass import StagePanel
from remix_video.stagebase import SLOT_H, SLOT_W, slot_position
from remix_video.theme import (
    FAINT, INK, MUTED, PAPER, STAGE_COLORS, STAGE_NAMES, T_SMALL, T_TINY, txt,
)


class Assemble(Scene):
    def construct(self):
        self.camera.background_color = PAPER

        rail = VGroup()
        for i in range(5):
            p = StagePanel(i + 1, STAGE_NAMES[i], STAGE_COLORS[i], SLOT_W, SLOT_H, label_size=0.62)
            p.move_to(slot_position(i))
            p.set_opacity(0.9)
            rail.add(p)
        self.add(rail)
        self.wait(0.4)

        # --- fly to centre and become the figure ---------------------------- #
        big = VGroup()
        for i in range(5):
            big.add(StagePanel(i + 1, STAGE_NAMES[i], STAGE_COLORS[i], 2.35, 2.5, label_size=1.0))
        big.arrange(RIGHT, buff=0.3).move_to(UP * 0.7)

        # Bolder connectors than the draft: at this size the thin ones vanished.
        links = VGroup(*[
            Arrow(
                big[i][0].get_right(), big[i + 1][0].get_left(),
                buff=0.04, color=INK, stroke_width=5,
                max_tip_length_to_length_ratio=0.45, tip_length=0.14,
            ).shift(UP * 0.16)
            for i in range(4)
        ])

        self.play(
            LaggedStart(*[ReplacementTransform(rail[i], big[i]) for i in range(5)], lag_ratio=0.1),
            run_time=1.5,
        )
        self.play(LaggedStart(*[GrowArrow(l) for l in links], lag_ratio=0.09), run_time=0.8)
        self.wait(0.5)

        # --- the summary, in real numbers ------------------------------------ #
        figs = VGroup(
            _fig(thousands(CATALOGUE_TOTAL), "clips enriched", STAGE_COLORS[0]),
            _fig(thousands(FIGURES["chains"]), "chains", STAGE_COLORS[2]),
            _fig(thousands(FIGURES["steps"]), "steps", STAGE_COLORS[1]),
            _fig(thousands(FIGURES["variants"]), "instructions", STAGE_COLORS[3]),
            _fig(f"{FIGURES['ac1_lo']}–{FIGURES['ac1_hi']}", "judge agreement", STAGE_COLORS[4]),
        ).arrange(RIGHT, buff=0.85).move_to(DOWN * 2.5)

        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.15) for f in figs], lag_ratio=0.1),
                  run_time=1.1)
        self.wait(1.6)

        # --- dissolve into the mark ------------------------------------------ #
        self.play(FadeOut(figs), FadeOut(links), run_time=0.5)

        logo = Logo(scale_factor=1.5).move_to(UP * 1.35)
        self.play(LaggedStart(*[FadeOut(p, scale=0.7) for p in big], lag_ratio=0.05), run_time=0.7)
        self.play(FadeIn(logo, scale=0.85), run_time=0.7)

        name = wordmark(0.92).next_to(logo, DOWN, buff=0.4)
        full = expansion(0.27).next_to(name, DOWN, buff=0.26)
        self.play(FadeIn(name, shift=UP * 0.1), run_time=0.6)
        self.play(FadeIn(full, shift=UP * 0.08), run_time=0.6)
        self.wait(0.5)

        pills = VGroup(
            link_pill("Paper", "paper", INK),
            link_pill("Dataset", "dataset", INK),
            link_pill("Code", "code", INK),
        ).arrange(RIGHT, buff=0.3).next_to(full, DOWN, buff=0.6)
        self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.12) for p in pills], lag_ratio=0.12),
                  run_time=0.7)
        self.wait(1.8)
        self.play(FadeOut(VGroup(logo, name, full, pills)), run_time=0.8)


def _fig(value: str, label: str, color: str) -> VGroup:
    v = txt(value, 0.42, color, BOLD)
    l = txt(label, T_TINY * 0.9, MUTED)
    return VGroup(v, l).arrange(DOWN, buff=0.08)

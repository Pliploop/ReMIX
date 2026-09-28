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
from remix_video.stagebase import done_dot, dot_position
from remix_video.theme import (
    BG, EASE_IN, EASE_MOVE, FAINT, HOLD_READ, INK2, LINE, STROKE, INK, MUTED, PAPER, STAGE_COLORS, STAGE_NAMES, STROKE_ARROW,
    T_ENTER, T_EXIT, T_SMALL, T_TINY, txt,
)


class Assemble(Scene):
    def construct(self):
        self.camera.background_color = BG

        # The finished progress rail, exactly as stage 5 left it.
        track = Line(dot_position(0), dot_position(4), color=LINE, stroke_width=STROKE)
        rail = VGroup(*[done_dot(i) for i in range(5)])
        self.add(track, rail)
        self.wait(0.2)

        # --- fly to centre and become the figure ---------------------------- #
        big = VGroup()
        for i in range(5):
            big.add(StagePanel(i + 1, STAGE_NAMES[i], STAGE_COLORS[i], 2.35, 2.5, label_size=1.0))
        big.arrange(RIGHT, buff=0.3).move_to(UP * 0.7)

        # Bolder connectors than the draft: at this size the thin ones vanished.
        links = VGroup(*[
            Arrow(
                big[i].bg.get_right(), big[i + 1].bg.get_left(),
                buff=0.06, color=MUTED, stroke_width=STROKE_ARROW,
                max_tip_length_to_length_ratio=0.35, tip_length=0.11,
            ).shift(UP * 0.16)
            for i in range(4)
        ])

        self.play(
            LaggedStart(*[ReplacementTransform(rail[i], big[i]) for i in range(5)], lag_ratio=0.08),
            FadeOut(track),
            run_time=1.6, rate_func=EASE_MOVE,
        )
        self.play(LaggedStart(*[GrowArrow(l) for l in links], lag_ratio=0.15), run_time=0.9, rate_func=EASE_IN)
        self.wait(0.6)

        # --- the summary, in real numbers ------------------------------------ #
        figs = VGroup(
            _fig(thousands(CATALOGUE_TOTAL), "clips enriched", STAGE_COLORS[0]),
            _fig(thousands(FIGURES["chains"]), "chains", STAGE_COLORS[2]),
            _fig(thousands(FIGURES["steps"]), "steps", STAGE_COLORS[1]),
            _fig(thousands(FIGURES["variants"]), "instructions", STAGE_COLORS[3]),
            _fig(f"{FIGURES['ac1_lo']}–{FIGURES['ac1_hi']}", "judge agreement", STAGE_COLORS[4]),
        ).arrange(RIGHT, buff=0.95, aligned_edge=UP).move_to(DOWN * 2.45)

        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.1) for f in figs], lag_ratio=0.15),
                  run_time=1.3, rate_func=EASE_IN)
        self.wait(HOLD_READ + 0.3)

        # --- dissolve into the mark ------------------------------------------ #
        self.play(FadeOut(VGroup(figs, links), shift=DOWN * 0.1), run_time=T_EXIT)

        logo = Logo(scale_factor=1.5).move_to(UP * 1.35)
        self.play(LaggedStart(*[FadeOut(p, scale=0.9) for p in big], lag_ratio=0.06), run_time=0.7, rate_func=EASE_MOVE)
        self.play(FadeIn(logo, scale=0.92), run_time=T_ENTER, rate_func=EASE_IN)

        name = wordmark(0.92).next_to(logo, DOWN, buff=0.4)
        full = expansion(0.27).next_to(name, DOWN, buff=0.26)
        self.play(LaggedStart(FadeIn(name, shift=UP * 0.08), FadeIn(full, shift=UP * 0.06), lag_ratio=0.3),
                  run_time=1.0, rate_func=EASE_IN)
        self.wait(0.4)

        pills = VGroup(
            link_pill("Paper", "paper", INK),
            link_pill("Dataset", "dataset", INK),
            link_pill("Code", "code", INK),
        ).arrange(RIGHT, buff=0.3).next_to(full, DOWN, buff=0.6)
        self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.08) for p in pills], lag_ratio=0.15),
                  run_time=0.8, rate_func=EASE_IN)
        self.wait(2.2)
        self.play(FadeOut(VGroup(logo, name, full, pills)), run_time=0.8, rate_func=EASE_MOVE)


def _fig(value: str, label: str, color: str) -> VGroup:
    """Same grammar as the stages' figures: accent tick, ink number, quiet label."""
    tick = Line(LEFT * 0.14, RIGHT * 0.14, color=color, stroke_width=2.4)
    v = txt(value, 0.44, INK, MEDIUM)
    l = txt(label, T_TINY * 0.95, INK2)
    return VGroup(tick, v, l).arrange(DOWN, buff=0.1)

"""Stage 5 — Validation & Benchmark.

Two LLM judges score every variant against the same rubric a human rater sees.
A gate keeps what passes. The accept rates and agreement are the real measured
numbers, not illustrations.
"""

from __future__ import annotations

from manim import *

from remix_video.facts import FIGURES, JUDGES, VARIANTS_PER_STEP
from remix_video.glass import organic_link
from remix_video.stagebase import StageScene, explain, stat_row
from remix_video.theme import (
    CHAIN, ENRICH, INK, MUTED, PAPER, T_TINY, VALIDATE, card, tint, txt,
)

RUBRIC = [
    ("meaningful change", 0.94),
    ("target follows", 0.86),
    ("source supported", 0.92),
    ("written as an edit", 0.79),
    ("clear", 0.90),
]


class Validation(StageScene):
    stage_index = 4

    def construct(self):
        rail, header = self.open_stage(upto=4)
        content = VGroup()

        # --- the variants come in ------------------------------------------- #
        variants = VGroup(*[
            RoundedRectangle(width=2.3, height=0.3, corner_radius=0.13,
                             fill_color=tint(VALIDATE, 0.10), fill_opacity=1,
                             stroke_color=VALIDATE, stroke_width=1.2, stroke_opacity=0.55)
            for _ in range(VARIANTS_PER_STEP)
        ]).arrange(DOWN, buff=0.13).move_to(LEFT * 4.7 + UP * 0.1)
        content.add(variants)

        line = explain("Two judges score every variant against the rubric a human rater sees.")
        content.add(line)
        self.play(LaggedStart(*[FadeIn(v, shift=RIGHT * 0.12) for v in variants], lag_ratio=0.07),
                  FadeIn(line), run_time=0.8)

        # --- the rubric ------------------------------------------------------- #
        panel = card(4.3, 2.5, VALIDATE, alpha=0.07, radius=0.14).move_to(LEFT * 0.9 + UP * 0.05)
        rows = VGroup()
        fills = []
        for label, frac in RUBRIC:
            name = txt(label, T_TINY * 0.85, MUTED)
            track = RoundedRectangle(width=1.5, height=0.11, corner_radius=0.05,
                                     fill_color=tint(VALIDATE, 0.16), fill_opacity=1, stroke_width=0)
            fill = RoundedRectangle(width=1.5 * frac, height=0.11, corner_radius=0.05,
                                    fill_color=VALIDATE, fill_opacity=1, stroke_width=0)
            fill.align_to(track, LEFT)
            bar = VGroup(track, fill)
            row = VGroup(name, bar).arrange(RIGHT, buff=0.24)
            name.align_to(row, LEFT)
            rows.add(row)
            fills.append((fill, track, frac))
        rows.arrange(DOWN, buff=0.19, aligned_edge=RIGHT).move_to(panel.get_center())
        content.add(panel, rows)

        feed = organic_link(variants.get_right() + RIGHT * 0.05, panel.get_left() + LEFT * 0.05,
                            VALIDATE, 2.4, bow=0.18, seed=0)
        content.add(feed)
        self.play(Create(feed), FadeIn(panel), run_time=0.5)

        # bars grow from empty -- the scoring happening
        for fill, track, frac in fills:
            fill.stretch_to_fit_width(0.01)
            fill.align_to(track, LEFT)
        self.play(LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.06), run_time=0.5)
        self.play(
            LaggedStart(*[
                fill.animate.stretch_to_fit_width(track.width * frac).align_to(track, LEFT)
                for fill, track, frac in fills
            ], lag_ratio=0.1),
            run_time=1.1,
        )
        self.wait(0.4)

        # --- the gate ---------------------------------------------------------- #
        yes = VGroup(
            Circle(radius=0.32, fill_color=tint(CHAIN, 0.16), fill_opacity=1,
                   stroke_color=CHAIN, stroke_width=2.4),
            Text("✓", font="sans-serif", color=CHAIN, weight=BOLD).scale(0.4),
        ).move_to(RIGHT * 3.9 + UP * 0.6)
        no = VGroup(
            Circle(radius=0.32, fill_color=tint(ENRICH, 0.10), fill_opacity=1,
                   stroke_color=ENRICH, stroke_width=2.0),
            Text("✕", font="sans-serif", color=ENRICH).scale(0.34),
        ).move_to(RIGHT * 3.9 + DOWN * 0.7)
        gate_l = txt("gate", T_TINY * 0.9, MUTED).next_to(VGroup(yes, no), DOWN, buff=0.2)
        content.add(yes, no, gate_l)

        out = VGroup(
            organic_link(panel.get_right() + RIGHT * 0.05, yes.get_left() + LEFT * 0.05,
                         CHAIN, 2.2, bow=0.16, seed=1),
            organic_link(panel.get_right() + RIGHT * 0.05, no.get_left() + LEFT * 0.05,
                         MUTED, 1.6, bow=0.16, seed=2).set_opacity(0.5),
        )
        content.add(out)

        line2 = explain("Only what passes becomes part of ReMIX.")
        content.add(line2)
        self.play(
            Create(out[0]), Create(out[1]),
            FadeIn(yes, scale=0.8), FadeIn(no, scale=0.8), FadeIn(gate_l),
            ReplacementTransform(line, line2),
            run_time=0.9,
        )
        self.wait(0.5)

        figs = stat_row(
            [(f"{FIGURES['accept_lo']}–{FIGURES['accept_hi']}%", "accepted overall"),
             (f"{FIGURES['ac1_lo']}–{FIGURES['ac1_hi']}", "judge agreement (AC1)")],
            VALIDATE, buff=1.6,
        ).move_to(DOWN * 2.75)
        content.add(figs)
        self.play(FadeIn(figs, shift=UP * 0.15), run_time=0.5)
        self.wait(1.0)

        self.close_stage(content, rail, header)

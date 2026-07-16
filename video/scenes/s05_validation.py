"""Stage 5 — Validation & Benchmark.

The same instruction goes to two LLM judges and to a human rater, against the
same rubric. What matters is not that a model scores it, but that the model and
the person agree -- so the agreement is the payoff of the scene.

Accept rates and AC1 are the measured numbers.
"""

from __future__ import annotations

from manim import *

from remix_video.chain import steps
from remix_video.facts import FIGURES, VARIANTS_PER_STEP
from remix_video.glass import elbow_link
from remix_video.parts import judge_logo
from remix_video.stagebase import StageScene, explain, stat_row
from remix_video.theme import (
    CHAIN, ENRICH, INK, MUTED, NEIGHBOUR, PAPER, T_TINY, VALIDATE, card, tint, txt,
)

RUBRIC = [
    ("meaningful change", 0.94),
    ("target follows", 0.86),
    ("grounded in source", 0.92),
    ("written as an edit", 0.79),
]


def person_icon(color: str, scale: float = 1.0) -> VGroup:
    head = Circle(radius=0.11, fill_color=color, fill_opacity=1, stroke_width=0)
    body = Arc(radius=0.19, start_angle=PI, angle=PI, color=color, stroke_width=3.4)
    body.next_to(head, DOWN, buff=0.03)
    return VGroup(head, body).scale(scale)


def rater_box(title: str, badge: Mobject, color: str) -> VGroup:
    box = card(2.5, 1.15, color, alpha=0.11, radius=0.13)
    t = txt(title, T_TINY * 0.95, color, BOLD).move_to(box.get_center() + DOWN * 0.38)
    g = Group(box, badge, t)
    badge.move_to(box.get_center() + UP * 0.18)
    return g


def score_pills(values, color: str) -> VGroup:
    return VGroup(*[
        VGroup(
            Circle(radius=0.15, fill_color=tint(color, 0.3), fill_opacity=1,
                   stroke_color=color, stroke_width=1.6),
            txt(str(v), T_TINY * 0.85, color, BOLD),
        ).arrange(ORIGIN)
        for v in values
    ]).arrange(RIGHT, buff=0.12)


class Validation(StageScene):
    stage_index = 4

    def construct(self):
        rail, header = self.open_stage(upto=4)
        st = steps()[1]

        # --- the instruction under test -------------------------------------- #
        real = txt(f'"{st["instruction"]}"', 0.28, INK, BOLD)
        rbg = RoundedRectangle(
            width=real.width + 0.5, height=real.height + 0.38, corner_radius=0.13,
            fill_color=tint(VALIDATE, 0.1), fill_opacity=1,
            stroke_color=VALIDATE, stroke_width=2,
        )
        instr = VGroup(rbg, real.move_to(rbg.get_center())).move_to(LEFT * 4.0 + UP * 0.5)

        line = explain("The same instruction goes to two judges and to a person.")
        self.play(FadeIn(instr, shift=RIGHT * 0.15), FadeIn(line), run_time=0.8)

        # --- both raters, same rubric ---------------------------------------- #
        logos = Group(judge_logo("qwen", 0.36), judge_logo("gemma", 0.36)).arrange(RIGHT, buff=0.18)
        llm = rater_box("LLM judges", logos, VALIDATE)
        llm.move_to(RIGHT * 0.1 + UP * 1.45)

        human = rater_box("Human rater", person_icon(NEIGHBOUR, 1.0), NEIGHBOUR)
        human.move_to(RIGHT * 0.1 + DOWN * 0.85)

        feeds = VGroup(
            elbow_link(instr.get_right() + RIGHT * 0.04, llm[0].get_left() + LEFT * 0.03, VALIDATE, 2.2),
            elbow_link(instr.get_right() + RIGHT * 0.04, human[0].get_left() + LEFT * 0.03, NEIGHBOUR, 2.2),
        )
        self.play(*[Create(f) for f in feeds], run_time=0.6)
        self.play(FadeIn(llm), FadeIn(human), run_time=0.6)
        self.wait(0.4)

        # --- the rubric they share -------------------------------------------- #
        rows = VGroup()
        fills = []
        for label, frac in RUBRIC:
            name = txt(label, T_TINY * 0.8, MUTED)
            track = RoundedRectangle(width=1.3, height=0.09, corner_radius=0.04,
                                     fill_color=tint(VALIDATE, 0.16), fill_opacity=1, stroke_width=0)
            fill = RoundedRectangle(width=0.02, height=0.09, corner_radius=0.04,
                                    fill_color=VALIDATE, fill_opacity=1, stroke_width=0)
            fill.align_to(track, LEFT)
            row = VGroup(name, VGroup(track, fill)).arrange(RIGHT, buff=0.18)
            rows.add(row)
            fills.append((fill, track, frac))
        rows.arrange(DOWN, buff=0.14, aligned_edge=RIGHT)
        panel = card(3.6, rows.height + 0.6, VALIDATE, alpha=0.06, radius=0.13)
        rows.move_to(panel.get_center())
        rubric = VGroup(panel, rows).move_to(RIGHT * 4.1 + UP * 0.5)

        joins = VGroup(
            elbow_link(llm[0].get_right() + RIGHT * 0.03, panel.get_left() + LEFT * 0.03 + UP * 0.2,
                       VALIDATE, 2.0),
            elbow_link(human[0].get_right() + RIGHT * 0.03, panel.get_left() + LEFT * 0.03 + DOWN * 0.2,
                       NEIGHBOUR, 2.0),
        )
        line2 = explain("Both answer the same rubric.")
        self.play(*[Create(j) for j in joins], FadeIn(panel), ReplacementTransform(line, line2),
                  run_time=0.7)
        self.play(LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.06), run_time=0.4)
        self.play(
            LaggedStart(*[
                fill.animate.stretch_to_fit_width(track.width * frac).align_to(track, LEFT)
                for fill, track, frac in fills
            ], lag_ratio=0.1),
            run_time=1.0,
        )
        self.wait(0.5)

        # --- the payoff: they agree -------------------------------------------- #
        self.play(
            FadeOut(VGroup(instr, feeds, joins, rubric)),
            llm.animate.move_to(LEFT * 3.2 + UP * 0.5),
            human.animate.move_to(RIGHT * 3.2 + UP * 0.5),
            run_time=0.7,
        )
        a_scores = score_pills([5, 4, 5, 4], VALIDATE).next_to(llm, DOWN, buff=0.35)
        b_scores = score_pills([5, 4, 4, 4], NEIGHBOUR).next_to(human, DOWN, buff=0.35)
        self.play(FadeIn(a_scores, shift=UP * 0.1), FadeIn(b_scores, shift=UP * 0.1), run_time=0.6)

        agree = VGroup(
            txt("they agree", T_TINY * 0.95, MUTED),
            txt(f"AC1 {FIGURES['ac1_lo']}–{FIGURES['ac1_hi']}", 0.42, CHAIN, BOLD),
        ).arrange(DOWN, buff=0.12).move_to(UP * 0.5)
        bridge = VGroup(
            Line(a_scores.get_right() + RIGHT * 0.15, agree.get_left() + LEFT * 0.15,
                 color=CHAIN, stroke_width=2),
            Line(agree.get_right() + RIGHT * 0.15, b_scores.get_left() + LEFT * 0.15,
                 color=CHAIN, stroke_width=2),
        )
        line3 = explain("Where the judges and the person land in the same place, we keep it.")
        self.play(Create(bridge), FadeIn(agree, scale=0.9), ReplacementTransform(line2, line3),
                  run_time=0.8)
        self.wait(0.8)

        figs = stat_row(
            [(f"{FIGURES['accept_lo']}–{FIGURES['accept_hi']}%", "instructions accepted"),
             (str(VARIANTS_PER_STEP), "variants scored per step")],
            VALIDATE, buff=1.6,
        )
        self.play(FadeIn(figs, shift=UP * 0.15), run_time=0.5)
        self.wait(1.0)

        content = VGroup(a_scores, b_scores, agree, bridge, figs, line3)
        self.play(FadeOut(Group(llm, human)), run_time=0.01)
        self.close_stage(content, rail, header)

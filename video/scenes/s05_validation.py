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
from remix_video.parts import judge_logo, person
from remix_video.stagebase import StageScene, explain, stat_row
from remix_video.theme import (
    CHAIN, ENRICH, FONT, INK, MUTED, NEIGHBOUR, PAPER, T_TINY, VALIDATE, card, tint, txt,
)

RUBRIC = [
    ("meaningful change", 0.94),
    ("target follows", 0.86),
    ("grounded in source", 0.92),
    ("written as an edit", 0.79),
]


def person_icon(color: str, height: float = 0.62):
    """A real icon. The hand-drawn circle-and-arc did not read as a person."""
    return person(color, height)


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
        logos = Group(judge_logo("qwen", 0.46), judge_logo("gemma", 0.46)).arrange(RIGHT, buff=0.22)
        llm = rater_box("LLM judges", logos, VALIDATE)
        llm.move_to(RIGHT * 0.1 + UP * 1.45)

        # 0.5 units in a 1.15-unit box. At 1.0 the icon overflowed its own card.
        human = rater_box("Human rater", person_icon(NEIGHBOUR, 0.5), NEIGHBOUR)
        human.move_to(RIGHT * 0.1 + DOWN * 0.85)

        feeds = VGroup(
            elbow_link(instr.get_right() + RIGHT * 0.04, llm[0].get_left() + LEFT * 0.03, VALIDATE, 2.2),
            elbow_link(instr.get_right() + RIGHT * 0.04, human[0].get_left() + LEFT * 0.03, NEIGHBOUR, 2.2),
        )
        self.play(*[Create(f) for f in feeds], run_time=0.6)
        self.play(FadeIn(llm), FadeIn(human), run_time=0.6)
        self.wait(0.4)

        # --- the rubric they share -------------------------------------------- #
        # A ticked box and a bar that fills: the draft's bare bars did not say
        # they were answers to questions.
        rows = VGroup()
        fills = []
        ticks = VGroup()
        for label, frac in RUBRIC:
            box = RoundedRectangle(width=0.18, height=0.18, corner_radius=0.04,
                                   fill_color=PAPER, fill_opacity=1,
                                   stroke_color=VALIDATE, stroke_width=1.6)
            tick = Text("✓", font=FONT, color=VALIDATE, weight=BOLD).scale(0.16).move_to(box)
            tick.set_opacity(0)
            ticks.add(tick)
            name = txt(label, T_TINY * 0.8, MUTED)
            track = RoundedRectangle(width=1.25, height=0.1, corner_radius=0.045,
                                     fill_color=tint(VALIDATE, 0.16), fill_opacity=1, stroke_width=0)
            fill = RoundedRectangle(width=0.03, height=0.1, corner_radius=0.045,
                                    fill_color=VALIDATE, fill_opacity=1, stroke_width=0)
            fill.align_to(track, LEFT)
            row = VGroup(VGroup(box, tick), name, VGroup(track, fill)).arrange(RIGHT, buff=0.16)
            rows.add(row)
            fills.append((fill, track, frac))
        rows.arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        panel = card(rows.width + 0.6, rows.height + 0.6, VALIDATE, alpha=0.06, radius=0.13)
        rows.move_to(panel.get_center())
        rubric = VGroup(panel, rows).move_to(RIGHT * 4.2 + UP * 0.5)

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
                AnimationGroup(
                    fill.animate.stretch_to_fit_width(track.width * frac).align_to(track, LEFT),
                    ticks[i].animate.set_opacity(1),
                )
                for i, (fill, track, frac) in enumerate(fills)
            ], lag_ratio=0.12),
            run_time=1.1,
        )
        self.wait(0.6)

        # --- the payoff: they agree, question by question --------------------- #
        # One row per rubric question, with the two scores facing each other and a
        # link between them. Two diagonal lines to a blob said "connected"; this
        # says *what* agrees with what.
        self.play(
            FadeOut(VGroup(instr, feeds, joins, rubric)),
            llm.animate.scale(0.8).move_to(LEFT * 4.6 + UP * 1.65),
            human.animate.scale(0.8).move_to(RIGHT * 4.6 + UP * 1.65),
            run_time=0.7,
        )

        llm_scores = [5, 4, 5, 4]
        human_scores = [5, 4, 4, 4]
        pair_rows = VGroup()
        for (label, _), sa, sb in zip(RUBRIC, llm_scores, human_scores):
            a = score_pills([sa], VALIDATE)[0]
            b = score_pills([sb], NEIGHBOUR)[0]
            name = txt(label, T_TINY * 0.85, MUTED)
            link = Line(LEFT * 0.9, RIGHT * 0.9, color=CHAIN, stroke_width=2, stroke_opacity=0.55)
            row = VGroup(a, link, name, link.copy(), b).arrange(RIGHT, buff=0.14)
            # The name sits on the link, so the link reads as "these two, on this".
            name.move_to(row.get_center())
            pair_rows.add(row)
        pair_rows.arrange(DOWN, buff=0.28).move_to(UP * 0.05)

        line3 = explain("Question by question, the judges and the person land in the same place.")
        self.play(ReplacementTransform(line2, line3), run_time=0.4)
        self.play(
            LaggedStart(*[
                AnimationGroup(FadeIn(r[0], scale=0.7), FadeIn(r[4], scale=0.7), FadeIn(r[2]))
                for r in pair_rows
            ], lag_ratio=0.12),
            run_time=0.9,
        )
        self.play(
            LaggedStart(*[AnimationGroup(Create(r[1]), Create(r[3])) for r in pair_rows],
                        lag_ratio=0.12),
            run_time=0.8,
        )

        agree = txt(f"AC1 {FIGURES['ac1_lo']}–{FIGURES['ac1_hi']}", 0.4, CHAIN, BOLD)
        agree.next_to(pair_rows, DOWN, buff=0.4)
        self.play(FadeIn(agree, scale=0.9), run_time=0.5)
        self.wait(0.9)
        bridge = VGroup()
        a_scores = pair_rows
        b_scores = VGroup()

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

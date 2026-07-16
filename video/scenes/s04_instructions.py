"""Stage 4 — Instruction Generation.

Two metadata frames, a structured diff between them, then the LLM that turns the
diff into an instruction. Everything shown is the real record for the second turn
of chain_00000496.

The clause budget and the axis distribution are here because they are what stops
the instructions being mush: an instruction may only spend so many clauses, and
each clause has to land on a named axis.
"""

from __future__ import annotations

from manim import *

from remix_video.chain import steps
from remix_video.facts import FIGURES, VARIANTS_PER_STEP, thousands
from remix_video.glass import elbow_link
from remix_video.parts import json_frame, judge_logo
from remix_video.stagebase import StageScene, explain, stat_row
from remix_video.theme import (
    CHAIN, ENRICH, INK, INSTRUCT, MUTED, NEIGHBOUR, PAPER, T_TINY, card, tint, txt,
)


def axis_bars(width: float = 2.9) -> VGroup:
    """The real axis distribution, from the exported stats."""
    dist = FIGURES.get("axes") or []
    if not dist:
        dist = [("genre style", 1.0), ("mood", 0.66), ("instrumentation", 0.6),
                ("texture", 0.55), ("rhythm", 0.4)]
    top = dist[:5]
    peak = max(v for _, v in top) or 1.0

    rows = VGroup()
    for name, val in top:
        label = txt(name, T_TINY * 0.78, MUTED)
        track = RoundedRectangle(width=width, height=0.1, corner_radius=0.04,
                                 fill_color=tint(INSTRUCT, 0.14), fill_opacity=1, stroke_width=0)
        fill = RoundedRectangle(width=max(0.04, width * val / peak), height=0.1, corner_radius=0.04,
                                fill_color=INSTRUCT, fill_opacity=1, stroke_width=0)
        fill.align_to(track, LEFT)
        bar = VGroup(track, fill)
        row = VGroup(label, bar).arrange(RIGHT, buff=0.2)
        rows.add(row)
    rows.arrange(DOWN, buff=0.13, aligned_edge=RIGHT)
    return rows


class Instructions(StageScene):
    stage_index = 3

    def construct(self):
        rail, header = self.open_stage(upto=3)
        st = steps()[1]  # "Keep vocals, make them robotic and metal."

        # --- 1. two metadata frames ----------------------------------------- #
        src = json_frame(
            "source",
            [("genre", "industrial"), ("vocals", "spoken"), ("energy", "high")],
            NEIGHBOUR, width=2.9,
        ).move_to(LEFT * 4.6 + UP * 0.85)
        tgt = json_frame(
            "target",
            [("genre", "metal"), ("vocals", "robotic"), ("energy", "high")],
            CHAIN, width=2.9,
        ).move_to(LEFT * 4.6 + DOWN * 1.15)

        line = explain("Two clips, two structured descriptions.")
        self.play(FadeIn(src, shift=RIGHT * 0.15), FadeIn(tgt, shift=RIGHT * 0.15),
                  FadeIn(line), run_time=0.9)
        self.wait(0.5)

        # --- 2. the diff ------------------------------------------------------ #
        diff_call = txt("diff(source, target)", T_TINY * 0.95, INSTRUCT, BOLD)
        diff_call.move_to(LEFT * 1.55 + DOWN * 0.15)

        delta = json_frame(
            "semantic delta",
            [("lost", (st.get("lost") or ["industrial"])[0]),
             ("new", (st.get("new") or ["robotic vocals"])[0]),
             ("preserved", (st.get("preserved") or ["vocals"])[0])],
            INSTRUCT, width=3.2,
        ).move_to(RIGHT * 1.15 + DOWN * 0.15)

        into = VGroup(
            elbow_link(src.get_right() + RIGHT * 0.04, diff_call.get_left() + LEFT * 0.12, INSTRUCT, 2.0),
            elbow_link(tgt.get_right() + RIGHT * 0.04, diff_call.get_left() + LEFT * 0.12, INSTRUCT, 2.0),
        )
        out = elbow_link(diff_call.get_right() + RIGHT * 0.12, delta.get_left() + LEFT * 0.04, INSTRUCT, 2.0)

        line2 = explain("The difference between them is computed, not guessed.")
        self.play(*[Create(i) for i in into], FadeIn(diff_call), ReplacementTransform(line, line2),
                  run_time=0.8)
        self.play(Create(out), FadeIn(delta, shift=LEFT * 0.1), run_time=0.8)
        self.wait(0.8)

        # --- 3. the judges write the instruction ----------------------------- #
        self.play(
            FadeOut(VGroup(src, tgt, into, diff_call)),
            VGroup(delta, out).animate.move_to(LEFT * 4.4 + UP * 0.4).scale(0.9),
            run_time=0.7,
        )
        self.remove(out)

        llm_box = card(2.3, 1.35, INSTRUCT, alpha=0.12, radius=0.14).move_to(LEFT * 0.5 + UP * 0.4)
        logos = Group(judge_logo("qwen", 0.4), judge_logo("gemma", 0.4)).arrange(RIGHT, buff=0.25)
        logos.move_to(llm_box.get_center() + UP * 0.2)
        llm_t = txt("LLM", T_TINY * 1.05, INSTRUCT, BOLD).move_to(llm_box.get_center() + DOWN * 0.38)

        feed = elbow_link(delta.get_right() + RIGHT * 0.04, llm_box.get_left() + LEFT * 0.03, INSTRUCT, 2.2)

        real = txt(f'"{st["instruction"]}"', 0.27, INK, BOLD)
        rbg = RoundedRectangle(
            width=real.width + 0.5, height=real.height + 0.4, corner_radius=0.13,
            fill_color=tint(INSTRUCT, 0.13), fill_opacity=1,
            stroke_color=INSTRUCT, stroke_width=2,
        )
        picked = VGroup(rbg, real.move_to(rbg.get_center())).move_to(RIGHT * 3.9 + UP * 0.4)
        emit = elbow_link(llm_box.get_right() + RIGHT * 0.03, picked.get_left() + LEFT * 0.04, INSTRUCT, 2.2)

        line3 = explain(f"An LLM writes the edit — {VARIANTS_PER_STEP} variants per step.")
        self.play(Create(feed), FadeIn(llm_box), FadeIn(logos), FadeIn(llm_t),
                  ReplacementTransform(line2, line3), run_time=0.8)
        self.play(Create(emit), FadeIn(picked, shift=LEFT * 0.1), run_time=0.7)
        self.wait(0.9)

        # --- 4. clause budget + axis distribution ---------------------------- #
        self.play(
            FadeOut(VGroup(delta, feed, llm_box, llm_t, emit)),
            FadeOut(logos),
            picked.animate.move_to(UP * 1.85),
            run_time=0.6,
        )

        budget = VGroup(
            txt("clause budget", T_TINY * 0.85, MUTED),
            txt("≤ 4 clauses", 0.32, INSTRUCT, BOLD),
            txt("each on a named axis", T_TINY * 0.8, MUTED),
        ).arrange(DOWN, buff=0.1).move_to(LEFT * 4.1 + DOWN * 0.5)

        bars = axis_bars(2.6).move_to(RIGHT * 1.3 + DOWN * 0.5)
        bars_title = txt("what instructions change", T_TINY * 0.85, MUTED)
        bars_title.next_to(bars, UP, buff=0.22)

        line4 = explain("Every clause has to land on a named musical axis.")
        self.play(FadeIn(budget, shift=UP * 0.1), ReplacementTransform(line3, line4), run_time=0.6)
        self.play(FadeIn(bars_title), LaggedStart(*[FadeIn(r, shift=LEFT * 0.15) for r in bars],
                                                  lag_ratio=0.1), run_time=0.9)
        self.wait(0.9)

        figs = stat_row(
            [(thousands(FIGURES["variants"]), "instruction variants"),
             ("2", "phrasings: standalone / contextual")],
            INSTRUCT, buff=1.5,
        )
        self.play(FadeIn(figs, shift=UP * 0.15), run_time=0.5)
        self.wait(0.9)

        content = VGroup(picked, budget, bars, bars_title, figs, line4)
        self.close_stage(content, rail, header)

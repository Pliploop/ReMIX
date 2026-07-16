"""Stage 4 — Instruction Generation.

Diff the two clips into a semantic delta -- what was lost, what is new, what
survived -- then ask an LLM for the instruction that turns one into the other.
Five variants per step, and two phrasings: standalone, and one that may refer
back to earlier turns.

The instruction shown is the real one from chain_00000496.
"""

from __future__ import annotations

from manim import *

from remix_video.chain import steps
from remix_video.facts import FIGURES, VARIANTS_PER_STEP, thousands
from remix_video.glass import organic_link
from remix_video.stagebase import StageScene, explain, stat_row
from remix_video.theme import (
    CHAIN, ENRICH, INK, INSTRUCT, MUTED, NEIGHBOUR, PAPER, T_TINY, card, tint, txt,
)


def delta_block(title: str, items, color: str) -> VGroup:
    head = txt(title, T_TINY * 0.9, color, BOLD)
    rows = VGroup(*[txt(_clip(i, 22), T_TINY * 0.82, MUTED) for i in items[:3]])
    rows.arrange(DOWN, buff=0.09, aligned_edge=LEFT)
    g = VGroup(head, rows).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
    return g


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


class Instructions(StageScene):
    stage_index = 3

    def construct(self):
        rail, header = self.open_stage(upto=3)
        content = VGroup()
        st = steps()[1]  # "Keep vocals, make them robotic and metal."

        # --- the delta -------------------------------------------------------- #
        box = card(4.0, 2.5, INSTRUCT, alpha=0.07, radius=0.14).move_to(LEFT * 4.0 + DOWN * 0.15)
        blocks = VGroup(
            delta_block("lost", st.get("lost", ["distorted guitars"]), ENRICH),
            delta_block("new", st.get("new", ["robotic vocals"]), CHAIN),
            delta_block("preserved", st.get("preserved", ["vocals"]), NEIGHBOUR),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to(box.get_center())
        content.add(box, blocks)

        line = explain("Diff the pair: what was lost, what is new, what survived.")
        content.add(line)
        self.play(FadeIn(box), FadeIn(line), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.15) for b in blocks], lag_ratio=0.18),
                  run_time=1.0)
        self.wait(0.5)

        # --- the LLM ---------------------------------------------------------- #
        llm = card(1.9, 0.95, INSTRUCT, alpha=0.16, radius=0.12).move_to(LEFT * 0.7 + UP * 0.1)
        llm_t = txt("LLM", 0.3, INSTRUCT, BOLD).move_to(llm.get_center() + UP * 0.13)
        llm_s = txt("Qwen / Gemma", T_TINY * 0.8, MUTED).move_to(llm.get_center() + DOWN * 0.18)
        feed = organic_link(box.get_right() + RIGHT * 0.05, llm.get_left() + LEFT * 0.05,
                            INSTRUCT, 2.4, bow=0.2, seed=0)
        content.add(llm, llm_t, llm_s, feed)
        self.play(Create(feed), FadeIn(llm), FadeIn(llm_t), FadeIn(llm_s), run_time=0.7)

        # --- five variants ----------------------------------------------------- #
        variants = VGroup(*[
            RoundedRectangle(width=2.9, height=0.34, corner_radius=0.14,
                             fill_color=tint(INSTRUCT, 0.10), fill_opacity=1,
                             stroke_color=INSTRUCT, stroke_width=1.2, stroke_opacity=0.5)
            for _ in range(VARIANTS_PER_STEP)
        ]).arrange(DOWN, buff=0.14).move_to(RIGHT * 3.5 + UP * 0.15)
        fans = VGroup(*[
            organic_link(llm.get_right() + RIGHT * 0.05, v.get_left() + LEFT * 0.05,
                         INSTRUCT, 1.4, bow=0.14, seed=i + 3, tip=False).set_opacity(0.5)
            for i, v in enumerate(variants)
        ])
        content.add(variants, fans)

        line2 = explain(f"{VARIANTS_PER_STEP} variants drafted per step.")
        content.add(line2)
        self.play(
            LaggedStart(*[AnimationGroup(Create(fans[i]), FadeIn(variants[i], shift=LEFT * 0.1))
                          for i in range(VARIANTS_PER_STEP)], lag_ratio=0.1),
            ReplacementTransform(line, line2),
            run_time=1.2,
        )
        self.wait(0.4)

        # --- the real instruction ---------------------------------------------- #
        real = txt(f'"{st["instruction"]}"', 0.3, INK, BOLD)
        real_bg = RoundedRectangle(
            width=real.width + 0.5, height=real.height + 0.35, corner_radius=0.12,
            fill_color=tint(INSTRUCT, 0.14), fill_opacity=1,
            stroke_color=INSTRUCT, stroke_width=2,
        )
        picked = VGroup(real_bg, real.move_to(real_bg.get_center())).move_to(RIGHT * 3.5 + UP * 0.15)
        content.add(picked)
        self.play(ReplacementTransform(variants[2].copy(), picked), FadeOut(variants), run_time=0.7)
        self.wait(0.7)

        figs = stat_row(
            [(thousands(FIGURES["variants"]), "instruction variants"),
             ("2", "phrasings: standalone / contextual")],
            INSTRUCT, buff=1.5,
        ).move_to(DOWN * 2.75)
        content.add(figs)
        self.play(FadeIn(figs, shift=UP * 0.15), run_time=0.5)
        self.wait(0.9)

        self.close_stage(content, rail, header)

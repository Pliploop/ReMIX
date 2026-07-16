"""Stage 3 — Chain Sampling.

A stochastic walk over the pruned graph, weighted by similarity. Because
transitions are sampled in proportion to how plausible they are, the chains stay
musical while still surprising.
"""

from __future__ import annotations

import random

from manim import *

from remix_video.facts import FIGURES, thousands
from remix_video.glass import organic_link
from remix_video.stagebase import StageScene, explain, stat_row
from remix_video.theme import CHAIN, INK, MUTED, PAPER, T_TINY, card, tint, txt


class Chains(StageScene):
    stage_index = 2

    def construct(self):
        rail, header = self.open_stage(upto=2)
        content = VGroup()

        rng = random.Random(3)
        pts = [
            LEFT * 4.3 + DOWN * 0.7, LEFT * 2.6 + UP * 0.9, LEFT * 0.7 + DOWN * 0.3,
            RIGHT * 1.2 + UP * 1.1, RIGHT * 3.2 + DOWN * 0.5,
            LEFT * 3.0 + DOWN * 1.9, RIGHT * 0.2 + UP * 1.9, RIGHT * 2.4 + DOWN * 1.9,
            LEFT * 1.4 + DOWN * 1.5, RIGHT * 4.1 + UP * 1.3,
        ]
        nodes = VGroup(*[
            Circle(radius=0.13, fill_color=PAPER, fill_opacity=1,
                   stroke_color=MUTED, stroke_width=1.6).move_to(p)
            for p in pts
        ])
        # the ambient graph
        amb_pairs = [(0, 5), (1, 6), (2, 8), (3, 6), (4, 7), (5, 8), (3, 9), (2, 6), (4, 9)]
        ambient = VGroup(*[
            organic_link(pts[a], pts[b], MUTED, 1.2, bow=0.3, seed=i + 5, tip=False).set_opacity(0.35)
            for i, (a, b) in enumerate(amb_pairs)
        ])
        content.add(ambient, nodes)

        line = explain("A weighted random walk over the graph.")
        content.add(line)
        self.play(
            LaggedStart(*[GrowFromCenter(n) for n in nodes], lag_ratio=0.04),
            FadeIn(ambient), FadeIn(line),
            run_time=0.9,
        )

        # --- the walk ------------------------------------------------------- #
        path_idx = [0, 1, 2, 3, 4]
        hops = VGroup(*[
            organic_link(pts[path_idx[i]], pts[path_idx[i + 1]], CHAIN, 3.2, bow=0.36, seed=i)
            for i in range(len(path_idx) - 1)
        ])
        weights = VGroup(*[
            txt(f"{rng.uniform(0.55, 0.9):.2f}", T_TINY * 0.85, CHAIN, BOLD)
            .move_to(hops[i][0].point_from_proportion(0.5) + UP * 0.24)
            for i in range(len(hops))
        ])
        marks = VGroup(*[
            Circle(radius=0.17, fill_color=tint(CHAIN, 0.4), fill_opacity=1,
                   stroke_color=CHAIN, stroke_width=2.2).move_to(pts[i])
            for i in path_idx
        ])
        content.add(hops, weights, marks)

        # walker steps hop by hop -- the chain being drawn, not revealed
        self.play(GrowFromCenter(marks[0]), run_time=0.3)
        for i in range(len(hops)):
            self.play(
                Create(hops[i]),
                FadeIn(weights[i], scale=0.7),
                run_time=0.42,
            )
            self.play(GrowFromCenter(marks[i + 1]), run_time=0.22)

        line2 = explain("Sampled in proportion to similarity — plausible, but not obvious.")
        content.add(line2)
        self.play(ReplacementTransform(line, line2), run_time=0.5)
        self.wait(0.6)

        figs = stat_row(
            [(thousands(FIGURES["chains"]), "chains"),
             (thousands(FIGURES["steps"]), "steps"),
             ("1–6", "turns per chain")],
            CHAIN, buff=1.3,
        ).move_to(DOWN * 2.75)
        content.add(figs)
        self.play(FadeIn(figs, shift=UP * 0.15), run_time=0.5)
        self.wait(1.0)

        self.close_stage(content, rail, header)

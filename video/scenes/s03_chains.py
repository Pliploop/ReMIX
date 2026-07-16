"""Stage 3 — Chain Sampling.

The same graph as stage 2, unchanged. No new edges are drawn: the walk lights up
edges that were already there, which is the whole claim -- a chain is a path
through the neighbourhood, not something invented on top of it.

At each node the walk shows its options with their scores, then commits to one.
That is the stochastic part, and it is worth seeing rather than asserting.
"""

from __future__ import annotations

import random

from manim import *

from remix_video.facts import FIGURES, thousands
from remix_video.glass import GlassCard
from remix_video.parts import (
    GRAPH_EDGES, GRAPH_PTS, GRAPH_WALK, edge_index, graph_edges, graph_nodes, latent_backdrop,
)
from remix_video.stagebase import StageScene, explain, stat_row
from remix_video.theme import CHAIN, INK, MUTED, NEIGHBOUR, PAPER, T_TINY, tint, txt

# Edges pruned in stage 2 stay gone here, so the two views agree.
PRUNED = [(2, 7), (1, 3), (5, 8), (4, 2), (8, 7)]


class Chains(StageScene):
    stage_index = 2

    def construct(self):
        rail, header = self.open_stage(upto=2)
        rng = random.Random(5)

        # --- the same neighbourhood, handed over from stage 2 ---------------- #
        backdrop = latent_backdrop(9.6, 3.9, NEIGHBOUR).move_to(UP * 0.15)
        edges = graph_edges(NEIGHBOUR, 1.4, 0.4)
        nodes = graph_nodes(NEIGHBOUR)
        pruned_idx = {edge_index(*e) for e in PRUNED}
        for i in pruned_idx:
            edges[i].set_opacity(0)

        line = explain("The same graph. A walk over edges that already exist.")
        self.add(backdrop, edges, nodes)
        self.play(FadeIn(line), run_time=0.4)

        # --- the walk ------------------------------------------------------- #
        walk_marks = VGroup()
        start = nodes[GRAPH_WALK[0][0]]
        first_mark = Circle(radius=0.17, fill_color=tint(CHAIN, 0.45), fill_opacity=1,
                            stroke_color=CHAIN, stroke_width=2.4).move_to(start.get_center())
        walk_marks.add(first_mark)
        self.play(GrowFromCenter(first_mark), run_time=0.3)

        card = GlassCard("Rubicon of Gits.", "Conway Hambone", seed=3, color=CHAIN,
                         width=2.3).scale(0.55)
        card.next_to(first_mark, UL, buff=0.1)
        self.play(FadeIn(card, shift=DOWN * 0.08), run_time=0.4)

        lit = VGroup()
        committed: set[int] = set()
        for step_i, (a, b) in enumerate(GRAPH_WALK):
            # 1. show the options this node actually has, with scores.
            #    Edges already walked are excluded: they touch this node too, and
            #    including them reset the chain back to faint on every step, so
            #    only the last edge ever stayed green.
            options = [
                (x, y) for (x, y) in GRAPH_EDGES
                if (x == a or y == a)
                and edge_index(x, y) not in pruned_idx
                and edge_index(x, y) not in committed
            ]
            labels = VGroup()
            highlights = VGroup()
            for (x, y) in options:
                idx = edge_index(x, y)
                other = y if x == a else x
                score = 0.82 if other == b else rng.uniform(0.28, 0.6)
                mid = (GRAPH_PTS[a] + GRAPH_PTS[other]) / 2
                chosen = other == b
                lab = txt(f"{score:.2f}", T_TINY * 0.8, CHAIN if chosen else MUTED,
                          BOLD if chosen else NORMAL)
                lab.move_to(mid + UP * 0.16)
                labels.add(lab)
                highlights.add(edges[idx])

            self.play(
                *[e.animate.set_stroke(NEIGHBOUR, 2.0, 0.8) for e in highlights],
                LaggedStart(*[FadeIn(l, scale=0.7) for l in labels], lag_ratio=0.06),
                run_time=0.45,
            )

            # 2. commit: the winning edge goes green and bold, the rest fall back
            chosen_idx = edge_index(a, b)
            losers = [e for e in highlights if e is not edges[chosen_idx]]
            mark = Circle(radius=0.17, fill_color=tint(CHAIN, 0.45), fill_opacity=1,
                          stroke_color=CHAIN, stroke_width=2.4).move_to(GRAPH_PTS[b])
            self.play(
                edges[chosen_idx].animate.set_stroke(CHAIN, 3.6, 1.0),
                *[e.animate.set_stroke(NEIGHBOUR, 1.4, 0.4) for e in losers],
                *[FadeOut(l) for l in labels],
                GrowFromCenter(mark),
                run_time=0.5,
            )
            lit.add(edges[chosen_idx])
            committed.add(chosen_idx)
            walk_marks.add(mark)

        line2 = explain("Sampled in proportion to similarity — plausible, but not obvious.")
        self.play(ReplacementTransform(line, line2), run_time=0.4)
        self.wait(0.5)

        figs = stat_row(
            [(thousands(FIGURES["chains"]), "chains"),
             (thousands(FIGURES["steps"]), "steps"),
             ("1–6", "turns per chain")],
            CHAIN, buff=1.3,
        )
        self.play(FadeIn(figs, shift=UP * 0.15), run_time=0.5)
        self.wait(1.0)

        content = VGroup(backdrop, edges, nodes, walk_marks, card, figs, line2)
        self.close_stage(content, rail, header)

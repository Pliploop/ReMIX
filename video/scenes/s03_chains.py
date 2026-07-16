"""Stage 3 — Chain Sampling.

The same graph as stage 2, in the chain-sampling green. No new edges are drawn:
the walk lights up edges that were already there, which is the whole claim -- a
chain is a path through the neighbourhood, not something invented on top of it.

At each node the walk shows its options with their scores, then commits to one.
That is the stochastic part, and it is worth seeing rather than asserting.

It ends by dropping the graph entirely and leaving only the tracks, because that
is what a chain actually is once the machinery is out of the way.
"""

from __future__ import annotations

import random

from manim import *

from remix_video.chain import tracks
from remix_video.facts import FIGURES, thousands
from remix_video.glass import GlassCard
from remix_video.parts import (
    GRAPH_EDGES, GRAPH_PTS, GRAPH_WALK, edge_index, graph_edges, graph_nodes, latent_grid,
)
from remix_video.stagebase import StageScene, explain, stat_row
from remix_video.theme import CHAIN, INK, MUTED, PAPER, T_TINY, arrow, tint, txt

PRUNED = [(2, 7), (1, 3), (5, 8), (4, 2), (8, 7)]


class Chains(StageScene):
    stage_index = 2

    def construct(self):
        rail, header = self.open_stage(upto=2)
        rng = random.Random(5)
        tr = tracks()

        # --- the same neighbourhood, now in green --------------------------- #
        grid = latent_grid(10.4, 4.1, CHAIN).move_to(UP * 0.15)
        edges = graph_edges(CHAIN, 1.4, 0.35)
        nodes = graph_nodes(CHAIN)
        pruned_idx = {edge_index(*e) for e in PRUNED}
        for i in pruned_idx:
            edges[i].set_opacity(0)

        # Fade the graph in rather than cutting to it fully formed.
        line = explain("The same graph. A walk over edges that already exist.")
        self.play(
            FadeIn(grid), FadeIn(edges), FadeIn(nodes), FadeIn(line),
            run_time=0.7,
        )

        # --- the walk ------------------------------------------------------- #
        walk_marks = VGroup()
        first_mark = Circle(radius=0.17, fill_color=tint(CHAIN, 0.45), fill_opacity=1,
                            stroke_color=CHAIN, stroke_width=2.4).move_to(GRAPH_PTS[GRAPH_WALK[0][0]])
        walk_marks.add(first_mark)
        self.play(GrowFromCenter(first_mark), run_time=0.3)

        committed: set[int] = set()
        for (a, b) in GRAPH_WALK:
            # Options exclude edges already walked: they touch this node too, and
            # including them reset the chain to faint on every step.
            options = [
                (x, y) for (x, y) in GRAPH_EDGES
                if (x == a or y == a)
                and edge_index(x, y) not in pruned_idx
                and edge_index(x, y) not in committed
            ]
            labels = VGroup()
            highlights = VGroup()
            for (x, y) in options:
                other = y if x == a else x
                chosen = other == b
                score = 0.82 if chosen else rng.uniform(0.28, 0.6)
                lab = txt(f"{score:.2f}", T_TINY * 0.8, CHAIN if chosen else MUTED,
                          BOLD if chosen else NORMAL)
                lab.move_to((GRAPH_PTS[a] + GRAPH_PTS[other]) / 2 + UP * 0.16)
                labels.add(lab)
                highlights.add(edges[edge_index(x, y)])

            self.play(
                *[e.animate.set_stroke(CHAIN, 2.0, 0.8) for e in highlights],
                LaggedStart(*[FadeIn(l, scale=0.7) for l in labels], lag_ratio=0.06),
                run_time=0.45,
            )

            chosen_idx = edge_index(a, b)
            losers = [e for e in highlights if e is not edges[chosen_idx]]
            mark = Circle(radius=0.17, fill_color=tint(CHAIN, 0.45), fill_opacity=1,
                          stroke_color=CHAIN, stroke_width=2.4).move_to(GRAPH_PTS[b])
            self.play(
                edges[chosen_idx].animate.set_stroke(CHAIN, 3.6, 1.0),
                *[e.animate.set_stroke(CHAIN, 1.4, 0.35) for e in losers],
                *[FadeOut(l) for l in labels],
                GrowFromCenter(mark),
                run_time=0.5,
            )
            committed.add(chosen_idx)
            walk_marks.add(mark)

        line2 = explain("Sampled in proportion to similarity — plausible, but not obvious.")
        self.play(ReplacementTransform(line, line2), run_time=0.4)
        self.wait(0.6)

        # --- strip the machinery: a chain is its tracks ---------------------- #
        others = [edges[i] for i in range(len(edges)) if i not in committed and i not in pruned_idx]
        line3 = explain("That path is a chain.")
        self.play(
            FadeOut(grid),
            *[FadeOut(e) for e in others],
            FadeOut(VGroup(*[n for i, n in enumerate(nodes)])),
            ReplacementTransform(line2, line3),
            run_time=0.8,
        )

        # The chain, straightened out, with a card per node.
        cards = VGroup()
        for i, t in enumerate(tr[:5]):
            cards.add(
                GlassCard(t["title"], t["artist"], seed=3 + i * 7, color=CHAIN,
                          width=2.3, energy=0.9 + 0.08 * i).scale(0.62)
            )
        cards.arrange(RIGHT, buff=0.62)
        for c, dy in zip(cards, [0.42, -0.26, 0.38, -0.32, 0.24]):
            c.shift(UP * dy)
        cards.move_to(UP * 0.4)
        hops = VGroup(*[
            arrow(cards[i].get_right() + RIGHT * 0.03, cards[i + 1].get_left() + LEFT * 0.03,
                  CHAIN, 2.2)
            for i in range(len(cards) - 1)
        ])

        walk_line = VGroup(*[edges[i] for i in committed], walk_marks)
        self.play(
            FadeOut(walk_line),
            LaggedStart(*[FadeIn(c, shift=UP * 0.12) for c in cards], lag_ratio=0.1),
            run_time=1.0,
        )
        self.play(LaggedStart(*[GrowArrow(h) for h in hops], lag_ratio=0.12), run_time=0.7)
        self.wait(0.6)

        figs = stat_row(
            [(thousands(FIGURES["chains"]), "chains"),
             (thousands(FIGURES["steps"]), "steps"),
             ("1–6", "turns per chain")],
            CHAIN, buff=1.3,
        )
        self.play(FadeIn(figs, shift=UP * 0.15), run_time=0.5)
        self.wait(1.0)

        content = VGroup(cards, hops, figs, line3)
        self.close_stage(content, rail, header)

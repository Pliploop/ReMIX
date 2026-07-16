"""Stage 2 — Neighbourhood Building.

Each clip is embedded twice: audio with MuQ-MuLan, text with EmbeddingGemma. The
graph comes first, then the score that explains its edges -- showing the formula
before the thing it scores put the algebra ahead of the point.

The formula is set in Unicode, not LaTeX: no TeX install here, and Computer
Modern serif would fight the sans identity anyway.
"""

from __future__ import annotations

import random

from manim import *

from remix_video.facts import AUDIO_DIM, AUDIO_MODEL, TEXT_DIM, TEXT_MODEL
from remix_video.glass import GlassCard, elbow_link, fork_link
from remix_video.parts import (
    GRAPH_EDGES, GRAPH_PTS, edge_index, graph_edges, graph_nodes, latent_grid,
)
from remix_video.stagebase import StageScene, explain
from remix_video.theme import INK, MUTED, NEIGHBOUR, PAPER, T_TINY, card, tint, txt

# Edges the pruning removes. Stage 3 keeps the same list, so the two views agree.
WEAK = [(2, 7), (1, 3), (5, 8), (4, 2), (8, 7)]
SCORED = [(0, 1), (0, 3), (3, 5), (5, 6), (6, 7), (2, 6)]


def vector_strip(n: int = 7, color: str = NEIGHBOUR, seed: int = 0) -> VGroup:
    rng = random.Random(seed)
    cells = VGroup(*[
        Square(side_length=0.16, fill_color=tint(color, rng.uniform(0.15, 0.62)),
               fill_opacity=1, stroke_color=color, stroke_width=0.9)
        for _ in range(n)
    ]).arrange(RIGHT, buff=0.03)
    dots = txt("…", T_TINY, MUTED).next_to(cells, RIGHT, buff=0.08)
    return VGroup(cells, dots)


def model_box(name: str, dim: int, color: str) -> VGroup:
    box = card(2.1, 0.64, color, alpha=0.13, radius=0.12)
    t = txt(name, T_TINY * 1.05, color, BOLD).move_to(box.get_center() + UP * 0.08)
    d = txt(f"{dim}-d", T_TINY * 0.85, MUTED).move_to(box.get_center() + DOWN * 0.15)
    return VGroup(box, t, d)


class Neighbourhood(StageScene):
    stage_index = 1

    def construct(self):
        rail, header = self.open_stage(upto=1)

        # --- 1. embed the clip twice ---------------------------------------- #
        clip = GlassCard("Rubicon of Gits.", "Conway Hambone", seed=3, color=NEIGHBOUR,
                         width=2.7).scale(0.8).move_to(LEFT * 4.6 + UP * 0.15)
        audio_m = model_box(AUDIO_MODEL, AUDIO_DIM, NEIGHBOUR).move_to(LEFT * 1.3 + UP * 1.15)
        text_m = model_box(TEXT_MODEL, TEXT_DIM, NEIGHBOUR).move_to(LEFT * 1.3 + DOWN * 0.5)
        v1 = vector_strip(7, NEIGHBOUR, 1).next_to(audio_m, RIGHT, buff=0.35)
        v2 = vector_strip(7, NEIGHBOUR, 2).next_to(text_m, RIGHT, buff=0.35)

        feeds = fork_link(
            clip.get_right() + RIGHT * 0.05,
            [audio_m.get_left() + LEFT * 0.03, text_m.get_left() + LEFT * 0.03],
            NEIGHBOUR, 2.2, mid=0.45,
        )
        embed = VGroup(clip, audio_m, text_m, v1, v2, feeds)

        line = explain("Every clip is embedded twice — how it sounds, and how it is described.")
        self.play(FadeIn(clip, shift=RIGHT * 0.2), FadeIn(line), run_time=0.8)
        self.play(Create(feeds), FadeIn(audio_m, shift=UP * 0.1), FadeIn(text_m, shift=DOWN * 0.1),
                  run_time=0.9)
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.6) for c in v1[0]], lag_ratio=0.05),
            LaggedStart(*[FadeIn(c, scale=0.6) for c in v2[0]], lag_ratio=0.05),
            FadeIn(v1[1]), FadeIn(v2[1]),
            run_time=0.9,
        )
        self.wait(0.7)

        # --- 2. the graph, on a grid ---------------------------------------- #
        grid = latent_grid(10.4, 4.1, NEIGHBOUR).move_to(UP * 0.15)
        edges = graph_edges(NEIGHBOUR, 1.5, 0.5)
        nodes = graph_nodes(NEIGHBOUR)

        line2 = explain("Every clip is a point. Similar clips are neighbours.")
        self.play(FadeOut(embed, shift=LEFT * 0.3), ReplacementTransform(line, line2), run_time=0.6)
        self.play(FadeIn(grid), run_time=0.5)

        # A card simply sits beside a node -- no tether. The adjacency says it.
        anchor = GlassCard("Rubicon of Gits.", "Conway Hambone", seed=3, color=NEIGHBOUR,
                           width=2.3).scale(0.56)
        anchor.next_to(nodes[0], UL, buff=0.08)

        # Edges added *under* the nodes, so no edge crosses a node's face.
        self.add(edges, nodes)
        edges.set_opacity(0)
        self.play(LaggedStart(*[GrowFromCenter(n) for n in nodes], lag_ratio=0.05),
                  FadeIn(anchor, shift=DOWN * 0.1), run_time=0.9)
        edges.set_opacity(1)
        for e in edges:
            e.set_stroke(opacity=0)
        self.play(LaggedStart(*[e.animate.set_stroke(opacity=0.5) for e in edges], lag_ratio=0.03),
                  run_time=1.0)
        self.wait(0.4)

        # --- 3. the score that made those edges ------------------------------ #
        # Between the rail (ends at x≈-5.3) and the header (starts at x≈-0.1).
        fbox = card(3.4, 0.8, NEIGHBOUR, alpha=0.1, radius=0.12).move_to(UP * 2.5 + LEFT * 2.9)
        # Subscripts via U+2090/U+209C, not the Mathematical Bold block (U+1D400+):
        # the sans font has no glyphs for those and renders tofu.
        formula = txt("s(A,B) = ½ (sₐ + sₜ)", 0.33, NEIGHBOUR, BOLD).move_to(fbox.get_center())
        line3 = explain("Each edge is scored: sound and description, averaged.")
        self.play(FadeIn(fbox), Write(formula), ReplacementTransform(line2, line3), run_time=0.8)

        # The scores go on the edges, so it is unmistakable that what is scored is
        # the transition and not the node.
        rng = random.Random(4)
        weak_idx = {edge_index(*e) for e in WEAK}
        marks = VGroup()
        for (a, b) in SCORED:
            i = edge_index(a, b)
            val = rng.uniform(0.62, 0.9)
            m = txt(f"{val:.2f}", T_TINY * 0.78, NEIGHBOUR, BOLD)
            m.move_to((GRAPH_PTS[a] + GRAPH_PTS[b]) / 2 + UP * 0.16)
            marks.add(m)
        weak_marks = VGroup()
        for (a, b) in WEAK:
            m = txt(f"{rng.uniform(0.09, 0.28):.2f}", T_TINY * 0.78, MUTED)
            m.move_to((GRAPH_PTS[a] + GRAPH_PTS[b]) / 2 + UP * 0.16)
            weak_marks.add(m)
        self.play(LaggedStart(*[FadeIn(m, scale=0.7) for m in [*marks, *weak_marks]], lag_ratio=0.05),
                  run_time=0.9)
        self.wait(0.6)

        # --- 4. prune, and firm up what is left ------------------------------ #
        line4 = explain("Weak links are pruned. What is left is a graph of plausible moves.")
        self.play(
            *[FadeOut(edges[i]) for i in weak_idx],
            FadeOut(weak_marks),
            ReplacementTransform(line3, line4),
            run_time=0.8,
        )
        keep_idx = [i for i in range(len(edges)) if i not in weak_idx]
        self.play(
            *[edges[i].animate.set_stroke(NEIGHBOUR, 2.4, 1.0) for i in keep_idx],
            *[n.animate.set_stroke(NEIGHBOUR, 2.6) for n in nodes],
            run_time=0.6,
        )
        self.wait(0.9)

        # Everything leaves together; the graph is not left hanging while the
        # header flies to the rail.
        content = VGroup(grid, edges, nodes, anchor, fbox, formula, marks, line4)
        self.close_stage(content, rail, header)

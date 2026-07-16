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
from remix_video.glass import GlassCard, elbow_link
from remix_video.parts import (
    GRAPH_EDGES, GRAPH_PTS, edge_index, graph_edges, graph_nodes, latent_backdrop,
)
from remix_video.stagebase import StageScene, explain
from remix_video.theme import INK, MUTED, NEIGHBOUR, PAPER, T_TINY, card, tint, txt


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

        feeds = VGroup(
            elbow_link(clip.get_right() + RIGHT * 0.05, audio_m.get_left() + LEFT * 0.03, NEIGHBOUR, 2.2),
            elbow_link(clip.get_right() + RIGHT * 0.05, text_m.get_left() + LEFT * 0.03, NEIGHBOUR, 2.2),
        )
        embed = VGroup(clip, audio_m, text_m, v1, v2, feeds)

        line = explain("Every clip is embedded twice — how it sounds, and how it is described.")
        self.play(FadeIn(clip, shift=RIGHT * 0.2), FadeIn(line), run_time=0.8)
        self.play(
            *[Create(f) for f in feeds],
            FadeIn(audio_m, shift=UP * 0.1), FadeIn(text_m, shift=DOWN * 0.1),
            run_time=0.9,
        )
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.6) for c in v1[0]], lag_ratio=0.05),
            LaggedStart(*[FadeIn(c, scale=0.6) for c in v2[0]], lag_ratio=0.05),
            FadeIn(v1[1]), FadeIn(v2[1]),
            run_time=0.9,
        )
        self.wait(0.7)

        # --- 2. the graph, in a latent space -------------------------------- #
        # Graph before formula: the score only means something once you can see
        # what it is scoring.
        backdrop = latent_backdrop(9.6, 3.9, NEIGHBOUR).move_to(UP * 0.15)
        edges = graph_edges(NEIGHBOUR, 1.5, 0.5)
        nodes = graph_nodes(NEIGHBOUR)

        line2 = explain("Every clip is a point. Similar clips are neighbours.")
        self.play(FadeOut(embed, shift=LEFT * 0.3), ReplacementTransform(line, line2), run_time=0.6)
        self.play(FadeIn(backdrop), run_time=0.5)
        self.play(LaggedStart(*[GrowFromCenter(n) for n in nodes], lag_ratio=0.05), run_time=0.8)

        # A card hangs off one node, so it stays obvious these points are tracks.
        anchor = GlassCard("Rubicon of Gits.", "Conway Hambone", seed=3, color=NEIGHBOUR,
                           width=2.4).scale(0.58)
        anchor.next_to(nodes[0], UL, buff=0.12)
        tether = DashedLine(anchor.get_bottom() + DOWN * 0.02, nodes[0].get_top(),
                            color=NEIGHBOUR, stroke_width=1.2, dash_length=0.05)
        self.play(FadeIn(anchor, shift=DOWN * 0.1), Create(tether), run_time=0.6)
        self.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.03), run_time=1.1)
        self.wait(0.5)

        # --- 3. now the score that made those edges ------------------------- #
        # Parked up beside the header, clear of the explain band -- sharing that
        # band made the sentence and the formula cross-fade through each other.
        fbox = card(3.5, 0.8, NEIGHBOUR, alpha=0.1, radius=0.12).move_to(UP * 2.5 + LEFT * 3.9)
        # Subscripts via U+2090/U+209C, not the Mathematical Bold block (U+1D400+):
        # the sans font has no glyphs for those and renders tofu.
        formula = txt("s(A,B) = ½ (sₐ + sₜ)", 0.33, NEIGHBOUR, BOLD).move_to(fbox.get_center())
        line3 = explain("Each edge is scored: sound and description, averaged.")
        self.play(FadeIn(fbox), Write(formula), ReplacementTransform(line2, line3), run_time=0.8)
        self.wait(0.7)

        # --- 4. prune ------------------------------------------------------- #
        weak = [edge_index(*e) for e in [(2, 7), (1, 3), (5, 8), (4, 2), (8, 7)]]
        self.play(
            *[edges[i].animate.set_stroke(MUTED, 1.0, 0.25) for i in weak],
            run_time=0.5,
        )
        line4 = explain("Weak links are pruned. What is left is a graph of plausible moves.")
        self.play(
            *[FadeOut(edges[i]) for i in weak],
            ReplacementTransform(line3, line4),
            run_time=0.7,
        )
        self.wait(0.9)

        content = VGroup(backdrop, nodes, edges, anchor, tether, fbox, formula, line4)
        self.close_stage(content, rail, header)

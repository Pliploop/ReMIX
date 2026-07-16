"""Stage 1 — Dataset Enrichment.

Two open catalogues go in. Every clip comes out with a caption and a transcript,
in one structured manifest. That is the whole stage, and the numbers are real.
"""

from __future__ import annotations

from manim import *

from remix_video.facts import FIGURES, M4A_CATALOGUE, MTG_CATALOGUE, CATALOGUE_TOTAL, thousands
from remix_video.glass import StatBadge, organic_link
from remix_video.stagebase import StageScene, explain, stat_row
from remix_video.theme import (
    ENRICH, FAINT, HAIR, INK, MUTED, PAPER, T_SMALL, T_TINY, card, tint, txt,
)


def cylinder(label: str, count: str, color: str, w: float = 1.55, h: float = 1.5) -> VGroup:
    top = Ellipse(width=w, height=w * 0.32, fill_color=tint(color, 0.22), fill_opacity=1,
                  stroke_color=color, stroke_width=2)
    body = Rectangle(width=w, height=h, fill_color=tint(color, 0.12), fill_opacity=1,
                     stroke_width=0)
    left = Line(body.get_corner(UL), body.get_corner(DL), color=color, stroke_width=2)
    right = Line(body.get_corner(UR), body.get_corner(DR), color=color, stroke_width=2)
    bottom = Arc(radius=w / 2, start_angle=PI, angle=PI, color=color, stroke_width=2).stretch(0.32, 1)
    bottom.move_to(body.get_bottom())
    top.move_to(body.get_top())

    g = VGroup(body, left, right, bottom, top)
    name = txt(label, T_TINY, INK, BOLD).next_to(g, DOWN, buff=0.16)
    n = txt(count, T_TINY * 0.92, color, BOLD).next_to(name, DOWN, buff=0.06)
    return VGroup(g, name, n)


def proc_box(name: str, produces: str, color: str) -> VGroup:
    box = card(2.0, 0.78, color, alpha=0.12, radius=0.12)
    t = txt(name, T_TINY * 1.1, color, BOLD).move_to(box.get_center() + UP * 0.11)
    s = txt(produces, T_TINY * 0.85, MUTED).move_to(box.get_center() + DOWN * 0.16)
    return VGroup(box, t, s)


def manifest_table(rows: int = 5) -> VGroup:
    frame = RoundedRectangle(width=2.5, height=2.0, corner_radius=0.1,
                             fill_color=PAPER, fill_opacity=1,
                             stroke_color=ENRICH, stroke_width=1.8)
    head = txt("structured manifest", T_TINY * 0.8, MUTED)
    head.next_to(frame.get_top(), DOWN, buff=0.13)

    cells = VGroup()
    for r in range(rows):
        row = VGroup(*[
            RoundedRectangle(width=0.44, height=0.16, corner_radius=0.04,
                             fill_color=tint(ENRICH, 0.30 if c == 0 else 0.16),
                             fill_opacity=1, stroke_width=0)
            for c in range(4)
        ]).arrange(RIGHT, buff=0.07)
        cells.add(row)
    cells.arrange(DOWN, buff=0.11).move_to(frame.get_center() + DOWN * 0.13)
    return VGroup(frame, head, cells)


class Enrich(StageScene):
    stage_index = 0

    def construct(self):
        rail, header = self.open_stage(upto=0)
        content = VGroup()

        # --- catalogues ---------------------------------------------------- #
        cats = VGroup(
            cylinder("Music4All", thousands(M4A_CATALOGUE), ENRICH),
            cylinder("MTG-Jamendo", thousands(MTG_CATALOGUE), ENRICH),
        ).arrange(DOWN, buff=0.55).move_to(LEFT * 4.6 + DOWN * 0.15)
        content.add(cats)

        line = explain("Two open catalogues go in — nothing licensed, nothing scraped.")
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.25) for c in cats], lag_ratio=0.2),
                  FadeIn(line), run_time=1.2)
        self.wait(1.2)

        # --- the two enrichers --------------------------------------------- #
        afnext = proc_box("AFNext", "captions", ENRICH).move_to(LEFT * 1.0 + UP * 0.95)
        whisper = proc_box("Whisper", "lyrics / transcripts", ENRICH).move_to(LEFT * 1.0 + DOWN * 0.95)
        content.add(afnext, whisper)

        feeds = VGroup(
            organic_link(cats.get_right() + RIGHT * 0.05 + UP * 0.4, afnext.get_left() + LEFT * 0.05,
                         ENRICH, 2.4, bow=0.22, seed=0),
            organic_link(cats.get_right() + RIGHT * 0.05 + DOWN * 0.4, whisper.get_left() + LEFT * 0.05,
                         ENRICH, 2.4, bow=0.22, seed=1),
        )
        content.add(feeds)

        line2 = explain("Every clip is captioned and transcribed.")
        self.play(
            *[Create(f) for f in feeds],
            FadeIn(afnext, shift=UP * 0.15), FadeIn(whisper, shift=DOWN * 0.15),
            ReplacementTransform(line, line2),
            run_time=1.1,
        )
        self.wait(0.6)

        # --- into one manifest ---------------------------------------------- #
        table = manifest_table().move_to(RIGHT * 3.0 + DOWN * 0.1)
        content.add(table)
        joins = VGroup(
            organic_link(afnext.get_right() + RIGHT * 0.05, table.get_left() + LEFT * 0.05 + UP * 0.3,
                         ENRICH, 2.4, bow=0.2, seed=2),
            organic_link(whisper.get_right() + RIGHT * 0.05, table.get_left() + LEFT * 0.05 + DOWN * 0.3,
                         ENRICH, 2.4, bow=0.2, seed=3),
        )
        content.add(joins)

        line3 = explain("Audio, caption, lyrics and tags land in one row per clip.")
        self.play(
            *[Create(j) for j in joins],
            FadeIn(table[0]), FadeIn(table[1]),
            ReplacementTransform(line2, line3),
            run_time=0.9,
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in table[2]], lag_ratio=0.12),
                  run_time=0.9)
        self.wait(0.5)

        # --- what a row actually holds --------------------------------------- #
        legend = VGroup(*[
            VGroup(
                RoundedRectangle(width=0.2, height=0.2, corner_radius=0.05,
                                 fill_color=tint(ENRICH, 0.34), fill_opacity=1, stroke_width=0),
                txt(l, T_TINY * 0.9, MUTED),
            ).arrange(RIGHT, buff=0.12)
            for l in ("audio", "caption", "lyrics", "tags + metadata")
        ]).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        legend.next_to(table, RIGHT, buff=0.4)
        content.add(legend)
        self.play(LaggedStart(*[FadeIn(x, shift=LEFT * 0.15) for x in legend], lag_ratio=0.14),
                  run_time=1.0)
        self.wait(0.8)

        # --- the numbers ------------------------------------------------------ #
        # Below the explain line, clear of the Whisper box.
        figs = stat_row(
            [(thousands(CATALOGUE_TOTAL), "clips enriched"),
             (thousands(FIGURES["artists"]), "artists in the chains")],
            ENRICH, buff=1.6,
        ).move_to(DOWN * 2.65)
        content.add(figs)
        self.play(FadeIn(figs, shift=UP * 0.2), run_time=0.7)
        self.wait(1.6)

        content.add(line3)
        self.close_stage(content, rail, header)

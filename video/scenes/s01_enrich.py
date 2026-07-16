"""Stage 1 — Dataset Enrichment.

Two open catalogues go in. Every clip comes out with a caption and a transcript,
in one structured manifest.

~10s, the per-stage budget for a 90s film. Three beats, one number.
"""

from __future__ import annotations

from manim import *

from remix_video.facts import CATALOGUE_TOTAL, FIGURES, M4A_CATALOGUE, MTG_CATALOGUE, thousands
from remix_video.glass import organic_link
from remix_video.stagebase import StageScene, explain, stat_row
from remix_video.theme import ENRICH, INK, MUTED, PAPER, T_TINY, card, tint, txt


def cylinder(label: str, count: str, color: str, w: float = 1.35, h: float = 1.25) -> VGroup:
    body = Rectangle(width=w, height=h, fill_color=tint(color, 0.12), fill_opacity=1, stroke_width=0)
    left = Line(body.get_corner(UL), body.get_corner(DL), color=color, stroke_width=2)
    right = Line(body.get_corner(UR), body.get_corner(DR), color=color, stroke_width=2)
    bottom = Arc(radius=w / 2, start_angle=PI, angle=PI, color=color, stroke_width=2).stretch(0.32, 1)
    bottom.move_to(body.get_bottom())
    top = Ellipse(width=w, height=w * 0.32, fill_color=tint(color, 0.22), fill_opacity=1,
                  stroke_color=color, stroke_width=2).move_to(body.get_top())

    g = VGroup(body, left, right, bottom, top)
    name = txt(label, T_TINY, INK, BOLD).next_to(g, DOWN, buff=0.14)
    n = txt(count, T_TINY * 0.95, color, BOLD).next_to(name, DOWN, buff=0.05)
    return VGroup(g, name, n)


def proc_box(name: str, produces: str, color: str) -> VGroup:
    box = card(1.95, 0.74, color, alpha=0.12, radius=0.12)
    t = txt(name, T_TINY * 1.1, color, BOLD).move_to(box.get_center() + UP * 0.1)
    s = txt(produces, T_TINY * 0.82, MUTED).move_to(box.get_center() + DOWN * 0.15)
    return VGroup(box, t, s)


def manifest_table(rows: int = 5) -> VGroup:
    frame = RoundedRectangle(width=2.4, height=1.9, corner_radius=0.1,
                             fill_color=PAPER, fill_opacity=1,
                             stroke_color=ENRICH, stroke_width=1.8)
    head = txt("structured manifest", T_TINY * 0.78, MUTED).next_to(frame.get_top(), DOWN, buff=0.12)
    cells = VGroup()
    for _ in range(rows):
        cells.add(VGroup(*[
            RoundedRectangle(width=0.42, height=0.15, corner_radius=0.04,
                             fill_color=tint(ENRICH, 0.30 if c == 0 else 0.16),
                             fill_opacity=1, stroke_width=0)
            for c in range(4)
        ]).arrange(RIGHT, buff=0.06))
    cells.arrange(DOWN, buff=0.1).move_to(frame.get_center() + DOWN * 0.12)
    return VGroup(frame, head, cells)


class Enrich(StageScene):
    stage_index = 0

    def construct(self):
        rail, header = self.open_stage(upto=0)
        content = VGroup()

        cats = VGroup(
            cylinder("Music4All", thousands(M4A_CATALOGUE), ENRICH),
            cylinder("MTG-Jamendo", thousands(MTG_CATALOGUE), ENRICH),
        ).arrange(DOWN, buff=0.5).move_to(LEFT * 4.7 + DOWN * 0.15)

        afnext = proc_box("AFNext", "captions", ENRICH).move_to(LEFT * 1.2 + UP * 0.85)
        whisper = proc_box("Whisper", "lyrics", ENRICH).move_to(LEFT * 1.2 + DOWN * 0.85)
        table = manifest_table().move_to(RIGHT * 2.9 + DOWN * 0.05)

        feeds = VGroup(
            organic_link(cats.get_right() + RIGHT * 0.05 + UP * 0.35, afnext.get_left() + LEFT * 0.05,
                         ENRICH, 2.4, bow=0.2, seed=0),
            organic_link(cats.get_right() + RIGHT * 0.05 + DOWN * 0.35, whisper.get_left() + LEFT * 0.05,
                         ENRICH, 2.4, bow=0.2, seed=1),
        )
        joins = VGroup(
            organic_link(afnext.get_right() + RIGHT * 0.05, table.get_left() + LEFT * 0.05 + UP * 0.28,
                         ENRICH, 2.4, bow=0.18, seed=2),
            organic_link(whisper.get_right() + RIGHT * 0.05, table.get_left() + LEFT * 0.05 + DOWN * 0.28,
                         ENRICH, 2.4, bow=0.18, seed=3),
        )
        content.add(cats, afnext, whisper, table, feeds, joins)

        line = explain("Open catalogues in — a caption and a transcript for every clip.")
        content.add(line)

        # beat 1: catalogues
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in cats], lag_ratio=0.18),
                  FadeIn(line), run_time=1.0)
        self.wait(0.5)

        # beat 2: enrich
        self.play(
            *[Create(f) for f in feeds],
            FadeIn(afnext, shift=UP * 0.1), FadeIn(whisper, shift=DOWN * 0.1),
            run_time=1.1,
        )
        self.wait(0.5)

        # beat 3: manifest
        self.play(
            *[Create(j) for j in joins],
            FadeIn(table[0]), FadeIn(table[1]),
            run_time=0.8,
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.15) for r in table[2]], lag_ratio=0.1),
                  run_time=0.8)

        figs = stat_row(
            [(thousands(CATALOGUE_TOTAL), "clips enriched"),
             (thousands(FIGURES["artists"]), "artists")],
            ENRICH, buff=1.6,
        ).move_to(DOWN * 2.6)
        content.add(figs)
        self.play(FadeIn(figs, shift=UP * 0.15), run_time=0.5)
        self.wait(1.0)

        self.close_stage(content, rail, header)

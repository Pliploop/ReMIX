"""Stage 1 — Dataset Enrichment.

Two open catalogues go in. Every clip comes out with a caption and a transcript,
in one structured manifest — and the manifest's columns are named, so it is clear
what a row actually holds.
"""

from __future__ import annotations

from manim import *

from remix_video.facts import CATALOGUE_TOTAL, FIGURES, M4A_CATALOGUE, MTG_CATALOGUE, thousands
from remix_video.glass import _tip_dir, elbow_link, fork_link
from remix_video.parts import catalogue
from remix_video.stagebase import StageScene, explain, stat_row
from remix_video.theme import ENRICH, INK, MUTED, PAPER, T_TINY, card, tint, txt

COLUMNS = ("audio", "caption", "lyrics", "tags")


def proc_box(name: str, produces: str, color: str) -> VGroup:
    box = card(1.9, 0.72, color, alpha=0.12, radius=0.12)
    t = txt(name, T_TINY * 1.1, color, BOLD).move_to(box.get_center() + UP * 0.1)
    s = txt(produces, T_TINY * 0.8, MUTED).move_to(box.get_center() + DOWN * 0.15)
    return VGroup(box, t, s)


def manifest_table(rows: int = 4) -> VGroup:
    """Named columns: the draft's anonymous blocks never said what a row held."""
    heads = VGroup(*[txt(c, T_TINY * 0.72, ENRICH, BOLD) for c in COLUMNS])
    for h in heads:
        h.set(width=min(h.width, 0.52))
    heads.arrange(RIGHT, buff=0.16)

    body = VGroup()
    for r in range(rows):
        row = VGroup()
        for c, head in enumerate(heads):
            cell = RoundedRectangle(
                width=0.52, height=0.14, corner_radius=0.04,
                fill_color=tint(ENRICH, 0.32 if c == 0 else 0.16),
                fill_opacity=1, stroke_width=0,
            )
            cell.move_to([head.get_center()[0], 0, 0])
            row.add(cell)
        body.add(row)
    body.arrange(DOWN, buff=0.11)
    for row in body:
        for cell, head in zip(row, heads):
            cell.set_x(head.get_center()[0])

    stack = VGroup(heads, body).arrange(DOWN, buff=0.16)
    frame = RoundedRectangle(
        width=stack.width + 0.5, height=stack.height + 0.7, corner_radius=0.12,
        fill_color=PAPER, fill_opacity=1, stroke_color=ENRICH, stroke_width=1.8,
    )
    title = txt("structured manifest", T_TINY * 0.78, MUTED)
    title.move_to(frame.get_top() + DOWN * 0.2)
    stack.next_to(title, DOWN, buff=0.12)
    return VGroup(frame, title, stack)


class Enrich(StageScene):
    stage_index = 0

    def construct(self):
        rail, header = self.open_stage(upto=0)
        content = VGroup()

        cats = VGroup(
            catalogue("Music4All", thousands(M4A_CATALOGUE), ENRICH),
            catalogue("MTG-Jamendo", thousands(MTG_CATALOGUE), ENRICH),
        ).arrange(DOWN, buff=0.45).move_to(LEFT * 4.9 + UP * 0.1)

        afnext = proc_box("AFNext", "captions", ENRICH).move_to(LEFT * 1.5 + UP * 0.95)
        whisper = proc_box("Whisper", "lyrics", ENRICH).move_to(LEFT * 1.5 + DOWN * 0.7)
        table = manifest_table().move_to(RIGHT * 3.1 + UP * 0.1)

        # One bus out, one bus in. Independent elbows would share the trunk and
        # the turn, and their fillets would overlap into a bubble.
        feeds = fork_link(
            cats.get_right() + RIGHT * 0.05,
            [afnext.get_left() + LEFT * 0.03, whisper.get_left() + LEFT * 0.03],
            ENRICH, 2.2, mid=0.42,
        )
        joins = fork_link(
            table.get_left() + LEFT * 0.03,
            [afnext.get_right() + RIGHT * 0.03, whisper.get_right() + RIGHT * 0.03],
            ENRICH, 2.2, mid=0.42, tip=False,
        )
        joins.add(_tip_dir(table.get_left() + LEFT * 0.03, RIGHT, ENRICH))
        content.add(cats, afnext, whisper, table, feeds, joins)

        line = explain("Open catalogues in — a caption and a transcript for every clip.")
        content.add(line)

        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in cats], lag_ratio=0.18),
                  FadeIn(line), run_time=1.0)
        self.wait(0.4)

        self.play(
            *[Create(f) for f in feeds],
            FadeIn(afnext, shift=UP * 0.1), FadeIn(whisper, shift=DOWN * 0.1),
            run_time=1.0,
        )
        self.wait(0.4)

        self.play(*[Create(j) for j in joins], FadeIn(table[0]), FadeIn(table[1]), run_time=0.7)
        self.play(FadeIn(table[2][0]), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.12) for r in table[2][1]], lag_ratio=0.11),
                  run_time=0.8)

        figs = stat_row(
            [(thousands(CATALOGUE_TOTAL), "clips enriched"),
             (thousands(FIGURES["artists"]), "artists")],
            ENRICH, buff=1.6,
        )
        content.add(figs)
        self.play(FadeIn(figs, shift=UP * 0.15), run_time=0.5)
        self.wait(1.0)

        self.close_stage(content, rail, header)

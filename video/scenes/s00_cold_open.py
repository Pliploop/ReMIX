"""Cold open: what the dataset is *for*, shown before anything is explained.

No pipeline, no jargon. One track, an instruction, a different track -- the thing
a person actually does when looking for music. The stages only earn attention
once the viewer wants this to exist.
"""

from __future__ import annotations

from manim import *

from remix_video.chain import steps, tracks
from remix_video.facts import FIGURES
from remix_video.components import InstructionBubble,  TrackCard, title_card, txt
from remix_video.theme import (
    CHAIN, FAINT, INK, INSTRUCT, MUTED, NEIGHBOUR, PAPER, T_BODY, T_SMALL, VALIDATE, arrow,
)


class ColdOpen(Scene):
    def construct(self):
        self.camera.background_color = PAPER
        st = steps()
        tr = tracks()

        # --- 1. one track, sitting there ---------------------------------- #
        a = TrackCard(tr[0]["title"], tr[0]["artist"], seed=3, color=NEIGHBOUR,
                      tags=tr[0].get("tags", []), energy=1.0)
        a.move_to(LEFT * 3.4)

        lede = txt("You found something close.", T_BODY, MUTED).move_to(UP * 2.4)
        self.play(FadeIn(a, shift=UP * 0.25), FadeIn(lede), run_time=0.9)
        self.play(a.pulse(), run_time=0.5)
        self.wait(0.3)

        # --- 2. but not right. so you say what to change ------------------- #
        want = txt("But not quite right.", T_BODY, MUTED).move_to(UP * 2.4)
        self.play(ReplacementTransform(lede, want), run_time=0.5)

        bubble = InstructionBubble(f'"{st[0]["instruction"]}"', width=4.6)
        bubble.move_to(RIGHT * 3.0 + UP * 0.9)
        self.play(FadeIn(bubble, shift=LEFT * 0.3), run_time=0.7)
        self.wait(0.5)

        # --- 3. and you get a different track ------------------------------ #
        b = TrackCard(tr[1]["title"], tr[1]["artist"], seed=11, color=NEIGHBOUR,
                      tags=tr[1].get("tags", []), energy=1.25)
        b.move_to(RIGHT * 3.4 + DOWN * 1.0)

        hop = arrow(a.get_right() + RIGHT * 0.1, b.get_left() + LEFT * 0.1, INSTRUCT, 3.2)
        self.play(GrowArrow(hop), run_time=0.5)
        self.play(FadeIn(b, shift=UP * 0.25), run_time=0.7)
        self.play(b.pulse(), run_time=0.5)
        self.wait(0.6)

        # --- 4. the point: it keeps going, and it composes ------------------ #
        self.play(
            FadeOut(want), FadeOut(hop), FadeOut(bubble),
            a.animate.scale(0.62).move_to(LEFT * 2.4 + UP * 1.9),
            b.animate.scale(0.62).move_to(RIGHT * 2.4 + UP * 1.9),
            run_time=0.8,
        )
        pair_link = arrow(a.get_right() + RIGHT * 0.08, b.get_left() + LEFT * 0.08, INSTRUCT, 2.6)
        self.play(GrowArrow(pair_link), run_time=0.35)

        # The second instruction is the thesis: keep one thing, change another.
        keep = InstructionBubble(f'"{st[1]["instruction"]}"', width=4.4)
        keep.move_to(DOWN * 0.35)
        self.play(FadeIn(keep, shift=UP * 0.2), run_time=0.6)

        # Say the quiet part out loud.
        kw = VGroup(
            txt("keep", T_SMALL, CHAIN, BOLD),
            txt("one thing,", T_SMALL, MUTED),
            txt("change", T_SMALL, INSTRUCT, BOLD),
            txt("another", T_SMALL, MUTED),
        ).arrange(RIGHT, buff=0.14).next_to(keep, DOWN, buff=0.45)
        self.play(FadeIn(kw, shift=UP * 0.15), run_time=0.6)
        self.wait(0.8)

        # --- 5. the chain continues ---------------------------------------- #
        self.play(FadeOut(kw), FadeOut(keep), FadeOut(pair_link), run_time=0.4)

        rest = VGroup()
        for i, t in enumerate(tr[2:], start=2):
            c = TrackCard(t["title"], t["artist"], seed=17 + i * 5, color=NEIGHBOUR,
                          tags=t.get("tags", []), energy=1.0 + 0.1 * i).scale(0.52)
            rest.add(c)

        # arrange() mutates in place, so compute the target layout on copies and
        # animate the real cards to it -- otherwise a and b teleport.
        layout = VGroup(a.copy().scale(0.52 / 0.62), b.copy().scale(0.52 / 0.62), *rest.copy())
        layout.arrange(RIGHT, buff=0.5).move_to(UP * 1.2)

        for c, target in zip(rest, layout[2:]):
            c.move_to(target)

        self.play(
            a.animate.scale(0.52 / 0.62).move_to(layout[0]),
            b.animate.scale(0.52 / 0.62).move_to(layout[1]),
            run_time=0.6,
        )

        row = VGroup(a, b, *rest)
        links = VGroup(*[
            arrow(row[i].get_right() + RIGHT * 0.04, row[i + 1].get_left() + LEFT * 0.04, INSTRUCT, 2.4)
            for i in range(len(row) - 1)
        ])
        # Turn numbers, not the instruction text: at this size the sentences are
        # unreadable and collide with the cards. They were already read above.
        turn_marks = VGroup(*[
            VGroup(
                Circle(radius=0.13, fill_color=INSTRUCT, fill_opacity=1, stroke_width=0),
                txt(str(i + 1), 0.15, PAPER, BOLD),
            ).arrange(ORIGIN).next_to(links[i], UP, buff=0.1)
            for i in range(len(links))
        ])

        self.play(
            LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in rest], lag_ratio=0.16),
            run_time=1.0,
        )
        self.play(
            LaggedStart(
                *[AnimationGroup(GrowArrow(links[i]), FadeIn(turn_marks[i], shift=DOWN * 0.1))
                  for i in range(len(links))],
                lag_ratio=0.22,
            ),
            run_time=1.3,
        )
        instrs = VGroup()
        self.wait(0.4)

        # --- 6. name the problem, not the project --------------------------- #
        # No logo here, by design: the mark lands at the very end, once the
        # pipeline has earned it.
        claim = title_card("Finding music is a conversation.", VALIDATE, 0.5).move_to(DOWN * 1.4)
        self.play(FadeIn(claim, shift=UP * 0.2), run_time=0.7)
        self.wait(1.4)

        # Hand off to stage 1 with the question the pipeline answers.
        ask = txt(f"So where do {_n(FIGURES['chains'])} of these come from?", 0.44, INK, BOLD)
        ask.move_to(DOWN * 1.4)
        self.play(
            FadeOut(VGroup(row, links, turn_marks), shift=UP * 0.4),
            ReplacementTransform(claim, ask),
            run_time=0.9,
        )
        self.wait(1.3)
        self.play(FadeOut(ask), run_time=0.5)


def _n(v: int) -> str:
    return f"{v:,}"


def _short(s: str, n: int = 38) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"

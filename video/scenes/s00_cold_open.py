"""Cold open: what the dataset is *for*, shown before anything is explained.

No pipeline, no jargon. A track, an instruction, a different track -- the thing a
person actually does when looking for music. The stages only earn attention once
the viewer wants this to exist.

The instruction always sits *between* the two tracks, with an arrow in and an
arrow out. It is the cause of the transition, so it is drawn as the cause, and
both pairs use the identical shot so the repetition is the point.
"""

from __future__ import annotations

from manim import *

from remix_video.chain import steps, tracks
from remix_video.components import InstructionBubble, title_card, txt
from remix_video.glass import GlassCard
from remix_video.theme import (
    CHAIN, INK, INSTRUCT, MUTED, NEIGHBOUR, PAPER, STAGE_COLORS,
    T_BODY, T_SMALL, VALIDATE, arrow,
)

CARD_Y = 0.55


class ColdOpen(Scene):
    def construct(self):
        self.camera.background_color = PAPER
        st = steps()
        tr = tracks()

        # --- 1. a track, playing ------------------------------------------- #
        a = GlassCard(tr[0]["title"], tr[0]["artist"], seed=3, color=NEIGHBOUR,
                      tags=tr[0].get("tags", []), playing=True, width=3.2)
        a.move_to(LEFT * 4.3 + UP * CARD_Y)

        lede = txt("You found something close.", T_BODY, MUTED).move_to(UP * 2.6)
        self.play(FadeIn(a, shift=UP * 0.25), FadeIn(lede), run_time=0.9)
        # Let it actually play before anything else happens.
        self.play(a.pulse(1.06, 0.7), run_time=0.7)
        self.play(a.pulse(1.05, 0.7), run_time=0.7)
        self.wait(0.4)

        # --- 2. the instruction, between, as the cause ---------------------- #
        want = txt("But not quite right. So you say what to change.", T_BODY, MUTED).move_to(UP * 2.6)
        self.play(ReplacementTransform(lede, want), run_time=0.5)

        bubble = InstructionBubble(f'"{st[0]["instruction"]}"', width=4.3)
        bubble.move_to(UP * CARD_Y)
        in_arrow = arrow(a.get_right() + RIGHT * 0.06, bubble.get_left() + LEFT * 0.06, INSTRUCT, 3.0)
        self.play(GrowArrow(in_arrow), run_time=0.35)
        self.play(FadeIn(bubble, scale=0.92), run_time=0.6)
        self.wait(0.7)

        # --- 3. ...and a different track comes out -------------------------- #
        b = GlassCard(tr[1]["title"], tr[1]["artist"], seed=11, color=NEIGHBOUR,
                      tags=tr[1].get("tags", []), playing=True, width=3.2, energy=1.2)
        b.move_to(RIGHT * 4.3 + UP * CARD_Y)

        out_arrow = arrow(bubble.get_right() + RIGHT * 0.06, b.get_left() + LEFT * 0.06, INSTRUCT, 3.0)
        self.play(GrowArrow(out_arrow), run_time=0.35)
        self.play(FadeIn(b, shift=LEFT * 0.2), run_time=0.6)
        self.play(b.pulse(1.06, 0.7), run_time=0.7)
        self.wait(0.6)

        # --- 4. again, identically: it composes ----------------------------- #
        want2 = txt("And again. Each turn edits the last result.", T_BODY, MUTED).move_to(UP * 2.6)
        b2 = GlassCard(tr[2]["title"], tr[2]["artist"], seed=23, color=NEIGHBOUR,
                       tags=tr[2].get("tags", []), playing=True, width=3.2, energy=1.1)
        b2.move_to(RIGHT * 4.3 + UP * CARD_Y)

        keep = InstructionBubble(f'"{st[1]["instruction"]}"', width=4.3)
        keep.move_to(UP * CARD_Y)

        self.play(
            ReplacementTransform(want, want2),
            FadeOut(a, shift=LEFT * 0.3),
            b.animate.move_to(LEFT * 4.3 + UP * CARD_Y),
            FadeOut(bubble, scale=0.9),
            run_time=0.7,
        )
        self.play(FadeIn(keep, scale=0.92), run_time=0.5)
        self.play(FadeIn(b2, shift=LEFT * 0.2), run_time=0.5)
        self.play(b2.pulse(1.06, 0.6), run_time=0.6)

        # the thesis, said plainly
        kw = VGroup(
            txt("keep", T_SMALL, CHAIN, BOLD),
            txt("one thing,", T_SMALL, MUTED),
            txt("change", T_SMALL, INSTRUCT, BOLD),
            txt("another", T_SMALL, MUTED),
        ).arrange(RIGHT, buff=0.14).move_to(DOWN * 1.85)
        self.play(FadeIn(kw, shift=UP * 0.15), run_time=0.5)
        self.wait(1.0)

        # --- 5. the chain, staggered ---------------------------------------- #
        self.play(
            FadeOut(VGroup(kw, keep, in_arrow, out_arrow, want2)),
            FadeOut(VGroup(b, b2)),
            run_time=0.5,
        )

        minis = VGroup()
        for i, t in enumerate(tr[:5]):
            minis.add(
                GlassCard(t["title"], t["artist"], seed=3 + i * 7, color=NEIGHBOUR,
                          width=2.45, energy=0.9 + 0.08 * i).scale(0.72)
            )
        minis.arrange(RIGHT, buff=0.42)
        # Stagger vertically so the run reads as a walk, not a conveyor belt.
        for c, dy in zip(minis, [0.5, -0.3, 0.45, -0.4, 0.3]):
            c.shift(UP * dy)
        minis.move_to(UP * 0.5)

        links = VGroup(*[
            arrow(minis[i].get_right() + RIGHT * 0.03, minis[i + 1].get_left() + LEFT * 0.03,
                  NEIGHBOUR, 2.4)
            for i in range(len(minis) - 1)
        ])
        marks = VGroup(*[
            VGroup(
                Circle(radius=0.14, fill_color=NEIGHBOUR, fill_opacity=1, stroke_width=0),
                txt(str(i + 1), 0.16, PAPER, BOLD),
            ).arrange(ORIGIN).move_to(links[i].get_center() + UP * 0.28)
            for i in range(len(links))
        ])

        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in minis], lag_ratio=0.13),
                  run_time=1.1)
        self.play(
            LaggedStart(*[AnimationGroup(GrowArrow(links[i]), FadeIn(marks[i], scale=0.7))
                          for i in range(len(links))], lag_ratio=0.18),
            run_time=1.2,
        )
        self.wait(0.5)

        claim = title_card("Finding music is a conversation.", VALIDATE, 0.5).move_to(DOWN * 2.2)
        self.play(FadeIn(claim, shift=UP * 0.2), run_time=0.6)
        self.wait(1.3)

        # --- 6. name it, and hand off to stage 1 ---------------------------- #
        # The five stage colours arrive here so the cut into stage 1 is a colour
        # match rather than a jump. No numbers: this claim is qualitative.
        self.play(FadeOut(VGroup(minis, links, marks)), FadeOut(claim), run_time=0.6)

        name = txt("ReMIX", 0.95, INK, BOLD).move_to(UP * 0.75)
        sub = VGroup(
            txt("a dataset of", T_BODY, MUTED),
            txt("grounded transitions", T_BODY, INK, BOLD),
        ).arrange(RIGHT, buff=0.16).next_to(name, DOWN, buff=0.35)

        dots = VGroup(*[
            Circle(radius=0.1, fill_color=c, fill_opacity=1, stroke_width=0)
            for c in STAGE_COLORS
        ]).arrange(RIGHT, buff=0.22).next_to(sub, DOWN, buff=0.6)

        self.play(FadeIn(name, shift=UP * 0.15), run_time=0.6)
        self.play(FadeIn(sub, shift=UP * 0.1), run_time=0.5)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.1), run_time=0.7)
        self.wait(0.9)

        # The red dot swells and takes the frame: stage 1 begins inside it.
        red = dots[0]
        self.play(
            FadeOut(VGroup(name, sub)),
            FadeOut(VGroup(*dots[1:])),
            red.animate.scale(70),
            run_time=0.9,
        )
        self.play(FadeOut(red), run_time=0.4)

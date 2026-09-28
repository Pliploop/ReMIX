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
    CHAIN, EASE_IN, EASE_MOVE, HOLD, HOLD_READ, INK, INSTRUCT, LAG, MUTED, NEIGHBOUR, PAPER,
    STAGE_COLORS, T_BODY, T_ENTER, T_EXIT, T_MOVE, T_SMALL, VALIDATE, arrow,
)

CARD_Y = 0.25


class ColdOpen(Scene):
    def construct(self):
        self.camera.background_color = PAPER
        st = steps()
        tr = tracks()

        # --- 1. a track, playing ------------------------------------------- #
        a = GlassCard(tr[0]["title"], tr[0]["artist"], seed=3, color=NEIGHBOUR,
                      tags=tr[0].get("tags", []), playing=True, width=3.5)
        a.move_to(LEFT * 4.6 + UP * CARD_Y)

        lede = txt("You found something close.", T_BODY, MUTED).move_to(UP * 2.3)
        self.play(FadeIn(a, shift=UP * 0.15), FadeIn(lede, shift=UP * 0.08), run_time=T_ENTER, rate_func=EASE_IN)
        # Let it actually play before anything else happens.
        self.play(a.pulse(1.04, 0.8), run_time=0.8)
        self.wait(0.3)

        # --- 2. the instruction, between, as the cause ---------------------- #
        want = txt("But not quite right. So you say what to change.", T_BODY, MUTED).move_to(UP * 2.3)
        self.play(FadeOut(lede, shift=UP * 0.1), FadeIn(want, shift=UP * 0.1), run_time=T_ENTER, rate_func=EASE_MOVE)

        # Grey bubble, grey arrows: the only colours in the instruction are the
        # ones that mean something -- green for what it keeps, orange for what it
        # changes.
        bubble = InstructionBubble(
            "", width=4.6, segments=_segments(st[0]["instruction"]),
        ).move_to(UP * CARD_Y)
        in_arrow = arrow(a.get_right() + RIGHT * 0.06, bubble.get_left() + LEFT * 0.06, MUTED, 3.0)
        self.play(LaggedStart(GrowArrow(in_arrow), FadeIn(bubble, scale=0.96), lag_ratio=0.35),
                  run_time=0.9, rate_func=EASE_IN)
        self.wait(0.9)

        # --- 3. ...and a different track comes out -------------------------- #
        b = GlassCard(tr[1]["title"], tr[1]["artist"], seed=11, color=NEIGHBOUR,
                      tags=tr[1].get("tags", []), playing=True, width=3.5, energy=1.2)
        b.move_to(RIGHT * 4.6 + UP * CARD_Y)

        out_arrow = arrow(bubble.get_right() + RIGHT * 0.06, b.get_left() + LEFT * 0.06, MUTED, 3.0)
        self.play(LaggedStart(GrowArrow(out_arrow), FadeIn(b, shift=LEFT * 0.15), lag_ratio=0.35),
                  run_time=0.9, rate_func=EASE_IN)
        self.wait(0.6)

        # --- 4. again, identically: it composes ----------------------------- #
        want2 = txt("And again. Each turn edits the last result.", T_BODY, MUTED).move_to(UP * 2.3)
        b2 = GlassCard(tr[2]["title"], tr[2]["artist"], seed=23, color=NEIGHBOUR,
                       tags=tr[2].get("tags", []), playing=True, width=3.5, energy=1.1)
        b2.move_to(RIGHT * 4.6 + UP * CARD_Y)

        keep = InstructionBubble(
            "", width=4.6, segments=_segments(st[1]["instruction"]),
        ).move_to(UP * CARD_Y)

        # The arrows have to go *with* the bubble. Leaving them up while the card
        # slides left left them anchored to positions nothing occupied any more,
        # which is the flicker.
        # Clear the old pair first, then slide: the card must not pass over a bubble
        # that is still fading.
        self.play(
            LaggedStart(
                AnimationGroup(FadeOut(want, shift=UP * 0.1), FadeOut(a, shift=LEFT * 0.3),
                               FadeOut(VGroup(bubble, in_arrow, out_arrow), scale=0.96)),
                AnimationGroup(FadeIn(want2, shift=UP * 0.1), b.animate.move_to(LEFT * 4.6 + UP * CARD_Y)),
                lag_ratio=0.45,
            ),
            run_time=T_MOVE + 0.4, rate_func=EASE_MOVE,
        )
        in2 = arrow(b.get_right() + RIGHT * 0.06, keep.get_left() + LEFT * 0.06, MUTED, 3.0)
        out2 = arrow(keep.get_right() + RIGHT * 0.06, b2.get_left() + LEFT * 0.06, MUTED, 3.0)
        self.play(LaggedStart(GrowArrow(in2), FadeIn(keep, scale=0.96), GrowArrow(out2),
                              FadeIn(b2, shift=LEFT * 0.15), lag_ratio=0.3),
                  run_time=1.3, rate_func=EASE_IN)

        # the thesis, said plainly
        kw = VGroup(
            txt("keep", T_SMALL, CHAIN, SEMIBOLD),
            txt("one thing,", T_SMALL, MUTED),
            txt("change", T_SMALL, INSTRUCT, SEMIBOLD),
            txt("another", T_SMALL, MUTED),
        ).arrange(RIGHT, buff=0.14).move_to(DOWN * 1.5)
        self.play(FadeIn(kw, shift=UP * 0.1), run_time=T_ENTER, rate_func=EASE_IN)
        self.wait(1.1)

        # --- 5. the chain, staggered, built one hop at a time ---------------- #
        self.play(FadeOut(VGroup(kw, keep, in2, out2, want2, b, b2), shift=DOWN * 0.1),
                  run_time=T_EXIT)

        minis = VGroup()
        for i, t in enumerate(tr[:5]):
            minis.add(
                GlassCard(t["title"], t["artist"], seed=3 + i * 7, color=NEIGHBOUR,
                          width=2.3, energy=0.9 + 0.08 * i).scale(0.66)
            )
        minis.arrange(RIGHT, buff=0.78)
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
                txt(str(i + 1), 0.16, PAPER, SEMIBOLD),
            ).arrange(ORIGIN).move_to(links[i].get_center() + UP * 0.3)
            for i in range(len(links))
        ])

        # card, arrow, card, arrow... The chain is built, not revealed: showing
        # every card first and then every arrow says "layout", not "walk".
        # One continuous stagger (card, hop, card, ...) instead of eight stop-start plays.
        beats = [FadeIn(minis[0], shift=UP * 0.1)]
        for i in range(len(links)):
            beats += [AnimationGroup(GrowArrow(links[i]), FadeIn(marks[i], scale=0.8)),
                      FadeIn(minis[i + 1], shift=UP * 0.1)]
        self.play(LaggedStart(*beats, lag_ratio=0.45), run_time=2.0, rate_func=linear)
        self.wait(0.3)

        claim = title_card("Finding music is a conversation.", VALIDATE, 0.5).move_to(DOWN * 2.2)
        self.play(FadeIn(claim, shift=UP * 0.12), run_time=T_ENTER, rate_func=EASE_IN)
        self.wait(1.4)

        # --- 6. name it, and hand off to stage 1 ---------------------------- #
        # The five stage colours arrive here so the cut into stage 1 is a colour
        # match rather than a jump. No numbers: this claim is qualitative.
        self.play(FadeOut(VGroup(minis, links, marks, claim), shift=UP * 0.1), run_time=T_EXIT)

        name = txt("ReMIX", 0.95, INK, SEMIBOLD).move_to(UP * 0.9)
        sub = VGroup(
            txt("a dataset of", T_BODY, MUTED),
            txt("grounded transitions", T_BODY, INK, SEMIBOLD),
        ).arrange(RIGHT, buff=0.16).next_to(name, DOWN, buff=0.32)
        sub2 = txt("for composed music retrieval", T_BODY, MUTED).next_to(sub, DOWN, buff=0.16)

        dots = VGroup(*[
            Circle(radius=0.1, fill_color=c, fill_opacity=1, stroke_width=0)
            for c in STAGE_COLORS
        ]).arrange(RIGHT, buff=0.22).next_to(sub2, DOWN, buff=0.6)

        self.play(LaggedStart(FadeIn(name, shift=UP * 0.12), FadeIn(sub, shift=UP * 0.08),
                              FadeIn(sub2, shift=UP * 0.06), lag_ratio=0.25), run_time=1.1, rate_func=EASE_IN)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=LAG), run_time=0.6)
        self.wait(0.9)

        # One fade for the whole title: the next frame is stage 1's header, so no
        # dead white frames in between.
        self.play(FadeOut(VGroup(name, sub, sub2, dots), shift=UP * 0.1), run_time=T_EXIT)


KEEP_WORDS = ("keep", "keeps", "keeping", "retain", "preserve", "maintain")
CHANGE_WORDS = ("swap", "make", "shift", "switch", "add", "ditch", "drop", "slow",
                "speed", "turn", "strip", "lower", "raise", "polish", "shout")


def _segments(instruction: str):
    """Colour the instruction by what it does: green for a clause that keeps
    something, orange for one that changes something.

    Split on commas, since these instructions are clause-per-comma by
    construction -- that is what the clause budget in stage 4 enforces. A clause
    is coloured only when it opens with a keep or change verb; guessing beyond
    that would miscolour more often than it would help.
    """
    out = []
    clauses = [c.strip() for c in instruction.strip().rstrip(".").split(",") if c.strip()]
    for i, clause in enumerate(clauses):
        head = clause.split()[0].lower().strip('"')
        if head in KEEP_WORDS:
            color = CHAIN
        elif head in CHANGE_WORDS:
            color = INSTRUCT
        else:
            color = None
        out.append((clause + ("," if i < len(clauses) - 1 else ""), color))
    return out

"""The spine of the film: stages play full-frame, then shrink into a rail at the
top left, and at the end the rail flies back to centre and *is* the paper's main
figure.

Every stage scene shares this so the motion is identical each time -- that
repetition is what makes the accumulation legible instead of busy.
"""

from __future__ import annotations

from typing import List

from manim import *

from .theme import (
    BG, EASE_IN, EASE_MOVE, FAINT, INK, INK2, LINE, MUTED, PAPER, STAGE_COLORS, STAGE_NAMES, STROKE,
    T_ENTER, T_MOVE, T_SMALL, T_TINY, Y_EXPLAIN, Y_FIGURES, Y_HEADER, eyebrow, txt,
)

# The progress rail: five small dots top-right, one per stage. Finished stages are
# filled in their colour, the current one is a ring, the rest are grey. At the end
# the dots grow into the five panels of the paper's main figure (s06_assemble).
DOT_GAP = 0.34
DOT_Y = 3.3
MARGIN_X = 6.3
# Kept for the assembled figure's panel size.
SLOT_W = 0.92
SLOT_H = 0.92


def dot_position(i: int) -> np.ndarray:
    return np.array([MARGIN_X - (4 - i) * DOT_GAP, DOT_Y, 0.0])


def slot_position(i: int) -> np.ndarray:   # backwards-compatible name
    return dot_position(i)


def done_dot(i: int) -> Dot:
    return Dot(dot_position(i), radius=0.055, color=STAGE_COLORS[i])


def current_dot(i: int) -> VGroup:
    return VGroup(
        Circle(radius=0.09, fill_color=PAPER, fill_opacity=1, stroke_color=STAGE_COLORS[i], stroke_width=1.6),
        Dot(radius=0.035, color=STAGE_COLORS[i]),
    ).move_to(dot_position(i))


class StageScene(Scene):
    """Base for the five stage scenes.

    Subclasses set `stage_index` and implement `body()`. The header, the
    progress rail, and the close are handled here.
    """

    stage_index: int = 0

    def setup(self):
        self.camera.background_color = BG

    # House motion for every stage: the shared easing unless a call picks its own, and no
    # step shorter than MIN_STEP (sub-half-second steps read as stutter, not speed).
    # run_time below 0.05 is a deliberate instant swap and is left alone.
    MIN_STEP = 0.5

    def play(self, *animations, **kwargs):
        kwargs.setdefault("rate_func", EASE_MOVE)
        rt = kwargs.get("run_time")
        if rt is not None and 0.05 <= rt < self.MIN_STEP:
            kwargs["run_time"] = self.MIN_STEP
        return super().play(*animations, **kwargs)

    # --- shared furniture ------------------------------------------------- #
    def color(self) -> str:
        return STAGE_COLORS[self.stage_index]

    def name(self) -> str:
        return STAGE_NAMES[self.stage_index]

    def build_rail(self, upto: int) -> VGroup:
        """Hairline track with a dot per stage: 0..upto-1 filled, the rest grey."""
        track = Line(dot_position(0), dot_position(4), color=LINE, stroke_width=STROKE)
        dots = VGroup(*[
            done_dot(i) if i < upto else Dot(dot_position(i), radius=0.045, color=FAINT)
            for i in range(5)
        ])
        return VGroup(track, dots)

    def open_stage(self, upto: int) -> tuple[VGroup, VGroup]:
        """Show the rail so far, then announce this stage: a small tracked eyebrow
        in the stage colour over the title in ink, set flush left.

        Returns (rail, header) so the body can position around them.
        """
        rail = self.build_rail(upto)
        self.add(rail)
        self.ring = current_dot(self.stage_index)

        header = VGroup(
            eyebrow(f"Stage {self.stage_index + 1:02d}", self.color()),
            txt(self.name(), 0.42, INK, MEDIUM),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        header.move_to(np.array([-MARGIN_X, 3.42, 0.0]), aligned_edge=UL)
        self.play(FadeIn(header, shift=RIGHT * 0.12), FadeIn(self.ring, scale=0.6),
                  run_time=T_ENTER, rate_func=EASE_IN)
        return rail, header

    def close_stage(self, content: VGroup, rail: VGroup, header: VGroup):
        """Clear the stage and mark it done on the rail.

        Fades whatever is actually on screen rather than whatever `content`
        happens to list: FadeOut re-adds a mobject that was already removed, and
        animating a VGroup's children individually adds the children, not the
        wrapper, so a `content`-based list silently missed them and they hard-cut
        at the scene boundary instead of fading.
        """
        done = done_dot(self.stage_index)
        protected = {rail, self.ring}
        live = Group(*[m for m in self.mobjects if m not in protected])
        self.play(
            FadeOut(live, shift=UP * 0.1),
            Transform(self.ring, done),
            run_time=T_MOVE, rate_func=EASE_MOVE,
        )
        return done


def stat_row(pairs: List[tuple[str, str]], color: str = INK, buff: float = 1.1) -> VGroup:
    """A row of figures, parked on the figures band so it can never land on the
    explain line. Numbers only where we actually have them."""
    from .glass import StatBadge

    g = VGroup(*[StatBadge(v, l, color, 0.46) for v, l in pairs]).arrange(RIGHT, buff=buff, aligned_edge=UP)
    return g.move_to(UP * Y_FIGURES)


def explain(text: str, at=None, size: float = T_SMALL) -> Text:
    """One plain sentence per stage, on its own band. The film is silent; this
    carries it, so it must never collide with the figures below."""
    return txt(text, size * 1.08, INK2).move_to(at if at is not None else UP * Y_EXPLAIN)

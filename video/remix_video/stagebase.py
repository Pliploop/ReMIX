"""The spine of the film: stages play full-frame, then shrink into a rail at the
top left, and at the end the rail flies back to centre and *is* the paper's main
figure.

Every stage scene shares this so the motion is identical each time -- that
repetition is what makes the accumulation legible instead of busy.
"""

from __future__ import annotations

from typing import List

from manim import *

from .glass import StagePanel
from .theme import (
    INK, MUTED, PAPER, RAIL_LEFT, RAIL_TOP, STAGE_COLORS, STAGE_NAMES,
    T_SMALL, T_TINY, Y_EXPLAIN, Y_FIGURES, Y_HEADER, txt,
)

# Where finished stages stack: a discreet rail of square cards, padded down from
# the top edge and kept tight to the left so it never crowds the stage content.
SLOT_W = 1.15
SLOT_H = 1.15
SLOT_GAP = 0.16


def slot_position(i: int) -> np.ndarray:
    return np.array([RAIL_LEFT, RAIL_TOP, 0.0]) + RIGHT * (i * (SLOT_W + SLOT_GAP) + SLOT_W / 2)


class StageScene(Scene):
    """Base for the five stage scenes.

    Subclasses set `stage_index` and implement `body()`. The header, the
    accumulated rail, and the shrink-away are handled here.
    """

    stage_index: int = 0

    def setup(self):
        self.camera.background_color = PAPER

    # --- shared furniture ------------------------------------------------- #
    def color(self) -> str:
        return STAGE_COLORS[self.stage_index]

    def name(self) -> str:
        return STAGE_NAMES[self.stage_index]

    def build_rail(self, upto: int) -> VGroup:
        """Panels for stages already finished (0..upto-1), parked top-left."""
        rail = VGroup()
        for i in range(upto):
            p = StagePanel(i + 1, STAGE_NAMES[i], STAGE_COLORS[i], SLOT_W, SLOT_H, label_size=0.62)
            p.move_to(slot_position(i))
            p.set_opacity(0.9)
            rail.add(p)
        return rail

    def title_in(self) -> VGroup:
        """Big stage title, centred, that then retreats to make room."""
        n = txt(str(self.stage_index + 1), 1.5, self.color(), BOLD)
        name = txt(self.name(), 0.62, INK, BOLD)
        g = VGroup(n, name).arrange(RIGHT, buff=0.34)
        return g

    def open_stage(self, upto: int) -> tuple[VGroup, VGroup]:
        """Show the rail so far, then announce this stage.

        The whole film is 60s, so five stages get ~7s each. There is no room for
        a centred title card that then retreats -- the header goes straight in.

        Returns (rail, header) so the body can position around them.
        """
        rail = self.build_rail(upto)
        if len(rail):
            self.add(rail)

        header = VGroup(
            txt(f"{self.stage_index + 1}", 0.5, self.color(), BOLD),
            txt(self.name(), 0.4, INK, BOLD),
        ).arrange(RIGHT, buff=0.2)
        # Right of the rail, never over it.
        header.move_to(RIGHT * 1.6 + UP * Y_HEADER)
        self.play(FadeIn(header, shift=DOWN * 0.2), run_time=0.4)
        return rail, header

    def close_stage(self, content: VGroup, rail: VGroup, header: VGroup):
        """Collapse this stage into its slot and join the rail."""
        panel = StagePanel(self.stage_index + 1, self.name(), self.color(),
                           SLOT_W, SLOT_H, label_size=0.62)
        panel.move_to(slot_position(self.stage_index))
        panel.set_opacity(0.9)

        # Only fade what is still on screen: FadeOut re-adds a mobject that was
        # already removed, which is how the encoders reappeared at the end of
        # stage 2 in the first draft.
        live = VGroup(*[m for m in content if m in self.mobjects])
        self.play(
            FadeOut(live, shift=DOWN * 0.25),
            ReplacementTransform(header, panel),
            run_time=0.7,
        )
        self.wait(0.15)
        return panel


def stat_row(pairs: List[tuple[str, str]], color: str = INK, buff: float = 1.1) -> VGroup:
    """A row of figures. Numbers only where we actually have them."""
    from .glass import StatBadge

    return VGroup(*[StatBadge(v, l, color, 0.46) for v, l in pairs]).arrange(RIGHT, buff=buff)


def explain(text: str, at=None, size: float = T_SMALL) -> Text:
    """One plain sentence per stage, on its own band. The film is silent; this
    carries it, so it must never collide with the figures below."""
    return txt(text, size, MUTED).move_to(at if at is not None else UP * Y_EXPLAIN)

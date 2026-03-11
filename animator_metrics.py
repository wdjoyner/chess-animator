"""
animator_metrics.py

Draws the full-width metrics strip at the bottom of the chess video frame.
Four side-by-side sub-plots are revealed one move at a time as the game
is animated:

    Eval  |  Space  |  Mobility  |  KS (king safety)

Each sub-plot is built from individual Manim Line segments rather than a
parametric function, which lets us add exactly one segment per move inside
the animation loop with no re-rendering of earlier data.

Design decisions
----------------
- All data series are pre-computed from the full move list at construction
  time (the JSON is already loaded, so this costs nothing).
- Each sub-plot maintains a VGroup of Line segments.  advance_to_move(idx)
  adds the next segment(s) and moves the cursor line; it returns a single
  Manim AnimationGroup so the caller can run it in parallel with the board
  and panel animations.
- Eval uses two colored half-segments per step: green above zero, red below,
  with a white segment crossing the zero line when the sign changes.
- Space and Mobility each show two lines: White (light gray) and Black
  (steel blue) as per-side raw values, so the viewer sees both sides moving.
- FTI shows a single net line (FTI1 Harmonious by default) in accent red.
- A dim horizontal zero-line and a thin white vertical cursor are drawn for
  every sub-plot.

Public API (used by animator_game.py)
--------------------------------------
    panel = MetricPlotPanel(all_moves)   # construct once
    scene.add(panel.get_mobject())       # add static elements to scene
    anim  = panel.advance_to_move(idx)   # call inside the animation loop
    scene.play(anim, ...)
"""

from __future__ import annotations

from typing import List, Tuple

from manim import *

from animator_layout import (
    COLORS, FONTS,
    METRICS_TOP_Y, METRICS_BOTTOM_Y, METRICS_CENTER_Y,
    METRICS_LEFT_X, METRICS_RIGHT_X, METRICS_WIDTH,
    METRICS_SUBPLOT_WIDTH, METRICS_SUBPLOT_GAP,
    METRICS_SUBPLOT_LABELS, METRICS_SUBPLOT_LEFT_EDGES,
    METRICS_EVAL_CENTER_X, METRICS_SPACE_CENTER_X,
    METRICS_MOBILITY_CENTER_X,
    get_metrics_rect,
)

# Import MoveData type for annotations only (avoid circular import at runtime)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from animator_game import MoveData


# =============================================================================
# Layout constants local to this module
# =============================================================================

# Internal vertical margins within each sub-plot
_PLOT_TOP_MARGIN    = 0.22   # space above the axes for the label
_PLOT_BOTTOM_MARGIN = 0.18   # space below for x-axis tick labels (move numbers)
_PLOT_H_MARGIN      = 0.08   # small horizontal inset inside each sub-plot

# Derived plot drawing area (same for all four sub-plots; only x-origin differs)
_PLOT_HEIGHT = (METRICS_TOP_Y - METRICS_BOTTOM_Y
                - _PLOT_TOP_MARGIN - _PLOT_BOTTOM_MARGIN)
_PLOT_DRAW_TOP    = METRICS_TOP_Y    - _PLOT_TOP_MARGIN
_PLOT_DRAW_BOTTOM = METRICS_BOTTOM_Y + _PLOT_BOTTOM_MARGIN
_PLOT_DRAW_WIDTH  = METRICS_SUBPLOT_WIDTH - 2 * _PLOT_H_MARGIN

# Fixed y-axis range for Eval only (well-defined ± pawns scale)
_EVAL_RANGE = 5.0    # ± pawns

# Padding fraction added above/below the data range for Space, Mobility, FTI
# e.g. 0.15 means 15% of the data range is added as breathing room each side
_AUTOSCALE_PAD = 0.15

# Minimum span for auto-scaled axes — prevents a completely flat line
# when values barely change (e.g. space = 0.4 the whole game)
_MIN_SPAN = 0.1

# Stroke widths
_LINE_WIDTH  = 1.5   # data series
_ZERO_WIDTH  = 0.8   # zero reference line
_CURSOR_WIDTH = 1.0  # vertical cursor


# =============================================================================
# Coordinate helpers
# =============================================================================

def _plot_origin_x(subplot_index: int) -> float:
    """Left drawing edge (inside margins) for sub-plot number subplot_index."""
    return METRICS_SUBPLOT_LEFT_EDGES[subplot_index] + _PLOT_H_MARGIN


def _x_coord(subplot_index: int, move_idx: int, total_moves: int) -> float:
    """
    Map a move index (0-based) to an x-coordinate within a sub-plot.
    move_idx == 0 → left edge;  move_idx == total_moves-1 → right edge.
    """
    origin = _plot_origin_x(subplot_index)
    if total_moves <= 1:
        return origin
    frac = move_idx / (total_moves - 1)
    return origin + frac * _PLOT_DRAW_WIDTH


def _y_coord(value: float, y_min: float, y_max: float) -> float:
    """
    Map a data value to a y-coordinate within the plot drawing area.
    Clamps value to [y_min, y_max].
    """
    value  = max(y_min, min(y_max, value))
    frac   = (value - y_min) / (y_max - y_min)
    return _PLOT_DRAW_BOTTOM + frac * _PLOT_HEIGHT


def _zero_y(y_min: float, y_max: float) -> float:
    """Y-coordinate of the zero line (clamped to drawing area)."""
    return _y_coord(0.0, y_min, y_max)


# =============================================================================
# SubPlot helper class
# =============================================================================

class _SubPlot:
    """
    Manages one sub-plot inside the metrics strip.

    Each series is a VGroup of Line segments grown incrementally.
    The cursor is a single Line that is repositioned each step.
    """

    def __init__(self,
                 subplot_index: int,
                 label: str,
                 y_min: float,
                 y_max: float,
                 series_specs: List[Tuple[str, str]]):
        """
        Args:
            subplot_index: 0–3 (left to right)
            label:         Display label shown above the plot
            y_min, y_max:  Data range (determines scale)
            series_specs:  List of (series_name, hex_color) pairs.
                           Series are drawn in order; names are used
                           only for external reference.
        """
        self.index       = subplot_index
        self.label       = label
        self.y_min       = y_min
        self.y_max       = y_max
        self.series_names = [s[0] for s in series_specs]
        self.series_colors = [s[1] for s in series_specs]
        self.n_series    = len(series_specs)

        # One VGroup of Line segments per series
        self.series_groups: List[VGroup] = [VGroup() for _ in series_specs]

        # Previous y-coordinates for each series (for segment start points)
        self.prev_y: List[float | None] = [None] * self.n_series

        # Cursor line (repositioned each step)
        self.cursor = Line(
            start=[0, _PLOT_DRAW_BOTTOM, 0],
            end=[0, _PLOT_DRAW_TOP, 0],
            stroke_color=COLORS.plot_cursor,
            stroke_width=_CURSOR_WIDTH,
            stroke_opacity=0.5,
        )

        # Static elements (background, axes, label, zero line)
        self.static_group = VGroup()
        self._build_static()

        # Dynamic group (series lines + cursor)
        self.dynamic_group = VGroup(*self.series_groups, self.cursor)

    # ------------------------------------------------------------------
    # Static elements
    # ------------------------------------------------------------------

    def _build_static(self):
        """Build axes background, zero line, and label."""
        lx = METRICS_SUBPLOT_LEFT_EDGES[self.index]
        rx = lx + METRICS_SUBPLOT_WIDTH
        cx = lx + METRICS_SUBPLOT_WIDTH / 2

        # Sub-plot background (slightly different shade)
        bg = Rectangle(
            width=METRICS_SUBPLOT_WIDTH,
            height=METRICS_TOP_Y - METRICS_BOTTOM_Y,
            fill_color=COLORS.panel_bg,
            fill_opacity=0.6,
            stroke_color=COLORS.panel_border,
            stroke_width=1,
        )
        bg.move_to([cx, METRICS_CENTER_Y, 0])
        self.static_group.add(bg)

        # Zero reference line
        zero_y = _zero_y(self.y_min, self.y_max)
        zero_line = Line(
            start=[_plot_origin_x(self.index),                zero_y, 0],
            end=[_plot_origin_x(self.index) + _PLOT_DRAW_WIDTH, zero_y, 0],
            stroke_color=COLORS.plot_zero_line,
            stroke_width=_ZERO_WIDTH,
        )
        self.static_group.add(zero_line)

        # Label + range above the plot, arranged on one line
        label_text = Text(
            self.label,
            font=FONTS.body_font,
            font_size=FONTS.metric_label_size,
            color=COLORS.text_secondary,
        )
        range_text = Text(
            f"[{self.y_min:.2g}, {self.y_max:.2g}]",
            font=FONTS.body_font,
            font_size=FONTS.metric_label_size - 2,
            color=COLORS.text_secondary,
        )
        header = VGroup(label_text, range_text)
        header.arrange(RIGHT, buff=0.08)
        header.move_to([cx, METRICS_TOP_Y - _PLOT_TOP_MARGIN / 2, 0])
        self.static_group.add(header)

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def first_point(self, move_idx: int, total_moves: int,
                    values: List[float]):
        """
        Record the very first data point (move_idx == 0).
        No segment is drawn yet; just stores the starting y positions.
        Positions the cursor at x=left edge.
        """
        for i, val in enumerate(values):
            self.prev_y[i] = _y_coord(val, self.y_min, self.y_max)

        x = _x_coord(self.index, move_idx, total_moves)
        self.cursor.put_start_and_end_on(
            [x, _PLOT_DRAW_BOTTOM, 0],
            [x, _PLOT_DRAW_TOP,    0],
        )

    def add_segments(self, move_idx: int, total_moves: int,
                     values: List[float],
                     split_colors: List[Tuple[str, str]] | None = None
                     ) -> List[Mobject]:
        """
        Add one line segment per series from the previous point to the
        new point, move the cursor, and return the new Mobjects so the
        caller can wrap them in a Manim animation.

        Args:
            move_idx:     Current move index (1-based here; must be > 0)
            total_moves:  Total number of moves in the game
            values:       New data values, one per series
            split_colors: Optional list of (color_pos, color_neg) pairs
                          per series.  When provided the segment is colored
                          by the sign of the *new* value rather than the
                          fixed series color.  Pass None for a fixed color.

        Returns:
            List of newly created Line objects (already added to their
            respective series VGroups).
        """
        x_new = _x_coord(self.index, move_idx, total_moves)
        x_old = _x_coord(self.index, move_idx - 1, total_moves)

        new_objects: List[Mobject] = []

        for i, (val, base_color) in enumerate(zip(values, self.series_colors)):
            y_new = _y_coord(val, self.y_min, self.y_max)
            y_old = self.prev_y[i] if self.prev_y[i] is not None else y_new

            # Choose color
            if split_colors and split_colors[i] is not None:
                color_pos, color_neg = split_colors[i]
                color = color_pos if val >= 0 else color_neg
            else:
                color = base_color

            seg = Line(
                start=[x_old, y_old, 0],
                end=[x_new,   y_new, 0],
                stroke_color=color,
                stroke_width=_LINE_WIDTH,
            )
            self.series_groups[i].add(seg)
            new_objects.append(seg)
            self.prev_y[i] = y_new

        # Move cursor
        self.cursor.put_start_and_end_on(
            [x_new, _PLOT_DRAW_BOTTOM, 0],
            [x_new, _PLOT_DRAW_TOP,    0],
        )
        new_objects.append(self.cursor)

        return new_objects

    # ------------------------------------------------------------------
    # VGroup access
    # ------------------------------------------------------------------

    def get_mobject(self) -> VGroup:
        """Return all Manim objects for this sub-plot."""
        return VGroup(self.static_group, self.dynamic_group)


def _autoscale(values: list, pad: float = _AUTOSCALE_PAD,
               min_span: float = _MIN_SPAN,
               force_include_zero: bool = False):
    """
    Compute y_min, y_max for a list of values with padding.

    Args:
        values:              All data values for this series.
        pad:                 Fractional padding added each side.
        min_span:            Minimum total span (avoids flat-line axes).
        force_include_zero:  If True, always include y=0 in the range
                             (useful for net-advantage series).

    Returns:
        (y_min, y_max) tuple
    """
    if not values:
        return -min_span / 2, min_span / 2

    lo, hi = min(values), max(values)

    if force_include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)

    span = hi - lo
    if span < min_span:
        mid = (hi + lo) / 2
        lo  = mid - min_span / 2
        hi  = mid + min_span / 2
        span = min_span

    lo -= span * pad
    hi += span * pad
    return lo, hi


# =============================================================================
# MetricPlotPanel  (public API)
# =============================================================================

class MetricPlotPanel:
    """
    Full-width metrics strip containing four sub-plots.

    Usage in animator_game.py::

        # Construction (once, before the animation loop)
        metric_panel = MetricPlotPanel(analysis.moves)
        self.add(metric_panel.get_mobject())

        # Inside the loop
        anim = metric_panel.advance_to_move(idx)
        self.play(..., anim, run_time=0.4)
    """

    def __init__(self, all_moves: "List[MoveData]"):
        self.all_moves   = all_moves
        self.total_moves = len(all_moves)

        # Pre-compute all data series
        self._eval     = self._extract_eval()
        self._space_w  = self._extract(lambda m: m.space_white)
        self._space_b  = self._extract(lambda m: m.space_black)
        self._mob_w    = self._extract(lambda m: m.mobility_white)
        self._mob_b    = self._extract(lambda m: m.mobility_black)
        self._ks_w     = self._extract(lambda m: m.king_safety_white)
        self._ks_b     = self._extract(lambda m: m.king_safety_black)

        # Build four SubPlot objects
        # Eval: fixed ± range (well-understood pawns scale)
        self._eval_plot = _SubPlot(
            subplot_index=0,
            label="Eval",
            y_min=-_EVAL_RANGE,
            y_max=+_EVAL_RANGE,
            series_specs=[("eval", COLORS.plot_net_pos)],
        )

        # Space: auto-scale across both White and Black series combined
        space_all = self._space_w + self._space_b
        sp_min, sp_max = _autoscale(space_all, force_include_zero=True)
        self._space_plot = _SubPlot(
            subplot_index=1,
            label="Space",
            y_min=sp_min,
            y_max=sp_max,
            series_specs=[
                ("white", COLORS.plot_white),
                ("black", COLORS.plot_black),
            ],
        )

        # Mobility: auto-scale across both series combined
        mob_all = self._mob_w + self._mob_b
        mob_min, mob_max = _autoscale(mob_all, force_include_zero=True)
        self._mob_plot = _SubPlot(
            subplot_index=2,
            label="Mobility",
            y_min=mob_min,
            y_max=mob_max,
            series_specs=[
                ("white", COLORS.plot_white),
                ("black", COLORS.plot_black),
            ],
        )

        # King Safety: auto-scale across both series combined
        ks_all = self._ks_w + self._ks_b
        ks_min, ks_max = _autoscale(ks_all, force_include_zero=True)
        self._ks_plot = _SubPlot(
            subplot_index=3,
            label="King Safety",
            y_min=ks_min,
            y_max=ks_max,
            series_specs=[
                ("white", COLORS.plot_white),
                ("black", COLORS.plot_black),
            ],
        )

        self._subplots = [
            self._eval_plot,
            self._space_plot,
            self._mob_plot,
            self._ks_plot,
        ]

        # Strip background (drawn once, behind all sub-plots)
        self._bg = get_metrics_rect()

        # Initialise first point (no segment drawn yet)
        if self.total_moves > 0:
            self._init_first_point()

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------

    def _extract(self, getter) -> List[float]:
        return [getter(m) for m in self.all_moves]

    def _extract_eval(self) -> List[float]:
        """Eval in pawns, clamped to ±EVAL_RANGE."""
        return [
            max(-_EVAL_RANGE, min(_EVAL_RANGE, m.eval_after / 100.0))
            for m in self.all_moves
        ]

    # ------------------------------------------------------------------
    # Initialise first data point (move index 0)
    # ------------------------------------------------------------------

    def _init_first_point(self):
        n = self.total_moves
        self._eval_plot.first_point(0, n, [self._eval[0]])
        self._space_plot.first_point(0, n, [self._space_w[0], self._space_b[0]])
        self._mob_plot.first_point(0, n, [self._mob_w[0], self._mob_b[0]])
        self._ks_plot.first_point(0, n, [self._ks_w[0], self._ks_b[0]])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_mobject(self) -> VGroup:
        """
        Return the complete VGroup for the metrics strip.
        Add this to the Manim scene once before the animation loop.
        """
        return VGroup(
            self._bg,
            *[sp.get_mobject() for sp in self._subplots],
        )

    def advance_to_move(self, idx: int) -> Animation:
        """
        Grow each sub-plot by one segment to include move number `idx`
        (0-based), and slide the cursor to the new position.

        Call this inside self.play(...) in the animation loop.

        Args:
            idx: Current move index (0-based).  Pass the same idx used
                 in the outer for-loop in AnimatedGame.construct().

        Returns:
            A Manim AnimationGroup that can be played in parallel with
            the board move and panel updates.
        """
        # idx == 0 was handled at construction (first_point); nothing to draw.
        if idx == 0 or idx >= self.total_moves:
            return Wait(0)

        n   = self.total_moves
        new_objects: List[Mobject] = []

        # Eval: color by sign of new value
        new_objects += self._eval_plot.add_segments(
            idx, n,
            values=[self._eval[idx]],
            split_colors=[(COLORS.plot_net_pos, COLORS.plot_net_neg)],
        )

        # Space: two fixed-color series (White / Black)
        new_objects += self._space_plot.add_segments(
            idx, n,
            values=[self._space_w[idx], self._space_b[idx]],
            split_colors=[None, None],
        )

        # Mobility: two fixed-color series
        new_objects += self._mob_plot.add_segments(
            idx, n,
            values=[self._mob_w[idx], self._mob_b[idx]],
            split_colors=[None, None],
        )

        # King Safety: two fixed-color series (White / Black)
        new_objects += self._ks_plot.add_segments(
            idx, n,
            values=[self._ks_w[idx], self._ks_b[idx]],
            split_colors=[None, None],
        )

        # Wrap new segments in FadeIn animations.
        # Cursors are repositioned in-place (put_start_and_end_on) so they
        # don't need an animation — exclude them from the group.
        segs = [obj for obj in new_objects
                if obj is not self._eval_plot.cursor
                and obj is not self._space_plot.cursor
                and obj is not self._mob_plot.cursor
                and obj is not self._ks_plot.cursor]
        if not segs:
            return Wait(0)
        return AnimationGroup(*[FadeIn(obj, run_time=0.0) for obj in segs],
                              run_time=0.0)


# =============================================================================
# Standalone debug scene
# =============================================================================

class MetricsDebug(Scene):
    """
    Renders the metrics strip with synthetic sinusoidal data so you can
    check layout and colors without needing a real game analysis file.

    Run with:
        manim -pql animator_metrics.py MetricsDebug
    """

    def construct(self):
        import math
        self.camera.background_color = COLORS.background

        # Generate synthetic MoveData-like objects
        N = 40

        class _FakeMove:
            def __init__(self, i):
                t = i / N
                self.eval_after          =  300 * math.sin(2 * math.pi * t)
                self.space_white         =  0.3 + 0.2 * math.sin(math.pi * t)
                self.space_black         =  0.3 - 0.15 * math.sin(math.pi * t)
                self.mobility_white      =  1.0 + 0.8 * math.cos(2 * math.pi * t)
                self.mobility_black      =  1.0 - 0.6 * math.cos(2 * math.pi * t)
                self.king_safety_white   =  0.5 + 0.4 * math.sin(1.5 * math.pi * t)
                self.king_safety_black   =  0.5 - 0.3 * math.cos(1.5 * math.pi * t)
                # unused but kept so MoveData-like duck typing stays valid
                self.fti1 = 0.0

        fake_moves = [_FakeMove(i) for i in range(N)]

        panel = MetricPlotPanel(fake_moves)
        self.add(panel.get_mobject())
        self.wait(0.5)

        for idx in range(1, N):
            self.play(panel.advance_to_move(idx), run_time=0.08)

        self.wait(2)

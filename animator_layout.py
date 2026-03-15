"""
animator_layout.py

Layout constants and configuration for the chess game video animator.
Defines the 16:9 frame geometry, positioning, colors, and fonts.

Color scheme: light warm background (#f0ede8) with dark text and plot lines
for maximum readability.

The layout divides the frame into two zones:

  ┌─────────────────────────────────────────────────────────┐
  │  Eval │                    │  Header                    │
  │  Bar  │   Chess Board      │  Move List                 │
  │       │                    │  Commentary                │
  ├───────────────────────────────────────────────────────── │
  │         Metrics Strip  (Eval · Space · Mobility · FTI)  │
  └─────────────────────────────────────────────────────────┘

Upper zone  — board + eval bar on the left, three stacked panels on the right.
Metrics strip — full-width horizontal band across the bottom of the frame,
                shared by both the board side and the panel side.

Key geometry decisions
----------------------
- Manim default coordinate system: x ∈ [-7.11, 7.11], y ∈ [-4.0, 4.0]
- The metrics strip is METRICS_HEIGHT tall and sits flush at the bottom.
- UPPER_BOTTOM_Y is the shared boundary: upper zone ends here, strip starts.
- A small GAP separates the strip from the upper zone for visual breathing room.
- The board is re-centred vertically within the upper zone.
- The three right panels are re-computed to fit the (now shorter) upper zone.
"""

from dataclasses import dataclass
from manim import *


# =============================================================================
# Frame Geometry (16:9 aspect ratio)
# =============================================================================

FRAME_WIDTH  = config.frame_width    # ~14.22 Manim units
FRAME_HEIGHT = config.frame_height   # 8.0 Manim units

# y-coordinates of the overall frame edges
FRAME_TOP_Y    =  FRAME_HEIGHT / 2   #  4.0
FRAME_BOTTOM_Y = -FRAME_HEIGHT / 2   # -4.0

# x-coordinates of the overall frame edges
FRAME_LEFT_X  = -FRAME_WIDTH / 2     # ~-7.11
FRAME_RIGHT_X =  FRAME_WIDTH / 2     # ~ 7.11

MARGIN = 0.3   # general margin from frame edges


# =============================================================================
# Metrics Strip  (Option B — full-width bottom band)
# =============================================================================

# Height of the metrics strip in Manim units.
# 1.6 gives four sub-plots enough room for axes labels and curves
# while leaving the upper zone well proportioned (≈ 75 % of frame height).
METRICS_HEIGHT = 1.6

# Gap between the bottom of the upper zone and the top of the strip
METRICS_GAP = 0.15

# Strip vertical boundaries
METRICS_BOTTOM_Y = FRAME_BOTTOM_Y + MARGIN          # -3.7
METRICS_TOP_Y    = METRICS_BOTTOM_Y + METRICS_HEIGHT # -2.1

# Strip horizontal boundaries (full width with margins)
METRICS_LEFT_X  = FRAME_LEFT_X  + MARGIN   # ~-6.81
METRICS_RIGHT_X = FRAME_RIGHT_X - MARGIN   # ~ 6.81
METRICS_WIDTH   = METRICS_RIGHT_X - METRICS_LEFT_X
METRICS_CENTER_X = 0.0                     # centred on frame
METRICS_CENTER_Y = (METRICS_TOP_Y + METRICS_BOTTOM_Y) / 2

# Individual sub-plot widths (four equal plots with small inter-plot gap)
METRICS_SUBPLOT_GAP   = 0.2
METRICS_SUBPLOT_WIDTH = (METRICS_WIDTH - 3 * METRICS_SUBPLOT_GAP) / 4

# Sub-plot left-edge x positions (left → right: Eval, Space, Mobility, FTI)
def _subplot_left(index: int) -> float:
    """Return the left-edge x-coordinate of sub-plot number `index` (0-based)."""
    return METRICS_LEFT_X + index * (METRICS_SUBPLOT_WIDTH + METRICS_SUBPLOT_GAP)

METRICS_EVAL_LEFT_X     = _subplot_left(0)
METRICS_SPACE_LEFT_X    = _subplot_left(1)
METRICS_MOBILITY_LEFT_X = _subplot_left(2)
METRICS_FTI_LEFT_X      = _subplot_left(3)

# Convenience: centre x of each sub-plot
METRICS_EVAL_CENTER_X     = METRICS_EVAL_LEFT_X     + METRICS_SUBPLOT_WIDTH / 2
METRICS_SPACE_CENTER_X    = METRICS_SPACE_LEFT_X    + METRICS_SUBPLOT_WIDTH / 2
METRICS_MOBILITY_CENTER_X = METRICS_MOBILITY_LEFT_X + METRICS_SUBPLOT_WIDTH / 2
METRICS_FTI_CENTER_X      = METRICS_FTI_LEFT_X      + METRICS_SUBPLOT_WIDTH / 2

# Sub-plot labels (used by animator_metrics.py)
METRICS_SUBPLOT_LABELS = ["Eval", "Space", "Mobility", "FTI"]
METRICS_SUBPLOT_LEFT_EDGES = [
    METRICS_EVAL_LEFT_X,
    METRICS_SPACE_LEFT_X,
    METRICS_MOBILITY_LEFT_X,
    METRICS_FTI_LEFT_X,
]


# =============================================================================
# Upper Zone  (board + panels, sits above the metrics strip)
# =============================================================================

UPPER_TOP_Y    = FRAME_TOP_Y    - MARGIN              #  3.7
UPPER_BOTTOM_Y = METRICS_TOP_Y  + METRICS_GAP         # -1.95  (approx)
UPPER_HEIGHT   = UPPER_TOP_Y - UPPER_BOTTOM_Y         #  5.65  (approx)
UPPER_CENTER_Y = (UPPER_TOP_Y + UPPER_BOTTOM_Y) / 2


# =============================================================================
# Board Layout  (left side of upper zone)
# =============================================================================

# Scale slightly tighter than before so the board fits the reduced upper zone.
# The original BOARD_SCALE = 0.80 was tuned for a taller zone; 0.74 keeps the
# board comfortably clear of the metrics strip with a little breathing room.
BOARD_SCALE = 0.74

# Centre the board within the upper zone vertically
BOARD_CENTER_X = -2.5
BOARD_CENTER_Y = UPPER_CENTER_Y   # tracks the upper zone, not a hard-coded 0.0

# Evaluation bar (to the left of the board)
EVAL_BAR_SCALE  = 0.72
EVAL_BAR_OFFSET = 0.3   # gap between eval bar and board left edge


# =============================================================================
# Right Panel Layout  (three stacked panels, right side of upper zone)
# =============================================================================

# Horizontal boundaries — unchanged from original
PANEL_LEFT_X   = 1.0
PANEL_RIGHT_X  = 6.5
PANEL_WIDTH    = PANEL_RIGHT_X - PANEL_LEFT_X
PANEL_CENTER_X = (PANEL_LEFT_X + PANEL_RIGHT_X) / 2

# Vertical boundaries now follow the upper zone, not the full frame
PANEL_TOP_Y    = UPPER_TOP_Y      #  3.7
PANEL_BOTTOM_Y = UPPER_BOTTOM_Y   # ~-1.95
PANEL_TOTAL_HEIGHT = PANEL_TOP_Y - PANEL_BOTTOM_Y

# Panel height ratios — same proportions as before
HEADER_RATIO     = 0.28
MOVE_LIST_RATIO  = 0.40
COMMENTARY_RATIO = 0.32   # remainder; not used directly in calculation below

# Calculate panel boundaries
HEADER_TOP_Y    = PANEL_TOP_Y
HEADER_BOTTOM_Y = PANEL_TOP_Y - (PANEL_TOTAL_HEIGHT * HEADER_RATIO)

MOVE_LIST_TOP_Y    = HEADER_BOTTOM_Y - 0.1
MOVE_LIST_BOTTOM_Y = MOVE_LIST_TOP_Y - (PANEL_TOTAL_HEIGHT * MOVE_LIST_RATIO)

COMMENTARY_TOP_Y    = MOVE_LIST_BOTTOM_Y - 0.1
COMMENTARY_BOTTOM_Y = PANEL_BOTTOM_Y      # commentary fills remaining space

# Panel vertical centres
HEADER_CENTER_Y     = (HEADER_TOP_Y     + HEADER_BOTTOM_Y)     / 2
MOVE_LIST_CENTER_Y  = (MOVE_LIST_TOP_Y  + MOVE_LIST_BOTTOM_Y)  / 2
COMMENTARY_CENTER_Y = (COMMENTARY_TOP_Y + COMMENTARY_BOTTOM_Y) / 2


# =============================================================================
# Colors
# =============================================================================

@dataclass
class ColorScheme:
    """Color scheme for the video — light background with dark text and plot lines."""
    background: str = "#f0ede8"       # Warm off-white background

    # Panel colors
    panel_bg:     str = "#e8e4de"     # Slightly darker than background
    panel_border: str = "#b0a898"     # Warm gray panel border

    # Text colors
    text_primary:   str = "#1a1a1a"   # Near-black for main text
    text_secondary: str = "#555555"   # Medium gray for secondary info
    text_accent:    str = "#b03040"   # Deep red accent for highlights

    # Player colors
    white_player: str = "#2a2a2a"     # Dark for White player label
    black_player: str = "#2a2a2a"     # Dark for Black player label

    # Move classification colors (darkened for legibility on light bg)
    brilliant:  str = "#0e7a76"       # Dark teal
    great:      str = "#2e5f8a"       # Dark blue
    best:       str = "#4a7a1e"       # Dark green
    excellent:  str = "#3d6b10"       # Darker green
    good:       str = "#4a6040"       # Muted dark green
    book:       str = "#6b5030"       # Dark brown
    inaccuracy: str = "#a07800"       # Dark amber
    mistake:    str = "#b05010"       # Dark orange
    blunder:    str = "#8b1a1a"       # Dark red
    missed_win: str = "#9b2020"       # Deep red

    # Metric plot line colors (dark, readable on light background)
    plot_white:      str = "#444444"  # Dark gray     — White's series
    plot_black:      str = "#1a3a6a"  # Dark navy     — Black's series
    plot_net_pos:    str = "#2e6b10"  # Dark green    — net advantage (positive)
    plot_net_neg:    str = "#8b1a1a"  # Dark red      — net advantage (negative)
    plot_fti:        str = "#7a1030"  # Deep crimson  — FTI line
    plot_zero_line:  str = "#999999"  # Medium gray   — y=0 reference line
    plot_cursor:     str = "#1a1a1a"  # Near-black    — current-move cursor


# Default color scheme
COLORS = ColorScheme()


# =============================================================================
# Typography
# =============================================================================

@dataclass
class Typography:
    """Font configuration."""
    heading_font: str = "Courier New"
    body_font:    str = "Courier New"
    mono_font:    str = "Courier New"

    # Sizes (Manim units; ~1 unit ≈ 36 pt at default resolution)
    title_size:       int = 20
    subtitle_size:    int = 14
    player_name_size: int = 14
    player_elo_size:  int = 14
    move_size:        int = 14
    commentary_size:  int = 14
    label_size:       int = 14

    # Header panel — compact sizes so content fits the short panel height
    header_player_size: int = 12   # player name lines
    header_vs_size:     int = 10   # "vs ♚ Black" line
    header_info_size:   int = 10   # event · date and opening lines

    # Smaller size for metric plot axis labels
    metric_label_size: int = 10


FONTS = Typography()


# =============================================================================
# Helper Functions
# =============================================================================

def get_panel_rect(top_y: float, bottom_y: float,
                   left_x: float = PANEL_LEFT_X,
                   right_x: float = PANEL_RIGHT_X) -> Rectangle:
    """
    Create a styled Rectangle for a panel background.

    Args:
        top_y:   Top edge y-coordinate
        bottom_y: Bottom edge y-coordinate
        left_x:  Left edge x-coordinate  (default: PANEL_LEFT_X)
        right_x: Right edge x-coordinate (default: PANEL_RIGHT_X)

    Returns:
        Manim Rectangle, positioned and styled as a panel background.
    """
    width    = right_x - left_x
    height   = top_y - bottom_y
    center_x = (left_x + right_x) / 2
    center_y = (top_y + bottom_y) / 2

    rect = Rectangle(
        width=width,
        height=height,
        fill_color=COLORS.panel_bg,
        fill_opacity=0.8,
        stroke_color=COLORS.panel_border,
        stroke_width=2,
    )
    rect.move_to([center_x, center_y, 0])
    return rect


def get_metrics_rect() -> Rectangle:
    """
    Create the background rectangle for the full-width metrics strip.

    Returns:
        Manim Rectangle spanning the entire metrics strip.
    """
    return get_panel_rect(
        top_y=METRICS_TOP_Y,
        bottom_y=METRICS_BOTTOM_Y,
        left_x=METRICS_LEFT_X,
        right_x=METRICS_RIGHT_X,
    )


def get_classification_color(classification: str) -> str:
    """
    Return the hex color string for a move classification.

    Args:
        classification: e.g. "blunder", "best", "inaccuracy"

    Returns:
        Hex color string.
    """
    key = classification.lower().replace(" ", "_")
    color_map = {
        "brilliant":  COLORS.brilliant,
        "great":      COLORS.great,
        "best":       COLORS.best,
        "excellent":  COLORS.excellent,
        "good":       COLORS.good,
        "book":       COLORS.book,
        "inaccuracy": COLORS.inaccuracy,
        "mistake":    COLORS.mistake,
        "blunder":    COLORS.blunder,
        "missed_win": COLORS.missed_win,
        "miss":       COLORS.missed_win,
    }
    return color_map.get(key, COLORS.text_primary)


def format_player_display(name: str, elo: str = None) -> str:
    """
    Format a player name with optional ELO rating.

    Returns:
        e.g. "Caruana, F (2820)" or "Caruana, F"
    """
    if elo and elo not in ("", "?"):
        return f"{name} ({elo})"
    return name


# =============================================================================
# Layout Debugging
# =============================================================================

def create_layout_guides() -> VGroup:
    """
    Create visual guide lines showing every layout boundary.
    Add to a scene during development to verify geometry.

    Returns:
        VGroup of Lines and Text labels.
    """
    guides = VGroup()
    style = dict(stroke_color=COLORS.text_secondary,
                 stroke_width=1, stroke_opacity=0.5)

    # ── Upper zone / metrics strip boundary ──────────────────────────────────
    guides.add(Line(
        start=[FRAME_LEFT_X, UPPER_BOTTOM_Y, 0],
        end=[FRAME_RIGHT_X,  UPPER_BOTTOM_Y, 0],
        **style
    ))

    # ── Metrics strip top & bottom ────────────────────────────────────────────
    for y in (METRICS_TOP_Y, METRICS_BOTTOM_Y):
        guides.add(Line(
            start=[METRICS_LEFT_X, y, 0],
            end=[METRICS_RIGHT_X,  y, 0],
            **style
        ))

    # ── Vertical dividers between sub-plots ──────────────────────────────────
    for i in range(1, 4):
        x = _subplot_left(i) - METRICS_SUBPLOT_GAP / 2
        guides.add(Line(
            start=[x, METRICS_TOP_Y,    0],
            end=[x,   METRICS_BOTTOM_Y, 0],
            **style
        ))

    # ── Sub-plot labels ───────────────────────────────────────────────────────
    for label, cx in zip(
        METRICS_SUBPLOT_LABELS,
        [METRICS_EVAL_CENTER_X, METRICS_SPACE_CENTER_X,
         METRICS_MOBILITY_CENTER_X, METRICS_FTI_CENTER_X]
    ):
        t = Text(label, font=FONTS.body_font,
                 font_size=FONTS.metric_label_size,
                 color=COLORS.text_secondary)
        t.move_to([cx, METRICS_TOP_Y - 0.15, 0])
        guides.add(t)

    # ── Vertical divider between board area and right panels ─────────────────
    guides.add(Line(
        start=[PANEL_LEFT_X - 0.2, UPPER_TOP_Y,    0],
        end=[PANEL_LEFT_X - 0.2,   UPPER_BOTTOM_Y, 0],
        **style
    ))

    # ── Horizontal dividers between right panels ──────────────────────────────
    for y in (HEADER_BOTTOM_Y, MOVE_LIST_BOTTOM_Y):
        guides.add(Line(
            start=[PANEL_LEFT_X, y, 0],
            end=[PANEL_RIGHT_X,  y, 0],
            **style
        ))

    return guides


# =============================================================================
# Module Info  (python animator_layout.py)
# =============================================================================

if __name__ == "__main__":
    print("Chess Animator Layout Configuration")
    print("=" * 48)
    print(f"Frame:          {FRAME_WIDTH:.2f} × {FRAME_HEIGHT:.2f} Manim units")
    print()
    print("Upper zone:")
    print(f"  y = {UPPER_TOP_Y:.2f}  to  {UPPER_BOTTOM_Y:.2f}"
          f"  (height {UPPER_HEIGHT:.2f})")
    print(f"  Board centre:  ({BOARD_CENTER_X}, {BOARD_CENTER_Y:.2f})")
    print(f"  Board scale:   {BOARD_SCALE}")
    print()
    print("Right panels:")
    print(f"  Header:      y = {HEADER_TOP_Y:.2f}  to  {HEADER_BOTTOM_Y:.2f}")
    print(f"  Move list:   y = {MOVE_LIST_TOP_Y:.2f}  to  {MOVE_LIST_BOTTOM_Y:.2f}")
    print(f"  Commentary:  y = {COMMENTARY_TOP_Y:.2f}  to  {COMMENTARY_BOTTOM_Y:.2f}")
    print(f"  Panel width: {PANEL_WIDTH:.2f},  centre x: {PANEL_CENTER_X:.2f}")
    print()
    print("Metrics strip:")
    print(f"  y = {METRICS_TOP_Y:.2f}  to  {METRICS_BOTTOM_Y:.2f}"
          f"  (height {METRICS_HEIGHT:.2f})")
    print(f"  x = {METRICS_LEFT_X:.2f}  to  {METRICS_RIGHT_X:.2f}"
          f"  (width {METRICS_WIDTH:.2f})")
    print(f"  Sub-plot width: {METRICS_SUBPLOT_WIDTH:.2f}")
    print()
    for label, lx in zip(METRICS_SUBPLOT_LABELS, METRICS_SUBPLOT_LEFT_EDGES):
        cx = lx + METRICS_SUBPLOT_WIDTH / 2
        print(f"  {label:<10}  left_x={lx:.2f},  centre_x={cx:.2f}")

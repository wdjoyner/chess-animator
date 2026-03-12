"""
animator_game.py

Animates a chess game with move-by-move analysis and commentary.
Integrates with chess_game_analyzer6y.py for Stockfish annotations.

Usage:
    # Set config path via environment variable, then run manim:
    export CHESS_ANIMATOR_CONFIG=game_animator_config.json
    manim -pql animator_game.py AnimatedGame

    # Or use run_animator.py which handles config setup automatically.

    # Quick demo (no files needed):
    manim -pql animator_game.py QuickDemo

Requirements:
    - manim, manim-chess, chess
    - chess_game_analyzer6y.py (for analysis)
    - Stockfish (if running live analysis)

Changes from previous version:
    - MoveData extended with full positional fields (space, mobility,
      king_safety, threats, fti1/2/3) read from positional_eval sub-dict
    - AnalysisData.from_json_file() correctly unpacks nested positional_eval
    - AnimatedGame reads config from CHESS_ANIMATOR_CONFIG env var (JSON file)
      instead of the broken --user_args Manim CLI approach
    - Animation loop accepts optional MetricPlotPanel (imported from
      animator_metrics.py when available)
    - QuickDemo updated with positional fields for self-contained testing
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

from manim import *
import manim_chess
import chess
import chess.pgn

from animator_layout import (
    COLORS, FONTS, FRAME_WIDTH,
    BOARD_SCALE, BOARD_CENTER_X, BOARD_CENTER_Y,
    EVAL_BAR_SCALE, EVAL_BAR_OFFSET,
    PANEL_LEFT_X, PANEL_RIGHT_X, PANEL_CENTER_X, PANEL_WIDTH,
    HEADER_TOP_Y, HEADER_BOTTOM_Y, HEADER_CENTER_Y,
    MOVE_LIST_TOP_Y, MOVE_LIST_BOTTOM_Y, MOVE_LIST_CENTER_Y,
    COMMENTARY_TOP_Y, COMMENTARY_BOTTOM_Y, COMMENTARY_CENTER_Y,
    get_panel_rect, get_classification_color, format_player_display
)

from animator_initial_frame import GameInfo, create_header_panel


# =============================================================================
# Scaled Evaluation Bar  (±4 pawns fills the bar instead of ±10)
# =============================================================================

class ScaledEvaluationBar(manim_chess.EvaluationBar):
    """
    Subclass of manim_chess.EvaluationBar with a calibrated ±4 pawn scale.

    Desired behaviour:
        eval =  0  →  white fills exactly half the bar
        eval = +4  →  white fills the full bar
        eval = -4  →  white fills none of the bar
        eval = +1  →  white advances by roughly one board square

    All heights are derived at runtime from self.black_rectangle.height
    (world units, after any scale() call) so the formula stays correct
    regardless of EVAL_BAR_SCALE or any other transform applied externally.
    Using the hardcoded raw value 6.18 as the max was the bug: after
    scale(0.72) the bar is only 4.45 world units tall, so a rect_height
    of 3.09 filled 69% of the bar instead of the intended 50%.
    """

    _MAX_PAWNS = 4.0   # ±this many pawns fills the bar
    _BAR_MIN_FRAC = 0.008  # tiny floor fraction so rect never disappears

    def set_evaluation(self, evaluation: float):
        self.evaluation = evaluation

        # Use get_height()/get_width(): these always return actual world dimensions
        # after any scale() call (unlike .height/.width which store unscaled values).
        H = self.black_rectangle.get_height()   # e.g. 6.4 * 0.72 = 4.608
        half = H / 2
        slope = half / self._MAX_PAWNS          # world units per pawn

        height_from_evaluation = slope * self.evaluation + half
        rect_height = min(max(self._BAR_MIN_FRAC * H, height_from_evaluation), H)
        pos = self.black_rectangle.get_bottom() + np.array([0, rect_height / 2, 0])
        W = self.black_rectangle.get_width()    # actual world width after scale()
        new_rect = (
            Rectangle(width=W, height=rect_height,
                      stroke_color=self.WHITE, fill_opacity=1)
            .set_fill(self.WHITE)
            .move_to(pos)
        )

        text_offset = H * 0.045   # ~4.5% of bar height for text nudge
        text_val = f'{self.evaluation:.1f}'
        if self.evaluation > 0:
            new_text = (
                Text(text_val, font="Arial")
                .move_to(self.black_rectangle.get_bottom() + np.array([0, text_offset, 0]))
                .set_fill(self.BLACK)
                .scale(0.2)
            )
        else:
            new_text = (
                Text(text_val, font="Arial")
                .move_to(self.black_rectangle.get_top() + np.array([0, -text_offset, 0]))
                .set_fill(self.WHITE)
                .scale(0.2)
            )

        return [Transform(self.white_rectangle, new_rect),
                Transform(self.bot_text, new_text)]

# Import the comment parser
from convert_script_to_comment_dict import parse_comments_file

# Optionally import MetricPlotPanel — graceful fallback if not yet written
try:
    from animator_metrics import MetricPlotPanel
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# FTI weight constants (mirrors chess_game_analyzer6y.py)
THREATS_SCALE_FACTOR = 20.0
FTI1_WEIGHTS = (0.25, 0.25, 0.25, 0.25)   # Harmonious
FTI2_WEIGHTS = (0.60, 0.10, 0.00, 0.30)   # Tactical
FTI3_WEIGHTS = (0.70, 0.10, 0.10, 0.10)   # Strategic


# =============================================================================
# FTI Computation
# =============================================================================

def compute_fti(space_adv: float, mob_adv: float, ks_adv: float,
                thr_adv: float, weights: Tuple[float, float, float, float]) -> float:
    """
    Compute the Fireteam Index from net positional advantages.

    Args:
        space_adv:  space_white   - space_black
        mob_adv:    mobility_white - mobility_black
        ks_adv:     king_safety_white - king_safety_black
        thr_adv:    (threats_white - threats_black) / THREATS_SCALE_FACTOR  (pre-scaled)
        weights:    (w_space, w_mobility, w_king_safety, w_threats)

    Returns:
        FTI score (positive = White advantage)
    """
    ws, wm, wk, wt = weights
    return ws * space_adv + wm * mob_adv + wk * ks_adv + wt * thr_adv


# =============================================================================
# Analysis Data Loading
# =============================================================================

@dataclass
class MoveData:
    """
    Simplified move data for animation.

    Positional fields (space, mobility, king_safety, threats) are per-side
    values read directly from the positional_eval sub-dict in the JSON.
    FTI values are computed at load time.
    """
    # Core move info
    ply: int
    move_san: str
    move_uci: str
    is_white_move: bool
    eval_before: float
    eval_after: float
    eval_loss: float
    classification: str
    best_move_san: str
    is_capture: bool
    is_check: bool
    pv_line: List[str]

    # Positional metrics (from positional_eval sub-dict)
    space_white: float = 0.0
    space_black: float = 0.0
    mobility_white: float = 0.0
    mobility_black: float = 0.0
    king_safety_white: float = 0.0
    king_safety_black: float = 0.0
    threats_white: float = 0.0
    threats_black: float = 0.0

    # Fireteam Index variants (computed at load time)
    fti1: float = 0.0   # Harmonious
    fti2: float = 0.0   # Tactical
    fti3: float = 0.0   # Strategic

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoveData":
        """
        Build a MoveData from a JSON-decoded dict.

        Handles two JSON shapes:
        1. Produced by chess_game_analyzer6y directly (nested positional_eval
           sub-dict with raw field names like space_white_mg).
        2. Produced by AnalysisData.save_to_json() via asdict() on MoveData
           (flat fields: space_white, mobility_white, etc. at top level).
        """
        # --- Try nested positional_eval sub-dict first (analyzer JSON) -------
        pe = d.get("positional_eval") or {}

        if pe:
            space_white    = float(pe.get("space_white_mg", 0.0))
            space_black    = float(pe.get("space_black_mg", 0.0))
            mob_w_mg       = float(pe.get("mobility_white_mg", 0.0))
            mob_w_eg       = float(pe.get("mobility_white_eg", 0.0))
            mob_b_mg       = float(pe.get("mobility_black_mg", 0.0))
            mob_b_eg       = float(pe.get("mobility_black_eg", 0.0))
            mobility_white = (mob_w_mg + mob_w_eg) / 2.0
            mobility_black = (mob_b_mg + mob_b_eg) / 2.0
            ks_white       = float(pe.get("king_safety_white_mg", 0.0))
            ks_black       = float(pe.get("king_safety_black_mg", 0.0))
            thr_white      = float(pe.get("threats_white_mg", 0.0))
            thr_black      = float(pe.get("threats_black_mg", 0.0))
        else:
            # --- Flat fields (saved by AnalysisData.save_to_json) ------------
            space_white    = float(d.get("space_white",    0.0))
            space_black    = float(d.get("space_black",    0.0))
            mobility_white = float(d.get("mobility_white", 0.0))
            mobility_black = float(d.get("mobility_black", 0.0))
            ks_white       = float(d.get("king_safety_white", 0.0))
            ks_black       = float(d.get("king_safety_black", 0.0))
            thr_white      = float(d.get("threats_white",  0.0))
            thr_black      = float(d.get("threats_black",  0.0))

        # --- Compute net advantages for FTI ----------------------------------
        space_adv = space_white - space_black
        mob_adv   = mobility_white - mobility_black
        ks_adv    = ks_white - ks_black
        thr_adv   = (thr_white - thr_black) / THREATS_SCALE_FACTOR

        fti1 = compute_fti(space_adv, mob_adv, ks_adv, thr_adv, FTI1_WEIGHTS)
        fti2 = compute_fti(space_adv, mob_adv, ks_adv, thr_adv, FTI2_WEIGHTS)
        fti3 = compute_fti(space_adv, mob_adv, ks_adv, thr_adv, FTI3_WEIGHTS)

        return cls(
            ply=d.get("ply", 0),
            move_san=d.get("move_san", ""),
            move_uci=d.get("move_uci", ""),
            is_white_move=d.get("is_white_move", True),
            eval_before=float(d.get("eval_before", 0.0)),
            eval_after=float(d.get("eval_after", 0.0)),
            eval_loss=float(d.get("eval_loss", 0.0)),
            classification=d.get("classification", ""),
            best_move_san=d.get("best_move_san", ""),
            is_capture=d.get("is_capture", False),
            is_check=d.get("is_check", False),
            pv_line=d.get("pv_line", []),
            space_white=space_white,
            space_black=space_black,
            mobility_white=mobility_white,
            mobility_black=mobility_black,
            king_safety_white=ks_white,
            king_safety_black=ks_black,
            threats_white=thr_white,
            threats_black=thr_black,
            fti1=fti1,
            fti2=fti2,
            fti3=fti3,
        )


@dataclass
class AnalysisData:
    """Container for game analysis data."""
    game_info: GameInfo
    moves: List[MoveData]
    white_accuracy: float
    black_accuracy: float

    @classmethod
    def from_json_file(cls, json_path: Path) -> "AnalysisData":
        """
        Load analysis from a JSON file produced by chess_game_analyzer6y.py.

        The JSON contains a 'moves' list where each element has a nested
        'positional_eval' dict — MoveData.from_dict() handles that unpacking.
        """
        with open(json_path) as f:
            data = json.load(f)

        game_info = GameInfo(
            white=data.get("white", "White"),
            black=data.get("black", "Black"),
            white_elo=str(data.get("white_elo", "") or ""),
            black_elo=str(data.get("black_elo", "") or ""),
            event=data.get("event", ""),
            site=data.get("site", ""),
            date=data.get("date", ""),
            round=data.get("round_num", ""),
            result=data.get("result", "*"),
            opening=data.get("opening_name", ""),
            eco=data.get("opening_eco", ""),
        )

        moves = [MoveData.from_dict(m) for m in data.get("moves", [])]

        white_stats = data.get("white_stats", {})
        black_stats = data.get("black_stats", {})

        return cls(
            game_info=game_info,
            moves=moves,
            white_accuracy=float(white_stats.get("accuracy", 0.0)),
            black_accuracy=float(black_stats.get("accuracy", 0.0)),
        )

    @classmethod
    def from_analyzer(cls, pgn_path: Path,
                      stockfish_path: str = "/usr/local/bin/stockfish",
                      depth: int = 20) -> "AnalysisData":
        """
        Run live analysis using chess_game_analyzer6y.py.
        Prefer pre-computed JSON (from_json_file) for iteration speed.
        """
        try:
            from chess_game_analyzer6y import EnhancedGameAnalyzer
        except ImportError:
            raise ImportError("chess_game_analyzer6y.py must be in the Python path")

        with EnhancedGameAnalyzer(stockfish_path, depth) as analyzer:
            result = analyzer.analyze_game(str(pgn_path))

        game_info = GameInfo(
            white=result.white,
            black=result.black,
            white_elo=str(result.white_elo) if result.white_elo else "",
            black_elo=str(result.black_elo) if result.black_elo else "",
            event=result.event,
            site=result.site,
            date=result.date,
            round=result.round_num,
            result=result.result,
            opening=result.opening_name,
            eco=result.opening_eco,
        )

        # Build MoveData objects directly from EnhancedMoveAnalysis
        moves = []
        for m in result.moves:
            pe = m.positional_eval

            space_white     = pe.space_white      if pe else 0.0
            space_black     = pe.space_black      if pe else 0.0
            mob_white       = pe.mobility_white   if pe else 0.0
            mob_black       = pe.mobility_black   if pe else 0.0
            ks_white        = pe.king_safety_white if pe else 0.0
            ks_black        = pe.king_safety_black if pe else 0.0
            thr_white       = pe.threats_white    if pe else 0.0
            thr_black       = pe.threats_black    if pe else 0.0

            space_adv = space_white - space_black
            mob_adv   = mob_white - mob_black
            ks_adv    = ks_white - ks_black
            thr_adv   = (thr_white - thr_black) / THREATS_SCALE_FACTOR

            moves.append(MoveData(
                ply=m.ply,
                move_san=m.move_san,
                move_uci=m.move_uci,
                is_white_move=m.is_white_move,
                eval_before=m.eval_before,
                eval_after=m.eval_after,
                eval_loss=m.eval_loss,
                classification=m.classification,
                best_move_san=m.best_move_san,
                is_capture=m.is_capture,
                is_check=m.is_check,
                pv_line=m.pv_line,
                space_white=space_white,
                space_black=space_black,
                mobility_white=mob_white,
                mobility_black=mob_black,
                king_safety_white=ks_white,
                king_safety_black=ks_black,
                threats_white=thr_white,
                threats_black=thr_black,
                fti1=compute_fti(space_adv, mob_adv, ks_adv, thr_adv, FTI1_WEIGHTS),
                fti2=compute_fti(space_adv, mob_adv, ks_adv, thr_adv, FTI2_WEIGHTS),
                fti3=compute_fti(space_adv, mob_adv, ks_adv, thr_adv, FTI3_WEIGHTS),
            ))

        return cls(
            game_info=game_info,
            moves=moves,
            white_accuracy=result.white_stats.get("accuracy", 0.0),
            black_accuracy=result.black_stats.get("accuracy", 0.0),
        )

    def save_to_json(self, output_path: Path):
        """Save analysis to JSON for reuse."""
        data = {
            "white": self.game_info.white,
            "black": self.game_info.black,
            "white_elo": self.game_info.white_elo,
            "black_elo": self.game_info.black_elo,
            "event": self.game_info.event,
            "site": self.game_info.site,
            "date": self.game_info.date,
            "round_num": self.game_info.round,
            "result": self.game_info.result,
            "opening_name": self.game_info.opening,
            "opening_eco": self.game_info.eco,
            "moves": [asdict(m) for m in self.moves],
            "white_stats": {"accuracy": self.white_accuracy},
            "black_stats": {"accuracy": self.black_accuracy},
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Dynamic Panel Components
# =============================================================================

class MoveListPanel:
    """
    Manages the move list display panel.
    Shows moves in standard paired notation (1. e4 e5) with scrolling.
    Blunders and mistakes get their own line for emphasis.
    """

    def __init__(self, max_visible_lines: int = 8):
        self.max_visible_lines = max_visible_lines
        self.moves: List[MoveData] = []
        self.lines: List[str] = []
        self.line_colors: List[str] = []
        self.panel_group = VGroup()
        self.moves_group = VGroup()
        self._create_panel()

    def _create_panel(self):
        bg = get_panel_rect(MOVE_LIST_TOP_Y, MOVE_LIST_BOTTOM_Y)
        self.panel_group.add(bg)

        title = Text(
            "Moves",
            font=FONTS.heading_font,
            font_size=FONTS.subtitle_size,
            color=COLORS.text_secondary
        )
        title.move_to([PANEL_CENTER_X, MOVE_LIST_TOP_Y - 0.25, 0])
        self.panel_group.add(title)
        self.panel_group.add(self.moves_group)

    def get_mobject(self) -> VGroup:
        return self.panel_group

    def _needs_separate_line(self, move: MoveData) -> bool:
        return move.classification in ("blunder", "mistake", "brilliant")

    def _format_move_text(self, move: MoveData) -> str:
        text = move.move_san
        symbols = {
            "blunder":    "??",
            "mistake":    "?",
            "inaccuracy": "?!",
            "brilliant":  "!!",
            "great":      "!",
        }
        return text + symbols.get(move.classification, "")

    def _rebuild_lines(self):
        self.lines = []
        self.line_colors = []

        i = 0
        while i < len(self.moves):
            move = self.moves[i]
            move_num = (move.ply + 1) // 2

            if move.is_white_move:
                white_text = self._format_move_text(move)
                line = f"{move_num}. {white_text}"
                color = get_classification_color(move.classification)

                if self._needs_separate_line(move):
                    self.lines.append(line)
                    self.line_colors.append(color)
                    i += 1
                    continue

                # Try to pair with black's reply on same line
                if i + 1 < len(self.moves) and not self.moves[i + 1].is_white_move:
                    black_move = self.moves[i + 1]
                    black_text = self._format_move_text(black_move)

                    if self._needs_separate_line(black_move):
                        self.lines.append(line)
                        self.line_colors.append(color)
                        self.lines.append(f"{move_num}... {black_text}")
                        self.line_colors.append(
                            get_classification_color(black_move.classification))
                        i += 2
                        continue
                    else:
                        line += f" {black_text}"
                        if black_move.classification in ("blunder", "mistake", "inaccuracy"):
                            if move.classification not in ("blunder", "mistake"):
                                color = get_classification_color(black_move.classification)
                        i += 2
                        self.lines.append(line)
                        self.line_colors.append(color)
                        continue

                self.lines.append(line)
                self.line_colors.append(color)
                i += 1

            else:
                black_text = self._format_move_text(move)
                self.lines.append(f"{move_num}... {black_text}")
                self.line_colors.append(get_classification_color(move.classification))
                i += 1

    def _render_lines(self) -> VGroup:
        """
        Build and return a fresh VGroup of Text objects for the currently
        visible lines.  Does NOT mutate self.moves_group — the caller
        decides what to do with the old and new groups.
        """
        visible_lines  = self.lines[-self.max_visible_lines:]
        visible_colors = self.line_colors[-self.max_visible_lines:]

        # Derive geometry from actual panel boundaries so lines never overflow.
        title_height  = 0.5   # room for the "Moves" title at the top
        bottom_margin = 0.15  # breathing room above the panel bottom edge
        usable_height = (MOVE_LIST_TOP_Y - MOVE_LIST_BOTTOM_Y
                         - title_height - bottom_margin)
        n = max(1, self.max_visible_lines)
        line_height = usable_height / n

        content_top = MOVE_LIST_TOP_Y - title_height

        group = VGroup()
        for i, (line, color) in enumerate(zip(visible_lines, visible_colors)):
            if len(line) > 25:
                line = line[:22] + "..."
            t = Text(
                line,
                font=FONTS.mono_font,
                font_size=FONTS.move_size,
                color=color,
            )
            y = content_top - (i + 0.5) * line_height
            t.move_to([PANEL_CENTER_X, y, 0])
            group.add(t)
        return group

    def add_move(self, move: MoveData) -> Animation:
        """Append a move and return the Manim animation for the panel update."""
        self.moves.append(move)
        self._rebuild_lines()

        new_group = self._render_lines()

        # Replace the old content in moves_group with the new content.
        # We do this by building the animation against the old state, then
        # swapping the children so the scene always holds the live objects.
        old_children = list(self.moves_group.submobjects)
        old_group    = VGroup(*[c.copy() for c in old_children])

        # Swap children into moves_group so the scene tracks the new objects
        self.moves_group.remove(*old_children)
        for obj in new_group.submobjects:
            self.moves_group.add(obj)

        if not old_children:
            return FadeIn(self.moves_group)
        return FadeTransform(old_group, self.moves_group)


class CommentaryPanel:
    """
    Manages the commentary display panel.
    Prioritizes human-written comments from a dictionary over engine analysis.
    """

    def __init__(self, custom_comments: dict = None):
        self.panel_group = VGroup()
        self.content_group = VGroup()
        self.custom_comments = custom_comments or {}
        self._create_panel()

    def _create_panel(self):
        bg = get_panel_rect(COMMENTARY_TOP_Y, COMMENTARY_BOTTOM_Y)
        self.panel_group.add(bg)

        title = Text(
            "Analysis & Commentary",
            font=FONTS.heading_font,
            font_size=FONTS.subtitle_size,
            color=COLORS.text_secondary
        )
        title.move_to([PANEL_CENTER_X, COMMENTARY_TOP_Y - 0.25, 0])
        self.panel_group.add(title)
        self.panel_group.add(self.content_group)

    def get_mobject(self) -> VGroup:
        return self.panel_group

    def _generate_commentary(self, move: MoveData) -> List[str]:
        """Return list of text lines for this move's commentary."""
        ply_key = str(move.ply)

        if ply_key in self.custom_comments:
            # Word-wrap the human comment to fit the panel
            words = self.custom_comments[ply_key].split()
            lines, current = [], ""
            for word in words:
                if len(current) + len(word) < 30:
                    current += word + " "
                else:
                    lines.append(current.strip())
                    current = word + " "
            if current.strip():
                lines.append(current.strip())
            return lines

        # Fallback: engine annotation
        lines = []
        if move.classification:
            loss_text = (f" ({move.eval_loss:.0f}cp lost)"
                         if move.eval_loss > 10 else "")
            lines.append(f"{move.classification.capitalize()}{loss_text}")

        eval_val = move.eval_after / 100.0
        if abs(eval_val) > 10:
            lines.append("White is winning" if eval_val > 0 else "Black is winning")
        else:
            lines.append(f"Eval: {eval_val:+.2f}")

        if move.best_move_san and move.classification in ("blunder", "mistake"):
            lines.append(f"Best: {move.best_move_san}")

        return lines

    def update_commentary(self, move: MoveData) -> Animation:
        """Update the commentary panel and return the transition animation."""
        lines = self._generate_commentary(move)

        # Derive geometry from actual panel boundaries
        title_height  = 0.5
        bottom_margin = 0.15
        usable_height = (COMMENTARY_TOP_Y - COMMENTARY_BOTTOM_Y
                         - title_height - bottom_margin)
        max_lines  = 5
        line_height = usable_height / max_lines
        content_top = COMMENTARY_TOP_Y - title_height

        new_texts = VGroup()
        ply_key = str(move.ply)
        for i, line in enumerate(lines[:max_lines]):
            color = COLORS.text_primary
            if ply_key not in self.custom_comments and i == 0:
                color = get_classification_color(move.classification)
            t = Text(line, font=FONTS.body_font,
                     font_size=FONTS.commentary_size, color=color)
            y = content_top - (i + 0.5) * line_height
            t.move_to([PANEL_CENTER_X, y, 0])
            new_texts.add(t)

        # Grab old children for the animation, then swap
        old_children = list(self.content_group.submobjects)
        old_group    = VGroup(*[c.copy() for c in old_children])

        self.content_group.remove(*old_children)
        for obj in new_texts.submobjects:
            self.content_group.add(obj)

        if not old_children:
            return FadeIn(self.content_group)
        return FadeTransform(old_group, self.content_group)


# =============================================================================
# Configuration Loading
# =============================================================================

def _load_animator_config() -> Dict[str, str]:
    """
    Load animator config from the JSON file whose path is in the
    CHESS_ANIMATOR_CONFIG environment variable.

    Returns a dict with optional keys:
        pgn_path, analysis_path, comments_path
    """
    config_path = os.environ.get("CHESS_ANIMATOR_CONFIG", "")
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


# =============================================================================
# Main Animated Scene
# =============================================================================

class AnimatedGame(Scene):
    """
    Main scene that animates a complete chess game with
    engine analysis and optional human commentary.

    Config is read from the JSON file at CHESS_ANIMATOR_CONFIG env var.
    See run_animator.py for how to set this up.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cfg = _load_animator_config()
        self.pgn_path      = cfg.get("pgn_path")
        self.analysis_path = cfg.get("analysis_path")
        self.comments_path = cfg.get("comments_path")
        self.custom_comments: Dict[str, str] = {}

    def _load_analysis(self) -> AnalysisData:
        """
        Load game analysis.  Priority order:
        1. Pre-computed JSON  (analysis_path or <game>_analysis.json)
        2. Live Stockfish run (pgn_path)
        3. Hard error
        """
        # Try analysis JSON first
        candidates = []
        if self.analysis_path:
            candidates.append(Path(self.analysis_path))
        if self.pgn_path:
            candidates.append(Path(self.pgn_path).with_suffix("").parent /
                               (Path(self.pgn_path).stem + "_analysis.json"))
        for p in ("game_analysis.json", "sample_game_analysis.json"):
            candidates.append(Path(p))

        for path in candidates:
            if path.exists():
                print(f"Loading analysis from {path}")
                return AnalysisData.from_json_file(path)

        # Fall back to live analysis
        pgn_candidates = []
        if self.pgn_path:
            pgn_candidates.append(Path(self.pgn_path))
        for p in ("game.pgn", "sample_game.pgn"):
            pgn_candidates.append(Path(p))

        for path in pgn_candidates:
            if path.exists():
                print(f"Running live Stockfish analysis on {path}…")
                return AnalysisData.from_analyzer(path)

        raise FileNotFoundError(
            "No analysis JSON or PGN found. "
            "Set CHESS_ANIMATOR_CONFIG or place game.pgn in the working directory."
        )

    def _load_custom_comments(self):
        """Load [KEY]-based text file into self.custom_comments."""
        txt_path = self.comments_path
        if not txt_path and self.pgn_path:
            candidate = Path(self.pgn_path).stem + "_notes.txt"
            if Path(candidate).exists():
                txt_path = candidate

        if txt_path and Path(txt_path).exists():
            print(f"Loading commentary from {txt_path}")
            self.custom_comments = parse_comments_file(txt_path)
        else:
            print("No commentary file found — using engine-only mode.")

    def construct(self):
        # ── 1. Initialise ────────────────────────────────────────────────────
        self.camera.background_color = COLORS.background
        analysis = self._load_analysis()
        self._load_custom_comments()

        # ── 2. Board & eval bar ──────────────────────────────────────────────
        board = manim_chess.Board()
        board.set_board_from_FEN()
        board.scale(BOARD_SCALE)
        board.move_to([BOARD_CENTER_X, BOARD_CENTER_Y, 0])

        eval_bar = ScaledEvaluationBar()
        eval_bar.scale(EVAL_BAR_SCALE)
        eval_bar.next_to(board, LEFT, buff=EVAL_BAR_OFFSET)

        # ── 3. Side panels ───────────────────────────────────────────────────
        header_panel = create_header_panel(analysis.game_info)
        move_list    = MoveListPanel()
        commentary   = CommentaryPanel(custom_comments=self.custom_comments)

        # ── 4. Optional metric panel ─────────────────────────────────────────
        metric_panel = None
        if METRICS_AVAILABLE:
            metric_panel = MetricPlotPanel(analysis.moves)

        # ── 5. Intro card ────────────────────────────────────────────────────
        if "intro" in self.custom_comments:
            intro_text = Text(
                self.custom_comments["intro"],
                font=FONTS.body_font,
                font_size=FONTS.commentary_size + 2,
            ).set_width(FRAME_WIDTH - 4)
            self.play(Write(intro_text))
            self.wait(2)
            self.play(FadeOut(intro_text))

        # Add all persistent objects
        objects_to_add = [board, eval_bar, header_panel,
                          move_list.get_mobject(), commentary.get_mobject()]
        if metric_panel is not None:
            objects_to_add.append(metric_panel.get_mobject())
        self.add(*objects_to_add)

        # ── 6. Animation loop ────────────────────────────────────────────────
        for idx, move in enumerate(analysis.moves):
            uci = move.move_uci
            from_sq, to_sq = uci[:2], uci[2:4]

            # Move the piece on the board
            manim_chess.play_game(
                scene=self,
                board=board,
                moves=[(from_sq, to_sq, "")]
            )

            # Clamp eval to ±10 pawns for the bar
            eval_pawns = max(-4.0, min(4.0, move.eval_after / 100.0))

            # Build simultaneous panel animations
            panel_anims = [
                eval_bar.set_evaluation(eval_pawns),
                move_list.add_move(move),
                commentary.update_commentary(move),
            ]
            if metric_panel is not None:
                panel_anims.append(metric_panel.advance_to_move(idx))

            self.play(*panel_anims, run_time=0.4)
            self.wait(0.2)

        # ── 7. Conclusion ────────────────────────────────────────────────────
        self.wait(1)

        final_result = self.custom_comments.get("result", analysis.game_info.result)
        result_text = Text(
            final_result,
            font=FONTS.heading_font,
            font_size=72
        ).move_to(board)
        dimmer = BackgroundRectangle(board, fill_opacity=0.8)
        self.play(FadeIn(dimmer), Write(result_text))

        if "conclusion" in self.custom_comments:
            conc_text = (
                Text(self.custom_comments["conclusion"], font_size=20)
                .set_width(PANEL_WIDTH - 0.5)
                .move_to(commentary.content_group)
            )
            self.play(ReplacementTransform(commentary.content_group, conc_text))

        self.wait(3)


# =============================================================================
# Quick Demo Scene (self-contained, no external files needed)
# =============================================================================

class QuickDemo(Scene):
    """
    Quick demo using hard-coded moves.
    Tests the full animation pipeline — board, eval bar, move list,
    commentary — without requiring any external files.

    MoveData objects now include zeroed positional fields so the
    dataclass constructor is satisfied; update with real values to
    test MetricPlotPanel rendering.
    """

    def construct(self):
        self.camera.background_color = COLORS.background

        board = manim_chess.Board()
        board.set_board_from_FEN()
        board.scale(BOARD_SCALE)
        board.move_to([BOARD_CENTER_X, BOARD_CENTER_Y, 0])

        eval_bar = ScaledEvaluationBar()
        eval_bar.scale(EVAL_BAR_SCALE)
        eval_bar.next_to(board, LEFT, buff=EVAL_BAR_OFFSET)

        title = Text(
            font=FONTS.heading_font,
            font_size=FONTS.title_size,
            color=COLORS.text_primary
        ).move_to([PANEL_CENTER_X, HEADER_CENTER_Y, 0])

        move_list  = MoveListPanel()
        commentary = CommentaryPanel()

        self.add(board, eval_bar, title,
                 move_list.get_mobject(), commentary.get_mobject())
        self.wait(1)

        # Minimal positional fields — all zero for demo purposes.
        # (ply, san, uci, is_white, ev_before, ev_after, ev_loss,
        #  classif, best_san, is_capture, is_check, pv_line)
        _z = dict(space_white=0, space_black=0,
                  mobility_white=0, mobility_black=0,
                  king_safety_white=0, king_safety_black=0,
                  threats_white=0, threats_black=0,
                  fti1=0, fti2=0, fti3=0)

        demo_moves = [
            MoveData(1,  "e4",    "e2e4", True,   0,   30,   0, "book",    "e4",  False, False, [], **_z),
            MoveData(2,  "e5",    "e7e5", False,  30,   25,   5, "book",    "e5",  False, False, [], **_z),
            MoveData(3,  "Nf3",   "g1f3", True,   25,   35,   0, "best",    "Nf3", False, False, [], **_z),
            MoveData(4,  "Nc6",   "b8c6", False,  35,   30,   5, "good",    "Nc6", False, False, [], **_z),
            MoveData(5,  "Bb5",   "f1b5", True,   30,   40,   0, "best",    "Bb5", False, False, [], **_z),
            MoveData(6,  "a6",    "a7a6", False,  40,   35,   5, "good",    "a6",  False, False, [], **_z),
            MoveData(7,  "Ba4",   "b5a4", True,   35,   40,   0, "good",    "Ba4", False, False, [], **_z),
            MoveData(8,  "Nf6",   "g8f6", False,  40,   35,   5, "best",    "Nf6", False, False, [], **_z),
            MoveData(9,  "O-O",   "e1g1", True,   35,   40,   0, "best",    "O-O", False, False, [], **_z),
            MoveData(10, "Be7",   "f8e7", False,  40,   35,   5, "good",    "Be7", False, False, [], **_z),
            MoveData(11, "Re1",   "f1e1", True,   35,   45,   0, "good",    "Re1", False, False, [], **_z),
            MoveData(12, "b5",    "b7b5", False,  45,   40,   5, "good",    "b5",  False, False, [], **_z),
            MoveData(13, "Bb3",   "a4b3", True,   40,   50,   0, "good",    "Bb3", False, False, [], **_z),
            MoveData(14, "d6",    "d7d6", False,  50,   45,   5, "good",    "d6",  False, False, [], **_z),
            MoveData(15, "c3",    "c2c3", True,   45,   55,   0, "good",    "c3",  False, False, [], **_z),
            MoveData(16, "O-O",   "e8g8", False,  55,   50,   5, "good",    "O-O", False, False, [], **_z),
            MoveData(17, "h3",    "h2h3", True,   50,   60,   0, "good",    "h3",  False, False, [], **_z),
            MoveData(18, "Na5??", "c6a5", False,  60,  180, 140, "blunder", "Nb8", False, False, [], **_z),
        ]

        for move in demo_moves:
            uci = move.move_uci
            from_sq, to_sq = uci[:2], uci[2:4]

            manim_chess.play_game(scene=self, board=board,
                                  moves=[(from_sq, to_sq, "")])

            eval_pawns = max(-4.0, min(4.0, move.eval_after / 100.0))
            self.play(eval_bar.set_evaluation(eval_pawns), run_time=0.3)
            self.play(move_list.add_move(move), run_time=0.3)
            self.play(commentary.update_commentary(move), run_time=0.3)
            self.wait(0.3)

        self.wait(2)


# =============================================================================
# Utility: Generate Analysis JSON
# =============================================================================

def generate_analysis_json(pgn_path: str, output_path: str = None,
                           stockfish_path: str = "/usr/local/bin/stockfish",
                           depth: int = 20):
    """
    Generate analysis JSON from a PGN using chess_game_analyzer6y.

    Run this once; then use the JSON with AnimatedGame for fast iteration.

    Usage:
        python animator_game.py --analyze game.pgn
        python animator_game.py --analyze game.pgn --depth 22
    """
    if output_path is None:
        output_path = Path(pgn_path).stem + "_analysis.json"

    print(f"Analyzing {pgn_path} with depth {depth}…")
    analysis = AnalysisData.from_analyzer(Path(pgn_path), stockfish_path, depth)
    analysis.save_to_json(Path(output_path))
    print(f"Analysis saved to {output_path}")
    return output_path


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        if len(sys.argv) < 3:
            print("Usage: python animator_game.py --analyze game.pgn [--depth 20]")
            sys.exit(1)

        pgn_path = sys.argv[2]
        depth = 20
        if "--depth" in sys.argv:
            depth = int(sys.argv[sys.argv.index("--depth") + 1])

        generate_analysis_json(pgn_path, depth=depth)

    else:
        print("Chess Game Animator")
        print("=" * 50)
        print()
        print("Scenes available:")
        print("  AnimatedGame  — full game animation")
        print("                  (needs CHESS_ANIMATOR_CONFIG env var)")
        print("  QuickDemo     — self-contained demo, no files needed")
        print()
        print("Run via Manim:")
        print("  manim -pql animator_game.py QuickDemo")
        print("  CHESS_ANIMATOR_CONFIG=my_config.json manim -pql animator_game.py AnimatedGame")
        print()
        print("Generate analysis JSON from a PGN:")
        print("  python animator_game.py --analyze game.pgn --depth 22")
        print()
        print("Config JSON format:")
        print('  {"pgn_path": "game.pgn",')
        print('   "analysis_path": "game_analysis.json",')
        print('   "comments_path": "game_notes.txt"}')

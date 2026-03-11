"""
animator_initial_frame.py

Generates the initial frame of a chess game video.
Displays the starting position with game metadata (players, event, date, opening).

Usage:
    manim -pql animator_initial_frame.py InitialFrame
    manim -pqh animator_initial_frame.py InitialFrame  # High quality
    
    # With a specific PGN file:
    manim -pql animator_initial_frame.py InitialFrame --pgn_file path/to/game.pgn
"""

import chess.pgn
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from manim import *
import manim_chess

from animator_layout import (
    COLORS, FONTS,
    BOARD_SCALE, BOARD_CENTER_X, BOARD_CENTER_Y,
    EVAL_BAR_SCALE, EVAL_BAR_OFFSET,
    PANEL_LEFT_X, PANEL_RIGHT_X, PANEL_CENTER_X, PANEL_WIDTH,
    HEADER_TOP_Y, HEADER_BOTTOM_Y, HEADER_CENTER_Y,
    MOVE_LIST_TOP_Y, MOVE_LIST_BOTTOM_Y, MOVE_LIST_CENTER_Y,
    COMMENTARY_TOP_Y, COMMENTARY_BOTTOM_Y, COMMENTARY_CENTER_Y,
    get_panel_rect, get_metrics_rect, format_player_display
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GameInfo:
    """Parsed game metadata from PGN headers."""
    white: str = "White"
    black: str = "Black"
    white_elo: str = ""
    black_elo: str = ""
    event: str = ""
    site: str = ""
    date: str = ""
    round: str = ""
    result: str = "*"
    opening: str = ""
    eco: str = ""
    
    @classmethod
    def from_pgn(cls, pgn_path: Path) -> "GameInfo":
        """
        Parse game info from a PGN file.
        
        Args:
            pgn_path: Path to the PGN file
            
        Returns:
            GameInfo with parsed metadata
        """
        with open(pgn_path) as f:
            game = chess.pgn.read_game(f)
        
        if game is None:
            return cls()
        
        headers = game.headers
        return cls(
            white=headers.get("White", "White"),
            black=headers.get("Black", "Black"),
            white_elo=headers.get("WhiteElo", ""),
            black_elo=headers.get("BlackElo", ""),
            event=headers.get("Event", ""),
            site=headers.get("Site", ""),
            date=headers.get("Date", ""),
            round=headers.get("Round", ""),
            result=headers.get("Result", "*"),
            opening=headers.get("Opening", ""),
            eco=headers.get("ECO", ""),
        )
    
    @classmethod  
    def from_pgn_string(cls, pgn_string: str) -> "GameInfo":
        """
        Parse game info from a PGN string.
        
        Args:
            pgn_string: PGN content as a string
            
        Returns:
            GameInfo with parsed metadata
        """
        import io
        pgn_io = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)
        
        if game is None:
            return cls()
        
        headers = game.headers
        return cls(
            white=headers.get("White", "White"),
            black=headers.get("Black", "Black"),
            white_elo=headers.get("WhiteElo", ""),
            black_elo=headers.get("BlackElo", ""),
            event=headers.get("Event", ""),
            site=headers.get("Site", ""),
            date=headers.get("Date", ""),
            round=headers.get("Round", ""),
            result=headers.get("Result", "*"),
            opening=headers.get("Opening", ""),
            eco=headers.get("ECO", ""),
        )
    
    def get_opening_display(self) -> str:
        """Get formatted opening name with ECO code if available."""
        if self.opening and self.eco:
            return f"{self.eco}: {self.opening}"
        elif self.opening:
            return self.opening
        elif self.eco:
            return self.eco
        return ""
    
    def get_event_display(self) -> str:
        """Get formatted event with round if available."""
        if self.event and self.round and self.round != "?":
            return f"{self.event}, Round {self.round}"
        return self.event


# =============================================================================
# Panel Creation Functions
# =============================================================================

def create_header_panel(game_info: GameInfo) -> VGroup:
    """
    Create the header panel showing game metadata.

    All content lives in a single VGroup arranged with a tight uniform
    buff so it fits compactly inside the (relatively short) header panel.

    Layout:
        ┌─────────────────────────────────┐
        │  ♔ White Player Name (ELO)      │
        │  vs  ♚ Black Player Name (ELO)  │
        │  Event Name, Round X  ·  Date   │
        │  ECO: Opening Name              │
        └─────────────────────────────────┘

    Tuning knobs (all in animator_layout.Typography):
        header_player_size  — player name font size  (default 12)
        header_vs_size      — "vs" separator size    (default 10)
        header_info_size    — event/date/opening size (default 10)
    """
    panel = VGroup()
    bg = get_panel_rect(HEADER_TOP_Y, HEADER_BOTTOM_Y)
    panel.add(bg)

    content = VGroup()

    # ── Player names ──────────────────────────────────────────────────────────
    white_display = format_player_display(game_info.white, game_info.white_elo)
    content.add(Text(
        f"♔  {white_display}",
        font=FONTS.heading_font,
        font_size=FONTS.header_player_size,
        color=COLORS.white_player,
    ))

    black_display = format_player_display(game_info.black, game_info.black_elo)
    content.add(Text(
        f"vs  ♚  {black_display}",
        font=FONTS.body_font,
        font_size=FONTS.header_vs_size,
        color=COLORS.text_secondary,
    ))

    # ── Event · Date on one line (saves vertical space) ──────────────────────
    event_display = game_info.get_event_display()
    date_display  = (game_info.date
                     if game_info.date and game_info.date != "????.??.??"
                     else "")
    event_date = "  ·  ".join(filter(None, [event_display, date_display]))
    if event_date:
        # Truncate if combined string is too wide
        if len(event_date) > 38:
            event_date = event_date[:35] + "…"
        content.add(Text(
            event_date,
            font=FONTS.body_font,
            font_size=FONTS.header_info_size,
            color=COLORS.text_secondary,
        ))

    # ── Opening ───────────────────────────────────────────────────────────────
    opening_display = game_info.get_opening_display()
    if opening_display:
        if len(opening_display) > 38:
            opening_display = opening_display[:35] + "…"
        content.add(Text(
            opening_display,
            font=FONTS.body_font,
            font_size=FONTS.header_info_size,
            color=COLORS.text_accent,
        ))

    # ── Arrange everything with a tight uniform spacing ───────────────────────
    content.arrange(DOWN, buff=0.08)
    content.move_to([PANEL_CENTER_X, HEADER_CENTER_Y, 0])
    panel.add(content)

    return panel


def create_move_list_panel() -> VGroup:
    """
    Create the move list panel (empty for initial frame).
    
    This will be populated as moves are played.
    """
    panel = VGroup()
    
    # Panel background
    bg = get_panel_rect(MOVE_LIST_TOP_Y, MOVE_LIST_BOTTOM_Y)
    panel.add(bg)
    
    # Title
    title = Text(
        "Moves",
        font=FONTS.heading_font,
        font_size=FONTS.subtitle_size,
        color=COLORS.text_secondary
    )
    title.move_to([PANEL_CENTER_X, MOVE_LIST_TOP_Y - 0.3, 0])
    panel.add(title)
    
    # Placeholder for empty state
    placeholder = Text(
        "Game starting...",
        font=FONTS.body_font,
        font_size=FONTS.move_size,
        color=COLORS.text_secondary,
        opacity=0.5
    )
    placeholder.move_to([PANEL_CENTER_X, MOVE_LIST_CENTER_Y, 0])
    panel.add(placeholder)
    
    return panel


def create_commentary_panel() -> VGroup:
    """
    Create the commentary panel (empty for initial frame).
    
    This will show annotations and engine analysis during playback.
    """
    panel = VGroup()
    
    # Panel background
    bg = get_panel_rect(COMMENTARY_TOP_Y, COMMENTARY_BOTTOM_Y)
    panel.add(bg)
    
    # Title
    title = Text(
        "Analysis",
        font=FONTS.heading_font,
        font_size=FONTS.subtitle_size,
        color=COLORS.text_secondary
    )
    title.move_to([PANEL_CENTER_X, COMMENTARY_TOP_Y - 0.3, 0])
    panel.add(title)
    
    # Placeholder
    placeholder = Text(
        "Waiting for first move...",
        font=FONTS.body_font,
        font_size=FONTS.commentary_size,
        color=COLORS.text_secondary,
        opacity=0.5
    )
    placeholder.move_to([PANEL_CENTER_X, COMMENTARY_CENTER_Y, 0])
    panel.add(placeholder)
    
    return panel


def create_board_with_eval() -> tuple[manim_chess.Board, manim_chess.EvaluationBar]:
    """
    Create the chess board and evaluation bar.
    
    Returns:
        Tuple of (board, eval_bar) Manim objects
    """
    # Create and configure the board
    board = manim_chess.Board()
    board.set_board_from_FEN()  # Starting position
    board.scale(BOARD_SCALE)
    board.move_to([BOARD_CENTER_X, BOARD_CENTER_Y, 0])
    
    # Create and configure the eval bar
    eval_bar = manim_chess.EvaluationBar()
    eval_bar.scale(EVAL_BAR_SCALE)
    eval_bar.next_to(board, LEFT, buff=EVAL_BAR_OFFSET)
    
    return board, eval_bar


# =============================================================================
# Main Scene
# =============================================================================

class InitialFrame(Scene):
    """
    Scene that displays the initial frame of a chess game.
    
    Shows:
    - Chess board in starting position
    - Evaluation bar (at 0.0)
    - Header panel with player names, event, date, opening
    - Empty move list panel
    - Empty commentary panel
    """
    
    def __init__(self, pgn_path: Optional[str] = None, **kwargs):
        """
        Initialize the scene.
        
        Args:
            pgn_path: Optional path to PGN file. If None, uses sample data.
        """
        super().__init__(**kwargs)
        self.pgn_path = pgn_path
    
    def construct(self):
        # Set background color
        self.camera.background_color = COLORS.background
        
        # Load game info - try multiple sources
        game_info = None
        
        # 1. Check for explicitly provided path
        if self.pgn_path and Path(self.pgn_path).exists():
            game_info = GameInfo.from_pgn(Path(self.pgn_path))
        
        # 2. Check for game.pgn in current directory
        elif Path("game.pgn").exists():
            game_info = GameInfo.from_pgn(Path("game.pgn"))
        
        # 3. Check for sample_game.pgn in current directory
        elif Path("sample_game.pgn").exists():
            game_info = GameInfo.from_pgn(Path("sample_game.pgn"))
        
        # 4. Fall back to sample data
        if game_info is None:
            game_info = GameInfo(
                white="Caruana, Fabiano",
                black="Carlsen, Magnus",
                white_elo="2820",
                black_elo="2863",
                event="Candidates Tournament",
                site="Toronto",
                date="2024.04.04",
                round="5",
                result="1/2-1/2",
                opening="Ruy Lopez, Berlin Defense",
                eco="C67"
            )
        
        # Create board and eval bar
        board, eval_bar = create_board_with_eval()
        
        # Create panels
        header_panel = create_header_panel(game_info)
        move_list_panel = create_move_list_panel()
        commentary_panel = create_commentary_panel()
        
        # Add everything to the scene
        self.add(board)
        self.add(eval_bar)
        self.add(header_panel)
        self.add(move_list_panel)
        self.add(commentary_panel)
        
        # Hold the frame
        self.wait(2)


class InitialFrameWithFEN(Scene):
    """
    Scene that displays a specific position from FEN.
    Useful for starting from a particular position.
    """
    
    def __init__(self, fen: str = None, **kwargs):
        super().__init__(**kwargs)
        # Default to a famous position if none provided
        self.fen = fen or "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    
    def construct(self):
        self.camera.background_color = COLORS.background
        
        # Sample game info
        game_info = GameInfo(
            white="Player 1",
            black="Player 2",
            opening="Italian Game",
            eco="C50"
        )
        
        # Create board with specific position
        board = manim_chess.Board()
        board.set_board_from_FEN(self.fen)
        board.scale(BOARD_SCALE)
        board.move_to([BOARD_CENTER_X, BOARD_CENTER_Y, 0])
        
        eval_bar = manim_chess.EvaluationBar()
        eval_bar.scale(EVAL_BAR_SCALE)
        eval_bar.next_to(board, LEFT, buff=EVAL_BAR_OFFSET)
        
        # Create panels
        header_panel = create_header_panel(game_info)
        move_list_panel = create_move_list_panel()
        commentary_panel = create_commentary_panel()
        
        self.add(board, eval_bar, header_panel, move_list_panel, commentary_panel)
        self.wait(2)


class LayoutDebug(Scene):
    """
    Debug scene showing layout guides and measurements.
    Useful for adjusting the layout configuration.
    """
    
    def construct(self):
        from animator_layout import create_layout_guides
        
        self.camera.background_color = COLORS.background
        
        # Create all elements
        board, eval_bar = create_board_with_eval()
        
        game_info = GameInfo(
            white="Test Player White",
            black="Test Player Black",
            white_elo="2700",
            black_elo="2750",
            event="Test Tournament",
            date="2024.01.01",
            opening="Test Opening Name That Is Long",
            eco="A00"
        )
        
        header_panel = create_header_panel(game_info)
        move_list_panel = create_move_list_panel()
        commentary_panel = create_commentary_panel()
        
        # Add layout guides
        guides = create_layout_guides()
        
        self.add(board, eval_bar)
        self.add(header_panel, move_list_panel, commentary_panel)
        self.add(guides)
        
        self.wait(3)


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("Chess Game Animator - Initial Frame Generator")
    print("=" * 50)
    print()
    print("Usage:")
    print("  manim -pql animator_initial_frame.py InitialFrame")
    print("  manim -pqh animator_initial_frame.py InitialFrame  # High quality")
    print("  manim -pql animator_initial_frame.py LayoutDebug   # Debug view")
    print()
    print("Scenes available:")
    print("  InitialFrame      - Main initial frame with sample data")
    print("  InitialFrameWithFEN - Initial frame with custom FEN position")
    print("  LayoutDebug       - Debug view showing layout guides")

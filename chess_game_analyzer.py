#!/usr/bin/env python3
"""
Enhanced Chess Game Analyzer with Positional Evaluation Metrics and Plots
==========================================================================

An extended version of the chess game analyzer that computes detailed positional
evaluation metrics directly from board analysis, including:
- **Space**: Control of territory, particularly in enemy's half of the board
- **Mobility**: Number of legal moves available to pieces (weighted by piece type)
- **King Safety**: Structural protection around the king
- **Pawn Structure**: Doubled pawns, isolated pawns, passed pawns

NEW IN THIS VERSION:
- Move-by-move matplotlib plots of evaluation, space, mobility, and king safety
- Optional ASCII plots for environments without matplotlib/graphics support
- Plots show White (solid) vs Black (dotted) on the same axis
- Game character classification (balanced/tense/tactical/chaotic, one-sided/seesaw)
- Multi-PV analysis: suggests playable alternative moves (within 50cp of best)
- Improved mistake annotations: "Consider instead: Nf3, Bg5." format

This module uses python-chess for direct board analysis to compute interpretable
positional metrics, while using Stockfish only for overall centipawn evaluation.
Ideal for creating educational chess content with detailed position assessment.

How the Code Computes These Metrics
------------------------------------

### Space
Space evaluation counts squares in the center (files c-f, ranks 2-4 for 
White, ranks 5-7 for Black) that are:
1. Attacked by at least one friendly pawn, AND
2. Not occupied by any enemy piece

The computation is weighted by the number of pieces behind the pawn chain. This
rewards positions where you control territory AND have pieces positioned to use
that space. The space term is scaled based on total piece count - it matters more
in closed positions with many pieces.

### Mobility
Mobility counts legal moves for each piece type, with weights:
- Knights: Safe squares attacked (excl. squares attacked by enemy pawns)
- Bishops: Safe squares on diagonals (bonus for long diagonals)
- Rooks: Squares on files/ranks (bonus for open files, 7th rank)
- Queens: Combination of bishop + rook mobility

"Safe" squares exclude those defended by enemy pawns or those that would allow
piece exchange with material loss. Mobility in the ENEMY's territory is valued
higher than mobility in your own territory.

### King Safety
A combination of:
1. Pawn shield strength (pawns on 2nd/3rd rank in front of king)
2. King tropism (distance of enemy pieces to your king)
3. Attack units (enemy pieces attacking squares near your king)
4. Safe checks available to opponent

### Threats
Evaluates tactical tension (computed directly from board position):
- Hanging pieces (pieces attacked but not defended) - weighted by piece value × 2
- Pieces attacked by lower-value pieces (e.g., queen attacked by knight)
- Attacks on squares near the enemy king (king zone pressure)
- Safe checks available to each side
- Weak squares (holes) in pawn structure that cannot be defended by pawns
- Available checking moves for the side to move

Usage:
    from chess_game_analyzer import (
        analyze_game_with_positional_metrics,
        PositionalEvaluation,
        EnhancedMoveAnalysis,
        analyze_games_to_book,
        classify_game_character
    )
    
    # Get analysis with full positional breakdown and plots
    game_pgn = "../wdj-games/james-rizzitano-vs-wdj-sussex-open-2026-01-10.pgn"
    report_output = "../wdj-games/james-rizzitano-vs-wdj-sussex-open-2026-01-10-analysis.tex"
    result = analyze_game_with_positional_metrics(
        pgn_source=game_pgn,
        output_path=report_output,
        stockfish_path = "/usr/local/bin/stockfish", # modify as needed
        include_plots=True,                          # matplotlib plots
        include_ascii_plots=False,                   # ASCII verbatim plots
        depth=22,
        time_limit=5.0,                              # longer for complex games
        plot_output_dir = "../wdj-games/plots/"
    )
    for move in result.moves:
        pos = move.positional_eval
        print(f"Move {move.ply}: Space W={pos.space_white:.2f} B={pos.space_black:.2f}")
        print(f"         Mobility W={pos.mobility_white:.2f} B={pos.mobility_black:.2f}")
    
    # Access game character classification
    gc = result.game_character
    print(f"Game character: {gc['spread_class']} ({gc['direction_class']})")
    print(f"  m1={gc['m1']:.2f}, m2={gc['m2']:.2f}, spread={gc['spread']:.2f}")

    >>> from chess_game_analyzer import (
    ...         analyze_game_with_positional_metrics,
    ...         PositionalEvaluation,
    ...         EnhancedMoveAnalysis,
    ...         analyze_games_to_book,
                classify_game_character
    ...     )
    >>> game_pgn = "../wdj-games/Joyner-vs-Goodson_2023-05-16.pgn"
    >>> report_output = "../wdj-games/Joyner-vs-Goodson_2023-05-16-analysis.tex"
    >>> result = analyze_game_with_positional_metrics(pgn_source=game_pgn,output_path=report_output,include_plots=True,include_ascii_plots=False, plot_output_dir = "../wdj-games/plots/")

NEW IN VERSION 6q:
- Raw positional data preserved after \\end{document} in machine-readable format
- Data includes: ply, SAN, eval_cp, space_w/b, mobility_w/b, king_safety_w/b, threats_w/b
- New utility functions: parse_raw_positional_data(), compute_fireteam_index()
- Enables downstream analysis like the "Fireteam Index" for win prediction

NEW IN VERSION 6r:
- Win prediction algorithms: predict_outcome_per_ply() and predict_outcome_windowed()
- Optional Fireteam Index prediction section in LaTeX reports (include_prediction=True)
- Fireteam Index plots (per-ply and smoothed versions) in reports
- Renamed berliner_color to player_color with optional player_name parameter
- Configurable weights, cutoff, margin, streak length, and window size
- BUG FIX: eval_loss calculation now uses proper sign convention for Black moves
- BUG FIX: eval_loss capped at MAX_EVAL_LOSS_FOR_ACCURACY (1500cp) to prevent
  mate score transitions from producing absurd values (8000+ cp) that distort
  accuracy statistics. Previously, games with missed mates could show <20% accuracy
  even for strong play, because a single "lose mate" move counted as 9000+ cp loss.

Author: Generated for David Joyner's chess analysis pipeline, 2026-01-24
distribution license: either modified BSD or MIT license, user's choice.
"""

import chess
import chess.pgn
import chess.engine
import io
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path

# Import plotting utilities
try:
    from chess_plotting import ChessPlotter, is_matplotlib_available
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    def is_matplotlib_available():
        return False


# =============================================================================
# PIECE VALUES (centipawns)
# =============================================================================

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# Threshold (in centipawns) for considering alternative moves as "playable"
# Moves within this threshold of the best move will be suggested as alternatives
PLAYABLE_THRESHOLD = 50

# Maximum eval_loss to count towards accuracy calculations (in centipawns)
# This prevents mate score transitions from producing absurd values (8000+ cp)
# that would completely distort accuracy statistics. A cap of 1500cp (15 pawns)
# still represents a catastrophic blunder but won't ruin the entire game's stats.
MAX_EVAL_LOSS_FOR_ACCURACY = 1500


def parse_elo(elo_str: str) -> Optional[int]:
    """
    Safely parse an ELO rating from a PGN header value.
    
    Handles common non-numeric values like "?", "*", "", "-", "N/A", etc.
    Returns None if the value cannot be parsed as a valid ELO rating.
    """
    if not elo_str:
        return None
    elo_str = elo_str.strip()
    if not elo_str or elo_str in ("?", "*", "-", "N/A", "n/a", "unknown", "Unknown"):
        return None
    try:
        elo = int(elo_str)
        # Sanity check: valid ELO ratings are typically between 100 and 4000
        if 100 <= elo <= 4000:
            return elo
        elif elo == 0:
            return None  # 0 often means "unknown"
        else:
            return elo  # Return anyway if outside typical range but parseable
    except ValueError:
        return None


# =============================================================================
# GAME CHARACTER CLASSIFICATION
# =============================================================================

def classify_game_character(moves: List) -> Dict[str, any]:
    """
    Classify the overall character of a game based on evaluation swings.
    
    Let m2 = maximum evaluation for White (in pawns)
    Let m1 = minimum evaluation for White (in pawns)
    Let d = m2 - m1 (the evaluation spread)
    
    Spread classifications:
    - balanced: d < 1 (minimal advantage shifts)
    - tense: 1 <= d < 3 (normal competitive tension)
    - tactical: 3 <= d < 6 (significant swings, tactical complications)
    - chaotic: d >= 6 (wild swings, likely blunders or speculative play)
    
    Directionality classifications:
    - one-sided: m1 > -0.5 OR m2 < 0.5 (advantage never truly changed hands)
    - seesaw: m1 < -1 AND m2 > 1 (advantage genuinely swung both ways)
    
    Args:
        moves: List of EnhancedMoveAnalysis objects
        
    Returns:
        Dictionary with:
        - m1: minimum evaluation (pawns)
        - m2: maximum evaluation (pawns)
        - spread: m2 - m1
        - spread_class: 'balanced', 'tense', 'tactical', or 'chaotic'
        - direction_class: 'one-sided', 'seesaw', or 'normal'
        - combined_description: human-readable summary
    """
    if not moves:
        return {
            'm1': 0.0, 'm2': 0.0, 'spread': 0.0,
            'spread_class': 'balanced',
            'direction_class': 'normal',
            'combined_description': 'No moves to analyze'
        }
    
    # Extract evaluations in pawns (eval_after is in centipawns)
    evals = []
    for m in moves:
        # Cap extreme evaluations (mate scores) at ±15 pawns for classification
        eval_pawns = m.eval_after / 100.0
        if abs(eval_pawns) > 15:
            eval_pawns = 15.0 if eval_pawns > 0 else -15.0
        evals.append(eval_pawns)
    
    m1 = min(evals)  # minimum (most favorable for Black)
    m2 = max(evals)  # maximum (most favorable for White)
    d = m2 - m1      # spread
    
    # Spread classification
    if d < 1:
        spread_class = 'balanced'
    elif d < 3:
        spread_class = 'tense'
    elif d < 6:
        spread_class = 'tactical'
    else:
        spread_class = 'chaotic'
    
    # Directionality classification
    if m1 > -0.5 or m2 < 0.5:
        direction_class = 'one-sided'
    elif m1 < -1 and m2 > 1:
        direction_class = 'seesaw'
    else:
        direction_class = 'normal'
    
    # Generate combined description
    if direction_class == 'one-sided':
        if m1 > -0.5:
            side_desc = "White maintained the advantage throughout"
        else:
            side_desc = "Black maintained the advantage throughout"
        combined = f"A {spread_class}, one-sided game. {side_desc}."
    elif direction_class == 'seesaw':
        combined = f"A {spread_class} seesaw battle with the advantage changing hands."
    else:
        combined = f"A {spread_class} game with moderate swings."
    
    return {
        'm1': m1,
        'm2': m2,
        'spread': d,
        'spread_class': spread_class,
        'direction_class': direction_class,
        'combined_description': combined
    }


# =============================================================================
# POSITIONAL EVALUATION DATA CLASSES
# =============================================================================

@dataclass
class PositionalEvaluation:
    """
    Detailed positional evaluation breakdown from Stockfish's classical eval.
    
    All values are in pawns (not centipawns) for readability.
    MG = Middlegame, EG = Endgame weights.
    
    Stockfish blends MG and EG based on remaining material (phase).
    """
    # Material and imbalance
    material_mg: float = 0.0
    material_eg: float = 0.0
    imbalance_mg: float = 0.0
    imbalance_eg: float = 0.0
    
    # Piece-specific terms (these include placement bonuses)
    pawns_white_mg: float = 0.0
    pawns_white_eg: float = 0.0
    pawns_black_mg: float = 0.0
    pawns_black_eg: float = 0.0
    
    knights_white_mg: float = 0.0
    knights_white_eg: float = 0.0
    knights_black_mg: float = 0.0
    knights_black_eg: float = 0.0
    
    bishops_white_mg: float = 0.0
    bishops_white_eg: float = 0.0
    bishops_black_mg: float = 0.0
    bishops_black_eg: float = 0.0
    
    rooks_white_mg: float = 0.0
    rooks_white_eg: float = 0.0
    rooks_black_mg: float = 0.0
    rooks_black_eg: float = 0.0
    
    queens_white_mg: float = 0.0
    queens_white_eg: float = 0.0
    queens_black_mg: float = 0.0
    queens_black_eg: float = 0.0
    
    # Key strategic terms (per-side)
    mobility_white_mg: float = 0.0
    mobility_white_eg: float = 0.0
    mobility_black_mg: float = 0.0
    mobility_black_eg: float = 0.0
    
    king_safety_white_mg: float = 0.0
    king_safety_white_eg: float = 0.0
    king_safety_black_mg: float = 0.0
    king_safety_black_eg: float = 0.0
    
    threats_white_mg: float = 0.0
    threats_white_eg: float = 0.0
    threats_black_mg: float = 0.0
    threats_black_eg: float = 0.0
    
    passed_white_mg: float = 0.0
    passed_white_eg: float = 0.0
    passed_black_mg: float = 0.0
    passed_black_eg: float = 0.0
    
    space_white_mg: float = 0.0
    space_white_eg: float = 0.0  # Usually 0, space matters in MG
    space_black_mg: float = 0.0
    space_black_eg: float = 0.0
    
    # Winnable term (adjustment factor)
    winnable_mg: float = 0.0
    winnable_eg: float = 0.0
    
    # Totals
    total_white_mg: float = 0.0
    total_white_eg: float = 0.0
    total_black_mg: float = 0.0
    total_black_eg: float = 0.0
    total_mg: float = 0.0
    total_eg: float = 0.0
    
    # Final evaluations
    classical_eval: float = 0.0
    nnue_eval: float = 0.0
    final_eval: float = 0.0
    
    # Computed summary metrics (convenience)
    @property
    def space_white(self) -> float:
        """White's space advantage (MG, since EG is usually 0)."""
        return self.space_white_mg
    
    @property
    def space_black(self) -> float:
        """Black's space control."""
        return self.space_black_mg
    
    @property
    def space_advantage(self) -> float:
        """Net space advantage (positive = White)."""
        return self.space_white_mg - self.space_black_mg
    
    @property
    def mobility_white(self) -> float:
        """White's mobility (average of MG and EG)."""
        return (self.mobility_white_mg + self.mobility_white_eg) / 2
    
    @property
    def mobility_black(self) -> float:
        """Black's mobility."""
        return (self.mobility_black_mg + self.mobility_black_eg) / 2
    
    @property
    def mobility_advantage(self) -> float:
        """Net mobility advantage (positive = White)."""
        return self.mobility_white - self.mobility_black
    
    @property
    def king_safety_white(self) -> float:
        """White's king safety (MG more relevant)."""
        return self.king_safety_white_mg
    
    @property
    def king_safety_black(self) -> float:
        """Black's king safety."""
        return self.king_safety_black_mg

    @property
    def threats_white(self) -> float:
        """White's threat level (MG more relevant for active threats)."""
        return self.threats_white_mg
    
    @property
    def threats_black(self) -> float:
        """Black's threat level."""
        return self.threats_black_mg
    
    @property
    def threats_advantage(self) -> float:
        """Net threats advantage (positive = White has more threats)."""
        return self.threats_white_mg - self.threats_black_mg


@dataclass
class EnhancedMoveAnalysis:
    """Analysis of a single move with positional breakdown."""
    ply: int
    move_san: str
    move_uci: str
    is_white_move: bool
    eval_before: float
    eval_after: float
    best_move_san: str
    best_move_uci: str
    best_eval: float
    eval_loss: float
    classification: str
    is_capture: bool
    is_check: bool
    material_balance: int
    fen_after: str
    pv_line: List[str]
    
    # Enhanced: Positional evaluation after this move
    positional_eval: Optional[PositionalEvaluation] = None
    
    # Alternative moves: list of (san, eval_cp) tuples for playable alternatives
    # Only includes moves within PLAYABLE_THRESHOLD of the best move
    alternative_moves: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class BrilliantSacrifice:
    """Details of a detected brilliant sacrifice."""
    ply: int
    move_san: str
    player: str
    piece_type: str
    material_lost: int
    eval_before: float
    eval_after: float
    eval_improvement: float
    is_sound: bool


@dataclass
class CriticalPosition:
    """A critical position worth showing a diagram for."""
    ply: int
    fen: str
    move_san: str
    eval_score: float
    reason: str
    best_continuation: List[str]
    positional_eval: Optional[PositionalEvaluation] = None
    is_biggest_swing: bool = False  # True if this is a top-N biggest evaluation swing
    eval_swing: float = 0.0  # Magnitude of the evaluation change
    alternative_moves: List[Tuple[str, float]] = field(default_factory=list)
    best_move_san: str = ""  # The best move instead of the played move

@dataclass
class EnhancedGameAnalysisResult:
    """Complete analysis result with positional metrics."""
    # Game metadata
    white: str
    black: str
    white_elo: Optional[int]
    black_elo: Optional[int]
    result: str
    date: str
    event: str
    site: str
    round_num: str
    opening_eco: str
    opening_name: str
    
    # Analysis data
    moves: List[EnhancedMoveAnalysis]
    brilliant_sacrifices: List[BrilliantSacrifice]
    critical_positions: List[CriticalPosition]
    
    # Statistics
    white_stats: Dict
    black_stats: Dict
    
    # Positional summaries
    positional_summary: Dict = field(default_factory=dict)
    
    # Game character classification
    game_character: Dict = field(default_factory=dict)
    
    # Metadata
    analysis_depth: int = 20
    analysis_time: float = 0.0
    engine_version: str = "Stockfish"

def compute_threats(board):
    """
    Compute threat metrics for both sides.
    Returns (white_threats, black_threats) as a tuple of scores.
    
    Threats include:
    - Hanging pieces (attacked and undefended)
    - Pieces attacked by lower-value pieces
    - Attacks on squares near enemy king
    - Safe checks available
    - Weak squares in pawn structure (holes)
    """
    white_threats = 0
    black_threats = 0
    
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # Don't count king threats
    }
    
    # 1. Hanging pieces and pieces attacked by lower-value pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.piece_type == chess.KING:
            continue
        
        # Get attackers and defenders
        attackers = board.attackers(not piece.color, square)
        defenders = board.attackers(piece.color, square)
        
        # Hanging piece (attacked but not defended)
        if attackers and not defenders:
            if piece.color == chess.WHITE:
                black_threats += piece_values[piece.piece_type] * 2
            else:
                white_threats += piece_values[piece.piece_type] * 2
        
        # Attacked by lower value piece
        elif attackers:
            min_attacker_value = min(
                piece_values[board.piece_at(sq).piece_type] 
                for sq in attackers
            )
            if min_attacker_value < piece_values[piece.piece_type]:
                if piece.color == chess.WHITE:
                    black_threats += piece_values[piece.piece_type] - min_attacker_value
                else:
                    white_threats += piece_values[piece.piece_type] - min_attacker_value
    
    # 2. Attacks near kings (king zone pressure)
    for color in [chess.WHITE, chess.BLACK]:
        king_square = board.king(color)
        if king_square is None:
            continue
        
        # Squares around the king
        king_zone = []
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                f, r = king_file + df, king_rank + dr
                if 0 <= f <= 7 and 0 <= r <= 7:
                    king_zone.append(chess.square(f, r))
        
        # Count enemy attacks on king zone
        enemy_attacks = sum(
            1 for sq in king_zone 
            if board.attackers(not color, sq)
        )
        
        if color == chess.WHITE:
            black_threats += enemy_attacks
        else:
            white_threats += enemy_attacks
    
    # 3. Safe checks available
    for color in [chess.WHITE, chess.BLACK]:
        enemy_king_sq = board.king(not color)
        if enemy_king_sq is None:
            continue
        
        safe_checks = 0
        # Find all squares that would give check
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            # Get squares from which this piece type could attack the enemy king
            if piece_type == chess.KNIGHT:
                check_squares = chess.SquareSet(chess.BB_KNIGHT_ATTACKS[enemy_king_sq])
            elif piece_type == chess.BISHOP:
                check_squares = board.attacks_mask(enemy_king_sq) & chess.BB_DIAG_ATTACKS[enemy_king_sq][0]
                # Use bishop attacks from enemy king square
                check_squares = chess.SquareSet(chess.BB_DIAG_MASKS[enemy_king_sq])
            elif piece_type == chess.ROOK:
                check_squares = chess.SquareSet(chess.BB_FILE_MASKS[enemy_king_sq] | chess.BB_RANK_MASKS[enemy_king_sq])
            else:  # QUEEN
                check_squares = chess.SquareSet(
                    chess.BB_DIAG_MASKS[enemy_king_sq] | 
                    chess.BB_FILE_MASKS[enemy_king_sq] | 
                    chess.BB_RANK_MASKS[enemy_king_sq]
                )
            
            for sq in check_squares:
                piece = board.piece_at(sq)
                if piece and piece.color == color and piece.piece_type == piece_type:
                    # This piece could potentially give check - check if square is safe
                    if not board.is_attacked_by(not color, sq):
                        # Verify it actually gives check (considering blockers)
                        if board.is_attacked_by(color, enemy_king_sq):
                            safe_checks += 1
        
        if color == chess.WHITE:
            white_threats += safe_checks * 0.5
        else:
            black_threats += safe_checks * 0.5
    
    # 4. Weak squares (holes in pawn structure)
    # A hole is a square that cannot be defended by pawns
    for color in [chess.WHITE, chess.BLACK]:
        enemy_color = not color
        # Check central and near-central squares for holes
        if color == chess.WHITE:
            # White looks for holes in Black's position (ranks 5-7)
            target_ranks = [4, 5, 6]
        else:
            # Black looks for holes in White's position (ranks 2-4, i.e., indices 1-3)
            target_ranks = [1, 2, 3]
        
        holes = 0
        for rank in target_ranks:
            for file in range(8):
                sq = chess.square(file, rank)
                # Check if any enemy pawn can ever defend this square
                can_be_defended = False
                
                # For a square to be defendable by a pawn, there must be a pawn
                # on an adjacent file that can advance to defend it
                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file <= 7:
                        # Check if there's an enemy pawn that could defend
                        if enemy_color == chess.WHITE:
                            # White pawns defend by moving up
                            for pawn_rank in range(rank):
                                pawn_sq = chess.square(adj_file, pawn_rank)
                                p = board.piece_at(pawn_sq)
                                if p and p.piece_type == chess.PAWN and p.color == enemy_color:
                                    can_be_defended = True
                                    break
                        else:
                            # Black pawns defend by moving down
                            for pawn_rank in range(rank + 1, 8):
                                pawn_sq = chess.square(adj_file, pawn_rank)
                                p = board.piece_at(pawn_sq)
                                if p and p.piece_type == chess.PAWN and p.color == enemy_color:
                                    can_be_defended = True
                                    break
                    if can_be_defended:
                        break
                
                if not can_be_defended:
                    # This is a hole - bonus if we control it with a piece
                    if board.is_attacked_by(color, sq):
                        holes += 0.3
                    else:
                        holes += 0.1
        
        if color == chess.WHITE:
            white_threats += holes
        else:
            black_threats += holes
    
    # 5. Check threats (if side to move can give check)
    if board.turn == chess.WHITE:
        # Count checking moves available to White
        check_moves = sum(1 for move in board.legal_moves if board.gives_check(move))
        white_threats += check_moves * 0.3
    else:
        # Count checking moves available to Black
        check_moves = sum(1 for move in board.legal_moves if board.gives_check(move))
        black_threats += check_moves * 0.3
    
    return (white_threats, black_threats)
    
# =============================================================================
# STOCKFISH EVAL PARSER
# =============================================================================

class StockfishEvalParser:
    """
    Parses Stockfish's `eval` command output to extract positional metrics.
    
    The `eval` command outputs a table showing contribution of each evaluation
    term for both White and Black in both middlegame (MG) and endgame (EG).
    
    Note: Different Stockfish versions may have slightly different output formats.
    This parser attempts to handle variations, but if you get all zeros, your
    Stockfish build may not output the classical eval table.
    """
    
    # Debug flag - set to True to print parsing details
    DEBUG = False
    
    @staticmethod
    def parse_eval_output(eval_text: str, debug: bool = False) -> PositionalEvaluation:
        """
        Parse the text output from Stockfish's eval command.
        
        Args:
            eval_text: Raw text from `eval` command
            debug: If True, print parsing details for troubleshooting
            
        Returns:
            PositionalEvaluation with all extracted metrics
        """
        debug = debug or StockfishEvalParser.DEBUG
        pos_eval = PositionalEvaluation()
        
        if debug:
            print(f"[DEBUG] Parsing eval output ({len(eval_text)} chars)")
            print(f"[DEBUG] First 500 chars:\n{eval_text[:500]}")
        
        # Check if this looks like valid Stockfish eval output
        if 'Contributing terms' not in eval_text and 'classical eval' not in eval_text.lower():
            if debug:
                print("[DEBUG] WARNING: Output doesn't appear to contain classical eval table")
            # Return empty evaluation - Stockfish may be NNUE-only build
            return pos_eval
        
        # Multiple regex patterns to handle different Stockfish versions
        # Pattern 1: Standard format with fixed-width columns
        # |      Space |  0.34  0.00 |  0.30  0.00 |  0.04  0.00 |
        patterns = [
            # Primary pattern - handles most cases
            re.compile(
                r'\|\s*([A-Za-z][A-Za-z ]*?)\s*\|'  # Term name
                r'\s*(-?[\d.]+|----)\s+(-?[\d.]+|----)\s*\|'  # White MG, EG  
                r'\s*(-?[\d.]+|----)\s+(-?[\d.]+|----)\s*\|'  # Black MG, EG
                r'\s*(-?[\d.]+|----)\s+(-?[\d.]+|----)\s*\|'  # Total MG, EG
            ),
            # Alternative pattern with looser spacing
            re.compile(
                r'\|\s*(\w+(?:\s+\w+)?)\s*\|'  # Term name
                r'\s*([\d.\-]+|----)\s+([\d.\-]+|----)\s*\|'  # White MG, EG
                r'\s*([\d.\-]+|----)\s+([\d.\-]+|----)\s*\|'  # Black MG, EG
                r'\s*([\d.\-]+|----)\s+([\d.\-]+|----)\s*\|'  # Total MG, EG
            ),
        ]
        
        # Term name normalization map
        term_aliases = {
            'king safety': 'king_safety',
            'kingsafety': 'king_safety',
            'king': 'king_safety',  # Some versions abbreviate
        }
        
        matched_terms = set()
        
        for line in eval_text.split('\n'):
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    raw_term = match.group(1).strip().lower()
                    term = term_aliases.get(raw_term, raw_term.replace(' ', '_'))
                    
                    w_mg = StockfishEvalParser._parse_value(match.group(2))
                    w_eg = StockfishEvalParser._parse_value(match.group(3))
                    b_mg = StockfishEvalParser._parse_value(match.group(4))
                    b_eg = StockfishEvalParser._parse_value(match.group(5))
                    t_mg = StockfishEvalParser._parse_value(match.group(6))
                    t_eg = StockfishEvalParser._parse_value(match.group(7))
                    
                    if debug:
                        print(f"[DEBUG] Matched term '{raw_term}' -> '{term}': "
                              f"W({w_mg}, {w_eg}) B({b_mg}, {b_eg}) T({t_mg}, {t_eg})")
                    
                    matched_terms.add(term)
                    
                    # Map terms to attributes
                    if term == 'material':
                        pos_eval.material_mg = t_mg
                        pos_eval.material_eg = t_eg
                    elif term == 'imbalance':
                        pos_eval.imbalance_mg = t_mg
                        pos_eval.imbalance_eg = t_eg
                    elif term == 'pawns':
                        pos_eval.pawns_white_mg = w_mg
                        pos_eval.pawns_white_eg = w_eg
                        pos_eval.pawns_black_mg = b_mg
                        pos_eval.pawns_black_eg = b_eg
                    elif term == 'knights':
                        pos_eval.knights_white_mg = w_mg
                        pos_eval.knights_white_eg = w_eg
                        pos_eval.knights_black_mg = b_mg
                        pos_eval.knights_black_eg = b_eg
                    elif term == 'bishops':
                        pos_eval.bishops_white_mg = w_mg
                        pos_eval.bishops_white_eg = w_eg
                        pos_eval.bishops_black_mg = b_mg
                        pos_eval.bishops_black_eg = b_eg
                    elif term == 'rooks':
                        pos_eval.rooks_white_mg = w_mg
                        pos_eval.rooks_white_eg = w_eg
                        pos_eval.rooks_black_mg = b_mg
                        pos_eval.rooks_black_eg = b_eg
                    elif term == 'queens':
                        pos_eval.queens_white_mg = w_mg
                        pos_eval.queens_white_eg = w_eg
                        pos_eval.queens_black_mg = b_mg
                        pos_eval.queens_black_eg = b_eg
                    elif term == 'mobility':
                        pos_eval.mobility_white_mg = w_mg
                        pos_eval.mobility_white_eg = w_eg
                        pos_eval.mobility_black_mg = b_mg
                        pos_eval.mobility_black_eg = b_eg
                    elif term == 'king_safety':
                        pos_eval.king_safety_white_mg = w_mg
                        pos_eval.king_safety_white_eg = w_eg
                        pos_eval.king_safety_black_mg = b_mg
                        pos_eval.king_safety_black_eg = b_eg
                    elif term == 'threats':
                        pos_eval.threats_white_mg = w_mg
                        pos_eval.threats_white_eg = w_eg
                        pos_eval.threats_black_mg = b_mg
                        pos_eval.threats_black_eg = b_eg
                    elif term == 'passed':
                        pos_eval.passed_white_mg = w_mg
                        pos_eval.passed_white_eg = w_eg
                        pos_eval.passed_black_mg = b_mg
                        pos_eval.passed_black_eg = b_eg
                    elif term == 'space':
                        pos_eval.space_white_mg = w_mg
                        pos_eval.space_white_eg = w_eg
                        pos_eval.space_black_mg = b_mg
                        pos_eval.space_black_eg = b_eg
                    elif term == 'winnable':
                        pos_eval.winnable_mg = t_mg
                        pos_eval.winnable_eg = t_eg
                    elif term == 'total':
                        pos_eval.total_mg = t_mg
                        pos_eval.total_eg = t_eg
                    
                    break  # Don't try other patterns if this one matched
        
        if debug:
            print(f"[DEBUG] Matched {len(matched_terms)} terms: {matched_terms}")
        
        # Parse final evaluation lines
        classical_match = re.search(r'Classical evaluation\s+([+-]?\d+\.?\d*)', eval_text)
        if classical_match:
            pos_eval.classical_eval = float(classical_match.group(1))
        
        nnue_match = re.search(r'NNUE evaluation\s+([+-]?\d+\.?\d*)', eval_text)
        if nnue_match:
            pos_eval.nnue_eval = float(nnue_match.group(1))
        
        final_match = re.search(r'Final evaluation\s+([+-]?\d+\.?\d*)', eval_text)
        if final_match:
            pos_eval.final_eval = float(final_match.group(1))
        
        return pos_eval
    
    @staticmethod
    def _parse_value(val_str: str) -> float:
        """Parse a value string, handling '----' as 0."""
        if val_str is None or val_str == '----' or val_str.strip() == '':
            return 0.0
        try:
            return float(val_str)
        except ValueError:
            return 0.0


# =============================================================================
# ENHANCED ANALYZER
# =============================================================================


class EnhancedGameAnalyzer:
    def __init__(self, stockfish_path: str = "/usr/local/bin/stockfish",
                 depth: int = 20, time_limit: float = 1.0,
                 extract_positional: bool = True):
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.time_limit = time_limit
        self.extract_positional = extract_positional
        self.engine = None
        self.engine_version = "Unknown"

    def __enter__(self):
        """Protocol to support 'with' statement."""
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self.engine_version = self.engine.id.get('name', 'Stockfish')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures the engine quits properly."""
        if self.engine:
            self.engine.quit()

    def _eval_to_cp(self, score: chess.engine.PovScore) -> float:
        white_score = score.white()
        if white_score.is_mate():
            mate_in = white_score.mate()
            return 10000 - abs(mate_in) * 10 if mate_in > 0 else -10000 + abs(mate_in) * 10
        return float(white_score.score() or 0)

    def _get_positional_eval(self, board: chess.Board) -> PositionalEvaluation:
        """New: Directly calculates metrics via python-chess."""
        pe = PositionalEvaluation()
        pe.space_white_mg = self._calculate_space(board, chess.WHITE)
        pe.space_black_mg = self._calculate_space(board, chess.BLACK)
        pe.mobility_white_mg = self._calculate_mobility(board, chess.WHITE)
        pe.mobility_black_mg = self._calculate_mobility(board, chess.BLACK)
        pe.king_safety_white_mg = self._calculate_king_safety(board, chess.WHITE)
        pe.king_safety_black_mg = self._calculate_king_safety(board, chess.BLACK)
        
        # Compute threats directly from board position
        white_threats, black_threats = compute_threats(board)
        pe.threats_white_mg = white_threats
        pe.threats_black_mg = black_threats
        
        return pe

    def _calculate_space(self, board: chess.Board, color: chess.Color) -> float:
        space = 0
        ranks = [1, 2, 3] if color == chess.WHITE else [4, 5, 6]
        files = [2, 3, 4, 5]
        for r in ranks:
            for f in files:
                sq = chess.square(f, r)
                pawn_attackers = [a for a in board.attackers(color, sq) 
                                 if board.piece_at(a).piece_type == chess.PAWN]
                occ = board.piece_at(sq)
                if pawn_attackers and (not occ or occ.color == color):
                    space += 1
        return space / 10.0

    def _calculate_mobility(self, board: chess.Board, color: chess.Color) -> float:
        mobility = 0
        for sq, piece in board.piece_map().items():
            if piece.color == color and piece.piece_type != chess.KING:
                for target in board.attacks(sq):
                    enemy_pawns = board.pieces(chess.PAWN, not color)
                    if not any(target in board.attacks(p) for p in enemy_pawns):
                        mobility += 0.1
        return mobility

    def _calculate_king_safety(self, board: chess.Board, color: chess.Color) -> float:
        king_sq = board.king(color)
        if king_sq is None: return 0.0
        score, k_file = 0.0, chess.square_file(king_sq)
        shield_rank = 1 if color == chess.WHITE else 6
        for f_off in [-1, 0, 1]:
            if 0 <= k_file + f_off <= 7:
                shield_sq = chess.square(k_file + f_off, shield_rank)
                p = board.piece_at(shield_sq)
                if p and p.piece_type == chess.PAWN and p.color == color:
                    score += 0.5
        for sq in chess.SQUARES:
            if chess.square_distance(king_sq, sq) <= 2:
                if board.is_attacked_by(not color, sq):
                    score -= 0.2
        return score

    def analyze_game(self, pgn_source: Union[str, io.StringIO], 
                     min_diagram_spacing: int = 6,
                     top_n_swings: int = 2) -> EnhancedGameAnalysisResult:
        """
        Complete analysis loop with manual metrics and state desync fixes.
        
        Args:
            pgn_source: PGN file path, PGN string, or StringIO object
            min_diagram_spacing: Minimum ply distance between critical position diagrams
            top_n_swings: Number of "biggest swing" positions to always include (default: 2)
        """
        # --- 1. Fix NameError: Initialize PGN Source ---
        if isinstance(pgn_source, str):
            if '\n' in pgn_source or pgn_source.startswith('['):
                pgn_io = io.StringIO(pgn_source)
            else:
                pgn_io = open(pgn_source, 'r')
        else:
            pgn_io = pgn_source
        
        try:
            game = chess.pgn.read_game(pgn_io)
            if not game:
                raise ValueError("Could not parse PGN")
            
            board = game.board()
            moves_analysis = []
            brilliant_sacrifices = []
            critical_positions = []
            
            # Collect all eval swings for top-N selection
            all_eval_swings = []
            
            start_time = time.time()
            prev_material = self._calculate_material(board)
            last_diagram_ply = -100
            
            # Initial evaluation
            info_init = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            prev_eval = self._eval_to_cp(info_init['score'])
            
            # --- 2. Main Move Loop ---
            for node in game.mainline():
                ply = board.ply() + 1
                is_white_move = board.turn == chess.WHITE
                
                # A. Analyze BEFORE push to get Best Move and alternatives (multipv=3)
                info_before_list = self.engine.analyse(
                    board, 
                    chess.engine.Limit(depth=self.depth),
                    multipv=3
                )
                # Handle both single dict (multipv=1) and list (multipv>1) returns
                if isinstance(info_before_list, dict):
                    info_before_list = [info_before_list]
                
                info_before = info_before_list[0]  # Best line
                best_move = info_before.get('pv', [None])[0]
                
                # Capture SAN strings while it's still the moving player's turn
                played_san = board.san(node.move)
                best_san = board.san(best_move) if best_move else "-"
                is_capture = board.is_capture(node.move)
                best_eval = self._eval_to_cp(info_before['score'])
                
                # Extract playable alternative moves (within PLAYABLE_THRESHOLD of best)
                alternative_moves = []
                for alt_info in info_before_list[1:]:  # Skip the best move (index 0)
                    alt_move = alt_info.get('pv', [None])[0]
                    if alt_move and alt_move in board.legal_moves:
                        alt_eval = self._eval_to_cp(alt_info['score'])
                        eval_diff = abs(best_eval - alt_eval)
                        if eval_diff <= PLAYABLE_THRESHOLD:
                            alt_san = board.san(alt_move)
                            alternative_moves.append((alt_san, alt_eval))
                
                # B. Execute the move
                move = node.move
                board.push(move)
                
                # C. Analyze AFTER push
                info_after = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
                current_eval = self._eval_to_cp(info_after['score'])
                current_material = self._calculate_material(board)
                
                # D. Calculate eval_loss properly
                # The key insight: eval_loss should measure how much WORSE the played move
                # is compared to the best move, from the perspective of the moving player.
                #
                # best_eval: evaluation if the best move was played (from White's perspective)
                # current_eval: evaluation after the actual move (from White's perspective)
                #
                # For White: a good move increases eval, so loss = best_eval - current_eval
                # For Black: a good move decreases eval, so loss = current_eval - best_eval
                #
                # We want loss >= 0 for bad moves, so:
                if is_white_move:
                    # White wants higher eval; if current < best, that's bad
                    raw_eval_loss = best_eval - current_eval
                else:
                    # Black wants lower eval; if current > best, that's bad
                    raw_eval_loss = current_eval - best_eval
                
                # Clamp negative values (move was better than engine's "best" - can happen 
                # due to search instability or horizon effects)
                raw_eval_loss = max(0, raw_eval_loss)
                
                # Cap eval_loss to avoid absurd values from mate score transitions
                # When positions swing between "mate" and "no mate", raw differences
                # can be 8000+ cp which distorts accuracy calculations.
                eval_loss = min(raw_eval_loss, MAX_EVAL_LOSS_FOR_ACCURACY)
                
                classification = self._classify_move(eval_loss, ply)
                
                # D. Fix AssertionError: Safely generate PV SAN line using a temp board
                temp_board = board.copy()
                pv_san = []
                for pv_move in info_after.get('pv', [])[:3]:
                    if pv_move in temp_board.legal_moves:
                        pv_san.append(temp_board.san(pv_move))
                        temp_board.push(pv_move)
                    else:
                        break
                
                # E. Calculate Manual Positional Metrics
                pos_eval = self._get_positional_eval(board)
                
                # F. Record Move Analysis
                move_analysis = EnhancedMoveAnalysis(
                    ply=ply, move_san=played_san, move_uci=move.uci(),
                    is_white_move=is_white_move, eval_before=prev_eval, eval_after=current_eval,
                    best_move_san=best_san, best_move_uci=best_move.uci() if best_move else "-",
                    best_eval=best_eval, eval_loss=eval_loss, classification=classification,
                    is_capture=is_capture, is_check=board.is_check(),
                    material_balance=current_material, fen_after=board.fen(),
                    pv_line=pv_san, positional_eval=pos_eval,
                    alternative_moves=alternative_moves
                )
                moves_analysis.append(move_analysis)
                
                # G. Detect Brilliant Sacrifices
                mat_diff = prev_material - current_material if is_white_move else current_material - prev_material
                if mat_diff >= 250:
                    eval_diff = current_eval - prev_eval if is_white_move else prev_eval - current_eval
                    if eval_diff >= -30:
                        brilliant_sacrifices.append(BrilliantSacrifice(
                            ply=ply, move_san=played_san, player="White" if is_white_move else "Black",
                            piece_type=self._classify_sacrifice_type(mat_diff),
                            material_lost=abs(mat_diff), eval_before=prev_eval, eval_after=current_eval,
                            eval_improvement=max(0, eval_diff), is_sound=eval_diff >= 0
                        ))

                # H. Detect Critical Positions (threshold-based)
                eval_swing = abs(current_eval - prev_eval)
                if eval_swing > 100 and ply - last_diagram_ply >= min_diagram_spacing:
                    critical_positions.append(CriticalPosition(
                        ply=ply, fen=board.fen(), move_san=played_san, eval_score=current_eval,
                        reason=self._get_critical_reason(prev_eval, current_eval, classification, is_white_move),
                        best_continuation=pv_san, positional_eval=pos_eval,
                        is_biggest_swing=False, eval_swing=eval_swing, alternative_moves=alternative_moves,
                        best_move_san=best_san
                    ))
                    last_diagram_ply = ply
                
                # I. Collect all eval swings for top-N selection (excluding already-added positions)
                all_eval_swings.append({
                    'ply': ply,
                    'fen': board.fen(),
                    'move_san': played_san,
                    'eval_score': current_eval,
                    'prev_eval': prev_eval,
                    'classification': classification,
                    'is_white_move': is_white_move,
                    'pv_san': pv_san,
                    'pos_eval': pos_eval,
                    'eval_swing': eval_swing,
                    'alternative_moves': alternative_moves,  # Include alternatives for biggest swing positions
                    'best_move_san': best_san  # Include best move for biggest swing positions
                })
                
                prev_eval = current_eval
                prev_material = current_material

            # --- 3. Add Top-N Biggest Swings ---
            # Get plies already in critical_positions
            existing_plies = {cp.ply for cp in critical_positions}
            
            # Sort all swings by magnitude (descending)
            all_eval_swings.sort(key=lambda x: x['eval_swing'], reverse=True)
            
            # Select top-N swings that aren't already included and respect spacing
            selected_plies = list(existing_plies)
            biggest_swing_positions = []
            
            for swing_data in all_eval_swings:
                if swing_data['ply'] in existing_plies:
                    continue  # Already a critical position
                
                # Check minimum spacing from all selected positions
                if all(abs(swing_data['ply'] - p) >= min_diagram_spacing for p in selected_plies):
                    reason = f"Biggest evaluation swing(s) ({swing_data['eval_swing']:.0f}cp)"
                    biggest_swing_positions.append(CriticalPosition(
                        ply=swing_data['ply'],
                        fen=swing_data['fen'],
                        move_san=swing_data['move_san'],
                        eval_score=swing_data['eval_score'],
                        reason=reason,
                        best_continuation=swing_data['pv_san'],
                        positional_eval=swing_data['pos_eval'],
                        is_biggest_swing=True,
                        eval_swing=swing_data['eval_swing'],
                        alternative_moves=swing_data.get('alternative_moves', []),
                        best_move_san=swing_data.get('best_move_san', '')
                    ))
                    selected_plies.append(swing_data['ply'])
                    
                    if len(biggest_swing_positions) >= top_n_swings:
                        break
            
            # Merge and sort all critical positions by ply
            critical_positions.extend(biggest_swing_positions)
            critical_positions.sort(key=lambda cp: cp.ply)

            # --- 4. Wrap Results ---
            white_moves = [m for m in moves_analysis if m.is_white_move]
            black_moves = [m for m in moves_analysis if not m.is_white_move]

            return EnhancedGameAnalysisResult(
                white=game.headers.get("White", "Unknown"),
                black=game.headers.get("Black", "Unknown"),
                white_elo=parse_elo(game.headers.get("WhiteElo", "")),
                black_elo=parse_elo(game.headers.get("BlackElo", "")),
                result=game.headers.get("Result", "*"),
                date=game.headers.get("Date", "????.??.??"),
                event=game.headers.get("Event", "Unknown"),
                site=game.headers.get("Site", "Unknown"),
                round_num=game.headers.get("Round", "?"),
                opening_eco=game.headers.get("ECO", "???"),
                opening_name=game.headers.get("Opening", "Unknown"),
                moves=moves_analysis,
                brilliant_sacrifices=brilliant_sacrifices,
                critical_positions=critical_positions,
                white_stats=self._calculate_player_stats(white_moves),
                black_stats=self._calculate_player_stats(black_moves),
                positional_summary=self._compute_positional_summary(moves_analysis),
                game_character=classify_game_character(moves_analysis),
                analysis_depth=self.depth,
                analysis_time=time.time() - start_time,
                engine_version=self.engine_version
            )
            
        finally:
            if isinstance(pgn_source, str) and not ('\n' in pgn_source or pgn_source.startswith('[')):
                pgn_io.close()

    def analyze_all_games(self, pgn_source: Union[str, io.StringIO],
                          min_diagram_spacing: int = 6,
                          top_n_swings: int = 2,
                          verbose: bool = True) -> List[EnhancedGameAnalysisResult]:
        """
        Analyze all games in a PGN file.
        
        Args:
            pgn_source: PGN file path, PGN string, or StringIO object
            min_diagram_spacing: Minimum ply distance between critical position diagrams
            top_n_swings: Number of "biggest swing" positions to always include
            verbose: Print progress messages
            
        Returns:
            List of EnhancedGameAnalysisResult objects, one per game
        """
        results = []
        
        # Open PGN source
        if isinstance(pgn_source, str):
            if '\n' in pgn_source or pgn_source.startswith('['):
                pgn_io = io.StringIO(pgn_source)
            else:
                pgn_io = open(pgn_source, 'r')
        else:
            pgn_io = pgn_source
        
        try:
            game_num = 0
            while True:
                game = chess.pgn.read_game(pgn_io)
                if game is None:
                    break
                
                game_num += 1
                if verbose:
                    white = game.headers.get("White", "Unknown")
                    black = game.headers.get("Black", "Unknown")
                    print(f"Analyzing game {game_num}: {white} vs {black}...")
                
                # Analyze this game using a StringIO of the game
                game_pgn = io.StringIO()
                exporter = chess.pgn.StringExporter()
                game_pgn.write(game.accept(exporter))
                game_pgn.seek(0)
                
                result = self.analyze_game(
                    game_pgn,
                    min_diagram_spacing=min_diagram_spacing,
                    top_n_swings=top_n_swings
                )
                results.append(result)
                
                if verbose:
                    print(f"  Completed in {result.analysis_time:.1f}s")
            
            if verbose:
                print(f"Analyzed {len(results)} game(s) total.")
                
        finally:
            if isinstance(pgn_source, str) and not ('\n' in pgn_source or pgn_source.startswith('[')):
                pgn_io.close()
        
        return results

    def _calculate_material(self, board: chess.Board) -> int:
        """Calculates the material balance (positive = White ahead)."""
        material = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            material += (white_count - black_count) * PIECE_VALUES[piece_type]
        return material

    def _classify_move(self, eval_loss: float, ply: int) -> str:
        """Classifies a move based on centipawn loss."""
        if ply <= 12 and eval_loss < 30:
            return "book"
        if eval_loss < 5:
            return "best"
        elif eval_loss < 15:
            return "excellent"
        elif eval_loss < 30:
            return "good"
        elif eval_loss < 60:
            return "inaccuracy"
        elif eval_loss < 120:
            return "mistake"
        else:
            return "blunder"

    def _classify_sacrifice_type(self, material: int) -> str:
        """Identifies what type of material was sacrificed."""
        if material >= 850: return "queen"
        elif material >= 450: return "rook"
        elif material >= 300: return "minor_piece"
        elif material >= 200: return "exchange"
        else: return "pawns"

    def _get_critical_reason(self, prev_eval: float, curr_eval: float,
                             classification: str, is_white: bool) -> str:
        """Generates a reason why a position is marked as critical."""
        swing = curr_eval - prev_eval
        if abs(curr_eval) >= 9900:
            return "Checkmate! " + ("White wins" if curr_eval > 0 else "Black wins")
        if classification == "blunder":
            player = "White" if is_white else "Black"
            return f"{player} blunders ({swing:+.0f}cp)"
        elif abs(swing) > 200:
            return "Position swings to " + ("White's" if swing > 0 else "Black's") + " favor"
        return "Critical moment"

    def _calculate_player_stats(self, moves: List[EnhancedMoveAnalysis]) -> Dict:
        """Calculates performance statistics for a specific player."""
        if not moves:
            return {
                'total_moves': 0,
                'accuracy': 0,
                'avg_centipawn_loss': 0,
                'best_moves': 0,
                'excellent_moves': 0,
                'good_moves': 0,
                'inaccuracies': 0,
                'mistakes': 0,
                'blunders': 0
            }
        
        import math
        avg_loss = sum(m.eval_loss for m in moves) / len(moves)
        accuracy = max(0, 100 * math.exp(-0.005 * avg_loss)) if avg_loss > 0 else 100.0
        
        counts = {'best': 0, 'excellent': 0, 'good': 0, 'book': 0, 'inaccuracy': 0, 'mistake': 0, 'blunder': 0}
        for m in moves:
            if m.classification in counts: counts[m.classification] += 1
            
        return {
            'total_moves': len(moves),
            'avg_centipawn_loss': avg_loss,
            'accuracy': accuracy,
            'best_moves': counts['best'],
            'excellent_moves': counts['excellent'],
            'good_moves': counts['good'] + counts['book'],
            'inaccuracies': counts['inaccuracy'],
            'mistakes': counts['mistake'],
            'blunders': counts['blunder']
        }

    def _compute_positional_summary(self, moves: List[EnhancedMoveAnalysis]) -> Dict:
        """
        Aggregates positional metrics and calculates the 'advantage' keys 
        required by the LaTeX generator.
        """
        summary = {
            'space': {'white': [], 'black': [], 'advantage': []},
            'mobility': {'white': [], 'black': [], 'advantage': []},
            'king_safety': {'white': [], 'black': []},
            'threats': {'white': [], 'black': [], 'advantage': []}
        }
        
        for m in moves:
            if m.positional_eval:
                pe = m.positional_eval
                # Space
                summary['space']['white'].append(pe.space_white)
                summary['space']['black'].append(pe.space_black)
                summary['space']['advantage'].append(pe.space_advantage)
                
                # Mobility
                summary['mobility']['white'].append(pe.mobility_white)
                summary['mobility']['black'].append(pe.mobility_black)
                summary['mobility']['advantage'].append(pe.mobility_advantage)
                
                # King Safety
                summary['king_safety']['white'].append(pe.king_safety_white)
                summary['king_safety']['black'].append(pe.king_safety_black)
                
                # Threats
                summary['threats']['white'].append(pe.threats_white)
                summary['threats']['black'].append(pe.threats_black)
                summary['threats']['advantage'].append(pe.threats_advantage)
        
        def get_stats(vals):
            """Helper to calculate min/max/avg."""
            if not vals:
                return {'avg': 0.0, 'min': 0.0, 'max': 0.0}
            return {
                'avg': sum(vals) / len(vals),
                'min': min(vals),
                'max': max(vals)
            }
        
        return {
            'space': {
                'white': get_stats(summary['space']['white']),
                'black': get_stats(summary['space']['black']),
                'advantage': get_stats(summary['space']['advantage']) # This fixes the KeyError
            },
            'mobility': {
                'white': get_stats(summary['mobility']['white']),
                'black': get_stats(summary['mobility']['black']),
                'advantage': get_stats(summary['mobility']['advantage']) # This fixes the KeyError
            },
            'king_safety': {
                'white': get_stats(summary['king_safety']['white']),
                'black': get_stats(summary['king_safety']['black'])
            },
            'threats': {
                'white': get_stats(summary['threats']['white']),
                'black': get_stats(summary['threats']['black']),
                'advantage': get_stats(summary['threats']['advantage'])
            }
        }

# =============================================================================
# ENHANCED LATEX REPORT GENERATOR
# =============================================================================

class EnhancedLaTeXReportGenerator:
    """Generates LaTeX reports with positional evaluation data."""
    
    @staticmethod
    def generate_raw_data_block(analysis: 'EnhancedGameAnalysisResult', game_num: int = None) -> str:
        """
        Generate a machine-readable data block to append after \\end{document}.
        
        This data is ignored by LaTeX but can be parsed by analysis scripts.
        Contains move-by-move positional metrics for computational analysis.
        
        Args:
            analysis: EnhancedGameAnalysisResult containing moves with positional data
            game_num: Optional game number for multi-game books
            
        Returns:
            String containing commented raw data in CSV-like format
        """
        lines = []
        
        # Header
        lines.append("")
        lines.append("% " + "=" * 70)
        if game_num is not None:
            lines.append(f"% RAW POSITIONAL DATA - GAME {game_num}")
        else:
            lines.append("% RAW POSITIONAL DATA FOR COMPUTATIONAL ANALYSIS")
        lines.append("% " + "=" * 70)
        lines.append("%")
        lines.append(f"% White: {analysis.white}")
        lines.append(f"% Black: {analysis.black}")
        lines.append(f"% Event: {analysis.event}")
        lines.append(f"% Date: {analysis.date}")
        lines.append(f"% Result: {analysis.result}")
        lines.append("%")
        lines.append("% Format: ply, SAN, eval_cp, space_w, space_b, mob_w, mob_b, ks_w, ks_b, threats_w, threats_b")
        lines.append("% Units: eval in centipawns (from White's perspective), others in Stockfish classical eval units")
        lines.append("%")
        lines.append("% GAME_DATA_START")
        
        for move in analysis.moves:
            pe = move.positional_eval
            
            # Handle None values and format the data
            eval_cp = int(move.eval_after) if move.eval_after is not None else "None"
            
            if pe is not None:
                space_w = f"{pe.space_white:.3f}"
                space_b = f"{pe.space_black:.3f}"
                mob_w = f"{pe.mobility_white_mg:.3f}"
                mob_b = f"{pe.mobility_black_mg:.3f}"
                ks_w = f"{pe.king_safety_white:.3f}"
                ks_b = f"{pe.king_safety_black:.3f}"
                threats_w = f"{pe.threats_white:.3f}"
                threats_b = f"{pe.threats_black:.3f}"
            else:
                space_w = space_b = mob_w = mob_b = "None"
                ks_w = ks_b = threats_w = threats_b = "None"
            
            # Escape any commas in SAN (shouldn't happen, but be safe)
            san = move.move_san.replace(",", ";")
            
            line = f"% {move.ply}, {san}, {eval_cp}, {space_w}, {space_b}, {mob_w}, {mob_b}, {ks_w}, {ks_b}, {threats_w}, {threats_b}"
            lines.append(line)
        
        lines.append("% GAME_DATA_END")
        lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def _escape_latex(text: str) -> str:
        """Escape special LaTeX characters."""
        if not text:
            return ""
        text = text.replace('#', r'\#')
        replacements = {
            '&': r'\&', '%': r'\%', '$': r'\$', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\^{}'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    @staticmethod
    def generate_report(analysis: EnhancedGameAnalysisResult,
                       include_diagrams: bool = True,
                       include_positional: bool = True,
                       include_methodology: bool = True,
                       include_plots: bool = True,
                       include_ascii_plots: bool = False,
                       include_prediction: bool = False,
                       prediction_player_color: str = None,
                       prediction_player_name: str = None,
                       plot_output_dir: str = None) -> str:
        """
        Generate a complete LaTeX report with positional analysis.
        
        Args:
            analysis: EnhancedGameAnalysisResult from analyzer
            include_diagrams: Include chess board diagrams
            include_positional: Include positional evaluation section
            include_methodology: Include explanation of how metrics are computed
            include_plots: Include matplotlib plots (requires matplotlib)
            include_ascii_plots: Include ASCII plots in verbatim environment
            include_prediction: Include Fireteam Index prediction section
            prediction_player_color: "W" or "B" - which player to analyze (required if include_prediction=True)
            prediction_player_name: Optional display name (e.g., "Berliner")
            plot_output_dir: Directory to save plot files (defaults to current dir)
            
        Returns:
            Complete LaTeX document as string
        """
        esc = EnhancedLaTeXReportGenerator._escape_latex
        lines = []
        
        # Set plot output directory
        if plot_output_dir is None:
            plot_output_dir = "."
        
        # Document preamble
        lines.extend([
            r"\documentclass[11pt]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{xskak}",
            r"\usepackage{amssymb}",
            r"\usepackage{chessboard}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{longtable}",
            r"\usepackage{booktabs}",
            r"\usepackage{hyperref}",
            r"\usepackage{xcolor}",
            r"\usepackage{pgfplots}",
            r"\usepackage{graphicx}",
            r"\pgfplotsset{compat=1.16}",
            r"",
            r"% Custom colors",
            r"\definecolor{brilliantcolor}{RGB}{0, 150, 150}",
            r"\definecolor{excellentcolor}{RGB}{0, 128, 0}",
            r"\definecolor{goodcolor}{RGB}{64, 160, 64}",
            r"\definecolor{inaccuracycolor}{RGB}{200, 180, 0}",
            r"\definecolor{mistakecolor}{RGB}{220, 120, 0}",
            r"\definecolor{blundercolor}{RGB}{200, 0, 0}",
            r"\definecolor{spacecolor}{RGB}{70, 130, 180}",
            r"\definecolor{mobilitycolor}{RGB}{60, 179, 113}",
            r"",
            r"\title{Chess Game Analysis with Positional Metrics\\",
            rf"{esc(analysis.white)} vs {esc(analysis.black)}}}",
            r"\author{Generated by " + esc(analysis.engine_version) + "}",
            r"\date{" + esc(analysis.date) + "}",
            r"",
            r"\begin{document}",
            r"\maketitle",
            r"\tableofcontents",
            r"\newpage",
            r"",
        ])
        
        # Game Information
        lines.extend([
            r"\section{Game Information}",
            r"\begin{tabular}{ll}",
            rf"\textbf{{Event}} & {esc(analysis.event)} \\",
            rf"\textbf{{Site}} & {esc(analysis.site)} \\",
            rf"\textbf{{Date}} & {esc(analysis.date)} \\",
            rf"\textbf{{Round}} & {esc(analysis.round_num)} \\",
        ])
        
        white_info = esc(analysis.white)
        if analysis.white_elo:
            white_info += f" ({analysis.white_elo})"
        black_info = esc(analysis.black)
        if analysis.black_elo:
            black_info += f" ({analysis.black_elo})"
        
        lines.extend([
            rf"\textbf{{White}} & {white_info} \\",
            rf"\textbf{{Black}} & {black_info} \\",
            rf"\textbf{{Result}} & {esc(analysis.result)} \\",
            rf"\textbf{{Opening}} & {esc(analysis.opening_eco)} -- {esc(analysis.opening_name)} \\",
            r"\end{tabular}",
            r"",
        ])
        
        # Player Statistics with positional metrics
        lines.extend([
            r"\section{Player Statistics}",
            r"",
            r"\subsection{" + esc(analysis.white) + " (White)}",
            r"\begin{itemize}",
            rf"\item Total moves: {analysis.white_stats['total_moves']}",
            rf"\item Accuracy: {analysis.white_stats['accuracy']:.1f}\%",
            rf"\item Average centipawn loss: {analysis.white_stats['avg_centipawn_loss']:.1f}",
            rf"\item Best/Excellent moves: {analysis.white_stats['best_moves']} / {analysis.white_stats['excellent_moves']}",
            rf"\item Good moves: {analysis.white_stats['good_moves']}",
            rf"\item Inaccuracies: {analysis.white_stats['inaccuracies']}",
            rf"\item Mistakes: {analysis.white_stats['mistakes']}",
            rf"\item Blunders: {analysis.white_stats['blunders']}",
        ])
        
        if analysis.white_stats.get('avg_space', 0) > 0:
            lines.append(rf"\item Average space control: {analysis.white_stats['avg_space']:.2f}")
        if analysis.white_stats.get('avg_mobility', 0) != 0:
            lines.append(rf"\item Average mobility: {analysis.white_stats['avg_mobility']:.2f}")
        
        lines.extend([
            r"\end{itemize}",
            r"",
            r"\subsection{" + esc(analysis.black) + " (Black)}",
            r"\begin{itemize}",
            rf"\item Total moves: {analysis.black_stats['total_moves']}",
            rf"\item Accuracy: {analysis.black_stats['accuracy']:.1f}\%",
            rf"\item Average centipawn loss: {analysis.black_stats['avg_centipawn_loss']:.1f}",
            rf"\item Best/Excellent moves: {analysis.black_stats['best_moves']} / {analysis.black_stats['excellent_moves']}",
            rf"\item Good moves: {analysis.black_stats['good_moves']}",
            rf"\item Inaccuracies: {analysis.black_stats['inaccuracies']}",
            rf"\item Mistakes: {analysis.black_stats['mistakes']}",
            rf"\item Blunders: {analysis.black_stats['blunders']}",
        ])
        
        if analysis.black_stats.get('avg_space', 0) > 0:
            lines.append(rf"\item Average space control: {analysis.black_stats['avg_space']:.2f}")
        if analysis.black_stats.get('avg_mobility', 0) != 0:
            lines.append(rf"\item Average mobility: {analysis.black_stats['avg_mobility']:.2f}")
        
        lines.extend([
            r"\end{itemize}",
            r"",
        ])
        
        # Positional Analysis Section with Plots
        if include_positional and analysis.positional_summary:
            lines.extend(EnhancedLaTeXReportGenerator._generate_positional_section(
                analysis,
                include_plots=include_plots,
                include_ascii_plots=include_ascii_plots,
                plot_output_dir=plot_output_dir
            ))
        
        # Methodology Section
        if include_methodology:
            lines.extend(EnhancedLaTeXReportGenerator._generate_methodology_section())
        
        # Brilliant Sacrifices
        if analysis.brilliant_sacrifices:
            lines.extend([
                r"\section{Brilliant Sacrifices}",
                r"",
                rf"This game features {len(analysis.brilliant_sacrifices)} brilliant sacrifice(s):",
                r"",
                r"\begin{itemize}",
            ])
            
            for sac in analysis.brilliant_sacrifices:
                move_num = (sac.ply + 1) // 2
                move_str = f"{move_num}. {sac.move_san}" if sac.player == "White" else f"{move_num}...{sac.move_san}"
                sound_str = "Sound sacrifice" if sac.is_sound else "Speculative sacrifice"
                
                lines.append(
                    rf"\item \textbf{{{esc(move_str)}}} -- {sac.player} sacrifices {sac.piece_type} "
                    rf"({sac.material_lost}cp). {sound_str}. "
                    rf"Evaluation: {sac.eval_before/100:+.2f} $\rightarrow$ {sac.eval_after/100:+.2f}"
                )
            
            lines.extend([r"\end{itemize}", r""])
        
        # Annotated Game
        lines.extend([
            r"\section{Annotated Game}",
            r"",
            r"\begin{quote}",
        ])
        
        current_line = ""
        for move in analysis.moves:
            move_num = (move.ply + 1) // 2
            
            if move.is_white_move:
                if current_line:
                    lines.append(current_line)
                current_line = f"{move_num}. {esc(move.move_san)}"
            else:
                if current_line:
                    current_line += f" {esc(move.move_san)}"
                else:
                    current_line = f"{move_num}...{esc(move.move_san)}"
            
            # NAG annotations
            if move.classification == "blunder":
                current_line += "??"
            elif move.classification == "mistake":
                current_line += "?"
            elif move.classification == "inaccuracy":
                current_line += "?!"
            elif move.classification == "best" and move.move_san == move.best_move_san:
                current_line += "!"
            
            # Comments for significant moves
            if move.classification in ["blunder", "mistake"]:
                lines.append(current_line)
                player = "White" if move.is_white_move else "Black"
                # Don't show alternatives when the played move equals the best move
                if move.move_san == move.best_move_san or move.move_uci == move.best_move_uci:
                    # Edge case: move classified as error but matches best move
                    comment = f"{player} loses {move.eval_loss:.0f}cp."
                else:
                    # Build list of moves to consider: best move + playable alternatives
                    moves_to_consider = [esc(move.best_move_san)]
                    for alt_san, alt_eval in move.alternative_moves:
                        if alt_san != move.best_move_san:  # Don't duplicate best move
                            moves_to_consider.append(esc(alt_san))
                    
                    if len(moves_to_consider) == 1:
                        comment = f"{player} loses {move.eval_loss:.0f}cp. Consider instead: {moves_to_consider[0]}."
                    else:
                        comment = f"{player} loses {move.eval_loss:.0f}cp. Consider instead: {', '.join(moves_to_consider)}."
                lines.append(rf"\textit{{{comment}}}")
                lines.append("")
                current_line = ""
        
        if current_line:
            lines.append(current_line)
        
        lines.extend([
            r"",
            esc(analysis.result),
            r"\end{quote}",
            r"",
        ])
        
        # Critical Positions with Positional Data
        if include_diagrams and analysis.critical_positions:
            lines.extend([
                r"\section{Critical Positions}",
                r"",
                r"This section highlights critical moments where the evaluation shifted substantially---either due to errors or missed opportunities. Each diagram shows the position \emph{after} the move was played.",
                r"",
                r"\begin{itemize}",
                r"\item \textbf{Instead of [move]}: The move(s) that should have been played instead.",
                r"\item \textbf{Best continuation}: The optimal sequence of moves from the diagrammed position.",
                r"\end{itemize}",
                r"",
                r"Positions marked with {\color{blue}$\bigstar$} represent the biggest evaluation swings in the game. The positional metrics table summarizes the spatial control, piece activity, king safety, and tactical threats for each side.",
                r"",
            ])
            
            for i, pos in enumerate(analysis.critical_positions, 1):
                move_num = (pos.ply + 1) // 2
                is_white = pos.ply % 2 == 1
                move_str = f"{move_num}. {pos.move_san}" if is_white else f"{move_num}...{pos.move_san}"
                
                # Special formatting for "biggest swing" positions
                if pos.is_biggest_swing:
                    subsection_title = rf"\subsection*{{\textcolor{{blue}}{{Position {i}: After {esc(move_str)} $\bigstar$}}}}"
                    reason_text = rf"\textit{{\textcolor{{blue}}{{{esc(pos.reason)}}}}}"
                else:
                    subsection_title = rf"\subsection*{{Position {i}: After {esc(move_str)}}}"
                    reason_text = rf"\textit{{{esc(pos.reason)}}}"
                
                lines.extend([
                    subsection_title,
                    reason_text,
                    r"",
                    rf"Evaluation: {pos.eval_score/100:+.2f}",
                    r"",
                    r"\chessboard[setfen=" + pos.fen + "]",
                    r"",
                ])
                
                # Add positional metrics for critical positions
                if pos.positional_eval:
                    pe = pos.positional_eval
                    lines.extend([
                        r"\begin{small}",
                        r"\begin{tabular}{lcc}",
                        r"\toprule",
                        r"Metric & White & Black \\",
                        r"\midrule",
                        rf"Space & {pe.space_white:.2f} & {pe.space_black:.2f} \\",
                        rf"Mobility (MG) & {pe.mobility_white_mg:.2f} & {pe.mobility_black_mg:.2f} \\",
                        rf"King Safety & {pe.king_safety_white:.2f} & {pe.king_safety_black:.2f} \\",
                        rf"Threats & {pe.threats_white:.2f} & {pe.threats_black:.2f} \\",
                        r"\bottomrule",
                        r"\end{tabular}",
                        r"\end{small}",
                        r"",
                    ])
                
                # Add "Instead of" moves: best move + alternatives (if different from played move)
                instead_moves = []
                if pos.best_move_san and pos.best_move_san != pos.move_san:
                    instead_moves.append(esc(pos.best_move_san))
                for alt_san, alt_eval in pos.alternative_moves[:2]:  # Add up to 2 more alternatives
                    if alt_san != pos.move_san and esc(alt_san) not in instead_moves:
                        instead_moves.append(esc(alt_san))
                if instead_moves:
                    instead_str = ", ".join(instead_moves[:3])  # Cap at 3 total
                    lines.append(rf"Instead of {esc(pos.move_san)}: \hspace{{0.5em}} {instead_str}\hspace{{4em}}")
                
                if pos.best_continuation:
                    cont_str = " ".join(esc(m) for m in pos.best_continuation[:5])
                    lines.append(rf"\hspace{{0.5em}}Best continuation: {cont_str}")
                
                lines.append(r"")
        
        # Fireteam Index Prediction Section (if requested)
        if include_prediction and prediction_player_color:
            prediction_lines = EnhancedLaTeXReportGenerator._generate_prediction_section(
                analysis,
                player_color=prediction_player_color,
                player_name=prediction_player_name,
                include_plots=include_plots,
                plot_output_dir=plot_output_dir
            )
            lines.extend(prediction_lines)
        
        # Analysis Metadata
        lines.extend([
            r"\section{Analysis Information}",
            r"\begin{itemize}",
            rf"\item Engine: {esc(analysis.engine_version)}",
            rf"\item Depth: {analysis.analysis_depth}",
            rf"\item Analysis time: {analysis.analysis_time:.1f} seconds",
            r"\end{itemize}",
            r"",
            r"\end{document}",
        ])
        
        # Append raw positional data after \end{document}
        # This is ignored by LaTeX but can be parsed by analysis scripts
        raw_data_block = EnhancedLaTeXReportGenerator.generate_raw_data_block(analysis)
        lines.append(raw_data_block)
        
        return "\n".join(lines)
    
    @staticmethod
    def _generate_positional_section(analysis: EnhancedGameAnalysisResult,
                                     include_plots: bool = True,
                                     include_ascii_plots: bool = False,
                                     plot_output_dir: str = ".") -> List[str]:
        """Generate the positional analysis section with optional plots."""
        lines = [
            r"\section{Positional Analysis}",
            r"",
            r"This section shows how key positional factors evolved throughout the game.",
            r"",
        ]
        
        ps = analysis.positional_summary
        
        # Generate plots if requested
        plot_files = {}
        if include_plots and PLOTTING_AVAILABLE and is_matplotlib_available() and analysis.moves:
            for metric in ['eval', 'space', 'mobility', 'king_safety', 'threats']:
                filename = f"plot_{metric}.pdf"
                filepath = os.path.join(plot_output_dir, filename)
                success = ChessPlotter.generate_matplotlib_plot(
                    analysis.moves, metric, filepath
                )
                if success:
                    plot_files[metric] = filepath
        
        # Evaluation plot section
        lines.extend([
            r"\subsection{Position Evaluation Over Time}",
            r"",
            r"The evaluation graph shows how the position assessment changed throughout the game. "
            r"Positive values favor White; negative values favor Black.",
            r"",
        ])
        
        # Add game character classification if available
        gc = analysis.game_character
        if gc:
            m1 = gc.get('m1', 0)
            m2 = gc.get('m2', 0)
            spread = gc.get('spread', 0)
            spread_class = gc.get('spread_class', 'unknown')
            direction_class = gc.get('direction_class', 'normal')
            description = gc.get('combined_description', '')
            
            lines.extend([
                r"\paragraph{Game Character Classification.}",
                rf"Evaluation range: minimum $m_1 = {m1:+.2f}$, maximum $m_2 = {m2:+.2f}$, "
                rf"spread $d = m_2 - m_1 = {spread:.2f}$.",
                r"",
                rf"This game is classified as \textbf{{{spread_class}}} "
                rf"({direction_class}). {description}",
                r"",
            ])
        
        if 'eval' in plot_files:
            lines.extend([
                r"\begin{center}",
                rf"\includegraphics[width=0.9\textwidth]{{{plot_files['eval']}}}",
                r"\end{center}",
                r"",
            ])
        
        if include_ascii_plots and analysis.moves and PLOTTING_AVAILABLE:
            ascii_plot = ChessPlotter.generate_ascii_plot(analysis.moves, 'eval')
            lines.extend([
                r"\begin{verbatim}",
                ascii_plot,
                r"\end{verbatim}",
                r"",
            ])
        
        # Space analysis
        if ps.get('space'):
            space = ps['space']
            lines.extend([
                r"\subsection{Space Control}",
                r"",
                r"Space measures control of territory, particularly in the center and enemy's half of the board.",
                r"",
                r"\begin{itemize}",
                rf"\item White average: {space['white']['avg']:.2f} (range: {space['white']['min']:.2f} to {space['white']['max']:.2f})",
                rf"\item Black average: {space['black']['avg']:.2f} (range: {space['black']['min']:.2f} to {space['black']['max']:.2f})",
                rf"\item Average advantage: {space['advantage']['avg']:.2f} " + 
                ("(White had more space)" if space['advantage']['avg'] > 0 else "(Black had more space)"),
                r"\end{itemize}",
                r"",
            ])
            
            if 'space' in plot_files:
                lines.extend([
                    r"\begin{center}",
                    rf"\includegraphics[width=0.9\textwidth]{{{plot_files['space']}}}",
                    r"\end{center}",
                    r"",
                ])
            
            if include_ascii_plots and analysis.moves and PLOTTING_AVAILABLE:
                ascii_plot = ChessPlotter.generate_ascii_plot(analysis.moves, 'space')
                lines.extend([
                    r"\begin{verbatim}",
                    ascii_plot,
                    r"\end{verbatim}",
                    r"",
                ])
        
        # Mobility analysis
        if ps.get('mobility'):
            mob = ps['mobility']
            lines.extend([
                r"\subsection{Mobility (Piece Activity)}",
                r"",
                r"Mobility counts the number of safe squares available to each piece, weighted by piece type.",
                r"",
                r"\begin{itemize}",
                rf"\item White average: {mob['white']['avg']:.2f} (range: {mob['white']['min']:.2f} to {mob['white']['max']:.2f})",
                rf"\item Black average: {mob['black']['avg']:.2f} (range: {mob['black']['min']:.2f} to {mob['black']['max']:.2f})",
                rf"\item Average advantage: {mob['advantage']['avg']:.2f} " +
                ("(White had more active pieces)" if mob['advantage']['avg'] > 0 else "(Black had more active pieces)"),
                r"\end{itemize}",
                r"",
            ])
            
            if 'mobility' in plot_files:
                lines.extend([
                    r"\begin{center}",
                    rf"\includegraphics[width=0.9\textwidth]{{{plot_files['mobility']}}}",
                    r"\end{center}",
                    r"",
                ])
            
            if include_ascii_plots and analysis.moves and PLOTTING_AVAILABLE:
                ascii_plot = ChessPlotter.generate_ascii_plot(analysis.moves, 'mobility')
                lines.extend([
                    r"\begin{verbatim}",
                    ascii_plot,
                    r"\end{verbatim}",
                    r"",
                ])
        
        # King safety analysis
        if ps.get('king_safety'):
            ks = ps['king_safety']
            lines.extend([
                r"\subsection{King Safety}",
                r"",
                r"King safety evaluates pawn shield strength, attacking pieces near the king, and available checks.",
                r"",
                r"\begin{itemize}",
                rf"\item White average: {ks['white']['avg']:.2f}",
                rf"\item Black average: {ks['black']['avg']:.2f}",
                r"\end{itemize}",
                r"",
            ])
            
            if 'king_safety' in plot_files:
                lines.extend([
                    r"\begin{center}",
                    rf"\includegraphics[width=0.9\textwidth]{{{plot_files['king_safety']}}}",
                    r"\end{center}",
                    r"",
                ])
            
            if include_ascii_plots and analysis.moves and PLOTTING_AVAILABLE:
                ascii_plot = ChessPlotter.generate_ascii_plot(analysis.moves, 'king_safety')
                lines.extend([
                    r"\begin{verbatim}",
                    ascii_plot,
                    r"\end{verbatim}",
                    r"",
                ])
        
        # Threats analysis
        if ps.get('threats'):
            th = ps['threats']
            lines.extend([
                r"\subsection{Threats}",
                r"",
                r"Threats evaluate tactical tension: hanging pieces, minor pieces attacking major pieces, "
                r"and attacks on squares near the enemy king. Higher values indicate more active threats.",
                r"",
                r"\begin{itemize}",
                rf"\item White average: {th['white']['avg']:.2f} (range: {th['white']['min']:.2f} to {th['white']['max']:.2f})",
                rf"\item Black average: {th['black']['avg']:.2f} (range: {th['black']['min']:.2f} to {th['black']['max']:.2f})",
            ])
            
            # Determine who maintained more threats on average
            if th['advantage']['avg'] > 0.1:
                lines.append(rf"\item Average advantage: {th['advantage']['avg']:.2f} (White maintained more threats)")
            elif th['advantage']['avg'] < -0.1:
                lines.append(rf"\item Average advantage: {th['advantage']['avg']:.2f} (Black maintained more threats)")
            else:
                lines.append(rf"\item Average advantage: {th['advantage']['avg']:.2f} (threats were balanced)")
            
            lines.extend([
                r"\end{itemize}",
                r"",
            ])
            
            # Generate threats plot if available
            if 'threats' in plot_files:
                lines.extend([
                    r"\begin{center}",
                    rf"\includegraphics[width=0.9\textwidth]{{{plot_files['threats']}}}",
                    r"\end{center}",
                    r"",
                ])
            
            if include_ascii_plots and analysis.moves and PLOTTING_AVAILABLE:
                ascii_plot = ChessPlotter.generate_ascii_plot(analysis.moves, 'threats')
                lines.extend([
                    r"\begin{verbatim}",
                    ascii_plot,
                    r"\end{verbatim}",
                    r"",
                ])
        
        return lines
    
    @staticmethod
    def _generate_prediction_section(
        analysis: EnhancedGameAnalysisResult,
        player_color: str,
        player_name: Optional[str] = None,
        include_plots: bool = True,
        plot_output_dir: str = "."
    ) -> List[str]:
        """
        Generate the Fireteam Index prediction section.
        
        This section shows:
        - The Fireteam Index formula and parameters
        - Predictions from both per-ply and windowed algorithms
        - Plots of FT over time (raw and smoothed)
        """
        esc = EnhancedLaTeXReportGenerator._escape_latex
        lines = []
        
        # Extract positional data from analysis
        data = compute_fireteam_index_for_analysis(analysis)
        
        if not data:
            lines.extend([
                r"\section{Fireteam Index Prediction}",
                r"",
                r"Insufficient positional data available for prediction analysis.",
                r"",
            ])
            return lines
        
        # Run both prediction algorithms
        result_per_ply = predict_outcome_per_ply(
            data, 
            player_color=player_color,
            player_name=player_name
        )
        result_windowed = predict_outcome_windowed(
            data,
            player_color=player_color,
            player_name=player_name
        )
        
        # Determine player display name
        if player_name:
            display_name = esc(player_name)
        elif player_color.upper() == "W":
            display_name = esc(analysis.white)
        else:
            display_name = esc(analysis.black)
        
        color_word = "White" if player_color.upper() == "W" else "Black"
        
        lines.extend([
            r"\section{Fireteam Index Prediction}",
            r"",
            r"The \textbf{Fireteam Index} is inspired by Rithmomachia's ``well-coordinated fireteam "
            r"in enemy territory'' victory condition. It combines four positional factors into a "
            r"single metric that predicts which side is likely to win.",
            r"",
            r"\subsection{The Fireteam Index Formula}",
            r"",
            r"\[",
            r"\text{FT} = \Delta\text{Space} + \Delta\text{Mobility} + \Delta\text{King Safety} "
            r"+ \frac{\Delta\text{Threats}}{10}",
            r"\]",
            r"",
            r"where each $\Delta$ represents (Player's value $-$ Opponent's value). "
            r"A positive FT indicates the player has assembled a well-coordinated position "
            r"with territorial control, piece activity, and active pressure.",
            r"",
            rf"\subsection{{Prediction for {display_name} ({color_word})}}",
            r"",
            r"\paragraph{Algorithm Parameters.}",
            r"\begin{itemize}",
            r"\item Opening cutoff: ply 16 (first 8 moves ignored)",
            r"\item Streak length: 10 plies (5 full moves required)",
            r"\item Margin ($\epsilon$): 0.0 (any positive advantage counts)",
            r"\item Weights: Space=1.0, Mobility=1.0, King Safety=1.0, Threats=0.1",
            r"\end{itemize}",
            r"",
        ])
        
        # Per-ply results
        lines.extend([
            r"\paragraph{Per-Ply Algorithm (Raw).}",
            r"This algorithm looks for 10 consecutive plies where the raw Fireteam Index exceeds zero.",
            r"",
            r"\begin{tabular}{ll}",
            rf"\textbf{{Prediction}} & \textbf{{{result_per_ply.prediction}}} \\",
        ])
        
        if result_per_ply.threshold_crossed_ply:
            lines.append(rf"Threshold crossed at & ply {result_per_ply.threshold_crossed_ply} \\")
        
        lines.extend([
            rf"Maximum streak & {result_per_ply.max_streak} plies",
        ])
        
        if result_per_ply.max_streak_start_ply:
            lines.append(rf" (starting ply {result_per_ply.max_streak_start_ply}) \\")
        else:
            lines.append(r" \\")
        
        lines.extend([
            rf"Peak FT value & {result_per_ply.peak_ft_value:+.3f}",
        ])
        
        if result_per_ply.peak_ft_ply:
            lines.append(rf" (ply {result_per_ply.peak_ft_ply}) \\")
        else:
            lines.append(r" \\")
        
        lines.extend([
            r"\end{tabular}",
            r"",
        ])
        
        # Windowed results
        lines.extend([
            r"\paragraph{Windowed Algorithm (Smoothed).}",
            r"This algorithm uses a 10-ply rolling average to smooth noise, then looks for "
            r"10 consecutive plies where the smoothed FT exceeds zero.",
            r"",
            r"\begin{tabular}{ll}",
            rf"\textbf{{Prediction}} & \textbf{{{result_windowed.prediction}}} \\",
        ])
        
        if result_windowed.threshold_crossed_ply:
            lines.append(rf"Threshold crossed at & ply {result_windowed.threshold_crossed_ply} \\")
        
        lines.extend([
            rf"Maximum streak & {result_windowed.max_streak} plies",
        ])
        
        if result_windowed.max_streak_start_ply:
            lines.append(rf" (starting ply {result_windowed.max_streak_start_ply}) \\")
        else:
            lines.append(r" \\")
        
        lines.extend([
            rf"Peak smoothed FT & {result_windowed.peak_ft_value:+.3f}",
        ])
        
        if result_windowed.peak_ft_ply:
            lines.append(rf" (ply {result_windowed.peak_ft_ply}) \\")
        else:
            lines.append(r" \\")
        
        lines.extend([
            r"\end{tabular}",
            r"",
        ])
        
        # Generate FT plots if requested
        if include_plots and PLOTTING_AVAILABLE and is_matplotlib_available():
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Per-ply plot
                if result_per_ply.ft_values:
                    plies = [p for p, _ in result_per_ply.ft_values]
                    ft_vals = [v for _, v in result_per_ply.ft_values]
                    
                    ax1.plot(plies, ft_vals, 'b-', linewidth=1.0, label='Fireteam Index')
                    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                    ax1.fill_between(plies, ft_vals, 0, 
                                    where=[v > 0 for v in ft_vals],
                                    alpha=0.3, color='green', label='Positive')
                    ax1.fill_between(plies, ft_vals, 0,
                                    where=[v <= 0 for v in ft_vals],
                                    alpha=0.3, color='red', label='Negative')
                    ax1.set_xlabel('Ply')
                    ax1.set_ylabel(f'FT ({display_name})')
                    ax1.set_title(f'Per-Ply Fireteam Index (Raw)')
                    ax1.legend(loc='best')
                    ax1.grid(True, alpha=0.3)
                
                # Windowed plot
                if result_windowed.ft_values:
                    plies_w = [p for p, _ in result_windowed.ft_values]
                    ft_vals_w = [v for _, v in result_windowed.ft_values]
                    
                    ax2.plot(plies_w, ft_vals_w, 'b-', linewidth=1.5, label='Smoothed FT (10-ply window)')
                    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                    ax2.fill_between(plies_w, ft_vals_w, 0,
                                    where=[v > 0 for v in ft_vals_w],
                                    alpha=0.3, color='green', label='Positive')
                    ax2.fill_between(plies_w, ft_vals_w, 0,
                                    where=[v <= 0 for v in ft_vals_w],
                                    alpha=0.3, color='red', label='Negative')
                    ax2.set_xlabel('Ply')
                    ax2.set_ylabel(f'FT ({display_name})')
                    ax2.set_title(f'Windowed Fireteam Index (Smoothed)')
                    ax2.legend(loc='best')
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                ft_plot_path = os.path.join(plot_output_dir, 'plot_fireteam_index.pdf')
                plt.savefig(ft_plot_path, format='pdf', bbox_inches='tight')
                plt.close()
                
                lines.extend([
                    r"\subsection{Fireteam Index Over Time}",
                    r"",
                    r"The plots below show the Fireteam Index evolution throughout the game. "
                    r"Green shading indicates positive (favorable) values; red indicates negative.",
                    r"",
                    r"\begin{center}",
                    rf"\includegraphics[width=0.95\textwidth]{{{ft_plot_path}}}",
                    r"\end{center}",
                    r"",
                ])
                
            except Exception as e:
                lines.extend([
                    r"\subsection{Fireteam Index Over Time}",
                    r"",
                    rf"(Plot generation failed: {esc(str(e))})",
                    r"",
                ])
        
        # Add interpretation
        lines.extend([
            r"\subsection{Interpretation}",
            r"",
        ])
        
        # Determine actual result for comparison
        actual_result = analysis.result
        player_won = (player_color.upper() == "W" and actual_result == "1-0") or \
                    (player_color.upper() == "B" and actual_result == "0-1")
        was_draw = actual_result == "1/2-1/2"
        
        if was_draw:
            actual_outcome = "DRAW"
        elif player_won:
            actual_outcome = "WIN"
        else:
            actual_outcome = "LOSS"
        
        per_ply_correct = (result_per_ply.prediction == "WIN" and player_won) or \
                         (result_per_ply.prediction == "DRAW" and (was_draw or not player_won))
        windowed_correct = (result_windowed.prediction == "WIN" and player_won) or \
                          (result_windowed.prediction == "DRAW" and (was_draw or not player_won))
        
        lines.extend([
            rf"Actual game result: \textbf{{{esc(actual_result)}}} ",
            rf"({display_name}'s outcome: \textbf{{{actual_outcome}}})",
            r"",
            r"\begin{itemize}",
            rf"\item Per-ply prediction: {result_per_ply.prediction} --- ",
        ])
        
        if per_ply_correct:
            lines.append(r"\textcolor{green}{$\checkmark$ Correct}")
        else:
            lines.append(r"\textcolor{red}{$\times$ Incorrect}")
        
        lines.extend([
            rf"\item Windowed prediction: {result_windowed.prediction} --- ",
        ])
        
        if windowed_correct:
            lines.append(r"\textcolor{green}{$\checkmark$ Correct}")
        else:
            lines.append(r"\textcolor{red}{$\times$ Incorrect}")
        
        lines.extend([
            r"\end{itemize}",
            r"",
        ])
        
        return lines
    
    @staticmethod
    def _generate_methodology_section() -> List[str]:
        """Generate explanation of how Stockfish computes positional metrics."""
        return [
            r"\section{Methodology: How Metrics Are Computed}",
            r"",
            r"This section explains how Stockfish's classical evaluation computes each positional factor. "
            r"Note that modern Stockfish primarily uses NNUE (neural network) evaluation, but the classical "
            r"evaluation terms provide interpretable metrics for educational purposes.",
            r"",
            r"\subsection{Space}",
            r"",
            r"Space evaluation counts squares in the center (files c-f, ranks 2-4 for White, 5-7 for Black) that are:",
            r"\begin{enumerate}",
            r"\item Attacked by at least one friendly pawn, AND",
            r"\item Not occupied by any enemy piece",
            r"\end{enumerate}",
            r"",
            r"The computation is weighted by the number of pieces behind the pawn chain---this rewards positions "
            r"where you control territory AND have pieces to exploit that space. The space term is scaled by "
            r"total piece count; it matters more in closed positions with many pieces.",
            r"",
            r"\subsection{Mobility}",
            r"",
            r"Mobility counts legal moves for each piece type, with different weights:",
            r"\begin{itemize}",
            r"\item \textbf{Knights}: Safe squares attacked (excluding squares defended by enemy pawns)",
            r"\item \textbf{Bishops}: Safe squares on diagonals (bonus for long, unobstructed diagonals)",
            r"\item \textbf{Rooks}: Squares on files/ranks (bonus for open files, 7th rank penetration)",
            r"\item \textbf{Queens}: Combination of bishop and rook mobility patterns",
            r"\end{itemize}",
            r"",
            r'``Safe squares" exclude those defended by enemy pawns or where exchanges would lose material. '
            r"Mobility in the opponent's territory is valued higher than mobility in your own.",
            r"",
            r"\subsection{King Safety}",
            r"",
            r"King safety combines several factors:",
            r"\begin{itemize}",
            r"\item \textbf{Pawn shield}: Pawns on 2nd/3rd rank directly in front of the castled king",
            r"\item \textbf{King tropism}: Distance of enemy pieces to your king (closer = more dangerous)",
            r"\item \textbf{Attack units}: Count of enemy pieces attacking squares adjacent to your king",
            r"\item \textbf{Safe checks}: Number of safe checking squares available to the opponent",
            r"\end{itemize}",
            r"",
            r"\subsection{Threats}",
            r"",
            r"Threat evaluation considers:",
            r"\begin{itemize}",
            r"\item Hanging pieces (attacked but not adequately defended)",
            r"\item Safe pawn advances that create threats",
            r"\item Minor pieces attacking major pieces",
            r"\item Weak squares in the pawn structure",
            r"\end{itemize}",
            r"",
            r"\subsection{Middlegame vs Endgame (MG/EG)}",
            r"",
            r"Each term has separate middlegame (MG) and endgame (EG) values. Stockfish blends these based on "
            r'remaining material (the ``phase"). With all pieces on the board, MG dominates; as pieces trade, '
            r"EG values become more important. For example, space matters greatly in the middlegame but is "
            r"nearly irrelevant in endgames.",
            r"",
            r"\subsection{Game Character Classification}",
            r"",
            r"Games are classified based on evaluation volatility and directionality. Let $m_2$ be the maximum "
            r"evaluation for White during the game (in pawns) and $m_1$ be the minimum. The \emph{spread} "
            r"$d = m_2 - m_1$ measures the total evaluation range.",
            r"",
            r"\paragraph{Spread Classifications.} Based on the magnitude of $d$:",
            r"\begin{itemize}",
            r"\item \textbf{Balanced} ($d < 1$): Minimal advantage shifts; controlled, technical play with "
            r"neither side gaining a significant edge.",
            r"\item \textbf{Tense} ($1 \le d < 3$): Normal competitive tension; typical of well-played games "
            r"where small advantages are contested.",
            r"\item \textbf{Tactical} ($3 \le d < 6$): Significant evaluation swings; tactical complications, "
            r"missed opportunities, or speculative play.",
            r"\item \textbf{Chaotic} ($d \ge 6$): Wild swings in evaluation; likely includes major blunders, "
            r"speculative sacrifices, or complex tactical melees.",
            r"\end{itemize}",
            r"",
            r"\paragraph{Directionality Classifications.} Based on whether the advantage changed hands:",
            r"\begin{itemize}",
            r"\item \textbf{One-sided}: Either $m_1 > -\frac{1}{2}$ (White always comfortable) or "
            r"$m_2 < \frac{1}{2}$ (Black always comfortable). The leading side maintained control throughout.",
            r"\item \textbf{Seesaw}: Both $m_1 < -1$ and $m_2 > 1$. The advantage genuinely changed hands, "
            r"with each side holding a significant advantage at some point.",
            r"\item \textbf{Normal}: Neither one-sided nor seesaw; moderate swings without complete reversals.",
            r"\end{itemize}",
            r"",
            r"These classifications combine to describe the overall game character. For example, a "
            r'``tactical seesaw battle" indicates significant evaluation swings ($3 \le d < 6$) with the '
            r"advantage changing hands multiple times.",
            r"",
            r"\subsection{Fireteam Index (FTI) and Win Prediction}",
            r"",
            r"The Fireteam Index is a composite metric inspired by the ancient mathematical board game "
            r"\textit{Rithmomachia}, where a ``progression victory'' is achieved when a coordinated group "
            r"of pieces (a ``fireteam'') establishes control deep in enemy territory. Similarly, in chess, "
            r"sustained positional dominance across multiple factors often precedes victory.",
            r"",
            r"\paragraph{Formula.} The Fireteam Index combines the four positional metrics into a single value "
            r"measuring overall positional advantage from a given player's perspective:",
            r"",
            r"\begin{equation}",
            r"\text{FT} = \Delta\text{Space} + \Delta\text{Mobility} + \Delta\text{King Safety} + \frac{\Delta\text{Threats}}{10}",
            r"\end{equation}",
            r"",
            r"where each $\Delta$ represents (Player's value $-$ Opponent's value). Positive FT indicates "
            r"the player has overall positional superiority; negative indicates the opponent is better.",
            r"",
            r"The Threats component is divided by 10 because its raw values are typically an order of magnitude "
            r"larger than the other metrics, and empirical testing showed this weighting provides better "
            r"predictive accuracy.",
            r"",
            r"\paragraph{Prediction Algorithms.} Two algorithms are used to predict game outcomes based on FT:",
            r"",
            r"\begin{enumerate}",
            r"\item \textbf{Per-Ply (Raw)}: Examines the raw FT value at each ply after the opening phase "
            r"(ply 16, approximately 8 moves). If FT exceeds zero for 10 consecutive plies (5 full moves), "
            r"a WIN is predicted for that player.",
            r"",
            r"\item \textbf{Windowed (Smoothed)}: Computes a 10-ply rolling average of FT before applying "
            r"the same streak detection. This smooths out tactical noise and captures sustained trends "
            r"rather than momentary fluctuations.",
            r"\end{enumerate}",
            r"",
            r"Both algorithms predict DRAW if no player achieves the required streak of positional dominance.",
            r"",
            r"\paragraph{Interpretation.} The FT prediction measures \emph{positional} dominance, which is a "
            r"necessary but not sufficient condition for winning. A player may dominate positionally yet fail "
            r"to convert due to:",
            r"\begin{itemize}",
            r"\item Tactical errors (blunders, missed combinations)",
            r"\item Time pressure in rapid/blitz games",
            r"\item Tenacious defense by the opponent",
            r"\item Drawing tendencies in certain structures (opposite-colored bishops, fortresses)",
            r"\end{itemize}",
            r"",
            r"Conversely, a player may win without ever achieving sustained FT dominance through a single "
            r"tactical strike or opponent error. The FT metric is most predictive in games where positional "
            r"understanding---rather than pure calculation---determines the outcome.",
            r"",
            r"\paragraph{For Draws.} When analyzing drawn games, the FT is computed from both players' "
            r"perspectives. This reveals whether:",
            r"\begin{itemize}",
            r"\item Neither side achieved dominance (a ``true draw'' with balanced play)",
            r"\item One side dominated but failed to convert (a ``missed win'')",
            r"\item Both sides dominated at different points (a ``mutual chances draw'')",
            r"\end{itemize}",
            r"",
        ]

    @staticmethod
    def generate_book_report(analyses: List[EnhancedGameAnalysisResult],
                             book_title: str = "Chess Game Collection Analysis",
                             author: str = None,
                             include_diagrams: bool = True,
                             include_positional: bool = True,
                             include_methodology: bool = True,
                             include_plots: bool = True,
                             include_ascii_plots: bool = False,
                             include_prediction: bool = True,
                             prediction_name: str = None,
                             prediction_winner: bool = True,
                             plot_output_dir: str = None,
                             verbose: bool = False) -> str:
        """
        Generate a LaTeX book with multiple games as chapters.
        
        Args:
            analyses: List of EnhancedGameAnalysisResult objects
            book_title: Title for the book
            author: Author name (defaults to engine version)
            include_diagrams: Include chess board diagrams
            include_positional: Include positional evaluation sections
            include_methodology: Include methodology explanation (once, as appendix)
            include_plots: Include matplotlib plots (requires matplotlib)
            include_ascii_plots: Include ASCII plots in verbatim environment
            include_prediction: Include Fireteam Index prediction sections
            prediction_name: Track specific player by name (e.g., "Berliner")
            prediction_winner: If True (default), track winner in decisive games
            plot_output_dir: Directory to save plot files (defaults to current dir)
            verbose: Print warnings to console
            
        Returns:
            Complete LaTeX book document as string
        """
        if not analyses:
            raise ValueError("No games to include in report")
        
        esc = EnhancedLaTeXReportGenerator._escape_latex
        lines = []
        
        # Set plot output directory
        if plot_output_dir is None:
            plot_output_dir = "."
        
        # Check if prediction_name matches any games
        name_matched_any = False
        if prediction_name and include_prediction:
            for analysis in analyses:
                if prediction_name.lower() in analysis.white.lower() or \
                   prediction_name.lower() in analysis.black.lower():
                    name_matched_any = True
                    break
            
            if not name_matched_any and verbose:
                print(f"WARNING: prediction_name '{prediction_name}' did not match any player "
                      f"in the {len(analyses)} games. Falling back to --prediction-winner mode.")
        
        # Determine author
        if author is None:
            author = f"Generated by {analyses[0].engine_version}"
        
        # Document preamble - using book class
        lines.extend([
            r"\documentclass[11pt]{book}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{xskak}",
            r"\usepackage{amssymb}",
            r"\usepackage{chessboard}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{longtable}",
            r"\usepackage{booktabs}",
            r"\usepackage{hyperref}",
            r"\usepackage{xcolor}",
            r"\usepackage{pgfplots}",
            r"\usepackage{graphicx}",
            r"\pgfplotsset{compat=1.16}",
            r"",
            r"% Custom colors",
            r"\definecolor{brilliantcolor}{RGB}{0, 150, 150}",
            r"\definecolor{excellentcolor}{RGB}{0, 128, 0}",
            r"\definecolor{goodcolor}{RGB}{64, 160, 64}",
            r"\definecolor{inaccuracycolor}{RGB}{200, 180, 0}",
            r"\definecolor{mistakecolor}{RGB}{220, 120, 0}",
            r"\definecolor{blundercolor}{RGB}{200, 0, 0}",
            r"\definecolor{spacecolor}{RGB}{70, 130, 180}",
            r"\definecolor{mobilitycolor}{RGB}{60, 179, 113}",
            r"",
            rf"\title{{{esc(book_title)}}}",
            rf"\author{{{esc(author)}}}",
            r"\date{\today}",
            r"",
            r"\begin{document}",
            r"\maketitle",
            r"\tableofcontents",
            r"",
        ])
        
        # Track prediction statistics for summary
        prediction_stats = {
            'total_games': len(analyses),
            'decisive_games': 0,
            'draws': 0,
            'per_ply_correct': 0,
            'windowed_correct': 0,
            'games_with_prediction': 0,
        }
        
        # Generate each game as a chapter
        for game_num, analysis in enumerate(analyses, 1):
            # Determine prediction player for this game
            game_prediction_color = None
            game_prediction_name = None
            is_draw = analysis.result == "1/2-1/2" or analysis.result == "1/2"
            
            if include_prediction:
                # First, try to match prediction_name
                if prediction_name and name_matched_any:
                    if prediction_name.lower() in analysis.white.lower():
                        game_prediction_color = 'W'
                        game_prediction_name = prediction_name
                    elif prediction_name.lower() in analysis.black.lower():
                        game_prediction_color = 'B'
                        game_prediction_name = prediction_name
                
                # If no name match, fall back to prediction_winner for decisive games
                # or analyze both sides for draws
                if game_prediction_color is None and prediction_winner:
                    if analysis.result == "1-0":
                        game_prediction_color = 'W'
                        game_prediction_name = analysis.white
                        prediction_stats['decisive_games'] += 1
                    elif analysis.result == "0-1":
                        game_prediction_color = 'B'
                        game_prediction_name = analysis.black
                        prediction_stats['decisive_games'] += 1
                    elif is_draw:
                        # For draws, we'll use special 'BOTH' marker
                        game_prediction_color = 'BOTH'
                        game_prediction_name = None
                        prediction_stats['draws'] += 1
            
            lines.extend(EnhancedLaTeXReportGenerator._generate_game_chapter(
                analysis,
                game_num=game_num,
                include_diagrams=include_diagrams,
                include_positional=include_positional,
                include_plots=include_plots,
                include_ascii_plots=include_ascii_plots,
                include_prediction=include_prediction and game_prediction_color is not None,
                prediction_player_color=game_prediction_color,
                prediction_player_name=game_prediction_name,
                plot_output_dir=plot_output_dir,
                prediction_stats=prediction_stats  # Pass for accumulation
            ))
        
        # Add Fireteam Index prediction summary chapter
        if include_prediction and prediction_stats['games_with_prediction'] > 0:
            lines.extend([
                r"\chapter{Fireteam Index Prediction Summary}",
                r"",
                r"This chapter summarizes the accuracy of the Fireteam Index prediction algorithms "
                r"across all games in this collection.",
                r"",
                r"\section{Overall Results}",
                r"",
                r"\begin{tabular}{lr}",
                r"\toprule",
                r"\textbf{Metric} & \textbf{Value} \\",
                r"\midrule",
                rf"Total games & {prediction_stats['total_games']} \\",
                rf"Decisive games & {prediction_stats['decisive_games']} \\",
                rf"Draws (both sides analyzed) & {prediction_stats['draws']} \\",
                rf"Games with prediction & {prediction_stats['games_with_prediction']} \\",
                r"\midrule",
            ])
            
            if prediction_stats['games_with_prediction'] > 0:
                pp_acc = 100 * prediction_stats['per_ply_correct'] / prediction_stats['games_with_prediction']
                w_acc = 100 * prediction_stats['windowed_correct'] / prediction_stats['games_with_prediction']
                lines.extend([
                    rf"Per-ply correct & {prediction_stats['per_ply_correct']}/{prediction_stats['games_with_prediction']} ({pp_acc:.1f}\%) \\",
                    rf"Windowed correct & {prediction_stats['windowed_correct']}/{prediction_stats['games_with_prediction']} ({w_acc:.1f}\%) \\",
                ])
            
            lines.extend([
                r"\bottomrule",
                r"\end{tabular}",
                r"",
            ])
            
            # Add interpretation
            if prediction_stats['games_with_prediction'] > 0:
                lines.extend([
                    r"\section{Interpretation}",
                    r"",
                ])
                
                if prediction_name and name_matched_any:
                    lines.append(
                        rf"These predictions tracked \textbf{{{esc(prediction_name)}}} across all games where they appeared. "
                    )
                else:
                    lines.append(
                        r"These predictions tracked the eventual winner in each decisive game. "
                    )
                
                lines.extend([
                    r"A ``WIN'' prediction means the Fireteam Index exceeded the threshold for the required "
                    r"number of consecutive plies, indicating sustained positional dominance.",
                    r"",
                    r"The per-ply algorithm uses raw positional deltas, while the windowed algorithm "
                    r"smooths values over a 10-ply (5 full move) rolling average to reduce noise.",
                    r"",
                ])
        
        # Add methodology as appendix (once for all games)
        if include_methodology:
            lines.extend([
                r"\appendix",
                r"\chapter{Methodology: How Metrics Are Computed}",
                r"",
                r"This appendix explains how the positional metrics are computed.",
                r"",
            ])
            # Reuse methodology content but convert sections to sections (not subsections)
            methodology = EnhancedLaTeXReportGenerator._generate_methodology_section()
            for line in methodology:
                # Skip the original \section line
                if line.startswith(r"\section{Methodology"):
                    continue
                # Convert \subsection to \section for appendix
                if line.startswith(r"\subsection{"):
                    line = line.replace(r"\subsection{", r"\section{")
                lines.append(line)
        
        lines.extend([
            r"\end{document}",
        ])
        
        # Append raw positional data for all games after \end{document}
        # This is ignored by LaTeX but can be parsed by analysis scripts
        lines.append("")
        lines.append("% " + "=" * 70)
        lines.append("% RAW POSITIONAL DATA FOR ALL GAMES")
        lines.append("% " + "=" * 70)
        for game_num, analysis in enumerate(analyses, 1):
            raw_data_block = EnhancedLaTeXReportGenerator.generate_raw_data_block(
                analysis, game_num=game_num
            )
            lines.append(raw_data_block)
        
        return "\n".join(lines)
    
    @staticmethod
    def _generate_game_chapter(analysis: EnhancedGameAnalysisResult,
                               game_num: int,
                               include_diagrams: bool = True,
                               include_positional: bool = True,
                               include_plots: bool = True,
                               include_ascii_plots: bool = False,
                               include_prediction: bool = False,
                               prediction_player_color: str = None,
                               prediction_player_name: str = None,
                               plot_output_dir: str = ".",
                               prediction_stats: Dict = None) -> List[str]:
        """Generate a chapter for a single game in the book."""
        esc = EnhancedLaTeXReportGenerator._escape_latex
        lines = []
        
        # Chapter title
        chapter_title = f"{esc(analysis.white)} vs {esc(analysis.black)}"
        lines.extend([
            rf"\chapter{{{chapter_title}}}",
            r"",
        ])
        
        # Game Information as section
        lines.extend([
            r"\section{Game Information}",
            r"\begin{tabular}{ll}",
            rf"\textbf{{Event}} & {esc(analysis.event)} \\",
            rf"\textbf{{Site}} & {esc(analysis.site)} \\",
            rf"\textbf{{Date}} & {esc(analysis.date)} \\",
            rf"\textbf{{Round}} & {esc(analysis.round_num)} \\",
        ])
        
        white_info = esc(analysis.white)
        if analysis.white_elo:
            white_info += f" ({analysis.white_elo})"
        black_info = esc(analysis.black)
        if analysis.black_elo:
            black_info += f" ({analysis.black_elo})"
        
        lines.extend([
            rf"\textbf{{White}} & {white_info} \\",
            rf"\textbf{{Black}} & {black_info} \\",
            rf"\textbf{{Result}} & {esc(analysis.result)} \\",
            rf"\textbf{{Opening}} & {esc(analysis.opening_eco)} -- {esc(analysis.opening_name)} \\",
            r"\end{tabular}",
            r"",
        ])
        
        # Player Statistics
        lines.extend([
            r"\section{Player Statistics}",
            r"",
            r"\subsection{" + esc(analysis.white) + " (White)}",
            r"\begin{itemize}",
            rf"\item Total moves: {analysis.white_stats['total_moves']}",
            rf"\item Accuracy: {analysis.white_stats['accuracy']:.1f}\%",
            rf"\item Average centipawn loss: {analysis.white_stats['avg_centipawn_loss']:.1f}",
            rf"\item Best/Excellent moves: {analysis.white_stats['best_moves']} / {analysis.white_stats['excellent_moves']}",
            rf"\item Good moves: {analysis.white_stats['good_moves']}",
            rf"\item Inaccuracies: {analysis.white_stats['inaccuracies']}",
            rf"\item Mistakes: {analysis.white_stats['mistakes']}",
            rf"\item Blunders: {analysis.white_stats['blunders']}",
            r"\end{itemize}",
            r"",
            r"\subsection{" + esc(analysis.black) + " (Black)}",
            r"\begin{itemize}",
            rf"\item Total moves: {analysis.black_stats['total_moves']}",
            rf"\item Accuracy: {analysis.black_stats['accuracy']:.1f}\%",
            rf"\item Average centipawn loss: {analysis.black_stats['avg_centipawn_loss']:.1f}",
            rf"\item Best/Excellent moves: {analysis.black_stats['best_moves']} / {analysis.black_stats['excellent_moves']}",
            rf"\item Good moves: {analysis.black_stats['good_moves']}",
            rf"\item Inaccuracies: {analysis.black_stats['inaccuracies']}",
            rf"\item Mistakes: {analysis.black_stats['mistakes']}",
            rf"\item Blunders: {analysis.black_stats['blunders']}",
            r"\end{itemize}",
            r"",
        ])
        
        # Positional Analysis Section with Plots (use game_num for unique plot filenames)
        if include_positional and analysis.positional_summary:
            lines.extend(EnhancedLaTeXReportGenerator._generate_positional_section_for_book(
                analysis,
                game_num=game_num,
                include_plots=include_plots,
                include_ascii_plots=include_ascii_plots,
                plot_output_dir=plot_output_dir
            ))
        
        # Brilliant Sacrifices
        if analysis.brilliant_sacrifices:
            lines.extend([
                r"\section{Brilliant Sacrifices}",
                r"",
                rf"This game features {len(analysis.brilliant_sacrifices)} brilliant sacrifice(s):",
                r"",
                r"\begin{itemize}",
            ])
            
            for sac in analysis.brilliant_sacrifices:
                move_num = (sac.ply + 1) // 2
                move_str = f"{move_num}. {sac.move_san}" if sac.player == "White" else f"{move_num}...{sac.move_san}"
                sound_str = "Sound sacrifice" if sac.is_sound else "Speculative sacrifice"
                
                lines.append(
                    rf"\item \textbf{{{esc(move_str)}}} -- {sac.player} sacrifices {sac.piece_type} "
                    rf"({sac.material_lost}cp). {sound_str}. "
                    rf"Evaluation: {sac.eval_before/100:+.2f} $\rightarrow$ {sac.eval_after/100:+.2f}"
                )
            
            lines.extend([r"\end{itemize}", r""])
        
        # Annotated Game
        lines.extend([
            r"\section{Annotated Game}",
            r"",
            r"\begin{quote}",
        ])
        
        current_line = ""
        for move in analysis.moves:
            move_num = (move.ply + 1) // 2
            
            if move.is_white_move:
                if current_line:
                    lines.append(current_line)
                current_line = f"{move_num}. {esc(move.move_san)}"
            else:
                if current_line:
                    current_line += f" {esc(move.move_san)}"
                else:
                    current_line = f"{move_num}...{esc(move.move_san)}"
            
            # NAG annotations
            if move.classification == "blunder":
                current_line += "??"
            elif move.classification == "mistake":
                current_line += "?"
            elif move.classification == "inaccuracy":
                current_line += "?!"
            elif move.classification == "best" and move.move_san == move.best_move_san:
                current_line += "!"
            
            # Comments for significant moves
            if move.classification in ["blunder", "mistake"]:
                lines.append(current_line)
                player = "White" if move.is_white_move else "Black"
                # Don't show alternatives when the played move equals the best move
                if move.move_san == move.best_move_san or move.move_uci == move.best_move_uci:
                    # Edge case: move classified as error but matches best move
                    comment = f"{player} loses {move.eval_loss:.0f}cp."
                else:
                    # Build list of moves to consider: best move + playable alternatives
                    moves_to_consider = [esc(move.best_move_san)]
                    for alt_san, alt_eval in move.alternative_moves:
                        if alt_san != move.best_move_san:  # Don't duplicate best move
                            moves_to_consider.append(esc(alt_san))
                    
                    if len(moves_to_consider) == 1:
                        comment = f"{player} loses {move.eval_loss:.0f}cp. Consider instead: {moves_to_consider[0]}."
                    else:
                        comment = f"{player} loses {move.eval_loss:.0f}cp. Consider instead: {', '.join(moves_to_consider)}."
                lines.append(rf"\textit{{{comment}}}")
                lines.append("")
                current_line = ""
        
        if current_line:
            lines.append(current_line)
        
        lines.extend([
            r"",
            esc(analysis.result),
            r"\end{quote}",
            r"",
        ])
        
        # Critical Positions
        if include_diagrams and analysis.critical_positions:
            lines.extend([
                r"\section{Critical Positions}",
                r"",
                r"This section highlights critical moments where the evaluation shifted substantially---either due to errors or missed opportunities. Each diagram shows the position \emph{after} the move was played.",
                r"",
                r"\begin{itemize}",
                r"\item \textbf{Instead of [move]}: The move(s) that should have been played instead.",
                r"\item \textbf{Best continuation}: The optimal sequence of moves from the diagrammed position.",
                r"\end{itemize}",
                r"",
                r"Positions marked with {\color{blue}$\bigstar$} represent the biggest evaluation swings in the game. The positional metrics table summarizes the spatial control, piece activity, king safety, and tactical threats for each side.",
                r"",
            ])
            
            for i, pos in enumerate(analysis.critical_positions, 1):
                move_num = (pos.ply + 1) // 2
                is_white = pos.ply % 2 == 1
                move_str = f"{move_num}. {pos.move_san}" if is_white else f"{move_num}...{pos.move_san}"
                
                if pos.is_biggest_swing:
                    subsection_title = rf"\subsection*{{\textcolor{{blue}}{{Position {i}: After {esc(move_str)} $\bigstar$}}}}"
                    reason_text = rf"\textit{{\textcolor{{blue}}{{{esc(pos.reason)}}}}}"
                else:
                    subsection_title = rf"\subsection*{{Position {i}: After {esc(move_str)}}}"
                    reason_text = rf"\textit{{{esc(pos.reason)}}}"
                
                lines.extend([
                    subsection_title,
                    reason_text,
                    r"",
                    rf"Evaluation: {pos.eval_score/100:+.2f}",
                    r"",
                    r"\chessboard[setfen=" + pos.fen + "]",
                    r"",
                ])
                
                if pos.positional_eval:
                    pe = pos.positional_eval
                    lines.extend([
                        r"\begin{small}",
                        r"\begin{tabular}{lcc}",
                        r"\toprule",
                        r"Metric & White & Black \\",
                        r"\midrule",
                        rf"Space & {pe.space_white:.2f} & {pe.space_black:.2f} \\",
                        rf"Mobility (MG) & {pe.mobility_white_mg:.2f} & {pe.mobility_black_mg:.2f} \\",
                        rf"King Safety & {pe.king_safety_white:.2f} & {pe.king_safety_black:.2f} \\",
                        rf"Threats & {pe.threats_white:.2f} & {pe.threats_black:.2f} \\",
                        r"\bottomrule",
                        r"\end{tabular}",
                        r"\end{small}",
                        r"",
                    ])
                
                # Add "Instead of" moves: best move + alternatives (if different from played move)
                instead_moves = []
                if pos.best_move_san and pos.best_move_san != pos.move_san:
                    instead_moves.append(esc(pos.best_move_san))
                for alt_san, alt_eval in pos.alternative_moves[:2]:  # Add up to 2 more alternatives
                    if alt_san != pos.move_san and esc(alt_san) not in instead_moves:
                        instead_moves.append(esc(alt_san))
                if instead_moves:
                    instead_str = ", ".join(instead_moves[:3])  # Cap at 3 total
                    lines.append(rf"Instead of {esc(pos.move_san)}: \hspace{{0.5em}} {instead_str}\hspace{{4em}}")
                
                if pos.best_continuation:
                    cont_str = " ".join(esc(m) for m in pos.best_continuation[:5])
                    lines.append(rf"\hspace{{0.5em}}Best continuation: {cont_str}")
                
                lines.append(r"")
        
        # Fireteam Index Prediction Section (if requested)
        if include_prediction and prediction_player_color:
            prediction_lines = EnhancedLaTeXReportGenerator._generate_prediction_section_for_book(
                analysis,
                game_num=game_num,
                player_color=prediction_player_color,
                player_name=prediction_player_name,
                include_plots=include_plots,
                plot_output_dir=plot_output_dir,
                prediction_stats=prediction_stats
            )
            lines.extend(prediction_lines)
        
        # Analysis Metadata for this game
        lines.extend([
            r"\section{Analysis Information}",
            r"\begin{itemize}",
            rf"\item Engine: {esc(analysis.engine_version)}",
            rf"\item Depth: {analysis.analysis_depth}",
            rf"\item Analysis time: {analysis.analysis_time:.1f} seconds",
            r"\end{itemize}",
            r"",
        ])
        
        return lines
    
    @staticmethod
    def _generate_positional_section_for_book(analysis: EnhancedGameAnalysisResult,
                                              game_num: int,
                                              include_plots: bool = True,
                                              include_ascii_plots: bool = False,
                                              plot_output_dir: str = ".") -> List[str]:
        """Generate the positional analysis section for a book chapter with unique plot filenames."""
        lines = [
            r"\section{Positional Analysis}",
            r"",
            r"This section shows how key positional factors evolved throughout the game.",
            r"",
        ]
        
        ps = analysis.positional_summary
        
        # Generate plots with game_num prefix for unique filenames
        plot_files = {}
        if include_plots and PLOTTING_AVAILABLE and is_matplotlib_available() and analysis.moves:
            for metric in ['eval', 'space', 'mobility', 'king_safety', 'threats']:
                filename = f"plot_game{game_num}_{metric}.pdf"
                filepath = os.path.join(plot_output_dir, filename)
                success = ChessPlotter.generate_matplotlib_plot(
                    analysis.moves, metric, filepath
                )
                if success:
                    plot_files[metric] = filepath
        
        # Evaluation plot section
        lines.extend([
            r"\subsection{Position Evaluation Over Time}",
            r"",
        ])
        
        # Add game character classification if available
        gc = analysis.game_character
        if gc:
            m1 = gc.get('m1', 0)
            m2 = gc.get('m2', 0)
            spread = gc.get('spread', 0)
            spread_class = gc.get('spread_class', 'unknown')
            direction_class = gc.get('direction_class', 'normal')
            description = gc.get('combined_description', '')
            
            lines.extend([
                r"\paragraph{Game Character Classification.}",
                rf"Evaluation range: minimum $m_1 = {m1:+.2f}$, maximum $m_2 = {m2:+.2f}$, "
                rf"spread $d = m_2 - m_1 = {spread:.2f}$.",
                r"",
                rf"This game is classified as \textbf{{{spread_class}}} "
                rf"({direction_class}). {description}",
                r"",
            ])
        
        if 'eval' in plot_files:
            lines.extend([
                r"\begin{center}",
                rf"\includegraphics[width=0.9\textwidth]{{{plot_files['eval']}}}",
                r"\end{center}",
                r"",
            ])
        
        if include_ascii_plots and analysis.moves and PLOTTING_AVAILABLE:
            ascii_plot = ChessPlotter.generate_ascii_plot(analysis.moves, 'eval')
            lines.extend([
                r"\begin{verbatim}",
                ascii_plot,
                r"\end{verbatim}",
                r"",
            ])
        
        # Space analysis
        if ps.get('space'):
            space = ps['space']
            lines.extend([
                r"\subsection{Space Control}",
                r"",
                r"\begin{itemize}",
                rf"\item White average: {space['white']['avg']:.2f} (range: {space['white']['min']:.2f} to {space['white']['max']:.2f})",
                rf"\item Black average: {space['black']['avg']:.2f} (range: {space['black']['min']:.2f} to {space['black']['max']:.2f})",
                r"\end{itemize}",
                r"",
            ])
            
            if 'space' in plot_files:
                lines.extend([
                    r"\begin{center}",
                    rf"\includegraphics[width=0.9\textwidth]{{{plot_files['space']}}}",
                    r"\end{center}",
                    r"",
                ])
        
        # Mobility analysis
        if ps.get('mobility'):
            mob = ps['mobility']
            lines.extend([
                r"\subsection{Mobility (Piece Activity)}",
                r"",
                r"\begin{itemize}",
                rf"\item White average: {mob['white']['avg']:.2f} (range: {mob['white']['min']:.2f} to {mob['white']['max']:.2f})",
                rf"\item Black average: {mob['black']['avg']:.2f} (range: {mob['black']['min']:.2f} to {mob['black']['max']:.2f})",
                r"\end{itemize}",
                r"",
            ])
            
            if 'mobility' in plot_files:
                lines.extend([
                    r"\begin{center}",
                    rf"\includegraphics[width=0.9\textwidth]{{{plot_files['mobility']}}}",
                    r"\end{center}",
                    r"",
                ])
        
        # King safety analysis
        if ps.get('king_safety'):
            ks = ps['king_safety']
            lines.extend([
                r"\subsection{King Safety}",
                r"",
                r"\begin{itemize}",
                rf"\item White average: {ks['white']['avg']:.2f}",
                rf"\item Black average: {ks['black']['avg']:.2f}",
                r"\end{itemize}",
                r"",
            ])
            
            if 'king_safety' in plot_files:
                lines.extend([
                    r"\begin{center}",
                    rf"\includegraphics[width=0.9\textwidth]{{{plot_files['king_safety']}}}",
                    r"\end{center}",
                    r"",
                ])
        
        # Threats analysis
        if ps.get('threats'):
            th = ps['threats']
            lines.extend([
                r"\subsection{Threats}",
                r"",
                r"\begin{itemize}",
                rf"\item White average: {th['white']['avg']:.2f} (range: {th['white']['min']:.2f} to {th['white']['max']:.2f})",
                rf"\item Black average: {th['black']['avg']:.2f} (range: {th['black']['min']:.2f} to {th['black']['max']:.2f})",
            ])
            
            if th['advantage']['avg'] > 0.1:
                lines.append(rf"\item Average advantage: {th['advantage']['avg']:.2f} (White maintained more threats)")
            elif th['advantage']['avg'] < -0.1:
                lines.append(rf"\item Average advantage: {th['advantage']['avg']:.2f} (Black maintained more threats)")
            else:
                lines.append(rf"\item Average advantage: {th['advantage']['avg']:.2f} (threats were balanced)")
            
            lines.extend([
                r"\end{itemize}",
                r"",
            ])
            
            if 'threats' in plot_files:
                lines.extend([
                    r"\begin{center}",
                    rf"\includegraphics[width=0.9\textwidth]{{{plot_files['threats']}}}",
                    r"\end{center}",
                    r"",
                ])
        
        return lines

    @staticmethod
    def _generate_prediction_section_for_book(
        analysis: EnhancedGameAnalysisResult,
        game_num: int,
        player_color: str,
        player_name: Optional[str] = None,
        include_plots: bool = True,
        plot_output_dir: str = ".",
        prediction_stats: Dict = None
    ) -> List[str]:
        """
        Generate the Fireteam Index prediction section for a book chapter.
        
        This is a condensed version of _generate_prediction_section for use in books.
        For draws (player_color='BOTH'), analyzes from both perspectives.
        """
        esc = EnhancedLaTeXReportGenerator._escape_latex
        lines = []
        
        # Extract positional data from analysis
        data = compute_fireteam_index_for_analysis(analysis)
        
        if not data:
            return lines
        
        # Check if this is a draw requiring dual analysis
        is_draw_dual = player_color == 'BOTH'
        was_draw = analysis.result in ("1/2-1/2", "1/2")
        
        if is_draw_dual:
            # Analyze from both perspectives for draws
            lines.extend([
                r"\section{Fireteam Index Prediction}",
                r"",
                r"\textit{This game ended in a draw. The Fireteam Index is analyzed from both players' "
                r"perspectives to determine whether either side achieved sustained positional dominance "
                r"that might have predicted a win.}",
                r"",
            ])
            
            # Run predictions for both sides
            result_white_pp = predict_outcome_per_ply(data, player_color='W')
            result_white_w = predict_outcome_windowed(data, player_color='W')
            result_black_pp = predict_outcome_per_ply(data, player_color='B')
            result_black_w = predict_outcome_windowed(data, player_color='B')
            
            white_name = esc(analysis.white)
            black_name = esc(analysis.black)
            
            # Results table for both players
            lines.extend([
                r"\begin{tabular}{llcc}",
                r"\toprule",
                r"\textbf{Player} & \textbf{Algorithm} & \textbf{Prediction} & \textbf{Max Streak} \\",
                r"\midrule",
                rf"{white_name} & Per-ply & {result_white_pp.prediction} & {result_white_pp.max_streak} plies \\",
                rf"{white_name} & Windowed & {result_white_w.prediction} & {result_white_w.max_streak} plies \\",
                r"\midrule",
                rf"{black_name} & Per-ply & {result_black_pp.prediction} & {result_black_pp.max_streak} plies \\",
                rf"{black_name} & Windowed & {result_black_w.prediction} & {result_black_w.max_streak} plies \\",
                r"\bottomrule",
                r"\end{tabular}",
                r"",
            ])
            
            # Interpretation for draws
            white_predicted_win = result_white_pp.prediction == "WIN" or result_white_w.prediction == "WIN"
            black_predicted_win = result_black_pp.prediction == "WIN" or result_black_w.prediction == "WIN"
            
            lines.append(r"\paragraph{Interpretation.}")
            if white_predicted_win and black_predicted_win:
                lines.append(
                    r"Both players achieved sustained positional dominance at different points in the game. "
                    r"The FT Index predicted wins for both sides, reflecting a hard-fought battle where "
                    r"momentum shifted. The draw is a reasonable outcome given the mutual winning chances."
                )
                # For stats: if both predicted WIN, we count this as "neither fully correct"
                # but it's a reasonable draw
                if prediction_stats is not None:
                    prediction_stats['games_with_prediction'] += 1
                    # Neither side's WIN prediction was correct (it was a draw)
            elif white_predicted_win:
                lines.append(
                    rf"White ({white_name}) achieved sustained positional dominance, with the FT Index "
                    r"predicting a win. However, the game ended in a draw, suggesting Black successfully "
                    r"defended or White failed to convert the advantage."
                )
                if prediction_stats is not None:
                    prediction_stats['games_with_prediction'] += 1
                    # White's WIN prediction was incorrect
            elif black_predicted_win:
                lines.append(
                    rf"Black ({black_name}) achieved sustained positional dominance, with the FT Index "
                    r"predicting a win. However, the game ended in a draw, suggesting White successfully "
                    r"defended or Black failed to convert the advantage."
                )
                if prediction_stats is not None:
                    prediction_stats['games_with_prediction'] += 1
                    # Black's WIN prediction was incorrect
            else:
                lines.append(
                    r"Neither player achieved sustained positional dominance according to the FT Index. "
                    r"Both algorithms predicted DRAW for both sides, which matches the actual result. "
                    r"This suggests a balanced game where neither side established lasting control."
                )
                if prediction_stats is not None:
                    prediction_stats['games_with_prediction'] += 1
                    prediction_stats['per_ply_correct'] += 1
                    prediction_stats['windowed_correct'] += 1
            
            lines.append(r"")
            
            # Generate dual FT plot if requested
            if include_plots and PLOTTING_AVAILABLE and is_matplotlib_available():
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # White's perspective
                    if result_white_w.ft_values:
                        plies_w = [p for p, _ in result_white_w.ft_values]
                        ft_vals_w = [v for _, v in result_white_w.ft_values]
                        ax1.plot(plies_w, ft_vals_w, 'b-', linewidth=2.0)
                        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                        ax1.fill_between(plies_w, ft_vals_w, 0,
                                        where=[v > 0 for v in ft_vals_w],
                                        alpha=0.3, color='green')
                        ax1.fill_between(plies_w, ft_vals_w, 0,
                                        where=[v <= 0 for v in ft_vals_w],
                                        alpha=0.3, color='red')
                    ax1.set_xlabel('Ply')
                    ax1.set_ylabel('Fireteam Index')
                    ax1.set_title(f"White's Perspective ({analysis.white})")
                    ax1.grid(True, alpha=0.3)
                    
                    # Black's perspective
                    if result_black_w.ft_values:
                        plies_b = [p for p, _ in result_black_w.ft_values]
                        ft_vals_b = [v for _, v in result_black_w.ft_values]
                        ax2.plot(plies_b, ft_vals_b, 'b-', linewidth=2.0)
                        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                        ax2.fill_between(plies_b, ft_vals_b, 0,
                                        where=[v > 0 for v in ft_vals_b],
                                        alpha=0.3, color='green')
                        ax2.fill_between(plies_b, ft_vals_b, 0,
                                        where=[v <= 0 for v in ft_vals_b],
                                        alpha=0.3, color='red')
                    ax2.set_xlabel('Ply')
                    ax2.set_ylabel('Fireteam Index')
                    ax2.set_title(f"Black's Perspective ({analysis.black})")
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    ft_plot_path = os.path.join(plot_output_dir, f'plot_fireteam_g{game_num}_dual.pdf')
                    plt.savefig(ft_plot_path, format='pdf', bbox_inches='tight')
                    plt.close()
                    
                    lines.extend([
                        r"\begin{center}",
                        rf"\includegraphics[width=0.95\textwidth]{{{ft_plot_path}}}",
                        r"\end{center}",
                        r"",
                    ])
                    
                except Exception as e:
                    lines.extend([
                        rf"(Plot generation failed: {esc(str(e))})",
                        r"",
                    ])
            
            return lines
        
        # Standard single-player analysis (non-draw or specific player requested)
        result_per_ply = predict_outcome_per_ply(
            data, 
            player_color=player_color,
            player_name=player_name
        )
        result_windowed = predict_outcome_windowed(
            data,
            player_color=player_color,
            player_name=player_name
        )
        
        # Determine player display name
        if player_name:
            display_name = esc(player_name)
        elif player_color.upper() == "W":
            display_name = esc(analysis.white)
        else:
            display_name = esc(analysis.black)
        
        color_word = "White" if player_color.upper() == "W" else "Black"
        
        lines.extend([
            r"\section{Fireteam Index Prediction}",
            r"",
            rf"Tracking: \textbf{{{display_name}}} ({color_word})",
            r"",
        ])
        
        # Compact results table
        lines.extend([
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"\textbf{Algorithm} & \textbf{Prediction} & \textbf{Max Streak} \\",
            r"\midrule",
            rf"Per-ply (raw) & {result_per_ply.prediction} & {result_per_ply.max_streak} plies \\",
            rf"Windowed (smoothed) & {result_windowed.prediction} & {result_windowed.max_streak} plies \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"",
        ])
        
        # Determine actual result for comparison
        actual_result = analysis.result
        player_won = (player_color.upper() == "W" and actual_result == "1-0") or \
                    (player_color.upper() == "B" and actual_result == "0-1")
        
        if was_draw:
            actual_outcome = "DRAW"
        elif player_won:
            actual_outcome = "WIN"
        else:
            actual_outcome = "LOSS"
        
        per_ply_correct = (result_per_ply.prediction == "WIN" and player_won) or \
                         (result_per_ply.prediction == "DRAW" and (was_draw or not player_won))
        windowed_correct = (result_windowed.prediction == "WIN" and player_won) or \
                          (result_windowed.prediction == "DRAW" and (was_draw or not player_won))
        
        # Update prediction stats if provided
        if prediction_stats is not None:
            prediction_stats['games_with_prediction'] += 1
            if per_ply_correct:
                prediction_stats['per_ply_correct'] += 1
            if windowed_correct:
                prediction_stats['windowed_correct'] += 1
        
        lines.extend([
            rf"Actual result: {esc(actual_result)} ({display_name}'s outcome: {actual_outcome})",
            r"",
        ])
        
        # Correctness indicators
        pp_mark = r"\textcolor{green}{$\checkmark$}" if per_ply_correct else r"\textcolor{red}{$\times$}"
        w_mark = r"\textcolor{green}{$\checkmark$}" if windowed_correct else r"\textcolor{red}{$\times$}"
        
        lines.extend([
            rf"Per-ply: {pp_mark} \quad Windowed: {w_mark}",
            r"",
        ])
        
        # Generate FT plot if requested
        if include_plots and PLOTTING_AVAILABLE and is_matplotlib_available():
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Per-ply values
                if result_per_ply.ft_values:
                    plies = [p for p, _ in result_per_ply.ft_values]
                    ft_vals = [v for _, v in result_per_ply.ft_values]
                    ax.plot(plies, ft_vals, 'b-', linewidth=0.8, alpha=0.5, label='Raw FT')
                
                # Windowed values (thicker line)
                if result_windowed.ft_values:
                    plies_w = [p for p, _ in result_windowed.ft_values]
                    ft_vals_w = [v for _, v in result_windowed.ft_values]
                    ax.plot(plies_w, ft_vals_w, 'b-', linewidth=2.0, label='Smoothed FT')
                
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                ax.fill_between(plies_w if result_windowed.ft_values else plies, 
                               ft_vals_w if result_windowed.ft_values else ft_vals, 0,
                               where=[v > 0 for v in (ft_vals_w if result_windowed.ft_values else ft_vals)],
                               alpha=0.3, color='green')
                ax.fill_between(plies_w if result_windowed.ft_values else plies,
                               ft_vals_w if result_windowed.ft_values else ft_vals, 0,
                               where=[v <= 0 for v in (ft_vals_w if result_windowed.ft_values else ft_vals)],
                               alpha=0.3, color='red')
                
                ax.set_xlabel('Ply')
                ax.set_ylabel(f'Fireteam Index ({display_name})')
                ax.set_title(f'Fireteam Index Evolution')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save with game_num for uniqueness
                ft_plot_path = os.path.join(plot_output_dir, f'plot_fireteam_g{game_num}.pdf')
                plt.savefig(ft_plot_path, format='pdf', bbox_inches='tight')
                plt.close()
                
                lines.extend([
                    r"\begin{center}",
                    rf"\includegraphics[width=0.85\textwidth]{{{ft_plot_path}}}",
                    r"\end{center}",
                    r"",
                ])
                
            except Exception as e:
                lines.extend([
                    rf"(Plot generation failed: {esc(str(e))})",
                    r"",
                ])
        
        return lines


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_game_with_positional_metrics(
    pgn_source: Union[str, io.StringIO],
    output_path: Optional[str] = None,
    stockfish_path: str = "/usr/games/stockfish",
    depth: int = 20,
    time_limit: float = 1.0,
    include_diagrams: bool = True,
    include_methodology: bool = True,
    include_plots: bool = True,
    include_ascii_plots: bool = False,
    plot_output_dir: str = None,
    top_n_swings: int = 2,
    verbose: bool = True
) -> Union[str, EnhancedGameAnalysisResult]:
    """
    Analyze a chess game with detailed positional metrics and generate LaTeX report.
    
    Args:
        pgn_source: Path to PGN file, or PGN string, or StringIO object
        output_path: Optional path to save LaTeX file (if None, returns result object)
        stockfish_path: Path to Stockfish executable
        depth: Analysis depth (default 20)
        time_limit: Time per position in seconds (default 1.0)
        include_diagrams: Include chess board diagrams in report
        include_methodology: Include explanation of how metrics are computed
        include_plots: Include matplotlib plots (requires matplotlib)
        include_ascii_plots: Include ASCII plots in verbatim environment
        plot_output_dir: Directory for plot files (defaults to same dir as output)
        top_n_swings: Number of "biggest swing" positions to always include (default: 2)
        verbose: Print progress messages
        
    Returns:
        If output_path provided: LaTeX document as string
        If output_path is None: EnhancedGameAnalysisResult object

    EXAMPLES:
        >>> from chess_game_analyzer5 import analyze_game_with_positional_metrics
        
        # Analyze a game with matplotlib plots
        analyze_game_with_positional_metrics(
            pgn_source="game.pgn",
            output_path="analysis.tex",
            stockfish_path="/usr/local/bin/stockfish",
            include_plots=True
        )
        
        # Analyze with ASCII plots only (no matplotlib needed)
        analyze_game_with_positional_metrics(
            pgn_source="game.pgn",
            output_path="analysis.tex",
            include_plots=False,
            include_ascii_plots=True
        )
        
        # Get both matplotlib and ASCII plots
        analyze_game_with_positional_metrics(
            pgn_source="game.pgn",
            output_path="analysis.tex",
            include_plots=True,
            include_ascii_plots=True
        )

        >>> path_to_stockfish =  "/usr/local/bin/stockfish" ## mac OS
        >>> game_pgn = "../wdj-games/culver-fortna-vs-wdj-sussex-open-2026-01-11.pgn"
        >>> report_output="../wdj-games/culver-fortna-vs-wdj-sussex-open-2026-01-11-report.tex"
        >>> result = analyze_game_with_positional_metrics(pgn_source=game_pgn,output_path=report_output,stockfish_path=path_to_stockfish,include_plots=True,include_ascii_plots=False, plot_output_dir = "../wdj-games/plots/")

    """
    if verbose:
        print(f"Starting enhanced analysis (depth={depth}, time={time_limit}s per position)...")
        if include_plots:
            if PLOTTING_AVAILABLE and is_matplotlib_available():
                print("Matplotlib plots: ENABLED")
            else:
                print("Matplotlib plots: DISABLED (matplotlib not available)")
        if include_ascii_plots:
            print("ASCII plots: ENABLED")
    
    with EnhancedGameAnalyzer(stockfish_path, depth, time_limit) as analyzer:
        if verbose:
            print(f"Engine: {analyzer.engine_version}")
        
        analysis = analyzer.analyze_game(pgn_source, top_n_swings=top_n_swings)
        
        if verbose:
            print(f"Analysis complete in {analysis.analysis_time:.1f}s")
            print(f"  {analysis.white} vs {analysis.black}")
            print(f"  Result: {analysis.result}")
            print(f"  White accuracy: {analysis.white_stats['accuracy']:.1f}%")
            print(f"  Black accuracy: {analysis.black_stats['accuracy']:.1f}%")
            if analysis.brilliant_sacrifices:
                print(f"  Brilliant sacrifices: {len(analysis.brilliant_sacrifices)}")
    
    if output_path is None:
        return analysis
    
    # Determine plot output directory
    if plot_output_dir is None:
        plot_output_dir = os.path.dirname(output_path) or "."
    
    latex_content = EnhancedLaTeXReportGenerator.generate_report(
        analysis,
        include_diagrams=include_diagrams,
        include_methodology=include_methodology,
        include_plots=include_plots,
        include_ascii_plots=include_ascii_plots,
        plot_output_dir=plot_output_dir
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    if verbose:
        print(f"LaTeX report saved to: {output_path}")
        if include_plots and PLOTTING_AVAILABLE and is_matplotlib_available():
            print(f"Plot files saved to: {plot_output_dir}")
    
    return latex_content


def analyze_games_to_book(
    pgn_source: Union[str, io.StringIO],
    output_path: str,
    book_title: str = "Chess Game Collection Analysis",
    author: str = None,
    stockfish_path: str = "/usr/games/stockfish",
    depth: int = 20,
    time_limit: float = 1.0,
    include_diagrams: bool = True,
    include_methodology: bool = True,
    include_plots: bool = True,
    include_ascii_plots: bool = False,
    plot_output_dir: str = None,
    top_n_swings: int = 2,
    verbose: bool = True
) -> Tuple[str, List[EnhancedGameAnalysisResult]]:
    """
    Analyze all games in a PGN file and generate a LaTeX book with each game as a chapter.
    
    Args:
        pgn_source: Path to PGN file containing one or more games
        output_path: Path to save LaTeX book file
        book_title: Title for the book
        author: Author name (defaults to engine version)
        stockfish_path: Path to Stockfish executable
        depth: Analysis depth (default 20)
        time_limit: Time per position in seconds (default 1.0)
        include_diagrams: Include chess board diagrams in report
        include_methodology: Include methodology explanation as appendix
        include_plots: Include matplotlib plots (requires matplotlib)
        include_ascii_plots: Include ASCII plots in verbatim environment
        plot_output_dir: Directory for plot files (defaults to same dir as output)
        top_n_swings: Number of "biggest swing" positions per game (default: 2)
        verbose: Print progress messages
        
    Returns:
        Tuple of (LaTeX content as string, list of EnhancedGameAnalysisResult objects)
        
    Example:
        >>> latex, results = analyze_games_to_book(
        ...     "tournament_games.pgn",
        ...     "tournament_analysis.tex",
        ...     book_title="Club Championship 2026",
        ...     author="Chess Club Analysis Team"
        ... )
        >>> print(f"Analyzed {len(results)} games")
    """
    if verbose:
        print(f"Starting multi-game book analysis (depth={depth}, time={time_limit}s per position)...")
        if include_plots:
            if PLOTTING_AVAILABLE and is_matplotlib_available():
                print("Matplotlib plots: ENABLED")
            else:
                print("Matplotlib plots: DISABLED (matplotlib not available)")
        if include_ascii_plots:
            print("ASCII plots: ENABLED")
    
    with EnhancedGameAnalyzer(stockfish_path, depth, time_limit) as analyzer:
        if verbose:
            print(f"Engine: {analyzer.engine_version}")
        
        analyses = analyzer.analyze_all_games(
            pgn_source,
            top_n_swings=top_n_swings,
            verbose=verbose
        )
        
        if not analyses:
            raise ValueError("No games found in PGN file")
        
        if verbose:
            total_time = sum(a.analysis_time for a in analyses)
            print(f"All games analyzed in {total_time:.1f}s total")
    
    # Determine plot output directory
    if plot_output_dir is None:
        plot_output_dir = os.path.dirname(output_path) or "."
    
    # Generate book report
    latex_content = EnhancedLaTeXReportGenerator.generate_book_report(
        analyses,
        book_title=book_title,
        author=author,
        include_diagrams=include_diagrams,
        include_methodology=include_methodology,
        include_plots=include_plots,
        include_ascii_plots=include_ascii_plots,
        plot_output_dir=plot_output_dir
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    if verbose:
        print(f"LaTeX book saved to: {output_path}")
        if include_plots and PLOTTING_AVAILABLE and is_matplotlib_available():
            print(f"Plot files saved to: {plot_output_dir}")
    
    return latex_content, analyses


def get_position_metrics(fen: str, stockfish_path: str = "/usr/games/stockfish") -> PositionalEvaluation:
    """
    Get detailed positional metrics for a single position.
    
    Args:
        fen: FEN string of the position
        stockfish_path: Path to Stockfish executable
        
    Returns:
        PositionalEvaluation with all metrics
    """
    commands = f"position fen {fen}\neval\nquit\n"
    result = subprocess.run(
        [stockfish_path],
        input=commands,
        capture_output=True,
        text=True,
        timeout=5
    )
    
    return StockfishEvalParser.parse_eval_output(result.stdout)


def diagnose_stockfish_eval(stockfish_path: str = "/usr/games/stockfish") -> Dict:
    """
    Diagnostic function to check if Stockfish eval parsing is working.
    
    Run this if you're getting all zeros in positional metrics.
    
    Args:
        stockfish_path: Path to Stockfish executable
        
    Returns:
        Dict with diagnostic information
    """
    import shutil
    
    result = {
        'stockfish_found': False,
        'stockfish_path': stockfish_path,
        'stockfish_version': None,
        'eval_output_length': 0,
        'has_classical_eval_table': False,
        'parsed_terms': [],
        'sample_metrics': {},
        'raw_output_sample': '',
        'error': None
    }
    
    # Check if stockfish exists
    if not shutil.which(stockfish_path) and not Path(stockfish_path).exists():
        result['error'] = f"Stockfish not found at {stockfish_path}"
        return result
    
    result['stockfish_found'] = True
    
    try:
        # Get version
        version_result = subprocess.run(
            [stockfish_path],
            input="quit\n",
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in version_result.stdout.split('\n'):
            if 'Stockfish' in line:
                result['stockfish_version'] = line.strip()
                break
        
        # Test eval command
        test_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        commands = f"position fen {test_fen}\neval\nquit\n"
        
        eval_result = subprocess.run(
            [stockfish_path],
            input=commands,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        output = eval_result.stdout
        result['eval_output_length'] = len(output)
        result['raw_output_sample'] = output[:1500]
        
        # Check for classical eval table
        if 'Contributing terms' in output or 'classical eval' in output.lower():
            result['has_classical_eval_table'] = True
        
        # Parse and check what we got
        StockfishEvalParser.DEBUG = True
        metrics = StockfishEvalParser.parse_eval_output(output)
        StockfishEvalParser.DEBUG = False
        
        result['sample_metrics'] = {
            'space_white': metrics.space_white_mg,
            'space_black': metrics.space_black_mg,
            'mobility_white': metrics.mobility_white_mg,
            'mobility_black': metrics.mobility_black_mg,
            'king_safety_white': metrics.king_safety_white_mg,
            'king_safety_black': metrics.king_safety_black_mg,
            'classical_eval': metrics.classical_eval,
            'nnue_eval': metrics.nnue_eval,
            'final_eval': metrics.final_eval,
        }
        
        # Check which terms parsed successfully
        if metrics.space_white_mg != 0 or metrics.space_black_mg != 0:
            result['parsed_terms'].append('space')
        if metrics.mobility_white_mg != 0 or metrics.mobility_black_mg != 0:
            result['parsed_terms'].append('mobility')
        if metrics.king_safety_white_mg != 0 or metrics.king_safety_black_mg != 0:
            result['parsed_terms'].append('king_safety')
        if metrics.material_mg != 0:
            result['parsed_terms'].append('material')
        
    except subprocess.TimeoutExpired:
        result['error'] = "Stockfish timed out"
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
    
    return result


def print_diagnostics(stockfish_path: str = "/usr/games/stockfish"):
    """
    Print diagnostic information about Stockfish eval parsing.
    
    Usage:
        from chess_game_analyzer_enhanced import print_diagnostics
        print_diagnostics("/path/to/stockfish")
    """
    print("=" * 60)
    print("STOCKFISH EVAL DIAGNOSTICS")
    print("=" * 60)
    
    diag = diagnose_stockfish_eval(stockfish_path)
    
    print(f"\nStockfish path: {diag['stockfish_path']}")
    print(f"Stockfish found: {diag['stockfish_found']}")
    print(f"Version: {diag['stockfish_version']}")
    
    if diag['error']:
        print(f"\n*** ERROR: {diag['error']} ***")
        return
    
    print(f"\nEval output length: {diag['eval_output_length']} chars")
    print(f"Has classical eval table: {diag['has_classical_eval_table']}")
    print(f"Successfully parsed terms: {diag['parsed_terms']}")
    
    print("\nSample metrics from test position:")
    for key, val in diag['sample_metrics'].items():
        print(f"  {key}: {val}")
    
    if not diag['has_classical_eval_table']:
        print("\n*** WARNING: No classical eval table found! ***")
        print("Your Stockfish build may be NNUE-only.")
        print("Positional metrics require the classical eval table.")
    
    if not diag['parsed_terms']:
        print("\n*** WARNING: No terms were parsed! ***")
        print("Raw output sample:")
        print("-" * 40)
        print(diag['raw_output_sample'])
        print("-" * 40)


# =============================================================================
# RAW DATA PARSING UTILITIES
# =============================================================================

def parse_raw_positional_data(tex_filepath: str, game_num: int = None) -> List[Dict]:
    """
    Extract raw positional data from a CGA-generated .tex file.
    
    The data is stored after \\end{document} in commented lines between
    GAME_DATA_START and GAME_DATA_END markers.
    
    Args:
        tex_filepath: Path to the .tex file
        game_num: For multi-game books, specify which game (1-indexed).
                  If None, returns the first (or only) game's data.
    
    Returns:
        List of dicts, one per half-move, with keys:
        - ply: Half-move number
        - san: Move in SAN notation
        - eval_cp: Centipawn evaluation (or None)
        - space_w, space_b: Space metrics
        - mob_w, mob_b: Mobility metrics
        - ks_w, ks_b: King safety metrics
        - threats_w, threats_b: Threat metrics
    """
    with open(tex_filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all data blocks
    import re
    pattern = r'% GAME_DATA_START\n(.*?)% GAME_DATA_END'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if not matches:
        return []
    
    # Select which game
    if game_num is not None:
        if 1 <= game_num <= len(matches):
            match = matches[game_num - 1]
        else:
            return []
    else:
        match = matches[0]
    
    data = []
    for line in match.group(1).strip().split('\n'):
        line = line.strip()
        if not line.startswith('%'):
            continue
        line = line[1:].strip()  # Remove leading %
        if not line or line.startswith('='):
            continue
        
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 11:
            continue
        
        try:
            def parse_float(s):
                return None if s == 'None' else float(s)
            
            def parse_int(s):
                return None if s == 'None' else int(s)
            
            data.append({
                'ply': int(parts[0]),
                'san': parts[1],
                'eval_cp': parse_int(parts[2]),
                'space_w': parse_float(parts[3]),
                'space_b': parse_float(parts[4]),
                'mob_w': parse_float(parts[5]),
                'mob_b': parse_float(parts[6]),
                'ks_w': parse_float(parts[7]),
                'ks_b': parse_float(parts[8]),
                'threats_w': parse_float(parts[9]),
                'threats_b': parse_float(parts[10]),
            })
        except (ValueError, IndexError):
            continue
    
    return data


def compute_fireteam_index(data: List[Dict], threats_divisor: float = 10.0) -> List[Dict]:
    """
    Compute the Fireteam Index for each position from raw positional data.
    
    The Fireteam Index is inspired by Rithmomachia's "well-coordinated fireteam
    in enemy territory" victory condition. It combines:
    - Space advantage (territorial control)
    - Mobility advantage (piece coordination)
    - King safety advantage (defensive solidity)
    - Threats advantage (active pressure, downweighted)
    
    Formula: FT = ΔSpace + ΔMobility + ΔKingSafety + ΔThreats/divisor
    
    Args:
        data: List of dicts from parse_raw_positional_data()
        threats_divisor: Divisor for threats term (default 10.0)
    
    Returns:
        List of dicts with keys:
        - ply: Half-move number
        - san: Move in SAN notation
        - eval_cp: Centipawn evaluation
        - ft_white: Fireteam Index from White's perspective
        - ft_black: Fireteam Index from Black's perspective
        - components: Dict with individual component values
    """
    result = []
    
    for row in data:
        # Skip rows with missing data
        if any(row[k] is None for k in ['space_w', 'space_b', 'mob_w', 'mob_b', 
                                         'ks_w', 'ks_b', 'threats_w', 'threats_b']):
            continue
        
        # Compute component differences (positive = White advantage)
        delta_space = row['space_w'] - row['space_b']
        delta_mob = row['mob_w'] - row['mob_b']
        delta_ks = row['ks_w'] - row['ks_b']
        delta_threats = (row['threats_w'] - row['threats_b']) / threats_divisor
        
        ft_white = delta_space + delta_mob + delta_ks + delta_threats
        
        result.append({
            'ply': row['ply'],
            'san': row['san'],
            'eval_cp': row['eval_cp'],
            'ft_white': ft_white,
            'ft_black': -ft_white,
            'components': {
                'space': delta_space,
                'mobility': delta_mob,
                'king_safety': delta_ks,
                'threats': delta_threats,
            }
        })
    
    return result


def get_game_count_in_tex(tex_filepath: str) -> int:
    """
    Return the number of games with raw data in a .tex file.
    
    Args:
        tex_filepath: Path to the .tex file
        
    Returns:
        Number of GAME_DATA blocks found
    """
    with open(tex_filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    import re
    matches = re.findall(r'% GAME_DATA_START', content)
    return len(matches)


# =============================================================================
# FIRETEAM INDEX PREDICTION ALGORITHMS
# =============================================================================

@dataclass
class PredictionResult:
    """
    Result of a win prediction algorithm.
    
    Attributes:
        prediction: "WIN" or "DRAW" (from player's perspective)
        player_color: "W" or "B"
        player_name: Optional name of the player being analyzed
        threshold_crossed_ply: Ply where the streak threshold was crossed (or None)
        max_streak: Maximum consecutive plies with positive index
        max_streak_start_ply: Starting ply of the maximum streak
        peak_ft_value: Maximum Fireteam Index value achieved
        peak_ft_ply: Ply where peak value occurred
        ft_values: List of (ply, ft_value) tuples for plotting
        algorithm: Name of the algorithm used ("per_ply" or "windowed")
        parameters: Dict of parameters used (cutoff, streak_length, epsilon, etc.)
    """
    prediction: str
    player_color: str
    player_name: Optional[str]
    threshold_crossed_ply: Optional[int]
    max_streak: int
    max_streak_start_ply: Optional[int]
    peak_ft_value: float
    peak_ft_ply: Optional[int]
    ft_values: List[Tuple[int, float]]
    algorithm: str
    parameters: Dict


def predict_outcome_per_ply(
    data: List[Dict],
    player_color: str,
    player_name: Optional[str] = None,
    cutoff_ply: int = 16,
    streak_length: int = 10,
    epsilon: float = 0.0,
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.1)
) -> PredictionResult:
    """
    Predict WIN if the specified player achieves a positive Fireteam Index
    for streak_length consecutive plies after the opening cutoff.
    
    This is the "per-ply" algorithm: no smoothing, just raw deltas with streak detection.
    
    Args:
        data: List of dicts from parse_raw_positional_data()
        player_color: "W" or "B" - which player's perspective to use
        player_name: Optional name for display (e.g., "Berliner", "Caruana")
        cutoff_ply: Ignore plies <= this value (opening phase)
        streak_length: Number of consecutive plies required (default 10 = 5 full moves)
        epsilon: Margin threshold; FT must be > epsilon (default 0.0)
        weights: Tuple (w_space, w_mobility, w_king_safety, w_threats)
                 Default (1.0, 1.0, 1.0, 0.1) - threats are downweighted
    
    Returns:
        PredictionResult with prediction and diagnostic info
    """
    w_s, w_m, w_k, w_t = weights
    
    streak = 0
    max_streak = 0
    max_streak_start = None
    current_streak_start = None
    
    threshold_crossed_ply = None
    peak_ft = float('-inf')
    peak_ply = None
    ft_values = []
    
    for row in data:
        ply = row['ply']
        
        # Skip if missing data
        if any(row[k] is None for k in ['space_w', 'space_b', 'mob_w', 'mob_b',
                                         'ks_w', 'ks_b', 'threats_w', 'threats_b']):
            continue
        
        # Compute deltas (White - Black)
        dS = row['space_w'] - row['space_b']
        dM = row['mob_w'] - row['mob_b']
        dKS = row['ks_w'] - row['ks_b']
        dT = row['threats_w'] - row['threats_b']
        
        # Flip to player's perspective if Black
        if player_color.upper() == "B":
            dS, dM, dKS, dT = -dS, -dM, -dKS, -dT
        
        # Weighted combination
        F = w_s * dS + w_m * dM + w_k * dKS + w_t * dT
        
        ft_values.append((ply, F))
        
        # Track peak
        if F > peak_ft:
            peak_ft = F
            peak_ply = ply
        
        # Skip opening phase for streak detection
        if ply <= cutoff_ply:
            continue
        
        # Streak logic
        if F > epsilon:
            if streak == 0:
                current_streak_start = ply
            streak += 1
            
            if streak > max_streak:
                max_streak = streak
                max_streak_start = current_streak_start
            
            if streak >= streak_length and threshold_crossed_ply is None:
                threshold_crossed_ply = ply
        else:
            streak = 0
    
    prediction = "WIN" if threshold_crossed_ply is not None else "DRAW"
    
    return PredictionResult(
        prediction=prediction,
        player_color=player_color.upper(),
        player_name=player_name,
        threshold_crossed_ply=threshold_crossed_ply,
        max_streak=max_streak,
        max_streak_start_ply=max_streak_start,
        peak_ft_value=peak_ft if peak_ft != float('-inf') else 0.0,
        peak_ft_ply=peak_ply,
        ft_values=ft_values,
        algorithm="per_ply",
        parameters={
            'cutoff_ply': cutoff_ply,
            'streak_length': streak_length,
            'epsilon': epsilon,
            'weights': weights,
        }
    )


def predict_outcome_windowed(
    data: List[Dict],
    player_color: str,
    player_name: Optional[str] = None,
    cutoff_ply: int = 16,
    window_size: int = 10,
    streak_length: int = 10,
    epsilon: float = 0.0,
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.1)
) -> PredictionResult:
    """
    Predict WIN using windowed (smoothed) averaging of the Fireteam Index.
    
    This algorithm computes a rolling average over window_size plies, then
    applies the same streak logic to the smoothed values. This reduces noise
    and captures sustained positional advantage.
    
    Args:
        data: List of dicts from parse_raw_positional_data()
        player_color: "W" or "B" - which player's perspective to use
        player_name: Optional name for display
        cutoff_ply: Ignore plies <= this value (opening phase)
        window_size: Number of plies to average over (default 10 = 5 full moves)
        streak_length: Number of consecutive plies required (default 10)
        epsilon: Margin threshold; averaged FT must be > epsilon
        weights: Tuple (w_space, w_mobility, w_king_safety, w_threats)
    
    Returns:
        PredictionResult with prediction and diagnostic info
    """
    from collections import deque
    
    w_s, w_m, w_k, w_t = weights
    
    # Rolling window for FT values
    ft_window = deque()
    ft_sum = 0.0
    
    streak = 0
    max_streak = 0
    max_streak_start = None
    current_streak_start = None
    
    threshold_crossed_ply = None
    peak_ft = float('-inf')
    peak_ply = None
    ft_values = []  # Store (ply, smoothed_ft) for plotting
    
    for row in data:
        ply = row['ply']
        
        # Skip if missing data
        if any(row[k] is None for k in ['space_w', 'space_b', 'mob_w', 'mob_b',
                                         'ks_w', 'ks_b', 'threats_w', 'threats_b']):
            continue
        
        # Compute deltas
        dS = row['space_w'] - row['space_b']
        dM = row['mob_w'] - row['mob_b']
        dKS = row['ks_w'] - row['ks_b']
        dT = row['threats_w'] - row['threats_b']
        
        # Flip to player's perspective if Black
        if player_color.upper() == "B":
            dS, dM, dKS, dT = -dS, -dM, -dKS, -dT
        
        # Weighted combination (per-ply value)
        F_raw = w_s * dS + w_m * dM + w_k * dKS + w_t * dT
        
        # Add to rolling window
        ft_window.append(F_raw)
        ft_sum += F_raw
        
        # Remove oldest if window too large
        if len(ft_window) > window_size:
            ft_sum -= ft_window.popleft()
        
        # Don't evaluate until window is full
        if len(ft_window) < window_size:
            continue
        
        # Compute windowed average
        F = ft_sum / window_size
        
        ft_values.append((ply, F))
        
        # Track peak
        if F > peak_ft:
            peak_ft = F
            peak_ply = ply
        
        # Skip opening phase for streak detection
        if ply <= cutoff_ply:
            continue
        
        # Streak logic on smoothed value
        if F > epsilon:
            if streak == 0:
                current_streak_start = ply
            streak += 1
            
            if streak > max_streak:
                max_streak = streak
                max_streak_start = current_streak_start
            
            if streak >= streak_length and threshold_crossed_ply is None:
                threshold_crossed_ply = ply
        else:
            streak = 0
    
    prediction = "WIN" if threshold_crossed_ply is not None else "DRAW"
    
    return PredictionResult(
        prediction=prediction,
        player_color=player_color.upper(),
        player_name=player_name,
        threshold_crossed_ply=threshold_crossed_ply,
        max_streak=max_streak,
        max_streak_start_ply=max_streak_start,
        peak_ft_value=peak_ft if peak_ft != float('-inf') else 0.0,
        peak_ft_ply=peak_ply,
        ft_values=ft_values,
        algorithm="windowed",
        parameters={
            'cutoff_ply': cutoff_ply,
            'window_size': window_size,
            'streak_length': streak_length,
            'epsilon': epsilon,
            'weights': weights,
        }
    )


def compute_fireteam_index_for_analysis(
    analysis: 'EnhancedGameAnalysisResult',
    threats_divisor: float = 10.0
) -> List[Dict]:
    """
    Compute Fireteam Index directly from an EnhancedGameAnalysisResult object.
    
    This is a convenience function that extracts positional data from the
    analysis result without needing to write/read a .tex file.
    
    Args:
        analysis: EnhancedGameAnalysisResult from the analyzer
        threats_divisor: Divisor for threats term (default 10.0)
    
    Returns:
        List of dicts suitable for predict_outcome_* functions
    """
    data = []
    
    for move in analysis.moves:
        if move.positional_eval is None:
            continue
        
        pe = move.positional_eval
        data.append({
            'ply': move.ply,
            'san': move.move_san,
            'eval_cp': int(move.eval_after) if move.eval_after else None,
            'space_w': pe.space_white,
            'space_b': pe.space_black,
            'mob_w': pe.mobility_white,
            'mob_b': pe.mobility_black,
            'ks_w': pe.king_safety_white,
            'ks_b': pe.king_safety_black,
            'threats_w': pe.threats_white,
            'threats_b': pe.threats_black,
        })
    
    return data


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze chess games with detailed positional metrics and plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with positional analysis and plots (single game)
  python chess_game_analyzer.py game.pgn -o analysis.tex
  
  # Analyze all games in PGN as a book (multiple games)
  python chess_game_analyzer.py games.pgn -o analysis.tex --book --book-title "My Games"
  
  # With ASCII plots only (no matplotlib needed)
  python chess_game_analyzer.py game.pgn -o analysis.tex --ascii-plots --no-plots
  
  # Both matplotlib and ASCII plots
  python chess_game_analyzer.py game.pgn -o analysis.tex --ascii-plots
  
  # Skip methodology explanation
  python chess_game_analyzer.py game.pgn -o analysis.tex --no-methodology
  
  # Get just the raw analysis data (JSON output)
  python chess_game_analyzer.py game.pgn --json-output analysis.json
        """
    )
    
    parser.add_argument("pgn_file", help="Path to PGN file (can contain multiple games)")
    parser.add_argument("-o", "--output", help="Output LaTeX file")
    parser.add_argument("--json-output", help="Output raw analysis as JSON")
    parser.add_argument("-s", "--stockfish", default="/usr/games/stockfish",
                       help="Path to Stockfish executable")
    parser.add_argument("-d", "--depth", type=int, default=20,
                       help="Analysis depth (default: 20)")
    parser.add_argument("-t", "--time", type=float, default=1.0,
                       help="Time per position in seconds (default: 1.0)")
    parser.add_argument("--no-diagrams", action="store_true",
                       help="Don't include position diagrams")
    parser.add_argument("--no-methodology", action="store_true",
                       help="Don't include methodology explanation")
    parser.add_argument("--no-plots", action="store_true",
                       help="Don't include matplotlib plots")
    parser.add_argument("--ascii-plots", action="store_true",
                       help="Include ASCII plots in verbatim environment")
    parser.add_argument("--plot-dir", default=None,
                       help="Directory for plot output files")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Suppress progress messages")
    
    # Fireteam Index prediction options (single game mode)
    parser.add_argument("--prediction", choices=['W', 'B'],
                       help="(Single game) Include Fireteam Index prediction for specified player (W=White, B=Black)")
    
    # Fireteam Index prediction options (book mode)
    parser.add_argument("--prediction-winner", action="store_true", default=True,
                       help="(Book mode) Track winner in decisive games (default: True)")
    parser.add_argument("--no-prediction", action="store_true",
                       help="Disable Fireteam Index prediction section entirely")
    parser.add_argument("--prediction-name", default=None,
                       help="Track specific player by name (e.g., 'Berliner') - matches against White/Black names")
    
    # Multi-game book options
    parser.add_argument("--book", action="store_true",
                       help="Analyze all games in PGN and generate a book (LaTeX book class)")
    parser.add_argument("--book-title", default="Chess Game Collection Analysis",
                       help="Title for the book (used with --book)")
    parser.add_argument("--book-author", default=None,
                       help="Author for the book (used with --book)")
    
    args = parser.parse_args()
    
    plot_dir = args.plot_dir or (os.path.dirname(args.output) if args.output else ".") or "."
    
    if args.book:
        # Multi-game book mode
        with EnhancedGameAnalyzer(args.stockfish, args.depth, args.time) as analyzer:
            if not args.quiet:
                print(f"Analyzing all games with {analyzer.engine_version}...")
            
            analyses = analyzer.analyze_all_games(args.pgn_file, verbose=not args.quiet)
            
            if not analyses:
                print("No games found in PGN file")
                return
            
            if not args.quiet:
                total_time = sum(a.analysis_time for a in analyses)
                print(f"All {len(analyses)} games analyzed in {total_time:.1f}s total")
        
        # Output LaTeX book
        if args.output:
            latex_content = EnhancedLaTeXReportGenerator.generate_book_report(
                analyses,
                book_title=args.book_title,
                author=args.book_author,
                include_diagrams=not args.no_diagrams,
                include_methodology=not args.no_methodology,
                include_plots=not args.no_plots,
                include_ascii_plots=args.ascii_plots,
                include_prediction=not args.no_prediction,
                prediction_name=args.prediction_name,
                prediction_winner=args.prediction_winner,
                plot_output_dir=plot_dir,
                verbose=not args.quiet
            )
            
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            if not args.quiet:
                print(f"LaTeX book saved to: {args.output}")
        
        # Output JSON (list of all analyses)
        if args.json_output:
            import json
            from dataclasses import asdict
            
            def to_dict(obj):
                if hasattr(obj, '__dataclass_fields__'):
                    return {k: to_dict(v) for k, v in asdict(obj).items()}
                elif isinstance(obj, list):
                    return [to_dict(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: to_dict(v) for k, v in obj.items()}
                return obj
            
            with open(args.json_output, 'w') as f:
                json.dump([to_dict(a) for a in analyses], f, indent=2)
            
            if not args.quiet:
                print(f"JSON output saved to: {args.json_output}")
        
        if not args.output and not args.json_output:
            # Print summary to stdout
            print(f"\nAnalyzed {len(analyses)} games:")
            for i, analysis in enumerate(analyses, 1):
                print(f"\n  Game {i}: {analysis.white} vs {analysis.black}")
                print(f"    Result: {analysis.result}")
                print(f"    White accuracy: {analysis.white_stats['accuracy']:.1f}%")
                print(f"    Black accuracy: {analysis.black_stats['accuracy']:.1f}%")
    
    else:
        # Single game mode (original behavior)
        with EnhancedGameAnalyzer(args.stockfish, args.depth, args.time) as analyzer:
            if not args.quiet:
                print(f"Analyzing with {analyzer.engine_version}...")
            
            analysis = analyzer.analyze_game(args.pgn_file)
            
            if not args.quiet:
                print(f"Analysis complete in {analysis.analysis_time:.1f}s")
        
        # Output LaTeX
        if args.output:
            latex_content = EnhancedLaTeXReportGenerator.generate_report(
                analysis,
                include_diagrams=not args.no_diagrams,
                include_methodology=not args.no_methodology,
                include_plots=not args.no_plots,
                include_ascii_plots=args.ascii_plots,
                include_prediction=args.prediction is not None,
                prediction_player_color=args.prediction,
                prediction_player_name=args.prediction_name,
                plot_output_dir=plot_dir
            )
            
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            if not args.quiet:
                print(f"LaTeX report saved to: {args.output}")
        
        # Output JSON
        if args.json_output:
            import json
            from dataclasses import asdict
            
            def to_dict(obj):
                if hasattr(obj, '__dataclass_fields__'):
                    return {k: to_dict(v) for k, v in asdict(obj).items()}
                elif isinstance(obj, list):
                    return [to_dict(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: to_dict(v) for k, v in obj.items()}
                return obj
            
            with open(args.json_output, 'w') as f:
                json.dump(to_dict(analysis), f, indent=2)
            
            if not args.quiet:
                print(f"JSON output saved to: {args.json_output}")
        
        if not args.output and not args.json_output:
            # Print summary to stdout
            print(f"\n{analysis.white} vs {analysis.black}")
            print(f"Result: {analysis.result}")
            print(f"\nWhite: {analysis.white_stats['accuracy']:.1f}% accuracy, "
                  f"avg space {analysis.white_stats.get('avg_space', 0):.2f}")
            print(f"Black: {analysis.black_stats['accuracy']:.1f}% accuracy, "
                  f"avg space {analysis.black_stats.get('avg_space', 0):.2f}")


if __name__ == "__main__":
    main()

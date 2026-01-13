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
        EnhancedMoveAnalysis
    )
    
    # Get analysis with full positional breakdown and plots
    game_pgn = "../wdj-games/james-rizzitano-vs-wdj-sussex-open-2026-01-10.pgn"
    report_output = "../wdj-games/james-rizzitano-vs-wdj-sussex-open-2026-01-10-analysis4.tex"
    result = analyze_game_with_positional_metrics(
        pgn_source=game_pgn,
        output_path=report_output,
        include_plots=True,        # matplotlib plots
        include_ascii_plots=False,   # ASCII verbatim plots
        plot_output_dir = "../wdj-games/plots/"
    )
    for move in result.moves:
        pos = move.positional_eval
        print(f"Move {move.ply}: Space W={pos.space_white:.2f} B={pos.space_black:.2f}")
        print(f"         Mobility W={pos.mobility_white:.2f} B={pos.mobility_black:.2f}")

    >>> from chess_game_analyzer import (
    ...         analyze_game_with_positional_metrics,
    ...         PositionalEvaluation,
    ...         EnhancedMoveAnalysis
    ...     )
    >>> game_pgn = "../wdj-games/Joyner-vs-Goodson_2023-05-16.pgn"
    >>> report_output = "../wdj-games/Joyner-vs-Goodson_2023-05-16-analysis.tex"
    >>> result = analyze_game_with_positional_metrics(pgn_source=game_pgn,output_path=report_output,include_plots=True,include_ascii_plots=False, plot_output_dir = "../wdj-games/plots/")

Author: Generated for David Joyner's chess analysis pipeline, 2026-01-12
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
                
                # A. Analyze BEFORE push to get Best Move and correct SAN strings
                info_before = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
                best_move = info_before.get('pv', [None])[0]
                
                # Capture SAN strings while it's still the moving player's turn
                played_san = board.san(node.move)
                best_san = board.san(best_move) if best_move else "-"
                is_capture = board.is_capture(node.move)
                best_eval = self._eval_to_cp(info_before['score'])
                
                # B. Execute the move
                move = node.move
                board.push(move)
                
                # C. Analyze AFTER push
                info_after = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
                current_eval = self._eval_to_cp(info_after['score'])
                current_material = self._calculate_material(board)
                eval_loss = abs(best_eval - current_eval)
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
                    pv_line=pv_san, positional_eval=pos_eval
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
                        is_biggest_swing=False, eval_swing=eval_swing
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
                    'eval_swing': eval_swing
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
                        eval_swing=swing_data['eval_swing']
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
                white_elo=int(game.headers.get("WhiteElo", 0)) or None,
                black_elo=int(game.headers.get("BlackElo", 0)) or None,
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
                analysis_depth=self.depth,
                analysis_time=time.time() - start_time,
                engine_version=self.engine_version
            )
            
        finally:
            if isinstance(pgn_source, str) and not ('\n' in pgn_source or pgn_source.startswith('[')):
                pgn_io.close()

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
            return {'total_moves': 0, 'accuracy': 0, 'avg_centipawn_loss': 0}
        
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
                current_line = f"{move_num}. {move.move_san}"
            else:
                if current_line:
                    current_line += f" {move.move_san}"
                else:
                    current_line = f"{move_num}...{move.move_san}"
            
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
                comment = f"{player} loses {move.eval_loss:.0f}cp. Better was {esc(move.best_move_san)}."
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
                r"\section{Critical Positions}\par This section lists the positions with the biggest evaluation swing ($\bigstar$) and those with evaluation jump $>100$ (if any).",
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
                
                if pos.best_continuation:
                    cont_str = " ".join(esc(m) for m in pos.best_continuation[:5])
                    lines.append(rf"Best continuation: {cont_str}")
                
                lines.append(r"")
        
        # Analysis Metadata
        lines.extend([
            r"\section{Analysis Information}",
            r"\begin{itemize}",
            rf"\item Engine: {esc(analysis.engine_version)}",
            rf"\item Depth: {analysis.analysis_depth}",
            rf"\item Analysis time: {analysis.analysis_time:.1f} seconds",
            r"\item Positional metrics extracted from classical evaluation",
            r"\end{itemize}",
            r"",
            r"\end{document}",
        ])
        
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
        ]


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
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze a chess game with detailed positional metrics and plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with positional analysis and plots
  python chess_game_analyzer5.py game.pgn -o analysis.tex
  
  # With ASCII plots only (no matplotlib needed)
  python chess_game_analyzer5.py game.pgn -o analysis.tex --ascii-plots --no-plots
  
  # Both matplotlib and ASCII plots
  python chess_game_analyzer5.py game.pgn -o analysis.tex --ascii-plots
  
  # Skip methodology explanation
  python chess_game_analyzer5.py game.pgn -o analysis.tex --no-methodology
  
  # Get just the raw analysis data (JSON output)
  python chess_game_analyzer5.py game.pgn --json-output analysis.json
        """
    )
    
    parser.add_argument("pgn_file", help="Path to PGN file")
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
    
    args = parser.parse_args()
    
    # Run analysis
    with EnhancedGameAnalyzer(args.stockfish, args.depth, args.time) as analyzer:
        if not args.quiet:
            print(f"Analyzing with {analyzer.engine_version}...")
        
        analysis = analyzer.analyze_game(args.pgn_file)
        
        if not args.quiet:
            print(f"Analysis complete in {analysis.analysis_time:.1f}s")
    
    # Output LaTeX
    if args.output:
        plot_dir = args.plot_dir or os.path.dirname(args.output) or "."
        
        latex_content = EnhancedLaTeXReportGenerator.generate_report(
            analysis,
            include_diagrams=not args.no_diagrams,
            include_methodology=not args.no_methodology,
            include_plots=not args.no_plots,
            include_ascii_plots=args.ascii_plots,
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
        
        # Convert to dict (handling dataclasses)
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

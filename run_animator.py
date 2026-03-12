"""
run_animator.py

CLI coordinator for the chess game video animator.
Writes a small JSON config file, sets CHESS_ANIMATOR_CONFIG in the
environment, then delegates to Manim.

Usage:
    python run_animator.py <game_id> [options]

    <game_id> is the base filename without extension.  The script looks for:
        {game_id}.pgn              — required for live analysis fallback
        {game_id}_analysis.json   — pre-computed analysis (preferred)
        {game_id}_notes.txt       — optional human commentary

    If none of those files exist the script exits with a clear error rather
    than letting Manim fail cryptically.

Options:
    --quality   low | medium | high | ultra   (default: low)
                Maps to Manim's -pql / -pqm / -pqh / -pqk flags.
    --scene     Manim scene class name         (default: AnimatedGame)
    --no-preview                               Don't open the video after render.
    --analyze   Run Stockfish analysis first, saving {game_id}_analysis.json,
                then animate.  Requires chess_game_analyzer6y.py on the path.
    --depth N   Stockfish search depth for --analyze  (default: 20)
    --stockfish PATH  Path to Stockfish binary  (default: /usr/local/bin/stockfish)

Examples:
    # Fast preview render (low quality)
    python run_animator.py sample_game

    # High-quality final render
    python run_animator.py sample_game --quality high

    # Analyze then animate in one step
    python run_animator.py sample_game --analyze --depth 22

    # Render the QuickDemo scene (no game files needed)
    python run_animator.py --scene QuickDemo
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Quality flag mapping
# ---------------------------------------------------------------------------

QUALITY_FLAGS = {
    "low":   "-pql",
    "medium": "-pqm",
    "high":  "-pqh",
    "ultra": "-pqk",
}


# ---------------------------------------------------------------------------
# Analysis helper
# ---------------------------------------------------------------------------

def run_analysis(pgn_path: Path, output_path: Path,
                 stockfish: str, depth: int) -> bool:
    """
    Run chess_game_analyzer on pgn_path and save JSON to output_path.
    Returns True on success.

    Deliberately does NOT import anything from animator_game so that
    manim / manim_chess are never touched during the analysis step.
    """
    print(f"Running Stockfish analysis (depth {depth}) on {pgn_path} …")
    try:
        from chess_game_analyzer import EnhancedGameAnalyzer
    except ImportError:
        print("Error: chess_game_analyzer.py not found on Python path.")
        return False

    try:
        with EnhancedGameAnalyzer(stockfish, depth) as analyzer:
            result = analyzer.analyze_game(str(pgn_path))
    except Exception as exc:
        print(f"Error during analysis: {exc}")
        return False

    # Constants duplicated here so we never need to import animator_game
    THREATS_SCALE_FACTOR = 20.0
    FTI1_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
    FTI2_WEIGHTS = (0.60, 0.10, 0.00, 0.30)
    FTI3_WEIGHTS = (0.70, 0.10, 0.10, 0.10)

    def compute_fti(sa, ma, ksa, ta, w):
        return w[0]*sa + w[1]*ma + w[2]*ksa + w[3]*ta

    try:
        moves_out = []
        for m in result.moves:
            pe = m.positional_eval
            sw  = float(pe.space_white       if pe else 0.0)
            sb  = float(pe.space_black       if pe else 0.0)
            mw  = float(pe.mobility_white    if pe else 0.0)
            mb  = float(pe.mobility_black    if pe else 0.0)
            ksw = float(pe.king_safety_white if pe else 0.0)
            ksb = float(pe.king_safety_black if pe else 0.0)
            tw  = float(pe.threats_white     if pe else 0.0)
            tb  = float(pe.threats_black     if pe else 0.0)

            sa  = sw  - sb
            ma  = mw  - mb
            ksa = ksw - ksb
            ta  = (tw - tb) / THREATS_SCALE_FACTOR

            moves_out.append({
                "ply":              m.ply,
                "move_san":         m.move_san,
                "move_uci":         m.move_uci,
                "is_white_move":    m.is_white_move,
                "eval_before":      float(m.eval_before),
                "eval_after":       float(m.eval_after),
                "eval_loss":        float(m.eval_loss),
                "classification":   m.classification,
                "best_move_san":    m.best_move_san or "",
                "is_capture":       m.is_capture,
                "is_check":         m.is_check,
                "pv_line":          m.pv_line or [],
                "space_white":      sw,
                "space_black":      sb,
                "mobility_white":   mw,
                "mobility_black":   mb,
                "king_safety_white": ksw,
                "king_safety_black": ksb,
                "threats_white":    tw,
                "threats_black":    tb,
                "fti1": compute_fti(sa, ma, ksa, ta, FTI1_WEIGHTS),
                "fti2": compute_fti(sa, ma, ksa, ta, FTI2_WEIGHTS),
                "fti3": compute_fti(sa, ma, ksa, ta, FTI3_WEIGHTS),
            })

        data = {
            "white":        result.white,
            "black":        result.black,
            "white_elo":    str(result.white_elo or ""),
            "black_elo":    str(result.black_elo or ""),
            "event":        result.event or "",
            "site":         result.site or "",
            "date":         result.date or "",
            "round_num":    result.round_num or "",
            "result":       result.result or "",
            "opening_name": result.opening_name or "",
            "opening_eco":  result.opening_eco or "",
            "moves":        moves_out,
            "white_stats":  {"accuracy": result.white_stats.get("accuracy", 0.0)},
            "black_stats":  {"accuracy": result.black_stats.get("accuracy", 0.0)},
        }

        output_path.write_text(json.dumps(data, indent=2))
        print(f"Analysis saved to {output_path}")
        return True

    except Exception as exc:
        print(f"Error saving analysis JSON: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render a chess game animation via Manim.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "game_id", nargs="?", default=None,
        help="Base filename without extension (e.g. 'my_game').",
    )
    parser.add_argument(
        "--quality", choices=QUALITY_FLAGS.keys(), default="low",
        help="Render quality (default: low).",
    )
    parser.add_argument(
        "--scene", default="AnimatedGame",
        help="Manim scene class to render (default: AnimatedGame).",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Don't open the video after rendering.",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run Stockfish analysis before animating.",
    )
    parser.add_argument(
        "--depth", type=int, default=20,
        help="Stockfish depth for --analyze (default: 20).",
    )
    parser.add_argument(
        "--stockfish", default="/usr/local/bin/stockfish",
        help="Path to Stockfish binary (default: /usr/local/bin/stockfish).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Special case: scene that needs no game files (e.g. QuickDemo)
    # ------------------------------------------------------------------
    if args.scene != "AnimatedGame":
        quality_flag = QUALITY_FLAGS[args.quality]
        if args.no_preview:
            quality_flag = quality_flag.replace("-p", "-")  # drop preview flag
        cmd = ["manim", quality_flag, "animator_game.py", args.scene]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        return

    # ------------------------------------------------------------------
    # AnimatedGame requires a game_id
    # ------------------------------------------------------------------
    if not args.game_id:
        parser.error("game_id is required when rendering AnimatedGame.")

    game_id = args.game_id
    pgn_path      = Path(f"{game_id}.pgn")
    analysis_path = Path(f"{game_id}_analysis.json")
    notes_path    = Path(f"{game_id}_notes.txt")

    # ------------------------------------------------------------------
    # Optional: run analysis first
    # ------------------------------------------------------------------
    if args.analyze:
        if not pgn_path.exists():
            print(f"Error: {pgn_path} not found — cannot run analysis.")
            sys.exit(1)
        ok = run_analysis(pgn_path, analysis_path, args.stockfish, args.depth)
        if not ok:
            sys.exit(1)

    # ------------------------------------------------------------------
    # Validate that at least one data source exists
    # ------------------------------------------------------------------
    has_analysis = analysis_path.exists()
    has_pgn      = pgn_path.exists()

    if not has_analysis and not has_pgn:
        print(f"Error: neither {analysis_path} nor {pgn_path} found.")
        print("       Run with --analyze to generate the analysis JSON first,")
        print("       or place the PGN in the current directory.")
        sys.exit(1)

    if not has_analysis:
        print(f"Note: {analysis_path} not found — AnimatedGame will run live "
              f"Stockfish analysis from {pgn_path}.")
        print("      This is slow. Consider running with --analyze first.")

    # ------------------------------------------------------------------
    # Write the config JSON and set the environment variable
    # ------------------------------------------------------------------
    config = {
        "pgn_path":      str(pgn_path)      if has_pgn      else None,
        "analysis_path": str(analysis_path) if has_analysis else None,
        "comments_path": str(notes_path)    if notes_path.exists() else None,
        "stockfish_path": args.stockfish,
    }

    config_file = Path(f"{game_id}_animator_config.json")
    config_file.write_text(json.dumps(config, indent=2))
    print(f"Config written to {config_file}")

    # Pass the config path to AnimatedGame via environment variable
    env = os.environ.copy()
    env["CHESS_ANIMATOR_CONFIG"] = str(config_file)

    # ------------------------------------------------------------------
    # Build and run the Manim command
    # ------------------------------------------------------------------
    quality_flag = QUALITY_FLAGS[args.quality]
    if args.no_preview:
        quality_flag = quality_flag.replace("-p", "-")

    cmd = ["manim", quality_flag, "animator_game.py", args.scene]
    print(f"Running: {' '.join(cmd)}")
    print(f"  CHESS_ANIMATOR_CONFIG={config_file}")
    print()

    result = subprocess.run(cmd, env=env)

    # ------------------------------------------------------------------
    # Clean up the ephemeral config file
    # ------------------------------------------------------------------
    try:
        config_file.unlink()
    except OSError:
        pass  # non-fatal

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

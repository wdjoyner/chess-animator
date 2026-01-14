# Chess Game Analyzer

A Python tool for deep analysis of chess games, generating professional LaTeX reports with positional metrics, evaluation graphs, and game character classification.

## Features

- **Stockfish Integration**: Move-by-move centipawn evaluation with configurable depth
- **Positional Metrics**: Space control, piece mobility, king safety, and threats computed directly from board position
- **Game Character Classification**: Automatic categorization of games as balanced, tense, tactical, or chaotic
- **Brilliant Sacrifice Detection**: Identifies and highlights spectacular sacrifices
- **Critical Position Identification**: Marks turning points and biggest evaluation swings
- **LaTeX Report Generation**: Publication-ready reports with chess diagrams and plots
- **Multi-Game Book Support**: Analyze entire PGN databases and generate book-format documents
- **Visualization**: Matplotlib plots and ASCII graphs showing how metrics evolve

## Installation

### Requirements

- Python 3.8+
- Stockfish chess engine

### Python Dependencies

```bash
pip install python-chess matplotlib
```

### Installing Stockfish

**Ubuntu/Debian:**
```bash
sudo apt install stockfish
```

**macOS:**
```bash
brew install stockfish
```

**Windows:**
Download from [stockfishchess.org](https://stockfishchess.org/download/)

### LaTeX Requirements (for PDF generation)

The generated `.tex` files require:
- `xskak` and `chessboard` packages for chess diagrams
- `pgfplots` for evaluation graphs
- Standard packages: `booktabs`, `longtable`, `hyperref`, `xcolor`, `graphicx`

## Quick Start

### Command Line

```bash
# Analyze a single game
python chess_game_analyzer6f.py game.pgn -o analysis.tex

# Analyze multiple games as a book
python chess_game_analyzer6f.py games.pgn -o book.tex --book --book-title "My Games"

# Export raw data as JSON
python chess_game_analyzer6f.py game.pgn --json-output analysis.json
```

### Python API

```python
from chess_game_analyzer6f import (
    analyze_game_with_positional_metrics,
    classify_game_character
)

# Analyze a game
result = analyze_game_with_positional_metrics(
    pgn_source="game.pgn",
    output_path="analysis.tex",
    include_plots=True,
    plot_output_dir="./plots/"
)

# Access results
print(f"White accuracy: {result.white_stats['accuracy']:.1f}%")
print(f"Black accuracy: {result.black_stats['accuracy']:.1f}%")

# Game character classification
gc = result.game_character
print(f"Game type: {gc['spread_class']} ({gc['direction_class']})")
print(f"Evaluation range: {gc['m1']:+.2f} to {gc['m2']:+.2f}")
```

## Positional Metrics

### Space
Measures control of central territory (files c-f), weighted by:
- Squares attacked by friendly pawns
- Pieces positioned to exploit the space
- Scaled by total piece count (more relevant in closed positions)

### Mobility
Counts safe squares available to each piece type:
- **Knights**: Squares not attacked by enemy pawns
- **Bishops**: Diagonal squares with bonus for long diagonals
- **Rooks**: File/rank squares with bonus for open files and 7th rank
- **Queens**: Combined bishop and rook mobility

### King Safety
Evaluates defensive structure:
- Pawn shield strength
- Enemy piece proximity (king tropism)
- Attack units near the king
- Available safe checks for opponent

### Threats
Tactical tension assessment:
- Hanging pieces (attacked but undefended)
- Pieces attacked by lower-value pieces
- King zone pressure
- Weak squares in pawn structure

## Game Character Classification

Games are automatically classified based on evaluation volatility:

| Spread (d) | Classification | Description |
|------------|----------------|-------------|
| d < 1 | **Balanced** | Minimal advantage shifts; controlled play |
| 1 ≤ d < 3 | **Tense** | Normal competitive tension |
| 3 ≤ d < 6 | **Tactical** | Significant swings; complications |
| d ≥ 6 | **Chaotic** | Wild swings; likely blunders |

Where:
- `m₂` = maximum evaluation for White
- `m₁` = minimum evaluation for White  
- `d = m₂ - m₁` (the spread)

### Directionality

- **One-sided**: `m₁ > -0.5` or `m₂ < 0.5` (one side dominated)
- **Seesaw**: `m₁ < -1` and `m₂ > 1` (advantage changed hands)

## Command Line Options

```
usage: chess_game_analyzer6f.py [-h] [-o OUTPUT] [--json-output JSON_OUTPUT]
                                 [-s STOCKFISH] [-d DEPTH] [-t TIME]
                                 [--no-diagrams] [--no-methodology] [--no-plots]
                                 [--ascii-plots] [--plot-dir PLOT_DIR]
                                 [--book] [--book-title BOOK_TITLE]
                                 [--book-author BOOK_AUTHOR] [-q]
                                 pgn_file

Arguments:
  pgn_file              Path to PGN file (can contain multiple games)

Options:
  -o, --output          Output LaTeX file
  --json-output         Output raw analysis as JSON
  -s, --stockfish       Path to Stockfish executable (default: /usr/games/stockfish)
  -d, --depth           Analysis depth (default: 20)
  -t, --time            Time per position in seconds (default: 1.0)
  --no-diagrams         Don't include position diagrams
  --no-methodology      Don't include methodology explanation
  --no-plots            Don't include matplotlib plots
  --ascii-plots         Include ASCII plots in verbatim environment
  --plot-dir            Directory for plot output files
  --book                Analyze all games and generate a book
  --book-title          Title for the book
  --book-author         Author for the book
  -q, --quiet           Suppress progress messages
```

## Output Examples

### Single Game Report
Generates an `article`-class LaTeX document with:
- Game information header
- Player statistics (accuracy, move classifications)
- Positional analysis with plots
- Game character classification
- Brilliant sacrifices (if any)
- Annotated game score
- Critical positions with diagrams
- Methodology appendix

### Multi-Game Book
Generates a `book`-class LaTeX document with:
- Table of contents
- Each game as a chapter
- Shared methodology appendix

## Data Classes

### `EnhancedGameAnalysisResult`
Complete analysis result containing:
- Game metadata (players, event, date, opening)
- List of `EnhancedMoveAnalysis` objects
- Player statistics
- `game_character` classification
- `positional_summary` aggregates
- Brilliant sacrifices and critical positions

### `EnhancedMoveAnalysis`
Per-move data:
- Evaluation before/after
- Best move and PV line
- Classification (best, excellent, good, inaccuracy, mistake, blunder)
- `PositionalEvaluation` breakdown

### `PositionalEvaluation`
Detailed positional metrics:
- Space (white/black)
- Mobility (middlegame/endgame weights)
- King safety
- Threats
- Pawn structure details

## License

Modified BSD or MIT License (user's choice)

## Author

Generated for David Joyner's chess analysis pipeline, 2026

## Contributing

Contributions welcome! Please ensure any changes maintain compatibility with the LaTeX output format and include appropriate tests.

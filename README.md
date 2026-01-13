# Chess Game Analyzer

A Python toolkit for analyzing chess games with detailed positional evaluation metrics and generating professional LaTeX reports with plots.

## Features

- **Move-by-move analysis** using Stockfish for centipawn evaluation
- **Positional metrics** computed directly from board position:
  - **Space**: Control of central territory
  - **Mobility**: Safe squares available to pieces
  - **King Safety**: Pawn shield strength and king zone pressure
  - **Threats**: Tactical tension including hanging pieces, attacks, and weak squares
- **LaTeX report generation** with:
  - Game metadata and player statistics
  - Annotated game score with NAG symbols
  - Critical position diagrams with `chessboard` package
  - Move-by-move plots (matplotlib PDF or ASCII)
  - Methodology explanations
- **Brilliant sacrifice detection**
- **Accuracy scoring** similar to chess.com/lichess

## Installation

### Requirements

- Python 3.8+
- Stockfish chess engine

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Stockfish

Install Stockfish for your platform:

```bash
# Ubuntu/Debian
sudo apt install stockfish

# macOS
brew install stockfish

# Windows: Download from https://stockfishchess.org/download/
```

## Quick Start

### Python API

```python
from chess_game_analyzer import analyze_game_with_positional_metrics

# Analyze a game and generate a LaTeX report
result = analyze_game_with_positional_metrics(
    pgn_source="game.pgn",
    output_path="analysis.tex",
    include_plots=True,
    include_ascii_plots=False,
    plot_output_dir="./plots/"
)

# Access move-by-move data
for move in result.moves:
    pos = move.positional_eval
    print(f"Move {move.ply}: Space W={pos.space_white:.2f} B={pos.space_black:.2f}")
    print(f"         Threats W={pos.threats_white:.2f} B={pos.threats_black:.2f}")
```

### Command Line

```bash
# Basic analysis with plots
python chess_game_analyzer.py game.pgn -o analysis.tex

# With ASCII plots (no matplotlib required)
python chess_game_analyzer.py game.pgn -o analysis.tex --ascii-plots --no-plots

# Custom Stockfish path and depth
python chess_game_analyzer.py game.pgn -o analysis.tex -s /path/to/stockfish -d 25

# JSON output for further processing
python chess_game_analyzer.py game.pgn --json-output analysis.json
```

## Positional Metrics

### Space
Counts squares in the center (files c-f, ranks 2-4 for White, 5-7 for Black) that are:
1. Attacked by at least one friendly pawn
2. Not occupied by any enemy piece

### Mobility
Counts safe squares available to each piece, excluding squares defended by enemy pawns. Weighted by piece type with bonuses for activity in enemy territory.

### King Safety
Evaluates:
- Pawn shield strength (pawns protecting the castled king)
- Enemy attacks on squares near the king

### Threats
Tactical tension computed directly from board position:
- **Hanging pieces**: Attacked but undefended pieces (weighted by piece value × 2)
- **Attacked by lower-value piece**: e.g., queen attacked by knight
- **King zone pressure**: Enemy attacks on squares near the king
- **Safe checks**: Available checking moves from safe squares
- **Weak squares (holes)**: Squares that cannot be defended by pawns
- **Check availability**: Legal moves that give check

## Output

### LaTeX Report Structure

1. **Game Information**: Event, players, date, opening
2. **Player Statistics**: Accuracy, move classifications, average positional metrics
3. **Positional Analysis**: Plots showing evaluation, space, mobility, king safety, and threats over time
4. **Methodology**: Explanation of how metrics are computed
5. **Brilliant Sacrifices**: Detected material sacrifices that maintain evaluation
6. **Annotated Game**: Full game score with annotations
7. **Critical Positions**: Diagrams of key moments with positional breakdowns

### Compiling LaTeX

The generated `.tex` file requires these LaTeX packages:
- `xskak`, `chessboard` (chess diagrams)
- `pgfplots`, `graphicx` (plots)
- `booktabs`, `longtable` (tables)
- `hyperref`, `xcolor` (formatting)

```bash
pdflatex analysis.tex
```

## Module Structure

- `chess_game_analyzer.py`: Main analysis engine and LaTeX generator
- `chess_plotting.py`: Matplotlib and ASCII plot generation

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stockfish_path` | `/usr/games/stockfish` | Path to Stockfish executable |
| `depth` | 20 | Analysis depth (higher = slower but more accurate) |
| `time_limit` | 1.0 | Seconds per position |
| `include_diagrams` | True | Include chess board diagrams |
| `include_plots` | True | Generate matplotlib plots |
| `include_ascii_plots` | False | Include ASCII plots in verbatim |
| `include_methodology` | True | Include methodology section |
| `top_n_swings` | 2 | Number of "biggest swing" positions to highlight |

## License

MIT License - see [LICENSE.txt](LICENSE.txt)

## Author

David Joyner

## Acknowledgments

- [python-chess](https://python-chess.readthedocs.io/) for chess logic
- [Stockfish](https://stockfishchess.org/) for position evaluation
- Claude (Anthropic) for development assistance

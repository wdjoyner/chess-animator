# Chess Game Analyzer (CGA) v6r

A comprehensive Python module for analyzing chess games with detailed positional metrics, generating publication-quality LaTeX reports with diagrams, and predicting game outcomes using the novel **Fireteam Index**.

## Features

- **Deep positional analysis** using Stockfish's classical evaluation terms (Space, Mobility, King Safety, Threats)
- **Publication-quality LaTeX reports** with chess diagrams, annotated moves, and metric plots
- **Multi-game book generation** with per-chapter analysis and aggregate statistics
- **Fireteam Index prediction** — a Rithmomachia-inspired formula for predicting winners
- **Game character classification** (balanced/tense/tactical/chaotic, one-sided/seesaw)
- **Brilliant sacrifice detection** and critical position highlighting
- **Raw data export** for downstream computational analysis

## Installation

### Requirements

```bash
pip install python-chess matplotlib
```

You'll also need [Stockfish](https://stockfishchess.org/download/) installed. The module looks for it at `/usr/games/stockfish` by default, but you can specify a custom path.

### Quick Start

```python
from chess_game_analyzer6r import analyze_game_with_positional_metrics

# Analyze a game and generate a LaTeX report
result = analyze_game_with_positional_metrics(
    pgn_source="my_game.pgn",
    output_path="analysis.tex",
    stockfish_path="/usr/local/bin/stockfish",
    depth=22,
    include_plots=True
)

# Compile to PDF
# pdflatex analysis.tex
```

## Command Line Usage

### Single Game Analysis

```bash
# Basic analysis with LaTeX output
python chess_game_analyzer6r.py game.pgn -o analysis.tex

# With Fireteam Index prediction (tracking White)
python chess_game_analyzer6r.py game.pgn -o analysis.tex --prediction W

# Higher depth, custom Stockfish path
python chess_game_analyzer6r.py game.pgn -o analysis.tex -d 25 -s /usr/local/bin/stockfish
```

### Multi-Game Book

```bash
# Generate a book from multiple games (prediction enabled by default)
python chess_game_analyzer6r.py games.pgn --book -o book.tex --book-title "My Tournament"

# Track a specific player across all games
python chess_game_analyzer6r.py games.pgn --book -o book.tex --prediction-name "Carlsen"

# Disable prediction section
python chess_game_analyzer6r.py games.pgn --book -o book.tex --no-prediction
```

### All Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output LaTeX file |
| `-s, --stockfish` | Path to Stockfish executable |
| `-d, --depth` | Analysis depth (default: 20) |
| `-t, --time` | Time per position in seconds (default: 1.0) |
| `--no-diagrams` | Omit chess board diagrams |
| `--no-methodology` | Omit methodology explanation section |
| `--no-plots` | Omit matplotlib plots |
| `--ascii-plots` | Include ASCII plots (for environments without graphics) |
| `--plot-dir` | Directory for plot output files |
| `--prediction W/B` | (Single game) Track specified player for Fireteam Index |
| `--prediction-name` | Track player by name (matches against White/Black) |
| `--prediction-winner` | (Book mode) Track winner in decisive games (default: True) |
| `--no-prediction` | Disable Fireteam Index prediction entirely |
| `--book` | Generate multi-game book (LaTeX book class) |
| `--book-title` | Title for the book |
| `--book-author` | Author for the book |
| `--json-output` | Export raw analysis data as JSON |
| `-q, --quiet` | Suppress progress messages |

## The Fireteam Index

Inspired by Rithmomachia's "progression victory" (a well-coordinated fireteam in enemy territory), the Fireteam Index combines four positional factors:

$$\text{FT} = \Delta\text{Space} + \Delta\text{Mobility} + \Delta\text{King Safety} + \frac{\Delta\text{Threats}}{10}$$

Where each Δ = (Your value − Opponent's value).

### Prediction Algorithms

**Per-Ply Algorithm**: Looks for 10 consecutive plies (5 full moves) where the raw FT exceeds zero after the opening (ply 16).

**Windowed Algorithm**: Uses a 10-ply rolling average to smooth noise before applying the same streak detection.

### Handling Draws

For drawn games, the Fireteam Index is analyzed from **both players' perspectives** to reveal:
- **True draw**: Neither side achieved sustained dominance
- **Missed win**: One side dominated but failed to convert
- **Mutual chances**: Both sides dominated at different points

The dual analysis helps identify whether draws were hard-fought battles or peaceful agreements.

### Usage

```python
from chess_game_analyzer6r import (
    parse_raw_positional_data,
    predict_outcome_per_ply,
    predict_outcome_windowed
)

# Parse data from a CGA-generated .tex file
data = parse_raw_positional_data("analysis.tex")

# Run predictions
result_raw = predict_outcome_per_ply(data, player_color='W', player_name='Carlsen')
result_smooth = predict_outcome_windowed(data, player_color='W', player_name='Carlsen')

print(f"Per-ply prediction: {result_raw.prediction}")
print(f"Windowed prediction: {result_smooth.prediction}")
print(f"Max streak: {result_raw.max_streak} plies")
print(f"Peak FT: {result_raw.peak_ft_value:.3f}")
```

## Positional Metrics

The module extracts four key positional factors from Stockfish's classical evaluation:

| Metric | Description |
|--------|-------------|
| **Space** | Control of central squares (files c-f, ranks 2-4/5-7) by pawns |
| **Mobility** | Legal moves available to pieces, weighted by piece type |
| **King Safety** | Pawn shield strength, king tropism, attack units near king |
| **Threats** | Hanging pieces, attacks by lower-value pieces, king zone pressure |

## Raw Data Format

Analysis data is preserved after `\end{document}` in the LaTeX file for computational reuse:

```
% GAME_DATA_START
% 1, e4, 35, 0.320, 0.390, 4.600, 4.150, 0.500, 0.380, 2.300, 1.900
% 2, e5, 19, 0.340, 0.380, 4.700, 4.100, 0.500, 0.360, 2.600, 2.000
...
% GAME_DATA_END
```

Format: `ply, SAN, eval_cp, space_w, space_b, mob_w, mob_b, ks_w, ks_b, threats_w, threats_b`

### Parsing Raw Data

```python
from chess_game_analyzer6r import parse_raw_positional_data, compute_fireteam_index

# Extract data
data = parse_raw_positional_data("analysis.tex", game_num=1)  # For multi-game books

# Compute Fireteam Index
ft_data = compute_fireteam_index(data, threats_divisor=10.0)

for row in ft_data:
    print(f"Ply {row['ply']}: FT={row['ft_white']:+.3f}")
```

## Game Character Classification

Games are automatically classified based on evaluation volatility:

| Classification | Spread (d) | Description |
|----------------|------------|-------------|
| Balanced | d < 1 | Minimal advantage shifts |
| Tense | 1 ≤ d < 3 | Normal competitive tension |
| Tactical | 3 ≤ d < 6 | Significant swings, complications |
| Chaotic | d ≥ 6 | Wild swings, likely blunders |

Directionality is also classified as **one-sided** (advantage never changed hands) or **seesaw** (advantage swung both ways).

## Version History

### v6r (Current)
- Win prediction algorithms: `predict_outcome_per_ply()` and `predict_outcome_windowed()`
- Fireteam Index prediction sections in LaTeX reports
- Book mode prediction with `--prediction-name` and `--prediction-winner`
- **Draw analysis**: Draws are now analyzed from both players' perspectives with interpretive text
- **Methodology section**: Added comprehensive FTI explanation to Appendix
- **Bug fix**: eval_loss now uses proper sign convention for Black moves
- **Bug fix**: eval_loss capped at 1500cp to prevent mate score transitions from distorting accuracy statistics

### v6q
- Raw positional data preserved after `\end{document}`
- New utility functions: `parse_raw_positional_data()`, `compute_fireteam_index()`

### Earlier Versions
- Multi-game book generation
- Game character classification
- Brilliant sacrifice detection
- Multi-PV analysis with alternative move suggestions
- Matplotlib and ASCII plot generation

## Example Output

The LaTeX reports include:

1. **Game Information** — Event, players, date, opening
2. **Player Statistics** — Accuracy, centipawn loss, move classifications
3. **Positional Analysis** — Space, mobility, king safety, threats with plots
4. **Annotated Game** — Move-by-move with error annotations
5. **Critical Positions** — Diagrams at key moments with best continuations
6. **Fireteam Index Prediction** — (Optional) Prediction results with FT evolution plot
7. **Methodology** — (Optional) Explanation of how metrics are computed

## API Reference

### Main Functions

```python
# High-level analysis
analyze_game_with_positional_metrics(pgn_source, output_path, ...)
analyze_games_to_book(pgn_source, output_path, book_title, ...)

# Prediction
predict_outcome_per_ply(data, player_color, player_name, ...)
predict_outcome_windowed(data, player_color, player_name, ...)

# Data extraction
parse_raw_positional_data(tex_filepath, game_num=None)
compute_fireteam_index(data, threats_divisor=10.0)
compute_fireteam_index_for_analysis(analysis, threats_divisor=10.0)
get_game_count_in_tex(tex_filepath)
```

### Key Classes

```python
EnhancedGameAnalyzer          # Main analysis engine
EnhancedGameAnalysisResult    # Complete analysis result
EnhancedMoveAnalysis          # Per-move analysis data
PositionalEvaluation          # Positional metrics breakdown
PredictionResult              # Fireteam Index prediction result
EnhancedLaTeXReportGenerator  # Report generation
```

## License

Modified BSD or MIT license, user's choice.

## Author

Generated for David Joyner's chess analysis pipeline, January 2026.

## Acknowledgments

- [python-chess](https://python-chess.readthedocs.io/) for chess logic and PGN parsing
- [Stockfish](https://stockfishchess.org/) for position evaluation
- The Fireteam Index concept is inspired by [Rithmomachia](https://en.wikipedia.org/wiki/Rithmomachia), the medieval "Philosophers' Game"
